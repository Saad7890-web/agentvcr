"""Record mode: capture a step, redact it, and decide which run it belongs to.

Two responsibilities (DESIGN.md §4, §11):

* **Redaction.** ``Authorization`` / ``x-api-key`` / ``api-key`` / cookie headers are
  forwarded upstream but replaced before anything touches disk. Secrets are never
  persisted — the Phase 1 acceptance check asserts this against the raw database file.
* **Run assignment.** An explicit id (``/r/<run>/…`` in the path, or the
  ``X-AgentVCR-Run`` header) wins; otherwise a request whose message list extends an
  active run's last message list chains onto that run, and anything else starts a new
  one. Runs idle for ``idle_timeout_s`` are closed.

Failed upstream calls are recorded as steps like any other. An SDK that retries a 429
therefore produces two steps, and replaying that tape positionally reproduces the same
429-then-success sequence the client already knows how to handle.

Each step that lands also gives :mod:`agentvcr.core.tools` the chance to materialize
the tool run it reports the result of, so the timeline is built as the tape is written
rather than reconstructed on read.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..providers import Provider, stable_hash
from . import tools
from .models import STATUS_COMPLETED, Run, Step
from .store import Store, utcnow

#: Header names never written to the database (DESIGN.md §11).
REDACTED_HEADERS = frozenset(
    {"authorization", "x-api-key", "api-key", "cookie", "set-cookie", "proxy-authorization"}
)
REDACTION_PLACEHOLDER = "<redacted>"

#: Headers that say nothing about the call and only add noise to a stored request.
_DROPPED_HEADERS = frozenset({"host", "content-length", "connection", "accept-encoding"})


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Lower-case header map with every credential replaced by a placeholder."""
    out: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in _DROPPED_HEADERS:
            continue
        out[lowered] = REDACTION_PLACEHOLDER if lowered in REDACTED_HEADERS else value
    return out


def message_keys(provider: Provider, body: Mapping[str, Any]) -> list[str]:
    """One stable hash per message — the unit the grouping heuristic compares."""
    return [stable_hash(m) for m in provider.messages_of(dict(body))]


def record_step(
    store: Store,
    *,
    run_id: str,
    provider: Provider,
    request_headers: Mapping[str, str],
    request_body: dict[str, Any],
    response: dict[str, Any] | None,
    status_code: int,
    latency_ms: int,
    chunks: list[bytes] | None = None,
    started_at: str | None = None,
    diverged: bool = False,
) -> Step:
    """Persist one LLM call as the next step of ``run_id``.

    Used for both a recorded call and a replayed one: a replay step holds the request
    the agent actually sent and the response the tape answered with, so the two runs
    can be diffed. ``diverged`` marks a replay step whose request drifted from the tape.
    """
    step = Step(
        run_id=run_id,
        idx=0,  # allocated under the store's write lock below
        request={"headers": redact_headers(request_headers), "body": request_body},
        response=response,
        response_chunks=b"".join(chunks) if chunks else None,
        fingerprint=provider.fingerprint(request_body),
        model=provider.model_of(request_body),
        usage=provider.usage_of(response) if response else None,
        latency_ms=latency_ms,
        status_code=status_code,
        diverged=diverged,
        started_at=started_at or utcnow(),
    )
    store.add_step(step, at_next_idx=True)
    # This request may carry the results of the previous step's tool calls; if it
    # does, that tool run becomes a row of its own (DESIGN.md §2).
    tools.materialize(store, provider=provider, step=step)
    return step


@dataclass
class _ActiveRun:
    """In-memory grouping state for a run the proxy is currently recording."""

    run_id: str
    keys: list[str] = field(default_factory=list)
    last_seen: float = 0.0


class RunRouter:
    """Decides which run an incoming request belongs to (DESIGN.md §4).

    Grouping state is per-process and advisory: losing it only costs a run boundary,
    never recorded data. The explicit path — ``agentvcr run`` or ``X-AgentVCR-Run`` —
    is what the docs recommend and what CI should use.
    """

    def __init__(self, store: Store, *, idle_timeout_s: float = 300.0, clock=time.monotonic):
        self.store = store
        self.idle_timeout_s = idle_timeout_s
        self.clock = clock
        self._active: dict[str, _ActiveRun] = {}
        #: tape run id -> the replay run recording the session replaying it.
        self._replays: dict[str, str] = {}

    # ------------------------------------------------------------------ resolving

    def resolve(
        self,
        *,
        provider: Provider,
        body: Mapping[str, Any],
        mode: str,
        run_id: str | None = None,
        upstream_url: str | None = None,
    ) -> Run:
        """The run this request belongs to, creating or reviving one as needed."""
        self.expire_idle()
        keys = message_keys(provider, body)

        if run_id is not None:
            run = self.store.get_run(run_id) or self.store.create_run(
                mode=mode, run_id=run_id, provider=provider.name, upstream_url=upstream_url
            )
            if run.provider is None:
                # `agentvcr run` creates the run before the agent has spoken, so it
                # cannot know the wire format. The first call settles it — and it has
                # to be stored, because `show` and `diff` read a run's provider to know
                # how to render what is on it.
                self.store.update_run(run.id, provider=provider.name, upstream_url=upstream_url)
                run.provider, run.upstream_url = provider.name, upstream_url
            self._track(run.id, keys)
            return run

        chained = self._chain(keys)
        if chained is not None:
            self._track(chained, keys)
            run = self.store.get_run(chained)
            if run is not None:
                return run

        run = self.store.create_run(mode=mode, provider=provider.name, upstream_url=upstream_url)
        self._track(run.id, keys)
        return run

    def replay_run_for(
        self, run: Run, *, provider: Provider, upstream_url: str | None = None
    ) -> Run:
        """The run that records this replay session (DESIGN.md §4).

        A replay is a run of its own so that divergence is recorded against it rather
        than onto the tape, and so a recording can be diffed against its own replay.
        ``agentvcr run --mode replay`` creates that run up front and points the agent at
        it; a client that instead points straight at a *recorded* run gets one created
        here on first call, for as long as the session stays warm.
        """
        if run.replay_of:
            return run
        existing = self._replays.get(run.id)
        if existing is not None:
            session = self.store.get_run(existing)
            if session is not None:
                self._track(session.id, [])
                return session
        session = self.store.create_run(
            mode="replay",
            replay_of=run.id,
            name=run.name,
            provider=provider.name,
            upstream_url=upstream_url,
        )
        self._replays[run.id] = session.id
        self._track(session.id, [])
        return session

    def _chain(self, keys: list[str]) -> str | None:
        """The active run whose last message list is the longest prefix of ``keys``."""
        best: _ActiveRun | None = None
        for active in self._active.values():
            if not active.keys:  # a replay session claims no conversation of its own
                continue
            if len(active.keys) > len(keys) or keys[: len(active.keys)] != active.keys:
                continue
            if len(active.keys) == len(keys) and not self._is_retry(active.run_id):
                continue
            if best is None or (len(active.keys), active.last_seen) > (
                len(best.keys),
                best.last_seen,
            ):
                best = active
        return best.run_id if best else None

    def _is_retry(self, run_id: str) -> bool:
        """Whether an identical repeat of a run's last request is a retry of it.

        An agent that continues a conversation always *extends* the message list, so an
        exactly-equal one is either an SDK retry or a second agent starting from the
        same prompt — indistinguishable by content. The last step tells them apart: a
        retry follows a failed call, whereas a repeat of a request the run already
        answered successfully is a different run beginning the same way. Without this,
        two concurrent runs of one agent would record onto a single tape.
        """
        last = self.store.last_step(run_id)
        return last is not None and not last.ok

    def _track(self, run_id: str, keys: list[str]) -> None:
        self._active[run_id] = _ActiveRun(run_id=run_id, keys=keys, last_seen=self.clock())

    # ------------------------------------------------------------------- lifecycle

    def expire_idle(self) -> list[str]:
        """Close runs that have been silent for longer than the idle timeout."""
        now = self.clock()
        stale = [a.run_id for a in self._active.values() if now - a.last_seen > self.idle_timeout_s]
        for run_id in stale:
            self.close(run_id)
        return stale

    def close(self, run_id: str, *, status: str = STATUS_COMPLETED) -> None:
        self._active.pop(run_id, None)
        for tape, session in list(self._replays.items()):
            if run_id in (tape, session):
                self._replays.pop(tape, None)
        if self.store.get_run(run_id) is not None:
            self.store.update_run(run_id, status=status)

    def close_all(self, run_ids: Iterable[str] | None = None) -> None:
        for run_id in list(run_ids if run_ids is not None else self._active):
            self.close(run_id)
