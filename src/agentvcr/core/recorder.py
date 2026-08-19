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
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..providers import Provider, stable_hash
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
) -> Step:
    """Persist one LLM call as the next step of ``run_id``."""
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
        started_at=started_at or utcnow(),
    )
    return store.add_step(step, at_next_idx=True)


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

    def _chain(self, keys: list[str]) -> str | None:
        """The active run whose last message list is the longest prefix of ``keys``."""
        best: _ActiveRun | None = None
        for active in self._active.values():
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
        if self.store.get_run(run_id) is not None:
            self.store.update_run(run_id, status=status)

    def close_all(self, run_ids: Iterable[str] | None = None) -> None:
        for run_id in list(run_ids if run_ids is not None else self._active):
            self.close(run_id)
