"""Replay mode: serve recorded responses without ever contacting an upstream.

Matching is **positional** (DESIGN.md §5): the *N*th LLM call of the replay session is
answered with the *N*th step of the tape. That is what makes replay survive the
harmless nondeterminism agents are full of — a timestamp in a system prompt, a
reordered tool list, a regenerated id. The request's fingerprint is compared as a
*signal*, not as the matching key, and what happens on a mismatch is policy:

``warn`` (default)
    serve the positional response anyway and flag the step as diverged.
``strict``
    refuse with a structured 409 — the CI setting.
``live-on-miss``
    stop replaying and go live from that step onward, i.e. auto-fork.

A replay is a **run of its own** (``replay_of`` points at the tape). The position is
therefore just how many steps the replay run has recorded so far — no session counter
to lose — divergence is recorded against the replay instead of scribbled onto the
original recording, and a tape can be diffed against its own replay.

Nothing here contacts the network. A replay costs zero tokens by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..providers import Provider
from .models import STATUS_DIVERGED, Run, Step
from .store import Store

#: ``Run.meta`` key marking the step from which a ``live-on-miss`` replay went live.
LIVE_FROM = "live_from"

JSON_MEDIA_TYPE = "application/json"
SSE_MEDIA_TYPE = "text/event-stream"


class ReplayError(Exception):
    """A replay that cannot be served, in a shape the proxy can return verbatim."""

    status_code = 409
    code = "replay_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class TapeExhausted(ReplayError):
    """The agent asked for more calls than the tape recorded."""

    code = "tape_exhausted"


class TapeMissing(ReplayError):
    """Nothing to replay: the run has no tape behind it."""

    code = "no_tape"


class FingerprintMismatch(ReplayError):
    """``strict`` policy: the request no longer matches what was recorded."""

    code = "fingerprint_mismatch"


@dataclass(frozen=True)
class Served:
    """The step answering a request, and whether it drifted from the request.

    ``edited`` marks the one step of a fork that came from an edit rather than from
    the tape (DESIGN.md §6) — the client is told so in a response header.
    """

    step: Step
    diverged: bool
    edited: bool = False


def tape_id(replay_run: Run) -> str:
    """The run whose steps ``replay_run`` serves from.

    Two lineages lead to a tape and both end here: a replay points at one through
    ``replay_of``, a fork through ``parent_run_id`` (DESIGN.md §4). Everything below
    this line is the same machinery either way — a fork *is* a replay, up to the step
    where its edit takes over.
    """
    assert_replayable(replay_run)
    return str(replay_run.replay_of or replay_run.parent_run_id)


def assert_replayable(run: Run) -> None:
    """Reject a run that can neither be a tape nor replay one.

    A run pointed at in replay mode is either a *recording* to replay (steps of its
    own, no ``replay_of``) or a replay run created ahead of time by ``agentvcr run``.
    A replay run with no tape behind it is neither: its steps are replays, not a
    recording, so replaying them would be replaying nothing. The same holds for a fork
    with no parent — there is nothing to branch away from.
    """
    if run.mode == "replay" and not run.replay_of:
        raise TapeMissing(
            f"run {run.id} is in replay mode but has no tape behind it; "
            f"replay a recording with `agentvcr run --mode replay --run <id>`",
            run=run.id,
        )
    if run.mode == "fork" and not run.parent_run_id:
        raise TapeMissing(
            f"run {run.id} is in fork mode but has no tape behind it; "
            f"branch off a recording with `agentvcr fork <run> --at <step>`",
            run=run.id,
        )


def is_live(run: Run) -> bool:
    """Whether a ``live-on-miss`` replay has already fallen through to the real API."""
    return run.meta.get(LIVE_FROM) is not None


def next_from_tape(
    store: Store,
    *,
    replay_run: Run,
    provider: Provider,
    body: dict[str, Any],
    policy: str,
) -> Served | None:
    """The step answering this request, or ``None`` when the proxy must go live.

    ``None`` is only ever returned under ``live-on-miss``; the other policies either
    serve a step or raise a :class:`ReplayError`.
    """
    if is_live(replay_run):
        return None

    source_id = tape_id(replay_run)
    idx = store.count_steps(replay_run.id)
    step = store.get_step(source_id, idx)
    if step is None:
        recorded = store.count_steps(source_id)
        raise TapeExhausted(
            f"tape {source_id} has {recorded} step(s); the agent asked for step {idx}. "
            f"The replayed agent is making more LLM calls than the recording did.",
            run=replay_run.id,
            tape=source_id,
            step=idx,
            recorded_steps=recorded,
        )

    if provider.fingerprint(body) == step.fingerprint:
        return Served(step=step, diverged=False)

    if policy == "strict":
        raise FingerprintMismatch(
            f"step {idx} of tape {source_id} was recorded for a different request; "
            f"the agent has drifted from the recording",
            run=replay_run.id,
            tape=source_id,
            step=idx,
        )
    if policy == "live-on-miss":
        _go_live(store, replay_run, idx)
        return None
    return Served(step=step, diverged=True)  # policy "warn"


def _go_live(store: Store, replay_run: Run, idx: int) -> None:
    """Record that this replay stopped being one, from ``idx`` onward."""
    replay_run.meta[LIVE_FROM] = idx
    replay_run.status = STATUS_DIVERGED
    store.update_run(replay_run.id, status=STATUS_DIVERGED, meta=replay_run.meta)


def response_body(step: Step, provider: Provider, *, streaming: bool) -> tuple[bytes, str]:
    """Bytes and media type for serving ``step`` back to the client.

    A recorded raw chunk tape is preferred whenever the client asked for a stream: it
    replays the provider's own bytes. Without one the stream is synthesized from the
    accumulated message, which is faithful but not byte-identical — ids and chunk
    boundaries are ours. A recorded *error* is served as the JSON it was, whatever the
    client asked for, because that is what the client saw the first time.
    """
    payload = step.response if step.response is not None else {}
    if not streaming or not step.ok:
        return json.dumps(payload, ensure_ascii=False).encode(), JSON_MEDIA_TYPE
    if step.response_chunks:
        return step.response_chunks, SSE_MEDIA_TYPE
    return b"".join(provider.synthesize_stream(payload)), SSE_MEDIA_TYPE
