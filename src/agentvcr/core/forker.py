"""Fork mode: replay a prefix, apply an edit, then go live on a child run.

The killer feature (DESIGN.md §6). A fork is a replay that stops being one: it serves
the tape up to the branch point, applies the edit, and from there makes real calls that
are recorded onto the child run. *The edit defines the fork point* — where the branch
leaves the tape follows from what was edited, not from a separate setting:

``response`` at step *k*
    The tape answers steps ``0 … k-1``; step *k* is answered with the **edited
    response**; ``k+1 …`` are live. The agent genuinely reacts to the edit.
``tool_result`` at step *k*
    A tool result physically lives inside request *k+1*, and the replayed prefix
    ignores request bodies — so this is not something to *serve*, it is an **outbound
    request patch**. The tape answers ``0 … k``, and from the first live call onward
    the matching tool-result message is rewritten on its way upstream.
``request_patch`` at step *k*
    A system or user prompt edit. It rewrites request *k* itself, so the tape answers
    ``0 … k-1`` and the branch goes live *at* *k*, carrying the patch.

**A fork run accumulates its own steps; it never starts with a copy of the prefix.**
Position on a tape is how many steps the run being served has recorded so far
(:mod:`agentvcr.core.replayer`, DESIGN.md §4) — a fork pre-loaded with *k* prefix rows
would have its *first* call answered with tape step *k*, and every step of the prefix
replay would be off by *k*. The prefix is *replayed onto* the child, so ``fork_step``
records where the branch leaves the tape, not how many rows were pre-inserted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ..providers import Provider
from . import replayer
from .models import EDIT_REQUEST_PATCH, EDIT_RESPONSE, EDIT_TOOL_RESULT, Edit, Run, Step
from .store import Store

#: Edit kinds whose branch point is *after* the step they name — see the module
#: docstring. Everything else rewrites the named step's own request, so it goes live
#: at that step rather than one past it.
_LIVE_AFTER_KINDS = frozenset({EDIT_RESPONSE, EDIT_TOOL_RESULT})


class CannotFork(ValueError):
    """A fork that cannot be created as described: bad step, unknown tool, conflict.

    Raised while *creating* the fork, where the user is still at a prompt and can be
    told what to do instead. By the time the proxy serves a fork, it is well-formed.
    """


@dataclass(frozen=True)
class EditSpec:
    """One edit as the command line collected it, before resolving it against the tape.

    ``target`` is the tool name for a ``tool_result`` edit and the message index for a
    ``request_patch``; a ``response`` edit needs no target.
    """

    kind: str
    value: Any
    target: str | None = None


@dataclass(frozen=True)
class Plan:
    """What serving one fork takes, resolved once per call from its stored edits."""

    tape: str
    fork_step: int
    #: First step the fork makes for real. Everything before it comes off the tape.
    live_from: int
    #: Assistant response to serve *instead of* the tape's, at ``fork_step``.
    response: dict[str, Any] | None = None
    #: ``(tool_call_id, result)`` pairs rewritten in every outbound request.
    tool_results: tuple[tuple[str, Any], ...] = ()
    #: Message/field patches applied to every outbound request.
    request_patches: tuple[dict[str, Any], ...] = ()

    @property
    def patches_requests(self) -> bool:
        return bool(self.tool_results or self.request_patches)


# ------------------------------------------------------------------------- creating


def create(
    store: Store,
    *,
    tape: Run,
    at: int,
    edits: list[EditSpec],
    provider: Provider,
    name: str | None = None,
) -> Run:
    """Create the child run for a fork of ``tape`` at step ``at`` and store its edits.

    ``at`` is a step number as ``agentvcr show`` prints it. The child holds no steps:
    it collects them as it is re-run.
    """
    recorded = store.count_steps(tape.id)
    if recorded == 0:
        raise CannotFork(f"run {tape.id} recorded no steps; there is nothing to fork")
    if not 0 <= at < recorded:
        raise CannotFork(
            f"run {tape.id} has {recorded} step(s), numbered 0…{recorded - 1}; "
            f"--at {at} is outside it"
        )
    step = store.get_step(tape.id, at)
    assert step is not None  # `at` was just bounds-checked against this run

    resolved = [_resolve(provider, tape=tape, step=step, spec=spec) for spec in edits]
    _check_combination(resolved, tape=tape, at=at)

    child = store.create_run(
        mode="fork",
        parent_run_id=tape.id,
        fork_step=at,
        name=name,
        provider=tape.provider,
        upstream_url=tape.upstream_url,
    )
    for kind, patch in resolved:
        store.add_edit(Edit(run_id=child.id, step_idx=at, kind=kind, patch=patch))
    return child


def _resolve(
    provider: Provider, *, tape: Run, step: Step, spec: EditSpec
) -> tuple[str, dict[str, Any]]:
    """Turn one command-line edit into the patch stored on the fork.

    Names are resolved to ids *here*, against the tape, so that serving a fork is a
    lookup rather than a search — and so that a typo is caught at a prompt.
    """
    if spec.kind == EDIT_RESPONSE:
        if not isinstance(spec.value, dict):
            raise CannotFork(
                "an edited response is a response body in the provider's own format; "
                "start from `agentvcr show <run> --json` and edit the step's response"
            )
        return EDIT_RESPONSE, spec.value

    if spec.kind == EDIT_TOOL_RESULT:
        calls = provider.extract_tool_calls(step.response or {})
        if not calls:
            raise CannotFork(
                f"step {step.idx} of {tape.id} requested no tool calls, "
                f"so it has no tool result to edit"
            )
        match = next((c for c in calls if c.get("tool_name") == spec.target), None)
        if match is None:
            names = ", ".join(sorted({str(c.get("tool_name")) for c in calls}))
            raise CannotFork(
                f"step {step.idx} of {tape.id} calls no tool named {spec.target!r}; "
                f"it calls: {names}"
            )
        return EDIT_TOOL_RESULT, {
            "tool_call_id": match.get("tool_call_id"),
            "tool_name": match.get("tool_name"),
            "result": spec.value,
        }

    if spec.kind == EDIT_REQUEST_PATCH:
        messages = provider.messages_of(step.request.get("body") or {})
        try:
            index = int(str(spec.target))
        except ValueError:
            raise CannotFork(f"message index must be a number, not {spec.target!r}") from None
        if not -len(messages) <= index < len(messages):
            raise CannotFork(
                f"request {step.idx} of {tape.id} carries {len(messages)} message(s), "
                f"numbered 0…{len(messages) - 1}; there is no message {index}"
            )
        return EDIT_REQUEST_PATCH, {"message": index, "content": spec.value}

    raise CannotFork(f"unknown edit kind {spec.kind!r}")


def _check_combination(resolved: list[tuple[str, dict[str, Any]]], *, tape: Run, at: int) -> None:
    """Reject edits that would silently cancel each other out.

    A fork has one branch point, and each kind puts it in a different place (see the
    module docstring). Mixing them is not a merge, it is a contradiction: an edited
    response at step *k* replaces the very tool calls a tool-result edit names, and a
    patched request *k* is one the tape was never going to answer. Several tool
    results at one step are the exception — a step can call more than one tool.
    """
    kinds = {kind for kind, _ in resolved}
    if len(kinds) > 1:
        raise CannotFork(
            f"a fork branches at one point, and these edits disagree about where: "
            f"{', '.join(sorted(kinds))} at step {at} of {tape.id}. Fork once per edit."
        )
    if EDIT_RESPONSE in kinds and len(resolved) > 1:
        raise CannotFork("a step has one response; pass --edit-response once")


# -------------------------------------------------------------------------- serving


def plan(store: Store, fork_run: Run) -> Plan:
    """Resolve a fork's stored edits into what serving its next call needs."""
    tape = replayer.tape_id(fork_run)
    fork_step = fork_run.fork_step if fork_run.fork_step is not None else 0

    response: dict[str, Any] | None = None
    tool_results: list[tuple[str, Any]] = []
    request_patches: list[dict[str, Any]] = []
    live_from = fork_step

    for edit in store.list_edits(fork_run.id):
        patch = edit.patch if isinstance(edit.patch, dict) else {}
        if edit.kind == EDIT_RESPONSE:
            response = edit.patch if isinstance(edit.patch, dict) else None
        elif edit.kind == EDIT_TOOL_RESULT:
            tool_results.append((str(patch.get("tool_call_id")), patch.get("result")))
        elif edit.kind == EDIT_REQUEST_PATCH:
            request_patches.append(patch)
        if edit.kind in _LIVE_AFTER_KINDS:
            live_from = max(live_from, edit.step_idx + 1)

    return Plan(
        tape=tape,
        fork_step=fork_step,
        live_from=live_from,
        response=response,
        tool_results=tuple(tool_results),
        request_patches=tuple(request_patches),
    )


def next_step(
    store: Store,
    *,
    fork_run: Run,
    fork_plan: Plan,
    provider: Provider,
    body: dict[str, Any],
    policy: str,
) -> replayer.Served | None:
    """The step answering this call, or ``None`` once the branch has gone live.

    Position is how many steps the *fork* has recorded, exactly as for a replay — the
    prefix is replayed onto the child rather than copied into it, so step *k* of the
    fork is step *k* of the tape with no offset to carry.
    """
    idx = store.count_steps(fork_run.id)
    if idx >= fork_plan.live_from:
        return None
    if fork_plan.response is not None and idx == fork_plan.fork_step:
        return replayer.Served(step=_edited_step(fork_plan), diverged=False, edited=True)
    return replayer.next_from_tape(
        store, replay_run=fork_run, provider=provider, body=body, policy=policy
    )


def _edited_step(fork_plan: Plan) -> Step:
    """The edited response, shaped as the step the proxy would have served."""
    return Step(
        run_id=fork_plan.tape,
        idx=fork_plan.fork_step,
        request={},
        response=fork_plan.response,
        status_code=200,
    )


def patch_outbound(
    fork_plan: Plan, provider: Provider, body: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """The request to actually send upstream, and a note of what was rewritten.

    Called for every live call of a fork, not just the first: every request carries
    the whole conversation, so an edited tool result has to be re-applied each time or
    the model would see the real one again on the very next turn.
    """
    if not fork_plan.patches_requests:
        return body, []
    patched = copy.deepcopy(body)
    applied: list[str] = []
    for tool_call_id, result in fork_plan.tool_results:
        if provider.patch_tool_result(patched, tool_call_id=tool_call_id, result=result):
            applied.append(f"tool_result {tool_call_id}")
    for patch in fork_plan.request_patches:
        if _apply_request_patch(provider, patched, patch):
            applied.append(f"message {patch['message']}")
    return patched, applied


def _apply_request_patch(provider: Provider, body: dict[str, Any], patch: dict[str, Any]) -> bool:
    """Rewrite one message of an outbound request. ``messages_of`` returns the request's
    own list, so assigning into it edits the body that is about to be sent."""
    messages = provider.messages_of(body)
    index = patch.get("message")
    if not isinstance(index, int) or not -len(messages) <= index < len(messages):
        return False
    message = messages[index]
    if not isinstance(message, dict):
        return False
    message["content"] = patch.get("content")
    return True


# -------------------------------------------------------------------------- display


def describe(edit: Edit) -> str:
    """One line naming an edit, for ``agentvcr fork`` and ``agentvcr show``."""
    patch = edit.patch if isinstance(edit.patch, dict) else {}
    if edit.kind == EDIT_TOOL_RESULT:
        return f"tool result of {patch.get('tool_name')} at step {edit.step_idx}"
    if edit.kind == EDIT_REQUEST_PATCH:
        return f"message {patch.get('message')} of request {edit.step_idx}"
    return f"response of step {edit.step_idx}"
