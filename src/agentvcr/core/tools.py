"""Reconstruct the tool timeline from the LLM boundary alone (DESIGN.md §2).

A proxy never sees a tool run: agents execute tools client-side. It does not need to.
A tool **call** appears in the assistant response of step *N*; its **result** appears
as a tool-result message inside the request of step *N+1*, because the agent appends
it to the conversation before asking what to do next. Pairing the two reconstructs the
interleaved timeline — LLM step, tool step, LLM step — with zero client
instrumentation.

Two details make that pairing survive contact with real traffic:

* **Every request carries the whole conversation**, so a request's tool results
  include every earlier turn's. The new ones are the suffix that the *source step's*
  own request did not have yet — the message-list diff of §2, in its simplest correct
  form.
* **A failed call is a step too** (§5), and carries no tool calls. So the step whose
  calls a request is answering is the last one that *succeeded*, not literally the
  previous one, and a retry must not materialize the same tool run twice.

Rows are written at record time, so ``tool_calls`` is populated for recordings and
replays alike and the differ can treat tool steps as first class.
"""

from __future__ import annotations

from typing import Any

from ..providers import Provider, stable_hash
from .models import Step, ToolCall
from .store import Store


def materialize(store: Store, *, provider: Provider, step: Step) -> list[ToolCall]:
    """Record the tool runs that happened between ``step`` and the step it answers.

    Called as each step lands. Returns the rows written, which is empty for the common
    case of a request that carries no new tool results.
    """
    source = _source_step(store, step.run_id, step.idx)
    if source is None or source.response is None:
        return []
    calls = provider.extract_tool_calls(source.response)
    if not calls:
        return []
    if store.count_tool_calls(step.run_id, after_step_idx=source.idx):
        return []  # already paired — this request is a retry of one we have seen
    if not _extends(provider, source, step):
        return []  # a different conversation that only shares this run

    results = _new_tool_results(provider, source=source, current=step)
    rows = [
        ToolCall(
            run_id=step.run_id,
            after_step_idx=source.idx,
            tool_name=call.get("tool_name"),
            args=call.get("args"),
            result=result,
            tool_call_id=call.get("tool_call_id"),
        )
        for call, result in _pair(calls, results)
    ]
    for row in rows:
        store.add_tool_call(row)
    return rows


def _source_step(store: Store, run_id: str, idx: int) -> Step | None:
    """The step whose tool calls a request at ``idx`` would be answering.

    Walks back over recorded *failures*: an error response carries no tool calls, and
    the client retrying it means this request is still answering the last call that
    actually succeeded.
    """
    cursor = idx - 1
    while cursor >= 0:
        step = store.get_step(run_id, cursor)
        if step is None or step.ok:
            return step
        cursor -= 1
    return None


def _messages(provider: Provider, step: Step) -> list[Any]:
    return provider.messages_of(step.request.get("body") or {})


def _extends(provider: Provider, source: Step, current: Step) -> bool:
    """Whether ``current`` continues the conversation ``source`` was part of."""
    before = [stable_hash(m) for m in _messages(provider, source)]
    after = [stable_hash(m) for m in _messages(provider, current)]
    return len(after) > len(before) and after[: len(before)] == before


def _new_tool_results(provider: Provider, *, source: Step, current: Step) -> list[dict[str, Any]]:
    """Tool results this request carries that the source step's request did not."""
    already = len(provider.extract_tool_results(source.request.get("body") or {}))
    return provider.extract_tool_results(current.request.get("body") or {})[already:]


def _pair(
    calls: list[dict[str, Any]], results: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], Any]]:
    """Match each call to its result by id, falling back to arrival order.

    A result the agent never returned pairs with ``None``: the call was made, and what
    came back is genuinely unknown rather than empty.
    """
    by_id = {r["tool_call_id"]: r for r in results if r.get("tool_call_id")}
    spare = [r for r in results if not r.get("tool_call_id")]
    paired: list[tuple[dict[str, Any], Any]] = []
    for call in calls:
        match = by_id.pop(call.get("tool_call_id"), None)
        if match is None and spare:
            match = spare.pop(0)
        paired.append((call, match["result"] if match else None))
    return paired
