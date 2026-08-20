"""Align two runs and diff them step by step (DESIGN.md §7).

Alignment is an LCS over step fingerprints — the same normalized-request hash replay
matches against — so a run that gained or lost a step still lines up around the change
instead of reporting every later step as different. Inside a stretch that has no
common fingerprint at all, steps pair up positionally, which is the right answer when
a prompt changed and *every* fingerprint moved.

What counts as a difference is deliberately **semantic, not byte-level**. Two live
calls to the same model differ in their response id, their timestamp and usually their
token counts, and a differ that reports those has no signal left for what the user
actually asked: did the agent say something different, call a different tool, or get a
different answer back? So a step is compared on its request messages, its assistant
text, its tool calls, its tool results and its HTTP status.

This is the engine behind ``agentvcr diff``, the UI's side-by-side view, and later the
CI check ("your prompt change altered the agent's decisions in 12 of 40 runs").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from ..providers import Provider, get_provider, stable_hash
from .models import Run, Step, ToolCall
from .store import Store

# Where a change was found. Listed in the order :func:`compare_steps` reports them,
# which is most-explanatory first — see its docstring.
WHERE_TOOL = "tool"
WHERE_RESPONSE = "response"
WHERE_STATUS = "status"
WHERE_REQUEST = "request"


@dataclass(frozen=True)
class Change:
    """One difference inside an aligned pair of steps."""

    where: str
    detail: str
    left: str | None = None
    right: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"where": self.where, "detail": self.detail, "left": self.left, "right": self.right}


@dataclass
class StepPair:
    """Two aligned steps — either may be ``None`` where a run has no counterpart."""

    left: Step | None
    right: Step | None
    changes: list[Change] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.left is not None and self.right is not None

    @property
    def same(self) -> bool:
        return self.matched and not self.changes

    @property
    def label(self) -> str:
        """How to name this position: ``6`` when aligned, ``6/-`` when one side is missing."""
        left = str(self.left.idx) if self.left else "-"
        right = str(self.right.idx) if self.right else "-"
        return left if left == right else f"{left}/{right}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_idx": self.left.idx if self.left else None,
            "right_idx": self.right.idx if self.right else None,
            "changes": [c.as_dict() for c in self.changes],
        }


@dataclass
class RunDiff:
    """The full comparison of two runs."""

    left: Run
    right: Run
    pairs: list[StepPair]

    @property
    def identical(self) -> bool:
        return all(pair.same for pair in self.pairs)

    @property
    def first_divergence(self) -> StepPair | None:
        return next((pair for pair in self.pairs if not pair.same), None)

    @property
    def summary(self) -> str:
        """One line: the headline a terminal or a CI log should lead with."""
        aligned = len(self.pairs)
        same = sum(1 for pair in self.pairs if pair.same)
        if self.identical:
            return f"identical — {aligned} step(s) aligned, no differences"
        pair = self.first_divergence
        assert pair is not None  # not identical, so there is one
        if not pair.matched:
            side = "only in the left run" if pair.left else "only in the right run"
            return f"runs diverge at step {pair.label} ({side}); {same}/{aligned} step(s) identical"
        reason = pair.changes[0].detail if pair.changes else "different"
        return f"runs diverge at step {pair.label} ({reason}); {same}/{aligned} step(s) identical"

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.id,
            "right": self.right.id,
            "identical": self.identical,
            "summary": self.summary,
            "steps": [pair.as_dict() for pair in self.pairs],
        }


class IncomparableRuns(ValueError):
    """Two runs whose contents this differ cannot meaningfully line up."""


def diff_runs(store: Store, left: Run, right: Run) -> RunDiff:
    """Compare two recorded runs, step by aligned step.

    Raises :class:`IncomparableRuns` for two runs in different wire formats. Reading an
    Anthropic response with the OpenAI provider finds no text and no tool calls, which
    would come back as "identical" — quietly wrong, and quietly wrong is the one thing
    this tool must never be. Comparing an agent across providers is a real thing to
    want; it needs the differ to hold a provider per side, and that is not this phase.
    """
    if left.provider and right.provider and left.provider != right.provider:
        raise IncomparableRuns(
            f"run {left.id} is {left.provider} and run {right.id} is {right.provider}; "
            f"agentvcr cannot diff two wire formats against each other yet"
        )
    provider = get_provider(left.provider or right.provider or "openai") or get_provider("openai")
    assert provider is not None  # the openai provider is always registered
    left_steps = store.list_steps(left.id)
    right_steps = store.list_steps(right.id)
    left_tools = _tools_by_step(store, left.id)
    right_tools = _tools_by_step(store, right.id)

    pairs = [
        StepPair(
            left=a,
            right=b,
            changes=compare_steps(
                provider,
                a,
                b,
                left_tools=left_tools.get(a.idx, []) if a else [],
                right_tools=right_tools.get(b.idx, []) if b else [],
            ),
        )
        for a, b in align(left_steps, right_steps)
    ]
    return RunDiff(left=left, right=right, pairs=_drop_echoes(pairs))


def _drop_echoes(pairs: list[StepPair]) -> list[StepPair]:
    """Report a changed request message once, at the step that introduced it.

    Every request carries the whole conversation, so a single edited system prompt
    otherwise reappears in all forty steps that follow it and buries everything else.
    The first occurrence is kept — which is the step the runs actually diverge at.
    """
    seen: set[tuple[str, str, str | None, str | None]] = set()
    for pair in pairs:
        kept = []
        for change in pair.changes:
            key = (change.where, change.detail, change.left, change.right)
            if change.where == WHERE_REQUEST and key in seen:
                continue
            seen.add(key)
            kept.append(change)
        pair.changes = kept
    return pairs


def align(left: list[Step], right: list[Step]) -> list[tuple[Step | None, Step | None]]:
    """Pair steps by LCS over their fingerprints, positionally where nothing matches."""
    matcher = SequenceMatcher(
        a=[s.fingerprint or f"?left{s.idx}" for s in left],
        b=[s.fingerprint or f"?right{s.idx}" for s in right],
        autojunk=False,
    )
    pairs: list[tuple[Step | None, Step | None]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend(zip(left[i1:i2], right[j1:j2], strict=True))
            continue
        # Nothing matched in this stretch: line the two sides up from the top and let
        # the longer one trail off. A rewritten prompt moves every fingerprint, and
        # step 3 is still the counterpart of step 3.
        block_left, block_right = left[i1:i2], right[j1:j2]
        overlap = min(len(block_left), len(block_right))
        pairs.extend(zip(block_left[:overlap], block_right[:overlap], strict=True))
        pairs.extend((step, None) for step in block_left[overlap:])
        pairs.extend((None, step) for step in block_right[overlap:])
    return pairs


def compare_steps(
    provider: Provider,
    left: Step | None,
    right: Step | None,
    *,
    left_tools: list[ToolCall],
    right_tools: list[ToolCall],
) -> list[Change]:
    """Every semantic difference between two aligned steps, most explanatory first.

    Order matters because the first change is the reason the summary reports. A tool
    that returned something else *causes* the request message that carries it to
    differ, and saying so is more use than saying a message changed.
    """
    if left is None or right is None:
        return []  # an unmatched step is a difference in itself, not a list of them
    changes: list[Change] = []
    changes.extend(_tool_changes(left_tools, right_tools))
    changes.extend(_response_changes(provider, left, right))
    if left.status_code != right.status_code:
        changes.append(
            Change(
                WHERE_STATUS,
                "HTTP status differs",
                str(left.status_code),
                str(right.status_code),
            )
        )
    changes.extend(_request_changes(provider, left, right))
    return changes


# --------------------------------------------------------------------------- requests


def _request_changes(provider: Provider, left: Step, right: Step) -> list[Change]:
    before = _body(left)
    after = _body(right)
    changes = [
        Change(WHERE_REQUEST, f"request field {key!r} differs", _short(old), _short(new))
        for key, old, new in _settings_changes(provider, before, after)
    ]
    changes.extend(_message_changes(provider, before, after))
    return changes


def _settings_changes(
    provider: Provider, before: dict[str, Any], after: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Normalized request fields other than the messages — model, tools, temperature…"""
    old = {k: v for k, v in provider.normalize(before).items() if k != "messages"}
    new = {k: v for k, v in provider.normalize(after).items() if k != "messages"}
    return [
        (key, old.get(key), new.get(key))
        for key in sorted(set(old) | set(new))
        if old.get(key) != new.get(key)
    ]


def _message_changes(
    provider: Provider, before: dict[str, Any], after: dict[str, Any]
) -> list[Change]:
    old = provider.messages_of(before)
    new = provider.messages_of(after)
    matcher = SequenceMatcher(
        a=[stable_hash(m) for m in old], b=[stable_hash(m) for m in new], autojunk=False
    )
    changes: list[Change] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        block_old, block_new = old[i1:i2], new[j1:j2]
        overlap = min(len(block_old), len(block_new))
        for one, two in zip(block_old[:overlap], block_new[:overlap], strict=True):
            changes.append(
                Change(WHERE_REQUEST, f"{_role(one)} message differs", _short(one), _short(two))
            )
        for message in block_old[overlap:]:
            changes.append(
                Change(WHERE_REQUEST, f"{_role(message)} message only on the left", _short(message))
            )
        for message in block_new[overlap:]:
            changes.append(
                Change(
                    WHERE_REQUEST,
                    f"{_role(message)} message only on the right",
                    None,
                    _short(message),
                )
            )
    return changes


# -------------------------------------------------------------------------- responses


def _response_changes(provider: Provider, left: Step, right: Step) -> list[Change]:
    old = left.response or {}
    new = right.response or {}
    changes: list[Change] = []

    old_calls = _safe(provider.extract_tool_calls, old)
    new_calls = _safe(provider.extract_tool_calls, new)
    if _call_key(old_calls) != _call_key(new_calls):
        changes.append(
            Change(
                WHERE_RESPONSE,
                "the model asked for different tool calls",
                _calls_label(old_calls),
                _calls_label(new_calls),
            )
        )

    old_text = _safe(provider.assistant_text, old)
    new_text = _safe(provider.assistant_text, new)
    if old_text != new_text:
        changes.append(
            Change(WHERE_RESPONSE, "response text differs", _short(old_text), _short(new_text))
        )
    return changes


def _call_key(calls: list[dict[str, Any]] | None) -> list[tuple[Any, str]]:
    return [(c.get("tool_name"), _canonical(c.get("args"))) for c in calls or []]


def _calls_label(calls: list[dict[str, Any]] | None) -> str:
    if not calls:
        return "(no tool calls)"
    return ", ".join(f"{c.get('tool_name')}({_short(c.get('args'))})" for c in calls)


# ------------------------------------------------------------------------ tool steps


def _tool_changes(left: list[ToolCall], right: list[ToolCall]) -> list[Change]:
    changes: list[Change] = []
    for one, two in zip(left, right, strict=False):
        if one.tool_name != two.tool_name:
            changes.append(Change(WHERE_TOOL, "a different tool ran", one.tool_name, two.tool_name))
            continue
        if _canonical(one.args) != _canonical(two.args):
            changes.append(
                Change(
                    WHERE_TOOL,
                    f"tool {one.tool_name} was called with different arguments",
                    _short(one.args),
                    _short(two.args),
                )
            )
        if _canonical(one.result) != _canonical(two.result):
            changes.append(
                Change(
                    WHERE_TOOL,
                    f"tool {one.tool_name} returned different results",
                    _short(one.result),
                    _short(two.result),
                )
            )
    for extra in left[len(right) :]:
        changes.append(Change(WHERE_TOOL, f"tool {extra.tool_name} ran only on the left"))
    for extra in right[len(left) :]:
        changes.append(Change(WHERE_TOOL, f"tool {extra.tool_name} ran only on the right"))
    return changes


def _tools_by_step(store: Store, run_id: str) -> dict[int, list[ToolCall]]:
    grouped: dict[int, list[ToolCall]] = {}
    for call in store.list_tool_calls(run_id):
        grouped.setdefault(call.after_step_idx, []).append(call)
    return grouped


# ----------------------------------------------------------------------------- helpers


def _body(step: Step) -> dict[str, Any]:
    body = step.request.get("body")
    return body if isinstance(body, dict) else {}


def _role(message: Any) -> str:
    if isinstance(message, dict) and isinstance(message.get("role"), str):
        return message["role"]
    return "a"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _short(value: Any, limit: int = 160) -> str:
    """A value in one readable line. ``None`` reads as absent, not as JSON ``null`` —
    it is what a step with no text, or a tool call with no result, actually means."""
    if value is None:
        return "(none)"
    text = value if isinstance(value, str) else _canonical(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _safe(extract: Any, response: dict[str, Any]) -> Any:
    """Providers whose renderers are not implemented yet must not break a diff."""
    try:
        return extract(response)
    except NotImplementedError:  # pragma: no cover - every shipped provider implements these
        return None
