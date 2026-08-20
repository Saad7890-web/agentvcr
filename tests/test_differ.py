"""Run alignment and step diffs (DESIGN.md §7).

The acceptance check Phase 3 is graded on lives at the bottom: a run diffed against
its **own replay** must report no divergence at all. That is what turns "the replay
did not crash" into "the replay reproduced the recording", and it is the same engine
that later answers "your prompt change altered the agent's decisions in 12 of 40 runs".
"""

from __future__ import annotations

import json

import pytest
import respx

from agentvcr.core import differ
from agentvcr.core.recorder import record_step
from agentvcr.core.store import Store
from agentvcr.providers import OPENAI
from test_replay import ANSWER, OPENING, TOOL_CALL, TOOL_RESULT, drive, record_the_loop, replay_run

SYSTEM = {"role": "system", "content": "You book flights."}
ASK = {"role": "user", "content": "Cheapest SFO to JFK?"}
CALL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_flights", "arguments": '{"origin":"SFO"}'},
        }
    ],
}


def answer(text: str) -> dict:
    return {
        "id": "chatcmpl-x",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
    }


CALL_RESPONSE = {"id": "chatcmpl-1", "choices": [{"index": 0, "message": CALL}]}


def add(
    store: Store,
    run_id: str,
    messages: list[dict],
    response: dict,
    status: int = 200,
    model: str = "m",
) -> None:
    record_step(
        store,
        run_id=run_id,
        provider=OPENAI,
        request_headers={},
        request_body={"model": model, "messages": messages},
        response=response,
        status_code=status,
        latency_ms=1,
    )


def tool_loop(
    store: Store, run_id: str, *, result: dict, final: str, system: dict = SYSTEM
) -> None:
    """A two-step recording: ask → tool call → result → final answer."""
    store.create_run(mode="record", run_id=run_id)
    add(store, run_id, [system, ASK], CALL_RESPONSE)
    add(
        store,
        run_id,
        [
            system,
            ASK,
            CALL,
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(result)},
        ],
        answer(final),
    )


def diff(store: Store, a: str, b: str) -> differ.RunDiff:
    left, right = store.get_run(a), store.get_run(b)
    assert left is not None and right is not None
    return differ.diff_runs(store, left, right)


# ------------------------------------------------------------------------- alignment


def test_two_identical_recordings_are_identical(store: Store) -> None:
    tool_loop(store, "A", result={"price": 289}, final="B6918, $289.")
    tool_loop(store, "B", result={"price": 289}, final="B6918, $289.")

    result = diff(store, "A", "B")

    assert result.identical
    assert result.first_divergence is None
    assert result.summary == "identical — 2 step(s) aligned, no differences"


def test_an_inserted_step_aligns_around_the_change(store: Store) -> None:
    """LCS over fingerprints: a run that gained a step still lines up on either side of
    it, instead of reporting every following step as different."""
    store.create_run(mode="record", run_id="A")
    store.create_run(mode="record", run_id="B")
    for run_id in ("A", "B"):
        add(store, run_id, [ASK], answer("one"))
    add(store, "B", [ASK, {"role": "user", "content": "wait"}], answer("interrupted"))
    for run_id in ("A", "B"):
        add(store, run_id, [ASK, {"role": "user", "content": "and?"}], answer("two"))

    result = diff(store, "A", "B")

    assert [
        (p.left.idx if p.left else None, p.right.idx if p.right else None) for p in result.pairs
    ] == [
        (0, 0),
        (None, 1),  # the inserted step, matched against nothing
        (1, 2),  # …and the run realigns after it
    ]
    assert result.pairs[0].same and result.pairs[2].same
    assert "present only in" not in result.summary
    assert result.summary.startswith("runs diverge at step -/1")


def test_a_step_the_other_run_never_made(store: Store) -> None:
    store.create_run(mode="record", run_id="A")
    store.create_run(mode="record", run_id="B")
    add(store, "A", [ASK], answer("one"))
    add(store, "B", [ASK], answer("one"))
    add(store, "A", [ASK, {"role": "user", "content": "more"}], answer("two"))

    result = diff(store, "A", "B")

    assert not result.identical
    assert result.pairs[1].left is not None and result.pairs[1].right is None
    assert "only in the left run" in result.summary


# ----------------------------------------------------------------------------- diffs


def test_a_changed_tool_result_is_the_reported_reason(store: Store) -> None:
    """The DESIGN.md §7 headline: 'runs diverge at step N (tool X returned different
    results)' — not 'a message changed', which is the same fact one level down."""
    tool_loop(store, "A", result={"price": 289}, final="B6918, $289.")
    tool_loop(store, "B", result={"price": 0}, final="No flights found.")

    result = diff(store, "A", "B")

    assert result.summary.startswith(
        "runs diverge at step 0 (tool search_flights returned different results)"
    )
    change = result.pairs[0].changes[0]
    assert change.where == differ.WHERE_TOOL
    assert change.left == '{"price": 289}'
    assert change.right == '{"price": 0}'
    # …and the step that acted on it reports what the model then said.
    assert any(
        c.where == differ.WHERE_RESPONSE and c.detail == "response text differs"
        for c in result.pairs[1].changes
    )


def test_a_different_tool_call_is_reported(store: Store) -> None:
    other = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search_trains", "arguments": '{"origin":"SFO"}'},
                        }
                    ],
                },
            }
        ],
    }
    store.create_run(mode="record", run_id="A")
    store.create_run(mode="record", run_id="B")
    add(store, "A", [ASK], CALL_RESPONSE)
    add(store, "B", [ASK], other)

    (change,) = diff(store, "A", "B").pairs[0].changes

    assert change.where == differ.WHERE_RESPONSE
    assert change.detail == "the model asked for different tool calls"
    assert change.left == 'search_flights({"origin": "SFO"})'
    assert change.right == 'search_trains({"origin": "SFO"})'


def test_a_changed_prompt_is_reported_once_not_forty_times(store: Store) -> None:
    """Every request carries the whole conversation, so an edited system prompt is in
    all of them. It is reported at the step that introduced it and nowhere else."""
    edited = {"role": "system", "content": "You book trains."}
    tool_loop(store, "A", result={"price": 289}, final="Same answer.")
    tool_loop(store, "B", result={"price": 289}, final="Same answer.", system=edited)

    result = diff(store, "A", "B")

    assert result.summary.startswith("runs diverge at step 0 (system message differs)")
    assert [c.detail for c in result.pairs[0].changes] == ["system message differs"]
    assert result.pairs[1].changes == []


def test_a_changed_request_setting_is_reported(store: Store) -> None:
    store.create_run(mode="record", run_id="A")
    store.create_run(mode="record", run_id="B")
    add(store, "A", [ASK], answer("hi"))
    record_step(
        store,
        run_id="B",
        provider=OPENAI,
        request_headers={},
        request_body={"model": "a-bigger-model", "messages": [ASK]},
        response=answer("hi"),
        status_code=200,
        latency_ms=1,
    )

    (change,) = diff(store, "A", "B").pairs[0].changes

    assert change.detail == "request field 'model' differs"
    assert (change.left, change.right) == ("m", "a-bigger-model")


def test_an_error_on_one_side_is_reported(store: Store) -> None:
    store.create_run(mode="record", run_id="A")
    store.create_run(mode="record", run_id="B")
    add(store, "A", [ASK], answer("hi"))
    add(store, "B", [ASK], {"error": {"message": "rate limited"}}, status=429)

    changes = diff(store, "A", "B").pairs[0].changes

    assert any(
        c.where == differ.WHERE_STATUS and c.left == "200" and c.right == "429" for c in changes
    )


def test_the_json_form_is_machine_readable(store: Store) -> None:
    tool_loop(store, "A", result={"price": 289}, final="B6918, $289.")
    tool_loop(store, "B", result={"price": 0}, final="No flights found.")

    payload = diff(store, "A", "B").as_dict()

    assert payload["identical"] is False
    assert payload["left"] == "A" and payload["right"] == "B"
    assert payload["steps"][0]["left_idx"] == 0
    assert payload["steps"][0]["changes"][0]["where"] == "tool"
    assert json.dumps(payload)  # nothing in it resists serialization


# ------------------------------------------------------- the Phase 3 acceptance check


def test_a_run_diffed_against_its_own_replay_reports_no_divergence(proxy) -> None:
    """This is what proves a replay *reproduced* the recording rather than merely not
    crashing — the property DESIGN.md §4 introduced replay-as-a-run-of-its-own for."""
    record_the_loop(proxy)

    with respx.mock(assert_all_called=False):  # no routes: any upstream call raises
        drive(proxy.client, replay_run(proxy))

    result = diff(proxy.store, "TAPE", "REPLAY")

    assert result.identical, result.summary
    assert len(result.pairs) == 2


def test_a_tweaked_re_record_pinpoints_the_diverging_step(proxy) -> None:
    """The other half of the check: a re-recording whose model answered differently is
    caught, and the step it changed at is named — the whole point of `agentvcr diff`."""
    record_the_loop(proxy)
    proxy.store.create_run(mode="record", run_id="AGAIN")
    add(proxy.store, "AGAIN", list(OPENING), TOOL_CALL, model="llama-3.1-8b")
    add(
        proxy.store,
        "AGAIN",
        [
            *OPENING,
            TOOL_CALL["choices"][0]["message"],
            {"role": "tool", "tool_call_id": "call_1", "content": TOOL_RESULT},
        ],
        {
            **ANSWER,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "No flights."}}],
        },
        model="llama-3.1-8b",
    )

    result = diff(proxy.store, "TAPE", "AGAIN")

    assert not result.identical
    assert result.pairs[0].same  # the agent asked the same question and got the same call
    assert result.summary.startswith("runs diverge at step 1 (response text differs)")
    (change,) = result.pairs[1].changes
    assert change.left == "The cheapest is B6918 at $289."
    assert change.right == "No flights."


def test_two_wire_formats_are_refused_rather_than_called_identical(store: Store) -> None:
    """Reading an Anthropic response with the OpenAI provider finds nothing to compare,
    which would come back as 'identical'. Saying so out loud is the only safe answer."""
    store.create_run(mode="record", run_id="OAI", provider="openai")
    store.create_run(mode="record", run_id="ANT", provider="anthropic")
    add(store, "OAI", [ASK], answer("hi"))

    with pytest.raises(differ.IncomparableRuns, match="cannot diff two wire formats"):
        diff(store, "OAI", "ANT")


def test_a_missing_side_reads_as_absent_not_as_null(store: Store) -> None:
    """A step that produced no text at all is a normal thing (it called a tool). It
    should not be reported as the JSON value `null`."""
    store.create_run(mode="record", run_id="A")
    store.create_run(mode="record", run_id="B")
    add(store, "A", [ASK], CALL_RESPONSE)  # a tool call, so no assistant text
    add(store, "B", [ASK], answer("here you go"))

    changes = diff(store, "A", "B").pairs[0].changes
    (text_change,) = [c for c in changes if c.detail == "response text differs"]

    assert text_change.left == "(none)"
    assert text_change.right == "here you go"
