"""The tool timeline reconstructed at the LLM boundary (DESIGN.md §2).

The pairing rule is one sentence — a tool call in response *N*, its result in request
*N+1* — and every test here is a way real traffic makes that sentence harder than it
sounds: retries that replay the same request, conversations that carry every earlier
turn's results forever, results that never come back.
"""

from __future__ import annotations

import json

from agentvcr.core.recorder import record_step
from agentvcr.core.store import Store
from agentvcr.providers import OPENAI

SYSTEM = {"role": "system", "content": "You book flights."}
ASK = {"role": "user", "content": "Cheapest SFO to JFK?"}


def call_message(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def response_with_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": f"chatcmpl-{call_id}",
        "choices": [{"index": 0, "message": call_message(call_id, name, args)}],
    }


TEXT_ANSWER = {
    "id": "chatcmpl-done",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "B6918, $289."}}],
}


def step(store: Store, run_id: str, messages: list[dict], response: dict | None, status: int = 200):
    return record_step(
        store,
        run_id=run_id,
        provider=OPENAI,
        request_headers={},
        request_body={"model": "m", "messages": messages},
        response=response,
        status_code=status,
        latency_ms=1,
    )


def result_message(call_id: str, payload: dict) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload)}


def test_a_tool_run_becomes_a_row_of_its_own(store: Store) -> None:
    run = store.create_run(mode="record")
    first = [SYSTEM, ASK]
    step(store, run.id, first, response_with_call("call_1", "search_flights", {"origin": "SFO"}))
    step(
        store,
        run.id,
        [
            *first,
            call_message("call_1", "search_flights", {"origin": "SFO"}),
            result_message("call_1", {"flights": [{"price": 289}]}),
        ],
        TEXT_ANSWER,
    )

    (tool_call,) = store.list_tool_calls(run.id)
    assert tool_call.after_step_idx == 0  # the step that asked for it
    assert tool_call.tool_name == "search_flights"
    assert tool_call.args == {"origin": "SFO"}
    assert tool_call.result == {"flights": [{"price": 289}]}
    assert tool_call.tool_call_id == "call_1"


def test_each_round_is_attributed_to_the_call_that_asked_for_it(store: Store) -> None:
    """Every request repeats the whole conversation, so the second request carries the
    first round's result too. It must not be paired a second time."""
    run = store.create_run(mode="record")
    one = call_message("call_1", "search_flights", {"origin": "SFO"})
    two = call_message("call_2", "book", {"flight": "B6918"})
    turn_one = [SYSTEM, ASK]
    turn_two = [*turn_one, one, result_message("call_1", {"price": 289})]
    turn_three = [*turn_two, two, result_message("call_2", {"booked": True})]

    step(store, run.id, turn_one, response_with_call("call_1", "search_flights", {"origin": "SFO"}))
    step(store, run.id, turn_two, response_with_call("call_2", "book", {"flight": "B6918"}))
    step(store, run.id, turn_three, TEXT_ANSWER)

    rows = store.list_tool_calls(run.id)
    assert [(r.after_step_idx, r.tool_name, r.result) for r in rows] == [
        (0, "search_flights", {"price": 289}),
        (1, "book", {"booked": True}),
    ]


def test_a_retried_request_does_not_record_the_tool_run_twice(store: Store) -> None:
    """A 429 is a step too (DESIGN.md §5), so the SDK's retry re-sends a request whose
    tool results are already on the timeline."""
    run = store.create_run(mode="record")
    turn_one = [SYSTEM, ASK]
    turn_two = [
        *turn_one,
        call_message("call_1", "search_flights", {"origin": "SFO"}),
        result_message("call_1", {"price": 289}),
    ]

    step(store, run.id, turn_one, response_with_call("call_1", "search_flights", {"origin": "SFO"}))
    step(store, run.id, turn_two, {"error": {"message": "rate limited"}}, status=429)
    step(store, run.id, turn_two, TEXT_ANSWER)

    (tool_call,) = store.list_tool_calls(run.id)
    assert tool_call.after_step_idx == 0
    assert tool_call.result == {"price": 289}


def test_a_call_whose_result_never_came_back_is_still_a_row(store: Store) -> None:
    """The call was made. What came back is unknown, which is not the same as empty."""
    run = store.create_run(mode="record")
    turn_one = [SYSTEM, ASK]
    step(store, run.id, turn_one, response_with_call("call_1", "search_flights", {"origin": "SFO"}))
    step(
        store,
        run.id,
        [*turn_one, call_message("call_1", "search_flights", {"origin": "SFO"}), ASK],
        TEXT_ANSWER,
    )

    (tool_call,) = store.list_tool_calls(run.id)
    assert tool_call.tool_name == "search_flights"
    assert tool_call.result is None


def test_parallel_tool_calls_in_one_step_each_get_a_row(store: Store) -> None:
    run = store.create_run(mode="record")
    both = {
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
                            "function": {"name": "search_flights", "arguments": '{"to":"JFK"}'},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "search_hotels", "arguments": '{"in":"NYC"}'},
                        },
                    ],
                },
            }
        ],
    }
    turn_one = [SYSTEM, ASK]
    step(store, run.id, turn_one, both)
    step(
        store,
        run.id,
        [
            *turn_one,
            both["choices"][0]["message"],
            # returned out of order, as an agent running them concurrently would
            result_message("call_2", {"hotels": 3}),
            result_message("call_1", {"flights": 2}),
        ],
        TEXT_ANSWER,
    )

    rows = store.list_tool_calls(run.id)
    assert [(r.tool_name, r.result) for r in rows] == [
        ("search_flights", {"flights": 2}),
        ("search_hotels", {"hotels": 3}),
    ]


def test_an_unrelated_conversation_on_the_same_run_is_not_paired(store: Store) -> None:
    """Run grouping is a heuristic (DESIGN.md §4); a wrong guess must not invent a
    tool run that never happened."""
    run = store.create_run(mode="record")
    step(store, run.id, [SYSTEM, ASK], response_with_call("call_1", "search_flights", {"o": "SFO"}))
    step(store, run.id, [{"role": "user", "content": "Something else entirely"}], TEXT_ANSWER)

    assert store.list_tool_calls(run.id) == []


def test_a_step_with_no_tool_calls_costs_nothing(store: Store) -> None:
    run = store.create_run(mode="record")
    step(store, run.id, [ASK], TEXT_ANSWER)
    step(store, run.id, [ASK, {"role": "user", "content": "thanks"}], TEXT_ANSWER)

    assert store.list_tool_calls(run.id) == []
    assert store.count_tool_calls(run.id) == 0
