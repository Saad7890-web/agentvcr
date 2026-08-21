"""Fork & edit: replay a prefix, apply the edit, then go live (PLAN.md phase 4).

The headline test is :func:`test_an_edited_tool_result_changes_the_branch` — the demo
the project exists for, headless: an agent that gave up because a tool came back empty
is forked with a tool result that isn't, and answers differently. What the assertions
watch for is that the edit reached *the model*, not just the database: the mocked
upstream is inspected for the body it actually received.

:func:`test_a_fork_does_not_start_with_a_copy_of_the_prefix` guards the one mistake
that would make every fork subtly wrong (PLAN.md phase 4): position on a tape is how
many steps the fork has recorded, so pre-loading it with the prefix would shift every
answer by the length of that prefix.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from agentvcr.core import differ, forker
from agentvcr.core.models import EDIT_REQUEST_PATCH, EDIT_RESPONSE, EDIT_TOOL_RESULT
from agentvcr.providers import OPENAI
from conftest import UPSTREAM

CHAT = f"{UPSTREAM}/chat/completions"

OPENING = [
    {"role": "system", "content": "You book flights."},
    {"role": "user", "content": "Cheapest SFO to JFK?"},
]

#: What the agent's own tool returns — the *real* result, the one the fork replaces.
NOTHING_FOUND = {"flights": []}
#: What the edit puts in its place.
FLIGHTS_FOUND = {"flights": [{"flight": "B6918", "price": 289}]}


def tool_call(call_id: str, name: str, args: dict) -> dict:
    """An assistant response asking for one tool call."""
    return {
        "id": f"chatcmpl-{call_id}",
        "object": "chat.completion",
        "created": 1,
        "model": "llama-3.1-8b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def answer(text: str) -> dict:
    """An assistant response that ends the loop."""
    return {
        "id": "chatcmpl-answer",
        "object": "chat.completion",
        "created": 2,
        "model": "llama-3.1-8b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


SEARCH = tool_call("call_0", "search_flights", {"origin": "SFO", "destination": "JFK"})
BOOK = tool_call("call_1", "book_flight", {"flight": "B6918"})
GAVE_UP = answer("Sorry, I could not find any flights.")
FOUND = answer("The cheapest is B6918 at $289.")
BOOKED = answer("Booked B6918.")


def drive(client, run_id: str, *, steps: int = 6, **extra) -> list[dict]:
    """Run the scripted tool loop against ``/r/<run_id>/…`` and return the transcript.

    The agent is the same code every time — its local tools keep returning
    :data:`NOTHING_FOUND`. Everything a fork changes, it changes at the proxy.
    """
    messages = list(OPENING)
    transcript: list[dict] = []
    for _ in range(steps):
        response = client.post(
            f"/r/{run_id}/openai/v1/chat/completions",
            json={"model": "llama-3.1-8b", "messages": messages, **extra},
            headers={"Authorization": "Bearer sk-test"},
        )
        response.raise_for_status()
        transcript.append(response.json())
        message = transcript[-1]["choices"][0]["message"]
        messages.append(message)
        if not message.get("tool_calls"):
            break
        for call in message["tool_calls"]:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(NOTHING_FOUND),
                }
            )
    return transcript


def record(proxy, responses: list[dict], run_id: str = "TAPE") -> list[dict]:
    """Record a run whose upstream answers with ``responses``, in order."""
    with respx.mock:
        respx.post(CHAT).mock(side_effect=[httpx.Response(200, json=r) for r in responses])
        proxy.store.create_run(mode="record", run_id=run_id, name="flights", provider="openai")
        return drive(proxy.client, run_id)


def make_fork(proxy, *, at: int, edits=(), tape: str = "TAPE"):
    """Fork ``tape`` the way ``agentvcr fork`` does, without going through the CLI."""
    return forker.create(
        proxy.store,
        tape=proxy.store.get_run(tape),
        at=at,
        edits=list(edits),
        provider=OPENAI,
    )


def upstream_bodies(route) -> list[dict]:
    """The request bodies the mocked upstream actually received."""
    return [json.loads(call.request.content) for call in route.calls]


# ----------------------------------------------------------------- the phase 4 contract


def test_an_edited_tool_result_changes_the_branch(proxy) -> None:
    """The demo, headless: an empty tool result made the agent give up; edit it and it
    answers instead — and the model is what sees the edit."""
    recorded = record(proxy, [SEARCH, GAVE_UP])
    assert recorded[-1]["choices"][0]["message"]["content"].startswith("Sorry")

    fork = make_fork(
        proxy,
        at=0,
        edits=[
            forker.EditSpec(kind=EDIT_TOOL_RESULT, target="search_flights", value=FLIGHTS_FOUND)
        ],
    )

    with respx.mock:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=FOUND))
        branch = drive(proxy.client, fork.id)

    # Step 0 came off the tape for free; only the step past the edit was paid for.
    assert route.call_count == 1
    assert branch[0] == SEARCH
    assert branch[-1]["choices"][0]["message"]["content"] == "The cheapest is B6918 at $289."

    # The edit reached the model: the agent still returned nothing, the proxy did not.
    (sent,) = upstream_bodies(route)
    (result,) = OPENAI.extract_tool_results(sent)
    assert result == {"tool_call_id": "call_0", "result": FLIGHTS_FOUND}

    steps = proxy.store.list_steps(fork.id)
    assert [s.response for s in steps] == [SEARCH, FOUND]
    # ...and the step recorded on the branch holds the request as it was sent, so the
    # tape shows what the model was asked, not what the agent typed.
    assert OPENAI.extract_tool_results(steps[1].request["body"]) == [
        {"tool_call_id": "call_0", "result": FLIGHTS_FOUND}
    ]


def test_the_recording_is_untouched_by_the_fork_of_it(proxy) -> None:
    record(proxy, [SEARCH, GAVE_UP])
    before = [(s.idx, s.response) for s in proxy.store.list_steps("TAPE")]

    fork = make_fork(
        proxy,
        at=0,
        edits=[
            forker.EditSpec(kind=EDIT_TOOL_RESULT, target="search_flights", value=FLIGHTS_FOUND)
        ],
    )
    with respx.mock:
        respx.post(CHAT).mock(return_value=httpx.Response(200, json=FOUND))
        drive(proxy.client, fork.id)

    assert [(s.idx, s.response) for s in proxy.store.list_steps("TAPE")] == before
    assert proxy.store.get_run("TAPE").parent_run_id is None


def test_an_edited_response_is_served_then_the_branch_goes_live(proxy) -> None:
    """DESIGN.md §6: the agent reacts to the edited response for real."""
    record(proxy, [SEARCH, GAVE_UP])
    fork = make_fork(proxy, at=1, edits=[forker.EditSpec(kind=EDIT_RESPONSE, value=BOOK)])

    with respx.mock:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=BOOKED))
        branch = drive(proxy.client, fork.id)

    assert branch == [SEARCH, BOOK, BOOKED]
    assert route.call_count == 1  # steps 0 and 1 cost nothing; only step 2 is live
    # The agent executed the tool the *edit* asked for and reported it upstream.
    (sent,) = upstream_bodies(route)
    assert [r["tool_call_id"] for r in OPENAI.extract_tool_results(sent)] == ["call_0", "call_1"]
    assert [s.response for s in proxy.store.list_steps(fork.id)] == [SEARCH, BOOK, BOOKED]


def test_the_edited_step_says_so_in_its_headers(proxy) -> None:
    record(proxy, [SEARCH, GAVE_UP])
    fork = make_fork(proxy, at=0, edits=[forker.EditSpec(kind=EDIT_RESPONSE, value=BOOK)])

    with respx.mock(assert_all_called=False):
        response = proxy.client.post(
            f"/r/{fork.id}/openai/v1/chat/completions",
            json={"model": "llama-3.1-8b", "messages": OPENING},
        )

    assert response.json() == BOOK
    assert response.headers["x-agentvcr-edited"] == "true"
    assert response.headers["x-agentvcr-replayed"] == "true"
    assert response.headers["x-agentvcr-tape"] == "TAPE"


def test_a_fork_does_not_start_with_a_copy_of_the_prefix(proxy) -> None:
    """Position is how many steps the *fork* has recorded (PLAN.md phase 4).

    A fork pre-loaded with its *k*-step prefix would answer its first call with tape
    step *k*. Forking at step 2 of a three-step tape is where that would show.
    """
    record(proxy, [SEARCH, BOOK, GAVE_UP])
    fork = make_fork(proxy, at=2)  # no edits: replay 0…1, then run live

    with respx.mock:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=FOUND))
        branch = drive(proxy.client, fork.id)

    assert branch == [SEARCH, BOOK, FOUND]  # not [GAVE_UP, …], which an offset would give
    assert route.call_count == 1
    assert proxy.store.get_run(fork.id).fork_step == 2


def test_a_prompt_edit_is_applied_on_the_way_upstream(proxy) -> None:
    """The same outbound-patch mechanism, pointed at a message instead of a result."""
    record(proxy, [SEARCH, GAVE_UP])
    fork = make_fork(
        proxy,
        at=0,
        edits=[forker.EditSpec(kind=EDIT_REQUEST_PATCH, target="0", value="You refuse to book.")],
    )

    with respx.mock:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=GAVE_UP))
        drive(proxy.client, fork.id)

    # A request patch rewrites step 0's own request, so the branch is live from step 0.
    assert route.call_count == 1
    (sent,) = upstream_bodies(route)
    assert sent["messages"][0] == {"role": "system", "content": "You refuse to book."}
    assert proxy.store.count_steps(fork.id) == 1


def test_the_edit_is_re_applied_to_every_live_call(proxy) -> None:
    """Every request carries the whole conversation, so patching only the first live
    call would hand the model the real tool result again on the very next turn."""
    record(proxy, [SEARCH, GAVE_UP])
    fork = make_fork(
        proxy,
        at=0,
        edits=[
            forker.EditSpec(kind=EDIT_TOOL_RESULT, target="search_flights", value=FLIGHTS_FOUND)
        ],
    )

    with respx.mock:
        route = respx.post(CHAT).mock(
            side_effect=[httpx.Response(200, json=BOOK), httpx.Response(200, json=BOOKED)]
        )
        drive(proxy.client, fork.id)

    assert route.call_count == 2
    for sent in upstream_bodies(route):
        results = {r["tool_call_id"]: r["result"] for r in OPENAI.extract_tool_results(sent)}
        assert results["call_0"] == FLIGHTS_FOUND


def test_an_edited_response_reaches_a_streaming_client(proxy) -> None:
    """The edit is stored as a final message; a streaming client gets it synthesized."""
    record(proxy, [SEARCH, GAVE_UP])
    fork = make_fork(proxy, at=0, edits=[forker.EditSpec(kind=EDIT_RESPONSE, value=FOUND)])

    with respx.mock(assert_all_called=False):
        response = proxy.client.post(
            f"/r/{fork.id}/openai/v1/chat/completions",
            json={"model": "llama-3.1-8b", "messages": OPENING, "stream": True},
        )

    assert response.headers["content-type"].startswith("text/event-stream")
    text = "".join(
        json.loads(line[len("data: ") :])["choices"][0]["delta"].get("content", "")
        for line in response.text.strip().split("\n\n")
        if line.startswith("data: ") and "[DONE]" not in line
    )
    assert text == "The cheapest is B6918 at $289."


# ---------------------------------------------------------------------------- lineage


def test_a_fork_records_where_it_came_from_and_what_was_edited(proxy) -> None:
    record(proxy, [SEARCH, GAVE_UP])
    fork = make_fork(
        proxy,
        at=0,
        edits=[
            forker.EditSpec(kind=EDIT_TOOL_RESULT, target="search_flights", value=FLIGHTS_FOUND)
        ],
    )

    stored = proxy.store.get_run(fork.id)
    assert stored.mode == "fork"
    assert stored.parent_run_id == "TAPE"
    assert stored.fork_step == 0
    assert stored.provider == "openai"
    assert proxy.store.count_steps(fork.id) == 0  # the prefix is replayed, never copied

    (edit,) = proxy.store.list_edits(fork.id)
    assert edit.kind == EDIT_TOOL_RESULT
    assert edit.step_idx == 0
    # The tool *name* is resolved to its call id when the fork is made, so serving it
    # later is a lookup and a typo is caught at the prompt.
    assert edit.patch == {
        "tool_call_id": "call_0",
        "tool_name": "search_flights",
        "result": FLIGHTS_FOUND,
    }
    assert "search_flights" in forker.describe(edit)


def test_diff_pinpoints_where_the_branch_left_the_tape(proxy) -> None:
    """The other half of the phase 4 acceptance check: the diff names the fork point."""
    record(proxy, [SEARCH, GAVE_UP])
    fork = make_fork(
        proxy,
        at=0,
        edits=[
            forker.EditSpec(kind=EDIT_TOOL_RESULT, target="search_flights", value=FLIGHTS_FOUND)
        ],
    )
    with respx.mock:
        respx.post(CHAT).mock(return_value=httpx.Response(200, json=FOUND))
        drive(proxy.client, fork.id)

    result = differ.diff_runs(
        proxy.store, proxy.store.get_run("TAPE"), proxy.store.get_run(fork.id)
    )

    assert not result.identical
    # The tool run belongs to the step whose call it answers, so the diff names the
    # fork point itself and says what changed there — not merely that step 1 differs.
    assert result.summary.startswith("runs diverge at step 0")
    assert "search_flights returned different results" in result.summary
    first_change = next(pair for pair in result.pairs if not pair.same)
    assert first_change.label == "0"


# --------------------------------------------------------------------- refusing a fork


def test_forking_past_the_end_of_the_tape_is_refused(proxy) -> None:
    record(proxy, [SEARCH, GAVE_UP])
    with pytest.raises(forker.CannotFork, match="0…1"):
        make_fork(proxy, at=7)


def test_forking_a_run_with_no_steps_is_refused(proxy) -> None:
    proxy.store.create_run(mode="record", run_id="EMPTY", provider="openai")
    with pytest.raises(forker.CannotFork, match="nothing to fork"):
        make_fork(proxy, at=0, tape="EMPTY")


def test_editing_a_tool_the_step_never_called_is_refused(proxy) -> None:
    record(proxy, [SEARCH, GAVE_UP])
    with pytest.raises(forker.CannotFork, match="search_flights"):
        make_fork(
            proxy,
            at=0,
            edits=[forker.EditSpec(kind=EDIT_TOOL_RESULT, target="book_flight", value={})],
        )


def test_editing_a_tool_result_of_a_step_that_called_none_is_refused(proxy) -> None:
    record(proxy, [SEARCH, GAVE_UP])
    with pytest.raises(forker.CannotFork, match="no tool calls"):
        make_fork(
            proxy,
            at=1,
            edits=[forker.EditSpec(kind=EDIT_TOOL_RESULT, target="search_flights", value={})],
        )


def test_editing_a_message_the_request_does_not_have_is_refused(proxy) -> None:
    record(proxy, [SEARCH, GAVE_UP])
    with pytest.raises(forker.CannotFork, match="no message 9"):
        make_fork(
            proxy, at=0, edits=[forker.EditSpec(kind=EDIT_REQUEST_PATCH, target="9", value="hi")]
        )


def test_edits_that_disagree_about_the_branch_point_are_refused(proxy) -> None:
    """An edited response replaces the very tool calls a tool-result edit names."""
    record(proxy, [SEARCH, GAVE_UP])
    with pytest.raises(forker.CannotFork, match="branches at one point"):
        make_fork(
            proxy,
            at=0,
            edits=[
                forker.EditSpec(kind=EDIT_RESPONSE, value=BOOK),
                forker.EditSpec(
                    kind=EDIT_TOOL_RESULT, target="search_flights", value=FLIGHTS_FOUND
                ),
            ],
        )


def test_two_tool_results_at_one_step_are_allowed(proxy) -> None:
    """A step can call more than one tool, so its results are the one exception."""
    both = tool_call("call_0", "search_flights", {"origin": "SFO"})
    both["choices"][0]["message"]["tool_calls"].append(
        {
            "id": "call_0b",
            "type": "function",
            "function": {"name": "check_weather", "arguments": "{}"},
        }
    )
    record(proxy, [both, GAVE_UP])

    fork = make_fork(
        proxy,
        at=0,
        edits=[
            forker.EditSpec(kind=EDIT_TOOL_RESULT, target="search_flights", value=FLIGHTS_FOUND),
            forker.EditSpec(kind=EDIT_TOOL_RESULT, target="check_weather", value={"sky": "clear"}),
        ],
    )

    with respx.mock:
        route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=FOUND))
        drive(proxy.client, fork.id)

    (sent,) = upstream_bodies(route)
    results = {r["tool_call_id"]: r["result"] for r in OPENAI.extract_tool_results(sent)}
    assert results == {"call_0": FLIGHTS_FOUND, "call_0b": {"sky": "clear"}}


def test_a_patched_live_call_can_stream(proxy) -> None:
    """The edit is applied to the body before it is forwarded, streaming or not.

    Both calls are made by hand here: with ``stream=True`` even the replayed prefix
    comes back as SSE, which the JSON-shaped loop in :func:`drive` cannot read.
    """
    record(proxy, [SEARCH, GAVE_UP])
    fork = make_fork(
        proxy,
        at=0,
        edits=[
            forker.EditSpec(kind=EDIT_TOOL_RESULT, target="search_flights", value=FLIGHTS_FOUND)
        ],
    )
    head = b'data: {"id":"c1","model":"m","choices":[{"index":0,'
    stream = (
        head
        + b'"delta":{"content":"B6918"}}]}\n\n'
        + head
        + b'"delta":{},"finish_reason":"stop"}]}\n\n'
        + b"data: [DONE]\n\n"
    )

    def post(messages):
        return proxy.client.post(
            f"/r/{fork.id}/openai/v1/chat/completions",
            json={"model": "llama-3.1-8b", "messages": messages, "stream": True},
        )

    with respx.mock:
        route = respx.post(CHAT).mock(
            return_value=httpx.Response(
                200, content=stream, headers={"content-type": "text/event-stream"}
            )
        )
        prefix = post(OPENING)  # step 0: off the tape, synthesized into a stream
        live = post(
            [
                *OPENING,
                SEARCH["choices"][0]["message"],
                {
                    "role": "tool",
                    "tool_call_id": "call_0",
                    "content": json.dumps(NOTHING_FOUND),
                },
            ]
        )

    assert prefix.headers["content-type"].startswith("text/event-stream")
    assert "search_flights" in prefix.text
    assert route.call_count == 1  # only the live call was paid for

    assert live.content == stream
    assert live.headers["x-agentvcr-patched"] == "tool_result call_0"
    (sent,) = upstream_bodies(route)
    assert sent["stream"] is True  # the client's own transport flags survive patching
    assert OPENAI.extract_tool_results(sent) == [
        {"tool_call_id": "call_0", "result": FLIGHTS_FOUND}
    ]
    assert proxy.store.list_steps(fork.id)[1].response_chunks == stream
