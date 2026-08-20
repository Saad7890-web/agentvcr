"""Record → replay parity for the Anthropic wire format (PLAN.md phase 3).

The point of these tests is what they *don't* touch. Not one line of
:mod:`agentvcr.core` learned that this format exists: the same recorder, the same
positional replayer, the same tool extractor and the same store serve a wire format
whose responses are block lists and whose streams are event-per-block. If the provider
abstraction were leaking, this is where it would show.
"""

from __future__ import annotations

import json

import httpx
import respx

from agentvcr.core.recorder import REDACTION_PLACEHOLDER
from agentvcr.providers import ANTHROPIC

UPSTREAM = "https://anthropic.test"
MESSAGES = f"{UPSTREAM}/v1/messages"
API_KEY = "sk-ant-super-secret-do-not-store"

TOOL_CALL = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "search_flights",
            "input": {"origin": "SFO", "destination": "JFK"},
        }
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 412, "output_tokens": 30},
}
ANSWER = {
    "id": "msg_02",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "The cheapest is B6918 at $289."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 503, "output_tokens": 14},
}
TOOL_RESULT = json.dumps({"flights": [{"flight": "B6918", "price": 289}]})


def drive(client, run_id: str, **extra) -> list[dict]:
    """Run an Anthropic-shaped tool loop against ``/r/<run_id>/anthropic``."""
    messages: list[dict] = [{"role": "user", "content": "Cheapest SFO to JFK?"}]
    transcript: list[dict] = []
    for _ in range(6):
        response = client.post(
            f"/r/{run_id}/anthropic/v1/messages",
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 1024,
                "messages": messages,
                **extra,
            },
            headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01"},
        )
        response.raise_for_status()
        message = response.json()
        transcript.append(message)
        messages.append({"role": "assistant", "content": message["content"]})
        calls = [b for b in message["content"] if b.get("type") == "tool_use"]
        if not calls:
            break
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call["id"], "content": TOOL_RESULT}
                    for call in calls
                ],
            }
        )
    return transcript


def record_the_loop(proxy) -> list[dict]:
    with respx.mock:
        respx.post(MESSAGES).mock(
            side_effect=[httpx.Response(200, json=TOOL_CALL), httpx.Response(200, json=ANSWER)]
        )
        proxy.store.create_run(mode="record", run_id="ATAPE", name="flights")
        return drive(proxy.client, "ATAPE")


def test_a_tool_loop_records_with_its_timeline(proxy) -> None:
    transcript = record_the_loop(proxy)

    assert transcript == [TOOL_CALL, ANSWER]
    steps = proxy.store.list_steps("ATAPE")
    assert [s.idx for s in steps] == [0, 1]
    assert [s.model for s in steps] == ["claude-sonnet-5", "claude-sonnet-5"]
    assert steps[0].usage == {"input_tokens": 412, "output_tokens": 30}
    assert steps[0].response == TOOL_CALL

    # The tool step between them, reconstructed from the LLM boundary (DESIGN.md §2).
    (tool_call,) = proxy.store.list_tool_calls("ATAPE")
    assert tool_call.after_step_idx == 0
    assert tool_call.tool_name == "search_flights"
    assert tool_call.args == {"origin": "SFO", "destination": "JFK"}
    assert tool_call.result == {"flights": [{"flight": "B6918", "price": 289}]}


def test_the_api_key_goes_upstream_but_never_to_disk(proxy) -> None:
    with respx.mock:
        route = respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=ANSWER))
        proxy.store.create_run(mode="record", run_id="ATAPE")
        drive(proxy.client, "ATAPE")

    assert route.calls.last.request.headers["x-api-key"] == API_KEY
    (step,) = proxy.store.list_steps("ATAPE")
    assert step.request["headers"]["x-api-key"] == REDACTION_PLACEHOLDER
    proxy.store.conn.execute("PRAGMA wal_checkpoint(FULL)")
    assert API_KEY.encode() not in proxy.settings.db_path.read_bytes()


def test_the_loop_replays_with_no_network(proxy) -> None:
    recorded = record_the_loop(proxy)

    # No routes registered at all: any upstream call raises rather than being served.
    with respx.mock(assert_all_called=False):
        proxy.store.create_run(mode="replay", run_id="AREPLAY", replay_of="ATAPE")
        replayed = drive(proxy.client, "AREPLAY")

    assert replayed == recorded
    assert [s.response for s in proxy.store.list_steps("AREPLAY")] == [TOOL_CALL, ANSWER]
    assert not any(s.diverged for s in proxy.store.list_steps("AREPLAY"))
    assert proxy.store.count_steps("ATAPE") == 2  # the tape is never written to


def test_a_recorded_stream_replays_byte_identically(proxy) -> None:
    stream = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_09",'
        b'"type":"message","role":"assistant","model":"claude-sonnet-5","content":[],'
        b'"usage":{"input_tokens":5,"output_tokens":1}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"text","text":""}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"Hi"}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta","delta":'
        b'{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    body = {"model": "claude-sonnet-5", "max_tokens": 16, "messages": [], "stream": True}

    with respx.mock:
        respx.post(MESSAGES).mock(
            return_value=httpx.Response(
                200, content=stream, headers={"content-type": "text/event-stream"}
            )
        )
        proxy.store.create_run(mode="record", run_id="ATAPE")
        recorded = proxy.client.post("/r/ATAPE/anthropic/v1/messages", json=body)

    assert recorded.content == stream
    (step,) = proxy.store.list_steps("ATAPE")
    assert step.response["content"] == [{"type": "text", "text": "Hi"}]
    assert step.usage == {"input_tokens": 5, "output_tokens": 2}

    with respx.mock(assert_all_called=False):
        proxy.store.create_run(mode="replay", run_id="AREPLAY", replay_of="ATAPE")
        replayed = proxy.client.post("/r/AREPLAY/anthropic/v1/messages", json=body)

    assert replayed.content == stream
    assert replayed.headers["content-type"] == "text/event-stream"


def test_a_non_streaming_tape_answers_a_streaming_client(proxy) -> None:
    """One tape serves both kinds of client (DESIGN.md §5): with no recorded chunks to
    replay, the stream is synthesized from the stored message. Asking for a stream is
    also not *drift* — ``stream`` is transport, so it never reaches the fingerprint."""
    record_the_loop(proxy)
    same_question = {
        "model": "claude-sonnet-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Cheapest SFO to JFK?"}],
        "stream": True,
    }

    with respx.mock(assert_all_called=False):
        proxy.store.create_run(mode="replay", run_id="AREPLAY", replay_of="ATAPE")
        replayed = proxy.client.post("/r/AREPLAY/anthropic/v1/messages", json=same_question)

    assert replayed.headers["content-type"] == "text/event-stream"
    assert replayed.headers["x-agentvcr-diverged"] == "false"
    assert b"event: content_block_start" in replayed.content
    # …and it is the recorded message, re-emitted as events the SDK can consume.
    assert ANTHROPIC.accumulate_stream([replayed.content]) == TOOL_CALL
