"""Corners of the Anthropic messages format that the shared contract cannot reach.

The interesting difference from OpenAI is the stream: Anthropic sends an event per
content block (``content_block_start`` → deltas → ``content_block_stop``) with the type
on the SSE ``event:`` line, and a tool call's arguments arrive as a JSON *string* split
across events. The sample below is shaped like the real thing, keep-alive ping and all.
"""

from __future__ import annotations

from agentvcr.providers.anthropic_messages import PROVIDER

EVENTS = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01",'
    b'"type":"message","role":"assistant","model":"claude-sonnet-5","content":[],'
    b'"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n\n',
    b'event: ping\ndata: {"type": "ping"}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"Let me "}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"look."}}\n\n',
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
    b'event: content_block_start\ndata: {"type":"content_block_start","index":1,'
    b'"content_block":{"type":"tool_use","id":"toolu_1","name":"search_flights","input":{}}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,'
    b'"delta":{"type":"input_json_delta","partial_json":"{\\"origin\\":"}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,'
    b'"delta":{"type":"input_json_delta","partial_json":"\\"SFO\\"}"}}\n\n',
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use",'
    b'"stop_sequence":null},"usage":{"output_tokens":42}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
)
STREAM = b"".join(EVENTS)


def test_accumulate_folds_block_events_into_one_message() -> None:
    final = PROVIDER.accumulate_stream([STREAM])

    assert final == {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        # message_start reports the input tokens, message_delta the final output count
        "usage": {"input_tokens": 25, "output_tokens": 42},
        "content": [
            {"type": "text", "text": "Let me look."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "search_flights",
                "input": {"origin": "SFO"},
            },
        ],
    }


def test_accumulate_survives_chunks_split_mid_line() -> None:
    """The proxy stores whatever the socket handed it; boundaries are not ours."""
    split = [STREAM[:137], STREAM[137:400], STREAM[400:]]
    assert PROVIDER.accumulate_stream(split) == PROVIDER.accumulate_stream([STREAM])


def test_synthesized_events_carry_the_type_the_sdk_dispatches_on() -> None:
    """The Anthropic SDK reads the ``event:`` line, not the payload's ``type``."""
    emitted = b"".join(PROVIDER.synthesize_stream(PROVIDER.accumulate_stream([STREAM])))

    for event in (b"message_start", b"content_block_start", b"content_block_delta"):
        assert b"event: " + event in emitted
    assert emitted.endswith(b'data: {"type":"message_stop"}\n\n')


def test_a_stream_that_died_mid_tool_call_keeps_the_partial_arguments() -> None:
    """A disconnect leaves a block with no ``content_block_stop``. Recording its
    arguments as ``{}`` would read as a call made with none — a different bug from the
    one the user is here to debug, so the partial JSON is kept as the string it is."""
    truncated = b"".join(EVENTS[:8])  # cut after the first half of the arguments

    (block,) = [
        b for b in PROVIDER.accumulate_stream([truncated])["content"] if b["type"] == "tool_use"
    ]
    assert block["input"] == '{"origin":'


def test_thinking_blocks_round_trip() -> None:
    message = {
        "id": "msg_02",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [
            {"type": "thinking", "thinking": "The user wants a flight.", "signature": "sig-abc"},
            {"type": "text", "text": "Sure."},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }
    assert PROVIDER.accumulate_stream(PROVIDER.synthesize_stream(message)) == message


def test_an_unknown_block_type_survives_a_round_trip() -> None:
    """A format we do not model yet must pass through, not be truncated away."""
    message = {
        "id": "msg_03",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "server_tool_use_v9", "payload": {"a": 1}}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
    }
    assert PROVIDER.accumulate_stream(PROVIDER.synthesize_stream(message)) == message


def test_assistant_text_joins_every_text_block() -> None:
    response = {
        "content": [
            {"type": "text", "text": "First."},
            {"type": "tool_use", "id": "t", "name": "x", "input": {}},
            {"type": "text", "text": "Second."},
        ]
    }
    assert PROVIDER.assistant_text(response) == "First. Second."
    assert PROVIDER.assistant_text({"content": []}) is None
    assert PROVIDER.assistant_text({"content": [{"type": "tool_use"}]}) is None


def test_a_tool_result_is_unwrapped_however_the_client_wrapped_it() -> None:
    as_blocks = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": '{"flights":[]}'}],
                    }
                ],
            }
        ]
    }
    assert PROVIDER.extract_tool_results(as_blocks) == [
        {"tool_call_id": "t1", "result": {"flights": []}}
    ]

    not_json = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "no flights"}],
            }
        ]
    }
    assert PROVIDER.extract_tool_results(not_json) == [
        {"tool_call_id": "t1", "result": "no flights"}
    ]


def test_a_plain_string_content_is_the_one_text_block_shorthand() -> None:
    assert PROVIDER.assistant_text({"content": "hello"}) == "hello"
    assert PROVIDER.extract_tool_results({"messages": [{"role": "user", "content": "hi"}]}) == []
