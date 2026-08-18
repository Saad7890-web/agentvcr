from __future__ import annotations

import json

from agentvcr.providers.openai_chat import PROVIDER

TOOL_RESPONSE = {
    "id": "chatcmpl-1",
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
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": '{"from":"SFO","to":"JFK"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def sse(*events: dict) -> list[bytes]:
    out = [f"data: {json.dumps(e)}\n\n".encode() for e in events]
    out.append(b"data: [DONE]\n\n")
    return out


def test_fingerprint_ignores_transport_fields() -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    streamed = {**body, "stream": True, "stream_options": {"include_usage": True}, "user": "u"}
    assert PROVIDER.fingerprint(body) == PROVIDER.fingerprint(streamed)
    assert PROVIDER.fingerprint(body) != PROVIDER.fingerprint({**body, "model": "other"})


def test_fingerprint_is_stable_across_key_order() -> None:
    a = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}
    b = {"temperature": 0, "messages": [{"role": "user", "content": "hi"}], "model": "m"}
    assert PROVIDER.fingerprint(a) == PROVIDER.fingerprint(b)


def test_accumulate_stream_joins_content_and_tool_call_deltas() -> None:
    chunks = sse(
        {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
        },
        {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search_", "arguments": '{"from":'},
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "flights", "arguments": '"SFO"}'}}
                        ]
                    },
                }
            ]
        },
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"total_tokens": 15},
        },
    )
    final = PROVIDER.accumulate_stream(chunks)

    message = final["choices"][0]["message"]
    assert message["content"] == "Hello"
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert final["usage"] == {"total_tokens": 15}
    (call,) = message["tool_calls"]
    # name and arguments both arrived split across chunks
    assert call == {
        "id": "call_1",
        "type": "function",
        "function": {"arguments": '{"from":"SFO"}', "name": "search_flights"},
    }


def test_accumulate_survives_split_and_malformed_chunks() -> None:
    raw = b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\ndata: {"cho'
    tail = b'ices":[{"index":0,"delta":{"content":"b"}}]}\n\ndata: not-json\n\ndata: [DONE]\n\n'
    final = PROVIDER.accumulate_stream([raw, tail])
    assert final["choices"][0]["message"]["content"] == "ab"


def test_synthesize_then_accumulate_round_trips() -> None:
    reproduced = PROVIDER.accumulate_stream(PROVIDER.synthesize_stream(TOOL_RESPONSE))
    assert reproduced["choices"] == TOOL_RESPONSE["choices"]
    assert reproduced["usage"] == TOOL_RESPONSE["usage"]
    assert reproduced["model"] == TOOL_RESPONSE["model"]


def test_extract_tool_calls_and_results() -> None:
    (call,) = PROVIDER.extract_tool_calls(TOOL_RESPONSE)
    assert call == {
        "tool_call_id": "call_1",
        "tool_name": "search_flights",
        "args": {"from": "SFO", "to": "JFK"},
    }
    (result,) = PROVIDER.extract_tool_results(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"flights":[]}'},
            ]
        }
    )
    assert result == {"tool_call_id": "call_1", "result": {"flights": []}}


def test_assistant_text_and_usage() -> None:
    assert PROVIDER.assistant_text(TOOL_RESPONSE) is None
    assert PROVIDER.usage_of(TOOL_RESPONSE)["total_tokens"] == 15
    text = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
    assert PROVIDER.assistant_text(text) == "done"
