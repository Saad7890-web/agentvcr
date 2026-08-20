"""The provider contract, asserted identically against every wire format.

:mod:`agentvcr.core` is written against this interface and nothing else, so anything
it relies on has to hold for all providers or the abstraction is a lie. Each format
supplies a *case* — one request carrying a tool result, one response carrying a tool
call — and every test below runs against all of them.

Format-specific corners (OpenAI's split tool-call deltas, Anthropic's block events)
live in ``test_providers_openai.py`` and ``test_providers_anthropic.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentvcr.providers import ANTHROPIC, OPENAI, REGISTRY, Provider


@dataclass(frozen=True)
class Case:
    """One provider plus the smallest traffic that exercises its whole interface."""

    provider: Provider
    request: dict[str, Any]
    response: dict[str, Any]
    #: Request fields that describe transport, not the question — fingerprints ignore them.
    volatile: dict[str, Any]
    text: str
    usage: dict[str, Any]


OPENAI_CASE = Case(
    provider=OPENAI,
    request={
        "model": "llama-3.1-8b",
        "messages": [
            {"role": "user", "content": "Cheapest SFO to JFK?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": '{"origin":"SFO"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"flights":[{"price":289}]}'},
        ],
    },
    response={
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "llama-3.1-8b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Looking now.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_flights",
                                "arguments": '{"origin":"SFO"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
    },
    volatile={"stream": True, "stream_options": {"include_usage": True}, "user": "u1"},
    text="Looking now.",
    usage={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
)

ANTHROPIC_CASE = Case(
    provider=ANTHROPIC,
    request={
        "model": "claude-sonnet-5",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Cheapest SFO to JFK?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "search_flights",
                        "input": {"origin": "SFO"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": '{"flights":[{"price":289}]}',
                    }
                ],
            },
        ],
    },
    response={
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [
            {"type": "text", "text": "Looking now."},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "search_flights",
                "input": {"origin": "SFO"},
            },
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 8, "output_tokens": 2},
    },
    volatile={"stream": True, "metadata": {"user_id": "u1"}},
    text="Looking now.",
    usage={"input_tokens": 8, "output_tokens": 2},
)

CASES = {"openai": OPENAI_CASE, "anthropic": ANTHROPIC_CASE}


@pytest.fixture(params=sorted(CASES), ids=sorted(CASES))
def case(request: pytest.FixtureRequest) -> Case:
    return CASES[request.param]


def test_every_registered_provider_has_a_case() -> None:
    """A new provider must arrive with its contract coverage, not after it."""
    assert set(REGISTRY) == set(CASES)


def test_a_provider_is_mounted_where_its_upstream_lives(case: Case) -> None:
    assert case.provider.mount_path.startswith("/")
    assert not case.provider.mount_path.endswith("/")
    assert all(path.startswith("/") for path in case.provider.recorded_paths)


def test_fingerprint_ignores_transport_fields(case: Case) -> None:
    """A streamed call and its non-streamed twin are the same question."""
    assert case.provider.fingerprint(
        {**case.request, **case.volatile}
    ) == case.provider.fingerprint(case.request)


def test_fingerprint_is_stable_across_key_order(case: Case) -> None:
    shuffled = dict(reversed(list(case.request.items())))
    assert case.provider.fingerprint(shuffled) == case.provider.fingerprint(case.request)


def test_fingerprint_moves_when_the_question_does(case: Case) -> None:
    drifted = {**case.request, "messages": case.request["messages"][:-1]}
    assert case.provider.fingerprint(drifted) != case.provider.fingerprint(case.request)


def test_streaming_flag_and_model(case: Case) -> None:
    assert case.provider.is_streaming(case.request) is False
    assert case.provider.is_streaming({**case.request, "stream": True}) is True
    assert case.provider.model_of(case.request) == case.request["model"]
    assert case.provider.model_of({}) is None
    assert case.provider.messages_of(case.request) == case.request["messages"]
    assert case.provider.messages_of({}) == []


def test_one_tape_answers_a_streaming_and_a_non_streaming_client(case: Case) -> None:
    """Synthesis is the fallback when no raw chunk tape was kept (DESIGN.md §5), so it
    has to preserve everything a caller can observe through the SDK."""
    reproduced = case.provider.accumulate_stream(case.provider.synthesize_stream(case.response))

    assert case.provider.assistant_text(reproduced) == case.provider.assistant_text(case.response)
    assert case.provider.extract_tool_calls(reproduced) == case.provider.extract_tool_calls(
        case.response
    )
    assert case.provider.usage_of(reproduced) == case.provider.usage_of(case.response)
    assert case.provider.model_of(reproduced) == case.provider.model_of(case.response)


def test_accumulating_nothing_is_not_a_crash(case: Case) -> None:
    """A client that disconnects one byte in still leaves a step behind (proxy.py)."""
    assert isinstance(case.provider.accumulate_stream([]), dict)
    assert isinstance(case.provider.accumulate_stream([b"data: not-json\n\n"]), dict)


def test_tool_calls_are_visible_in_the_response(case: Case) -> None:
    (call,) = case.provider.extract_tool_calls(case.response)
    assert call == {
        "tool_call_id": "call_1",
        "tool_name": "search_flights",
        "args": {"origin": "SFO"},
    }
    assert case.provider.extract_tool_calls({}) == []


def test_tool_results_are_visible_in_the_next_request(case: Case) -> None:
    """DESIGN.md §2: the result of a call rides inside the *following* request."""
    (result,) = case.provider.extract_tool_results(case.request)
    assert result == {"tool_call_id": "call_1", "result": {"flights": [{"price": 289}]}}
    assert case.provider.extract_tool_results({}) == []


def test_assistant_text_and_usage(case: Case) -> None:
    assert case.provider.assistant_text(case.response) == case.text
    assert case.provider.usage_of(case.response) == case.usage
    assert case.provider.usage_of({}) is None
