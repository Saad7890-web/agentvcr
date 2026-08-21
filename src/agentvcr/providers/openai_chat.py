"""OpenAI chat-completions wire format (``POST {mount}/chat/completions``).

Covers every OpenAI-compatible endpoint — Groq, Gemini's compat layer, OpenRouter,
Ollama, vLLM — which is why this is the Phase 1 provider. Implements
:class:`~agentvcr.providers.base.Provider`.

Paths are relative to the provider's *mount*, which mirrors the upstream base URL:
a client's ``base_url`` is ``http://host:8484/openai/v1`` and the SDK appends
``/chat/completions``, exactly as it would against ``https://api.openai.com/v1``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from .base import iter_sse_events, stable_hash

NAME = "openai"
MOUNT_PATH = "/openai/v1"
RECORDED_PATHS = ("/chat/completions",)

#: Request fields describing transport or bookkeeping rather than the question asked.
#: Stripping them keeps a streamed call's fingerprint equal to its non-streamed twin.
VOLATILE_REQUEST_FIELDS = frozenset({"stream", "stream_options", "user", "metadata", "store"})


def iter_sse_data(raw: bytes) -> Iterator[str]:
    """Yield the payload of every SSE message. OpenAI never sets an ``event:`` line —
    the chunk's type lives in the JSON — so only the data half is of interest here."""
    for _, data in iter_sse_events(raw):
        yield data.strip()


class OpenAIChatProvider:
    """The OpenAI chat-completions implementation of the provider interface."""

    name = NAME
    mount_path = MOUNT_PATH
    recorded_paths = RECORDED_PATHS

    # ------------------------------------------------------------------- requests

    def normalize(self, body: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in body.items() if k not in VOLATILE_REQUEST_FIELDS}

    def fingerprint(self, body: dict[str, Any]) -> str:
        return stable_hash(self.normalize(body))

    def is_streaming(self, body: dict[str, Any]) -> bool:
        return bool(body.get("stream"))

    def model_of(self, body: dict[str, Any]) -> str | None:
        model = body.get("model")
        return model if isinstance(model, str) else None

    def messages_of(self, body: dict[str, Any]) -> list[Any]:
        messages = body.get("messages")
        return messages if isinstance(messages, list) else []

    # ------------------------------------------------------------------ responses

    def usage_of(self, response: dict[str, Any]) -> dict[str, Any] | None:
        usage = response.get("usage")
        return usage if isinstance(usage, dict) else None

    def assistant_text(self, response: dict[str, Any]) -> str | None:
        for choice in response.get("choices") or []:
            content = (choice.get("message") or {}).get("content")
            if isinstance(content, str) and content:
                return content
        return None

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Tool calls requested by an assistant response (DESIGN.md §2)."""
        calls: list[dict[str, Any]] = []
        for choice in response.get("choices") or []:
            for call in (choice.get("message") or {}).get("tool_calls") or []:
                function = call.get("function") or {}
                calls.append(
                    {
                        "tool_call_id": call.get("id"),
                        "tool_name": function.get("name"),
                        "args": _maybe_json(function.get("arguments")),
                    }
                )
        return calls

    def extract_tool_results(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        """Tool results carried by a request's message list (DESIGN.md §2)."""
        results: list[dict[str, Any]] = []
        for message in self.messages_of(body):
            if isinstance(message, dict) and message.get("role") == "tool":
                results.append(
                    {
                        "tool_call_id": message.get("tool_call_id"),
                        "result": _maybe_json(message.get("content")),
                    }
                )
        return results

    def patch_tool_result(self, body: dict[str, Any], *, tool_call_id: str, result: Any) -> bool:
        """Rewrite a tool result on its way upstream (DESIGN.md §6)."""
        for message in self.messages_of(body):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            if message.get("tool_call_id") == tool_call_id:
                message["content"] = _as_content(result)
                return True
        return False

    # -------------------------------------------------------------------- streams

    def accumulate_stream(self, chunks: list[bytes]) -> dict[str, Any]:
        """Fold a recorded SSE chunk sequence into the final response body."""
        final: dict[str, Any] = {"object": "chat.completion"}
        choices: dict[int, dict[str, Any]] = {}
        for payload in iter_sse_data(b"".join(chunks)):
            if payload == "[DONE]" or not payload:
                continue
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            for key in ("id", "model", "created", "system_fingerprint", "service_tier"):
                if event.get(key) is not None:
                    final[key] = event[key]
            if isinstance(event.get("usage"), dict):
                final["usage"] = event["usage"]
            for choice in event.get("choices") or []:
                _merge_choice_delta(choices, choice)
        final["choices"] = [choices[i] for i in sorted(choices)]
        for choice in final["choices"]:
            _drop_tool_call_indexes(choice["message"])
        return final

    def synthesize_stream(self, response: dict[str, Any]) -> list[bytes]:
        """Re-emit a stored final response as an SSE chunk sequence."""
        head = {
            "id": response.get("id", "chatcmpl-agentvcr"),
            "object": "chat.completion.chunk",
            "created": response.get("created", 0),
            "model": response.get("model"),
        }
        out: list[bytes] = []

        def emit(choices: list[dict[str, Any]], **extra: Any) -> None:
            out.append(_sse({**head, **extra, "choices": choices}))

        for choice in response.get("choices") or []:
            index = choice.get("index", 0)
            message = choice.get("message") or {}
            emit([{"index": index, "delta": {"role": message.get("role", "assistant")}}])
            if message.get("content"):
                emit([{"index": index, "delta": {"content": message["content"]}}])
            for position, call in enumerate(message.get("tool_calls") or []):
                function = call.get("function") or {}
                emit(
                    [
                        {
                            "index": index,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": position,
                                        "id": call.get("id"),
                                        "type": call.get("type", "function"),
                                        "function": {
                                            "name": function.get("name"),
                                            "arguments": function.get("arguments", ""),
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                )
            emit(
                [{"index": index, "delta": {}, "finish_reason": choice.get("finish_reason")}],
                **({"usage": response["usage"]} if response.get("usage") else {}),
            )
        out.append(b"data: [DONE]\n\n")
        return out


def _sse(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"data: {body}\n\n".encode()


def _as_content(value: Any) -> str:
    """The inverse of :func:`_maybe_json`: a tool result as the string the wire wants."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _maybe_json(value: Any) -> Any:
    """Tool arguments and results travel as JSON-in-a-string; decode when they parse."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _merge_choice_delta(choices: dict[int, dict[str, Any]], choice: dict[str, Any]) -> None:
    index = choice.get("index", 0)
    accumulated = choices.setdefault(
        index,
        {"index": index, "message": {"role": "assistant", "content": None}, "finish_reason": None},
    )
    if choice.get("finish_reason") is not None:
        accumulated["finish_reason"] = choice["finish_reason"]
    if choice.get("logprobs") is not None:
        accumulated["logprobs"] = choice["logprobs"]

    message = accumulated["message"]
    delta = choice.get("delta") or {}
    if delta.get("role"):
        message["role"] = delta["role"]
    for key in ("content", "refusal"):
        if delta.get(key):
            message[key] = (message.get(key) or "") + delta[key]
    for call_delta in delta.get("tool_calls") or []:
        _merge_tool_call_delta(message, call_delta)


def _merge_tool_call_delta(message: dict[str, Any], delta: dict[str, Any]) -> None:
    calls: list[dict[str, Any]] = message.setdefault("tool_calls", [])
    index = delta.get("index", len(calls))
    while len(calls) <= index:
        calls.append(
            {"index": len(calls), "id": None, "type": "function", "function": {"arguments": ""}}
        )
    slot = calls[index]
    if delta.get("id"):
        slot["id"] = delta["id"]
    if delta.get("type"):
        slot["type"] = delta["type"]
    function = delta.get("function") or {}
    # Both name and arguments can arrive split across chunks; append, never assign.
    if function.get("name"):
        slot["function"]["name"] = slot["function"].get("name", "") + function["name"]
    if function.get("arguments"):
        slot["function"]["arguments"] += function["arguments"]


def _drop_tool_call_indexes(message: dict[str, Any]) -> None:
    """Streamed tool calls carry an ``index``; the non-streamed shape does not."""
    for call in message.get("tool_calls") or []:
        call.pop("index", None)


PROVIDER = OpenAIChatProvider()
