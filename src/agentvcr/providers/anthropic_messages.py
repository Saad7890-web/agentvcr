"""Anthropic messages wire format (``POST /anthropic/v1/messages``).

The second wire format the proxy speaks, and the one that proves the provider
abstraction carries its weight: nothing in :mod:`agentvcr.core` changed to record,
replay or diff an Anthropic run. Everything the format needs to explain about itself
is here.

Two shapes differ enough from OpenAI's to be worth naming:

* **Content is a list of blocks**, not a string. Text, tool calls and extended
  thinking are all blocks of the assistant's ``content``, so ``assistant_text`` joins
  rather than picks, and a tool call is a ``tool_use`` block rather than a field of
  its own.
* **A tool result is a user message.** It arrives as a ``tool_result`` block inside
  the *next* request's ``content``, which is exactly the §2 mechanism the tool
  timeline is reconstructed from — just spelled differently.

Streaming is an event-per-block protocol (``content_block_start`` → deltas →
``content_block_stop``) with the event type on the SSE ``event:`` line, which the
Anthropic SDK dispatches on. Synthesis re-emits both halves so a synthesized stream is
one a real client can consume.
"""

from __future__ import annotations

import json
from typing import Any

from .base import iter_sse_events, stable_hash

NAME = "anthropic"
MOUNT_PATH = "/anthropic"
RECORDED_PATHS = ("/v1/messages",)

#: Request fields describing transport or bookkeeping rather than the question asked.
VOLATILE_REQUEST_FIELDS = frozenset({"stream", "metadata"})

#: Content-block field each text-shaped delta appends to, by delta type.
_TEXT_DELTAS = {
    "text_delta": "text",
    "thinking_delta": "thinking",
    "signature_delta": "signature",
}


class AnthropicMessagesProvider:
    """The Anthropic messages implementation of the provider interface."""

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
        """Every ``text`` block joined — a reply can legitimately span several."""
        parts = [
            block["text"]
            for block in _blocks(response.get("content"))
            if block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        joined = " ".join(part for part in parts if part)
        return joined or None

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Tool calls requested by an assistant response (DESIGN.md §2)."""
        return [
            {
                "tool_call_id": block.get("id"),
                "tool_name": block.get("name"),
                "args": block.get("input"),
            }
            for block in _blocks(response.get("content"))
            if block.get("type") == "tool_use"
        ]

    def extract_tool_results(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        """Tool results carried by a request's message list (DESIGN.md §2).

        They ride inside a *user* message: the agent appends the result as a
        ``tool_result`` block before asking what to do next.
        """
        results: list[dict[str, Any]] = []
        for message in self.messages_of(body):
            if not isinstance(message, dict):
                continue
            for block in _blocks(message.get("content")):
                if block.get("type") == "tool_result":
                    results.append(
                        {
                            "tool_call_id": block.get("tool_use_id"),
                            "result": _result_content(block.get("content")),
                        }
                    )
        return results

    def patch_tool_result(self, body: dict[str, Any], *, tool_call_id: str, result: Any) -> bool:
        """Rewrite a tool result on its way upstream (DESIGN.md §6).

        The replacement goes in as a plain string, which the format accepts wherever a
        block list does — and which :func:`_result_content` reads back unchanged.
        """
        for message in self.messages_of(body):
            if not isinstance(message, dict):
                continue
            for block in _blocks(message.get("content")):
                if block.get("type") == "tool_result" and block.get("tool_use_id") == tool_call_id:
                    block["content"] = _as_content(result)
                    return True
        return False

    # -------------------------------------------------------------------- streams

    def accumulate_stream(self, chunks: list[bytes]) -> dict[str, Any]:
        """Fold a recorded SSE event sequence into the final message body."""
        final: dict[str, Any] = {"type": "message", "role": "assistant"}
        blocks: dict[int, dict[str, Any]] = {}
        partial_json: dict[int, str] = {}

        for _, payload in iter_sse_events(b"".join(chunks)):
            event = _parse(payload)
            if event is None:
                continue
            kind = event.get("type")
            if kind == "message_start" and isinstance(event.get("message"), dict):
                final.update(event["message"])
                final.pop("content", None)
            elif kind == "content_block_start":
                index = int(event.get("index", len(blocks)))
                blocks[index] = dict(event.get("content_block") or {})
                partial_json.setdefault(index, "")
            elif kind == "content_block_delta":
                _apply_delta(blocks, partial_json, event)
            elif kind == "content_block_stop":
                _close_block(blocks, partial_json, int(event.get("index", -1)))
            elif kind == "message_delta":
                final.update(event.get("delta") or {})
                if isinstance(event.get("usage"), dict):
                    final["usage"] = {**(final.get("usage") or {}), **event["usage"]}

        # A client that disconnects mid-stream leaves blocks that never got their
        # ``content_block_stop``. Close them anyway: a tool call recorded with empty
        # arguments would read as one made with none, which is a different bug.
        for index in sorted(partial_json):
            _close_block(blocks, partial_json, index)

        final["content"] = [blocks[i] for i in sorted(blocks)]
        return final

    def synthesize_stream(self, response: dict[str, Any]) -> list[bytes]:
        """Re-emit a stored final message as an SSE event sequence."""
        shell = {k: v for k, v in response.items() if k != "content"}
        opening = {"type": "message_start", "message": {**shell, "content": []}}
        out: list[bytes] = [_sse("message_start", opening)]

        for index, block in enumerate(_blocks(response.get("content"))):
            empty, deltas = _split_block(block)
            out.append(
                _sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": index, "content_block": empty},
                )
            )
            for delta in deltas:
                out.append(
                    _sse(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": index, "delta": delta},
                    )
                )
            out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}))

        message_delta: dict[str, Any] = {
            "type": "message_delta",
            "delta": {
                "stop_reason": response.get("stop_reason"),
                "stop_sequence": response.get("stop_sequence"),
            },
        }
        if isinstance(response.get("usage"), dict):
            message_delta["usage"] = response["usage"]
        out.append(_sse("message_delta", message_delta))
        out.append(_sse("message_stop", {"type": "message_stop"}))
        return out


def _blocks(content: Any) -> list[dict[str, Any]]:
    """Content as a block list. A plain string is the one-text-block shorthand."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _result_content(content: Any) -> Any:
    """A tool result's payload, unwrapped from however the client wrapped it."""
    if isinstance(content, list):
        texts = [b.get("text") for b in _blocks(content) if b.get("type") == "text"]
        if texts and all(isinstance(t, str) for t in texts):
            return _maybe_json("".join(texts))
        return content
    return _maybe_json(content)


def _as_content(value: Any) -> str:
    """The inverse of :func:`_maybe_json`: a tool result as the string the wire wants."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _maybe_json(value: Any) -> Any:
    """Tool results travel as JSON-in-a-string; decode when they parse."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _parse(payload: str) -> dict[str, Any] | None:
    try:
        event = json.loads(payload)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _apply_delta(
    blocks: dict[int, dict[str, Any]], partial_json: dict[int, str], event: dict[str, Any]
) -> None:
    index = int(event.get("index", 0))
    block = blocks.setdefault(index, {})
    delta = event.get("delta") or {}
    kind = delta.get("type")
    if kind == "input_json_delta":
        # Tool arguments stream as a JSON *string* split across events; it only
        # becomes an object once the block closes.
        partial_json[index] = partial_json.get(index, "") + (delta.get("partial_json") or "")
        return
    target = _TEXT_DELTAS.get(str(kind))
    if target is not None:
        block[target] = (block.get(target) or "") + (delta.get(target) or "")
        return
    # An unrecognized delta (a new block type) is kept rather than dropped.
    for key, value in delta.items():
        if key != "type":
            block[key] = value


def _close_block(
    blocks: dict[int, dict[str, Any]], partial_json: dict[int, str], index: int
) -> None:
    raw = partial_json.pop(index, "")
    block = blocks.get(index)
    if block is None or not raw:
        return
    parsed = _parse(raw)
    # Truncated arguments (a stream that died mid-block) stay as the string they were.
    block["input"] = parsed if parsed is not None else raw


def _split_block(block: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A block's empty opening shell and the deltas that refill it."""
    kind = block.get("type")
    if kind == "tool_use":
        start = {**block, "input": {}}
        arguments = json.dumps(block.get("input") or {}, separators=(",", ":"), ensure_ascii=False)
        return start, [{"type": "input_json_delta", "partial_json": arguments}]
    if kind == "text":
        return {**block, "text": ""}, [{"type": "text_delta", "text": block.get("text") or ""}]
    if kind == "thinking":
        start = {**block, "thinking": "", "signature": ""}
        deltas = [{"type": "thinking_delta", "thinking": block.get("thinking") or ""}]
        if block.get("signature"):
            deltas.append({"type": "signature_delta", "signature": block["signature"]})
        return start, deltas
    # A block shape we do not know how to split streams whole, in its start event.
    return dict(block), []


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n".encode()


PROVIDER = AnthropicMessagesProvider()
