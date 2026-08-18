"""Anthropic messages wire format (``POST /anthropic/v1/messages``).

Phase 1 mounts this prefix as an **unrecorded passthrough** so an Anthropic client can
already point at the proxy: ``recorded_paths`` is empty, so no call takes the recording
path and every response carries ``X-AgentVCR-Recorded: false``.

Phase 3 (PLAN.md) fills in SSE event accumulation, fingerprinting and record/replay
parity with the OpenAI provider — the shared provider tests are parameterized over
both — by implementing the methods below and populating ``recorded_paths``.
"""

from __future__ import annotations

from typing import Any

NAME = "anthropic"
MOUNT_PATH = "/anthropic"
#: Empty until Phase 3; ``("/v1/messages",)`` switches recording on.
RECORDED_PATHS: tuple[str, ...] = ()


class AnthropicMessagesProvider:
    """Passthrough-only until Phase 3 implements the wire format."""

    name = NAME
    mount_path = MOUNT_PATH
    recorded_paths = RECORDED_PATHS

    def _unimplemented(self, what: str) -> NotImplementedError:
        return NotImplementedError(f"anthropic {what} lands in Phase 3 (PLAN.md)")

    def normalize(self, body: dict[str, Any]) -> dict[str, Any]:
        raise self._unimplemented("normalization")

    def fingerprint(self, body: dict[str, Any]) -> str:
        raise self._unimplemented("fingerprinting")

    def is_streaming(self, body: dict[str, Any]) -> bool:
        return bool(body.get("stream"))

    def model_of(self, body: dict[str, Any]) -> str | None:
        model = body.get("model")
        return model if isinstance(model, str) else None

    def messages_of(self, body: dict[str, Any]) -> list[Any]:
        messages = body.get("messages")
        return messages if isinstance(messages, list) else []

    def usage_of(self, response: dict[str, Any]) -> dict[str, Any] | None:
        usage = response.get("usage")
        return usage if isinstance(usage, dict) else None

    def assistant_text(self, response: dict[str, Any]) -> str | None:
        raise self._unimplemented("response rendering")

    def accumulate_stream(self, chunks: list[bytes]) -> dict[str, Any]:
        raise self._unimplemented("stream accumulation")

    def synthesize_stream(self, response: dict[str, Any]) -> list[bytes]:
        raise self._unimplemented("stream synthesis")

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        raise self._unimplemented("tool-call extraction")

    def extract_tool_results(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        raise self._unimplemented("tool-result extraction")


PROVIDER = AnthropicMessagesProvider()
