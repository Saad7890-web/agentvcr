"""The provider interface — the load-bearing abstraction of the codebase.

Recording, replay, forking and diffing are format-agnostic; everything a wire format
needs to explain about itself lives behind this interface. Adding a provider (a new
API surface, a new vendor) must not require touching :mod:`agentvcr.core`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any, Protocol


class Provider(Protocol):
    """What ``core`` needs to know about a wire format."""

    #: Registry key and path segment, e.g. ``"openai"``.
    name: str
    #: Prefix the proxy mounts this provider under. It mirrors the upstream base URL,
    #: so a client's ``base_url`` swap is the only change: ``/openai/v1`` stands in for
    #: ``https://api.openai.com/v1``, ``/anthropic`` for ``https://api.anthropic.com``.
    mount_path: str
    #: Paths (relative to ``mount_path``) whose calls are recorded as steps. Everything
    #: else under the mount is proxied verbatim.
    recorded_paths: tuple[str, ...]

    def normalize(self, body: dict[str, Any]) -> dict[str, Any]:
        """Strip volatile fields so two equivalent requests compare equal."""

    def fingerprint(self, body: dict[str, Any]) -> str:
        """Stable hash of the normalized request (model + messages + tools)."""

    def is_streaming(self, body: dict[str, Any]) -> bool:
        """Whether the client asked for an SSE stream."""

    def accumulate_stream(self, chunks: list[bytes]) -> dict[str, Any]:
        """Fold a recorded SSE chunk sequence into the final response body."""

    def synthesize_stream(self, response: dict[str, Any]) -> list[bytes]:
        """Re-emit a stored final response as an SSE chunk sequence."""

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Tool calls requested by an assistant response (DESIGN.md §2)."""

    def extract_tool_results(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        """Tool results carried by a request's message list (DESIGN.md §2)."""

    def patch_tool_result(self, body: dict[str, Any], *, tool_call_id: str, result: Any) -> bool:
        """Rewrite the result of ``tool_call_id`` in an *outbound* request, in place.

        The inverse of :meth:`extract_tool_results`, and how a forked tool result
        reaches the model (DESIGN.md §6): the result lives inside the request that
        follows the call, so a fork edits it on the way upstream. Returns whether a
        matching result was found — a request from before the call carries none.
        """

    def model_of(self, body: dict[str, Any]) -> str | None:
        """Model name for display and step metadata."""

    def messages_of(self, body: dict[str, Any]) -> list[Any]:
        """The request's message list — what run grouping chains on (DESIGN.md §4)."""

    def assistant_text(self, response: dict[str, Any]) -> str | None:
        """Assistant text of a final response, for terminal and UI previews."""

    def usage_of(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """Token usage from a final response, if the format reports one."""


def stable_hash(payload: Any) -> str:
    """Shared fingerprint helper: sha256 over a canonical JSON encoding."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def iter_sse_events(raw: bytes) -> Iterator[tuple[str | None, str]]:
    """Yield ``(event, data)`` for every message in a raw SSE byte stream.

    Both wire formats stream over SSE, but they carry the event type differently:
    OpenAI puts it inside the JSON payload, Anthropic dispatches on the ``event:``
    line and its SDK reads that line. One parser serves both — and it takes the whole
    concatenated stream, so a chunk boundary landing mid-line costs nothing.
    """
    event: str | None = None
    data: list[str] = []
    for line in raw.replace(b"\r\n", b"\n").split(b"\n"):
        if not line.strip():
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(b":"):  # a comment, i.e. a keep-alive
            continue
        field, _, value = line.partition(b":")
        if value[:1] == b" ":  # a single leading space after the colon is separator, not data
            value = value[1:]
        decoded = value.decode("utf-8", "replace")
        if field == b"event":
            event = decoded.strip()
        elif field == b"data":
            data.append(decoded)
    if data:  # a stream that ended without its final blank line
        yield event, "\n".join(data)
