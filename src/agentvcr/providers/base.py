"""The provider interface — the load-bearing abstraction of the codebase.

Recording, replay, forking and diffing are format-agnostic; everything a wire format
needs to explain about itself lives behind this interface. Adding a provider (a new
API surface, a new vendor) must not require touching :mod:`agentvcr.core`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol


class Provider(Protocol):
    """What ``core`` needs to know about a wire format."""

    #: Path prefix the proxy mounts this provider under, e.g. ``"openai"``.
    name: str
    #: Paths (relative to the prefix) whose calls are recorded as steps.
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

    def model_of(self, body: dict[str, Any]) -> str | None:
        """Model name for display and step metadata."""

    def usage_of(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """Token usage from a final response, if the format reports one."""


def stable_hash(payload: Any) -> str:
    """Shared fingerprint helper: sha256 over a canonical JSON encoding."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
