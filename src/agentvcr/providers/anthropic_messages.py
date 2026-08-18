"""Anthropic messages wire format (``POST /v1/messages``).

Phase 3 (PLAN.md): its SSE event accumulation, fingerprinting, and record/replay
parity with the OpenAI provider — the shared provider tests are parameterized over
both. Implements :class:`~agentvcr.providers.base.Provider`.
"""

from __future__ import annotations

NAME = "anthropic"
RECORDED_PATHS = ("/v1/messages",)
