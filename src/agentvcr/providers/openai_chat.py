"""OpenAI chat-completions wire format (``POST /v1/chat/completions``).

Covers every OpenAI-compatible endpoint — Groq, Gemini's compat layer, OpenRouter,
Ollama, vLLM — which is why this is the Phase 1 provider. Implements
:class:`~agentvcr.providers.base.Provider`.
"""

from __future__ import annotations

NAME = "openai"
RECORDED_PATHS = ("/v1/chat/completions",)
