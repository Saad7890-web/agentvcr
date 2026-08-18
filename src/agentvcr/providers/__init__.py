"""Wire-format providers, keyed by the path prefix the proxy mounts them under."""

from __future__ import annotations

from .anthropic_messages import PROVIDER as ANTHROPIC
from .base import Provider, stable_hash
from .openai_chat import PROVIDER as OPENAI

#: Every wire format the proxy speaks, by name. Mount order is registration order.
REGISTRY: dict[str, Provider] = {OPENAI.name: OPENAI, ANTHROPIC.name: ANTHROPIC}


def get_provider(name: str) -> Provider | None:
    """The provider registered under ``name``, or ``None`` if the format is unknown."""
    return REGISTRY.get(name)


__all__ = ["ANTHROPIC", "OPENAI", "REGISTRY", "Provider", "get_provider", "stable_hash"]
