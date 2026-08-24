"""agentvcr — VCR for AI agents.

A drop-in proxy that records every LLM call (and, derived from them, every tool call)
of an agent run into a local SQLite tape, then replays it deterministically for free,
forks it at any step, and diffs two runs.

See DESIGN.md for the architecture and PLAN.md for the build order.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
