"""The proxy routes: ``/openai/*`` and ``/anthropic/*``.

Phase 1–4 (PLAN.md) fill this in: forward recorded formats to the configured upstream
with the client's own auth headers, stream SSE through to the client while accumulating
the final message, and — depending on the run's mode — serve from the tape instead of
going upstream. Unrecorded paths under a provider prefix are proxied verbatim.
"""

from __future__ import annotations
