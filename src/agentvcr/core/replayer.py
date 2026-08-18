"""Replay mode: serve recorded responses without ever contacting an upstream.

Phase 2 (PLAN.md) fills this in: positional matching as the primary rule, a
fingerprint comparison layered on top with the ``warn`` / ``strict`` / ``live-on-miss``
policies of DESIGN.md §5, SSE synthesis when the client asked for a stream, and a
structured error when the tape runs out of steps.
"""

from __future__ import annotations
