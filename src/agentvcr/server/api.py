"""Control REST API consumed by the web UI (``/api/*``).

Phase 5 (PLAN.md) fills this in: runs, steps, tool calls, diff, edits, fork, and
``POST /api/runs/{id}/rerun`` which respawns the stored command with the fork
environment (only available for runs launched via ``agentvcr run``).
"""

from __future__ import annotations
