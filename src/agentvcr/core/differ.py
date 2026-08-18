"""Align two runs and diff them step by step (DESIGN.md §7).

Phase 3 (PLAN.md) fills this in: LCS alignment over step fingerprints with a positional
fallback, then per aligned step a message-level request diff, a text/JSON response diff
and tool name/args/result diffs. Consumed by ``agentvcr diff`` and the UI's
side-by-side view.
"""

from __future__ import annotations
