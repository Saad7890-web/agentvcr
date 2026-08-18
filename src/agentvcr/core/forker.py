"""Fork mode: replay a prefix, apply an edit, then go live on a child run.

Phase 4 (PLAN.md) fills this in, per DESIGN.md §6: create the child run
(``parent_run_id`` / ``fork_step``), store the edits, serve an edited assistant
response at the fork step, and patch outbound request bodies for tool-result and
prompt edits from the first live call onward.
"""

from __future__ import annotations
