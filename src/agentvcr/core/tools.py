"""Reconstruct the tool timeline from the LLM boundary alone (DESIGN.md §2).

A tool call appears in the assistant response of step *N*; its result appears as a
tool-result message inside the request of step *N+1*. Diffing consecutive request
message lists therefore yields tool name, arguments, result and the step that
triggered it — with zero client instrumentation. Phase 3 (PLAN.md) fills this in and
materializes the rows into ``tool_calls`` at record time.
"""

from __future__ import annotations
