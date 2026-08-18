"""Record mode: capture a step, redact it, and decide which run it belongs to.

Phase 1 (PLAN.md) fills this in:

* ``redact_headers`` — drop ``authorization`` / ``x-api-key`` / ``api-key`` / cookies
  before anything touches disk. Secrets are forwarded upstream, never persisted.
* ``record_step`` — persist request, accumulated response, optional raw SSE tape,
  fingerprint, model, usage and latency as the next :class:`~agentvcr.core.models.Step`.
* run assignment — ``X-AgentVCR-Run`` when present, else the message-prefix chaining
  heuristic with an idle timeout (DESIGN.md §4).
"""

from __future__ import annotations

#: Header names never written to the database (DESIGN.md §11).
REDACTED_HEADERS = frozenset(
    {"authorization", "x-api-key", "api-key", "cookie", "set-cookie", "proxy-authorization"}
)
REDACTION_PLACEHOLDER = "<redacted>"
