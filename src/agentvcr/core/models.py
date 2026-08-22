"""Row dataclasses mirroring the schema in DESIGN.md §8.

These are plain data holders — no ORM, no lazy loading. ``from_row`` takes a
``sqlite3.Row`` (the store sets ``row_factory``) and JSON columns are decoded eagerly
into ``*_json``-free attribute names.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

# Run.status values
STATUS_ACTIVE = "active"
STATUS_COMPLETED = "completed"
STATUS_DIVERGED = "diverged"

# Edit.kind values (DESIGN.md §6)
EDIT_RESPONSE = "response"
EDIT_TOOL_RESULT = "tool_result"
EDIT_REQUEST_PATCH = "request_patch"


def _loads(value: str | bytes | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def _dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def total_tokens(usage: Mapping[str, Any] | None) -> int | None:
    """Total tokens from a usage block. OpenAI reports a total, Anthropic the halves."""
    if not usage:
        return None
    total = usage.get("total_tokens")
    if total is None and {"input_tokens", "output_tokens"} <= usage.keys():
        total = usage["input_tokens"] + usage["output_tokens"]
    return total if isinstance(total, int) else None


@dataclass
class Run:
    """One agent execution."""

    id: str
    created_at: str
    mode: str
    name: str | None = None
    parent_run_id: str | None = None
    fork_step: int | None = None
    #: Tape this run replays, when it is a replay run. Replay lineage is deliberately
    #: separate from fork lineage: a replay re-derives an existing tape, a fork branches
    #: away from one (DESIGN.md §4).
    replay_of: str | None = None
    command: list[str] | None = None
    provider: str | None = None
    upstream_url: str | None = None
    status: str = STATUS_ACTIVE
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Run:
        return cls(
            id=row["id"],
            created_at=row["created_at"],
            mode=row["mode"],
            name=row["name"],
            parent_run_id=row["parent_run_id"],
            fork_step=row["fork_step"],
            replay_of=row["replay_of"],
            command=_loads(row["command"]),
            provider=row["provider"],
            upstream_url=row["upstream_url"],
            status=row["status"],
            meta=_loads(row["meta_json"], {}) or {},
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "mode": self.mode,
            "parent_run_id": self.parent_run_id,
            "fork_step": self.fork_step,
            "replay_of": self.replay_of,
            "command": _dumps(self.command),
            "provider": self.provider,
            "upstream_url": self.upstream_url,
            "status": self.status,
            "meta_json": _dumps(self.meta),
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Step:
    """One LLM call within a run."""

    run_id: str
    idx: int
    request: dict[str, Any]
    response: dict[str, Any] | None = None
    response_chunks: bytes | None = None
    fingerprint: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    latency_ms: int | None = None
    #: Upstream HTTP status. Errors are recorded as steps so a retry sequence replays.
    status_code: int | None = None
    #: Set on a *replay* step whose request no longer matches the tape it was served
    #: from. Never written to the tape being replayed (DESIGN.md §5).
    diverged: bool = False
    started_at: str | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Step:
        return cls(
            id=row["id"],
            run_id=row["run_id"],
            idx=row["idx"],
            request=_loads(row["request_json"], {}) or {},
            response=_loads(row["response_json"]),
            response_chunks=row["response_chunks"],
            fingerprint=row["fingerprint"],
            model=row["model"],
            usage=_loads(row["usage_json"]),
            latency_ms=row["latency_ms"],
            status_code=row["status_code"],
            diverged=bool(row["diverged"]),
            started_at=row["started_at"],
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "idx": self.idx,
            "request_json": _dumps(self.request),
            "response_json": _dumps(self.response),
            "response_chunks": self.response_chunks,
            "fingerprint": self.fingerprint,
            "model": self.model,
            "usage_json": _dumps(self.usage),
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "diverged": int(self.diverged),
            "started_at": self.started_at,
        }

    @property
    def ok(self) -> bool:
        """Whether the upstream answered successfully (unknown status counts as ok)."""
        return self.status_code is None or self.status_code < 400

    @property
    def total_tokens(self) -> int | None:
        """Tokens this call cost, however the wire format chose to report them."""
        return total_tokens(self.usage)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("response_chunks")
        data["has_chunks"] = self.response_chunks is not None
        return data


@dataclass
class ToolCall:
    """A client-side tool execution, reconstructed at the LLM boundary (DESIGN.md §2)."""

    run_id: str
    after_step_idx: int
    tool_name: str
    args: Any = None
    result: Any = None
    tool_call_id: str | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ToolCall:
        return cls(
            id=row["id"],
            run_id=row["run_id"],
            after_step_idx=row["after_step_idx"],
            tool_name=row["tool_name"],
            args=_loads(row["args_json"]),
            result=_loads(row["result_json"]),
            tool_call_id=row["tool_call_id"],
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "after_step_idx": self.after_step_idx,
            "tool_name": self.tool_name,
            "args_json": _dumps(self.args),
            "result_json": _dumps(self.result),
            "tool_call_id": self.tool_call_id,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Edit:
    """An edit attached to a fork run; the edit defines the fork point (DESIGN.md §6)."""

    run_id: str
    step_idx: int
    kind: str
    patch: Any = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Edit:
        return cls(
            id=row["id"],
            run_id=row["run_id"],
            step_idx=row["step_idx"],
            kind=row["kind"],
            patch=_loads(row["patch_json"]),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_idx": self.step_idx,
            "kind": self.kind,
            "patch_json": _dumps(self.patch),
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunStats:
    """Aggregates describing one run, for listings and headers.

    Deliberately *not* derived from :class:`Step` objects. Every step's request carries
    the whole conversation up to it (DESIGN.md §8), so building a run list out of full
    steps would read every tape in the database to print a column of counts.
    """

    steps: int = 0
    tool_calls: int = 0
    tokens: int | None = None
    model: str | None = None
    diverged: bool = False
    errors: int = 0

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]], *, tool_calls: int = 0) -> RunStats:
        """Fold the light step columns (``model``, ``usage_json``, status, divergence)."""
        steps = tokens = errors = 0
        model: str | None = None
        counted = False
        diverged = False
        for row in rows:
            steps += 1
            model = model or row["model"]
            step_tokens = total_tokens(_loads(row["usage_json"]))
            if step_tokens is not None:
                tokens, counted = tokens + step_tokens, True
            status = row["status_code"]
            if status is not None and status >= 400:
                errors += 1
            diverged = diverged or bool(row["diverged"])
        return cls(
            steps=steps,
            tool_calls=tool_calls,
            tokens=tokens if counted else None,
            model=model,
            diverged=diverged,
            errors=errors,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
