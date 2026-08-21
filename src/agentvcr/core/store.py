"""SQLite storage layer and migration runner.

One database file per project (``.agentvcr/agentvcr.db`` by default). Plain ``sqlite3``
with a ``user_version``-based migration runner — no ORM (standing decision, PLAN.md).
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import STATUS_ACTIVE, Edit, Run, Step, ToolCall

# Crockford base32, as used by ULID: no I, L, O or U.
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id(*, now_ms: int | None = None) -> str:
    """A short, lexicographically sortable id (ULID-shaped: 10 chars time + 6 random)."""
    ms = int(time.time() * 1000) if now_ms is None else now_ms
    out = []
    for _ in range(10):
        ms, rem = divmod(ms, 32)
        out.append(_ULID_ALPHABET[rem])
    time_part = "".join(reversed(out))
    rand_part = "".join(secrets.choice(_ULID_ALPHABET) for _ in range(6))
    return time_part + rand_part


def utcnow() -> str:
    """UTC timestamp in ISO-8601 with a trailing ``Z`` — the one time format we store."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


MIGRATIONS: list[str] = [
    # v1 — initial schema (DESIGN.md §8)
    """
    CREATE TABLE runs (
      id            TEXT PRIMARY KEY,
      name          TEXT,
      created_at    TEXT NOT NULL,
      mode          TEXT NOT NULL,
      parent_run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
      fork_step     INTEGER,
      command       TEXT,
      provider      TEXT,
      upstream_url  TEXT,
      status        TEXT NOT NULL DEFAULT 'active',
      meta_json     TEXT
    );
    CREATE INDEX idx_runs_created_at ON runs(created_at DESC);
    CREATE INDEX idx_runs_parent ON runs(parent_run_id);

    CREATE TABLE steps (
      id              INTEGER PRIMARY KEY,
      run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      idx             INTEGER NOT NULL,
      request_json    TEXT,
      response_json   TEXT,
      response_chunks BLOB,
      fingerprint     TEXT,
      model           TEXT,
      usage_json      TEXT,
      latency_ms      INTEGER,
      diverged        INTEGER NOT NULL DEFAULT 0,
      started_at      TEXT,
      UNIQUE (run_id, idx)
    );
    CREATE INDEX idx_steps_fingerprint ON steps(fingerprint);

    CREATE TABLE tool_calls (
      id             INTEGER PRIMARY KEY,
      run_id         TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      after_step_idx INTEGER NOT NULL,
      tool_name      TEXT,
      args_json      TEXT,
      result_json    TEXT,
      tool_call_id   TEXT
    );
    CREATE INDEX idx_tool_calls_run ON tool_calls(run_id, after_step_idx);

    CREATE TABLE edits (
      id         INTEGER PRIMARY KEY,
      run_id     TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      step_idx   INTEGER NOT NULL,
      kind       TEXT NOT NULL,
      patch_json TEXT
    );
    CREATE INDEX idx_edits_run ON edits(run_id, step_idx);
    """,
    # v2 — upstream status per step. Failed calls are recorded like any other step so
    # that an SDK's retry sequence replays exactly as it was recorded (see recorder.py).
    """
    ALTER TABLE steps ADD COLUMN status_code INTEGER;
    """,
    # v3 — replay lineage. A replay is a run of its own, linked to the tape it served
    # from, so divergence is recorded against the replay instead of scribbled onto the
    # original recording, and a run can be diffed against its own replay (DESIGN.md §4).
    """
    ALTER TABLE runs ADD COLUMN replay_of TEXT REFERENCES runs(id) ON DELETE SET NULL;
    CREATE INDEX idx_runs_replay_of ON runs(replay_of);
    """,
]

SCHEMA_VERSION = len(MIGRATIONS)


class SchemaTooNewError(RuntimeError):
    """The database was written by a newer agentvcr than the one opening it."""


def _dumps_meta(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


class Store:
    """Thin wrapper around a SQLite connection holding all tape reads and writes."""

    def __init__(self, connection: sqlite3.Connection, *, path: Path | None = None) -> None:
        self.conn = connection
        self.path = path
        # One connection serves the whole process (the proxy handles calls concurrently),
        # so writes and explicit transactions are serialized here. WAL only isolates
        # separate connections; two BEGINs on one connection are an error.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def open(cls, db_path: str | os.PathLike[str] | None = None) -> Store:
        """Open (creating parent dirs) and migrate a database. ``None`` → in-memory."""
        if db_path is None:
            conn = sqlite3.connect(":memory:", isolation_level=None)
            path = None
        else:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if path is not None:
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        store = cls(conn, path=path)
        store.migrate()
        return store

    def migrate(self) -> int:
        """Apply pending migrations; returns the resulting schema version."""
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise SchemaTooNewError(
                f"tape was written by a newer agentvcr (schema v{version}, "
                f"this build understands v{SCHEMA_VERSION}); upgrade agentvcr to read it"
            )
        for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
            # executescript() commits any open transaction, so the BEGIN/COMMIT that
            # makes each migration atomic has to live inside the script itself.
            self.conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {i};\nCOMMIT;")
        return self.schema_version

    @property
    def schema_version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------------- runs

    def create_run(
        self,
        *,
        mode: str,
        run_id: str | None = None,
        name: str | None = None,
        parent_run_id: str | None = None,
        fork_step: int | None = None,
        replay_of: str | None = None,
        command: list[str] | None = None,
        provider: str | None = None,
        upstream_url: str | None = None,
        status: str = STATUS_ACTIVE,
        meta: dict[str, Any] | None = None,
    ) -> Run:
        run = Run(
            id=run_id or new_id(),
            created_at=utcnow(),
            mode=mode,
            name=name,
            parent_run_id=parent_run_id,
            fork_step=fork_step,
            replay_of=replay_of,
            command=command,
            provider=provider,
            upstream_url=upstream_url,
            status=status,
            meta=meta or {},
        )
        row = run.to_row()
        self._insert("runs", row)
        return run

    def get_run(self, run_id: str) -> Run | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return Run.from_row(row) if row else None

    def list_runs(self, *, limit: int = 50, parent_run_id: str | None = None) -> list[Run]:
        if parent_run_id is None:
            rows = self.conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM runs WHERE parent_run_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (parent_run_id, limit),
            ).fetchall()
        return [Run.from_row(r) for r in rows]

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "meta" in fields:
            fields["meta_json"] = _dumps_meta(fields.pop("meta"))
        if "command" in fields:
            fields["command"] = _dumps_meta(fields["command"])
        allowed = {
            "name",
            "mode",
            "status",
            "provider",
            "upstream_url",
            "command",
            "parent_run_id",
            "fork_step",
            "replay_of",
            "meta_json",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"cannot update run field(s): {sorted(unknown)}")
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self.conn.execute(
                f"UPDATE runs SET {assignments} WHERE id = ?", (*fields.values(), run_id)
            )

    # ---------------------------------------------------------------------- steps

    def next_step_idx(self, run_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(idx) + 1, 0) FROM steps WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row[0])

    def add_step(self, step: Step, *, at_next_idx: bool = False) -> Step:
        """Persist a step. ``at_next_idx`` allocates ``idx`` under the write lock, so
        concurrent recordings on one run cannot collide on ``UNIQUE (run_id, idx)``."""
        with self._lock:
            if at_next_idx:
                step.idx = self.next_step_idx(step.run_id)
            step.id = self._insert("steps", step.to_row())
        return step

    def get_step(self, run_id: str, idx: int) -> Step | None:
        row = self.conn.execute(
            "SELECT * FROM steps WHERE run_id = ? AND idx = ?", (run_id, idx)
        ).fetchone()
        return Step.from_row(row) if row else None

    def last_step(self, run_id: str) -> Step | None:
        row = self.conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY idx DESC LIMIT 1", (run_id,)
        ).fetchone()
        return Step.from_row(row) if row else None

    def list_steps(self, run_id: str) -> list[Step]:
        rows = self.conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY idx", (run_id,)
        ).fetchall()
        return [Step.from_row(r) for r in rows]

    def count_steps(self, run_id: str) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM steps WHERE run_id = ?", (run_id,)).fetchone()
        return int(row[0])

    def mark_diverged(self, run_id: str, idx: int) -> None:
        """Flag a *replay* step whose request drifted from the tape (DESIGN.md §5)."""
        with self._lock:
            self.conn.execute(
                "UPDATE steps SET diverged = 1 WHERE run_id = ? AND idx = ?", (run_id, idx)
            )

    # ----------------------------------------------------------------- tool calls

    def add_tool_call(self, tool_call: ToolCall) -> ToolCall:
        tool_call.id = self._insert("tool_calls", tool_call.to_row())
        return tool_call

    def count_tool_calls(self, run_id: str, *, after_step_idx: int | None = None) -> int:
        """How many tool runs are materialized for a run, or for one step of it."""
        if after_step_idx is None:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE run_id = ?", (run_id,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE run_id = ? AND after_step_idx = ?",
                (run_id, after_step_idx),
            ).fetchone()
        return int(row[0])

    def list_tool_calls(self, run_id: str) -> list[ToolCall]:
        rows = self.conn.execute(
            "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY after_step_idx, id", (run_id,)
        ).fetchall()
        return [ToolCall.from_row(r) for r in rows]

    # ---------------------------------------------------------------------- edits

    def add_edit(self, edit: Edit) -> Edit:
        edit.id = self._insert("edits", edit.to_row())
        return edit

    def list_edits(self, run_id: str) -> list[Edit]:
        rows = self.conn.execute(
            "SELECT * FROM edits WHERE run_id = ? ORDER BY step_idx, id", (run_id,)
        ).fetchall()
        return [Edit.from_row(r) for r in rows]

    # -------------------------------------------------------------------- helpers

    def _insert(self, table: str, row: dict[str, Any]) -> int:
        columns = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        with self._lock:
            cursor = self.conn.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(row.values())
            )
            return int(cursor.lastrowid or 0)
