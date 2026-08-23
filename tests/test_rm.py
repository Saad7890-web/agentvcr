"""Deleting runs — the only thing in agentvcr that destroys a tape.

Storage is quadratic in run length (DESIGN.md §8), so the first user to record a long
run needs a way out that is not ``rm -rf .agentvcr/``. What that costs is lineage: a
fork replays its prefix off its parent's tape, so these tests are mostly about the runs
that are *not* named on the command line.
"""

from __future__ import annotations

from pathlib import Path

from agentvcr.core.models import Edit, Step, ToolCall
from agentvcr.core.store import Store


def _tape(store: Store, run_id: str, *, steps: int = 2, created_at: str | None = None) -> None:
    """A recording with everything that hangs off one: steps, tool calls, an edit."""
    run = store.create_run(mode="record", run_id=run_id, provider="openai")
    if created_at is not None:
        store.conn.execute("UPDATE runs SET created_at = ? WHERE id = ?", (created_at, run.id))
    for idx in range(steps):
        store.add_step(
            Step(
                run_id=run.id,
                idx=idx,
                request={"messages": [{"role": "user", "content": f"q{idx}"}]},
                response={"choices": [{"message": {"content": f"a{idx}"}}]},
                fingerprint=f"fp{idx}",
                model="gpt-4o-mini",
            )
        )
        store.add_tool_call(
            ToolCall(run_id=run.id, after_step_idx=idx, tool_name="search", result={"hits": []})
        )
    store.add_edit(Edit(run_id=run.id, step_idx=0, kind="response", patch={"x": 1}))


def _counts(store: Store, run_id: str) -> tuple[int, int, int]:
    return (
        store.count_steps(run_id),
        store.count_tool_calls(run_id),
        len(store.list_edits(run_id)),
    )


# ------------------------------------------------------------------------------ store


def test_delete_takes_the_steps_tool_calls_and_edits_with_it(store: Store) -> None:
    _tape(store, "KEEP")
    _tape(store, "GONE")

    assert store.delete_runs(["GONE"]) == 1

    assert store.get_run("GONE") is None
    assert _counts(store, "GONE") == (0, 0, 0)
    assert _counts(store, "KEEP") == (2, 2, 1)  # the cascade stopped at the run it was given


def test_descendants_reach_through_a_replay_to_its_fork(store: Store) -> None:
    """A fork of a replay of a recording is two links from the recording, and just as
    dependent on it."""
    _tape(store, "TAPE")
    store.create_run(mode="replay", run_id="REPLAY", replay_of="TAPE")
    store.create_run(mode="fork", run_id="FORK", parent_run_id="REPLAY", fork_step=1)
    store.create_run(mode="record", run_id="UNRELATED")

    assert {r.id for r in store.run_descendants(["TAPE"])} == {"REPLAY", "FORK"}
    assert store.run_descendants(["UNRELATED"]) == []


def test_vacuum_returns_the_disk_a_deleted_tape_was_using(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    with Store.open(db) as store:
        _tape(store, "BIG", steps=0)
        for idx in range(40):
            store.add_step(
                Step(run_id="BIG", idx=idx, request={"messages": [{"content": "x" * 20_000}]})
            )
        # The rows are still in the write-ahead log until this checkpoint folds them in.
        store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        before = db.stat().st_size
        store.delete_runs(["BIG"])
        store.vacuum()

    assert db.stat().st_size < before / 2
