"""``agentvcr rm`` — the only thing in the CLI that destroys a tape.

Storage is quadratic in run length (DESIGN.md §8), so the first user to record a long
run needs a way out that is not ``rm -rf .agentvcr/``. What that costs is lineage: a
fork replays its prefix off its parent's tape, so these tests are mostly about the runs
that are *not* named on the command line.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agentvcr.cli import app
from agentvcr.core.models import Edit, Step, ToolCall
from agentvcr.core.store import Store

runner = CliRunner()


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


# -------------------------------------------------------------------------------- cli


def test_rm_deletes_a_named_run(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    with Store.open(db) as store:
        _tape(store, "GONE")
        _tape(store, "KEEP")

    result = runner.invoke(app, ["rm", "GONE", "--yes", "--db", str(db)])
    assert result.exit_code == 0
    assert "deleted 1 run(s), 2 step(s)" in result.output

    with Store.open(db) as store:
        assert [r.id for r in store.list_runs()] == ["KEEP"]


def test_rm_refuses_to_orphan_a_fork(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    with Store.open(db) as store:
        _tape(store, "TAPE")
        store.create_run(mode="fork", run_id="FORK", parent_run_id="TAPE", fork_step=1)

    result = runner.invoke(app, ["rm", "TAPE", "--yes", "--db", str(db)])
    assert result.exit_code == 1
    assert "FORK" in result.output
    assert "--recursive" in result.output

    with Store.open(db) as store:
        assert store.get_run("TAPE") is not None  # nothing was deleted on the way to refusing
        assert store.get_run("FORK").parent_run_id == "TAPE"


def test_rm_recursive_deletes_the_whole_family(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    with Store.open(db) as store:
        _tape(store, "TAPE")
        store.create_run(mode="replay", run_id="REPLAY", replay_of="TAPE")
        store.create_run(mode="fork", run_id="FORK", parent_run_id="REPLAY", fork_step=1)
        _tape(store, "OTHER")

    result = runner.invoke(app, ["rm", "TAPE", "--recursive", "--yes", "--db", str(db)])
    assert result.exit_code == 0
    assert "deleted 3 run(s)" in result.output

    with Store.open(db) as store:
        assert [r.id for r in store.list_runs()] == ["OTHER"]


def test_rm_before_keeps_what_was_recorded_that_day(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    with Store.open(db) as store:
        _tape(store, "OLD", created_at="2026-08-01T09:00:00.000Z")
        _tape(store, "NEW", created_at="2026-08-23T00:30:00.000Z")

    result = runner.invoke(app, ["rm", "--before", "2026-08-23", "--yes", "--db", str(db)])
    assert result.exit_code == 0

    with Store.open(db) as store:
        assert [r.id for r in store.list_runs()] == ["NEW"]


def test_rm_before_matching_nothing_is_not_an_error(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    with Store.open(db) as store:
        _tape(store, "NEW", created_at="2026-08-23T00:30:00.000Z")

    result = runner.invoke(app, ["rm", "--before", "2020-01-01", "--yes", "--db", str(db)])
    assert result.exit_code == 0
    assert "no runs recorded before" in result.output

    with Store.open(db) as store:
        assert [r.id for r in store.list_runs()] == ["NEW"]


def test_rm_rejects_an_unparseable_date(tmp_path: Path) -> None:
    result = runner.invoke(app, ["rm", "--before", "last tuesday", "--db", str(tmp_path / "a.db")])
    assert result.exit_code != 0


def test_rm_rejects_an_unknown_run(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    with Store.open(db) as store:
        _tape(store, "KEEP")

    result = runner.invoke(app, ["rm", "NOPE", "--yes", "--db", str(db)])
    assert result.exit_code == 1
    assert "NOPE" in result.output

    with Store.open(db) as store:
        assert [r.id for r in store.list_runs()] == ["KEEP"]


def test_rm_with_no_selection_says_what_to_pass(tmp_path: Path) -> None:
    result = runner.invoke(app, ["rm", "--db", str(tmp_path / "a.db")])
    assert result.exit_code == 1
    assert "--before" in result.output


def test_rm_lists_what_it_will_delete_and_a_no_keeps_it(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    with Store.open(db) as store:
        _tape(store, "TAPE")

    result = runner.invoke(app, ["rm", "TAPE", "--db", str(db)], input="n\n")
    assert result.exit_code == 1
    assert "TAPE" in result.output  # the run is shown before the question, not just counted

    with Store.open(db) as store:
        assert store.get_run("TAPE") is not None
