from __future__ import annotations

from pathlib import Path

from agentvcr.core.models import Edit, Step, ToolCall
from agentvcr.core.store import SCHEMA_VERSION, Store, new_id


def test_migrations_are_applied_and_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "agentvcr.db"
    with Store.open(db) as store:
        assert store.schema_version == SCHEMA_VERSION
        assert db.exists()
        assert store.migrate() == SCHEMA_VERSION

    with Store.open(db) as reopened:
        assert reopened.schema_version == SCHEMA_VERSION
        tables = {
            row[0]
            for row in reopened.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"runs", "steps", "tool_calls", "edits"} <= tables


def test_new_id_is_short_and_sortable() -> None:
    first = new_id(now_ms=1_700_000_000_000)
    second = new_id(now_ms=1_700_000_001_000)
    assert len(first) == 16
    assert first < second


def test_run_step_roundtrip(store: Store) -> None:
    run = store.create_run(mode="record", name="demo", command=["python", "agent.py"])
    assert store.next_step_idx(run.id) == 0

    store.add_step(
        Step(
            run_id=run.id,
            idx=0,
            request={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            response={"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
            fingerprint="abc123",
            model="gpt-4o-mini",
            usage={"total_tokens": 12},
            latency_ms=42,
        )
    )
    assert store.next_step_idx(run.id) == 1

    fetched = store.get_run(run.id)
    assert fetched is not None
    assert fetched.command == ["python", "agent.py"]
    assert fetched.status == "active"

    step = store.get_step(run.id, 0)
    assert step is not None
    assert step.request["messages"][0]["content"] == "hi"
    assert step.usage == {"total_tokens": 12}
    assert step.diverged is False

    store.mark_diverged(run.id, 0)
    assert store.list_steps(run.id)[0].diverged is True


def test_fork_lineage_tool_calls_and_edits(store: Store) -> None:
    parent = store.create_run(mode="record")
    fork = store.create_run(mode="fork", parent_run_id=parent.id, fork_step=6)

    store.add_tool_call(
        ToolCall(
            run_id=parent.id,
            after_step_idx=0,
            tool_name="search_flights",
            args={"from": "SFO"},
            result={"flights": []},
            tool_call_id="call_1",
        )
    )
    store.add_edit(Edit(run_id=fork.id, step_idx=6, kind="tool_result", patch={"flights": [1]}))

    assert [r.id for r in store.list_runs(parent_run_id=parent.id)] == [fork.id]
    (tool_call,) = store.list_tool_calls(parent.id)
    assert tool_call.tool_name == "search_flights"
    assert tool_call.args == {"from": "SFO"}
    (edit,) = store.list_edits(fork.id)
    assert edit.kind == "tool_result"
    assert edit.patch == {"flights": [1]}


def test_replay_lineage_is_separate_from_fork_lineage(store: Store) -> None:
    """A replay re-derives a tape; a fork branches away from one. Different columns."""
    tape = store.create_run(mode="record")
    replay = store.create_run(mode="replay", replay_of=tape.id)

    assert store.get_run(replay.id).replay_of == tape.id
    assert store.get_run(replay.id).parent_run_id is None
    assert store.list_runs(parent_run_id=tape.id) == []


def test_last_step_is_the_highest_indexed_one(store: Store) -> None:
    run = store.create_run(mode="record")
    assert store.last_step(run.id) is None

    store.add_step(Step(run_id=run.id, idx=0, request={}, status_code=200))
    store.add_step(Step(run_id=run.id, idx=1, request={}, status_code=429))

    last = store.last_step(run.id)
    assert last.idx == 1
    assert last.ok is False


def test_deleting_a_run_cascades(store: Store) -> None:
    run = store.create_run(mode="record")
    store.add_step(Step(run_id=run.id, idx=0, request={}))
    store.conn.execute("DELETE FROM runs WHERE id = ?", (run.id,))
    assert store.count_steps(run.id) == 0


def test_run_stats_folds_a_run_without_reading_its_conversations(store: Store) -> None:
    """What a run listing needs, from the narrow columns only (DESIGN.md §8)."""
    run = store.create_run(mode="record")
    store.add_step(
        Step(
            run_id=run.id,
            idx=0,
            request={"body": {"messages": ["…"]}},
            model="llama-3.1-8b",
            usage={"total_tokens": 100},
            status_code=200,
        )
    )
    store.add_step(
        Step(
            run_id=run.id,
            idx=1,
            request={},
            model="llama-3.1-8b",
            # Anthropic reports the halves rather than a total; both fold into one number.
            usage={"input_tokens": 20, "output_tokens": 5},
            status_code=429,
            diverged=True,
        )
    )
    store.add_tool_call(ToolCall(run_id=run.id, after_step_idx=0, tool_name="search"))

    stats = store.run_stats(run.id)
    assert stats.steps == 2
    assert stats.tool_calls == 1
    assert stats.tokens == 125
    assert stats.model == "llama-3.1-8b"
    assert stats.errors == 1
    assert stats.diverged is True


def test_run_stats_of_a_run_with_no_steps_reports_no_tokens(store: Store) -> None:
    """Zero tokens and "nobody counted" are different answers; a fresh fork is the latter."""
    run = store.create_run(mode="fork")
    assert store.run_stats(run.id).tokens is None
    assert store.run_stats(run.id).steps == 0
