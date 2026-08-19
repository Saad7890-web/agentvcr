from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from agentvcr import __version__
from agentvcr.cli import app
from agentvcr.core.models import Step
from agentvcr.core.store import Store

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_lists_serve() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.stdout


def test_serve_rejects_unknown_preset() -> None:
    result = runner.invoke(app, ["serve", "--preset", "nope"])
    assert result.exit_code != 0


def test_help_lists_the_phase_1_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("serve", "run", "runs", "show"):
        assert command in result.stdout


def test_run_requires_a_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--db", str(tmp_path / "a.db")])
    assert result.exit_code != 0


def test_run_records_argv_and_points_the_child_at_the_proxy(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    probe = tmp_path / "env.json"
    script = (
        "import json, os, pathlib, sys;"
        f"pathlib.Path({str(probe)!r}).write_text(json.dumps("
        "{k: os.environ[k] for k in ("
        "'AGENTVCR_RUN','AGENTVCR_MODE','OPENAI_BASE_URL','ANTHROPIC_BASE_URL')}))"
    )
    result = runner.invoke(
        app, ["run", "--name", "demo", "--db", str(db), "--", sys.executable, "-c", script]
    )
    assert result.exit_code == 0

    with Store.open(db) as store:
        (run,) = store.list_runs()
    assert run.name == "demo"
    assert run.mode == "record"
    assert run.status == "completed"
    assert run.meta["exit_code"] == 0
    assert run.command == [sys.executable, "-c", script]

    env = json.loads(probe.read_text())
    assert env["AGENTVCR_RUN"] == run.id
    assert env["AGENTVCR_MODE"] == "record"
    assert env["OPENAI_BASE_URL"] == f"http://127.0.0.1:8484/r/{run.id}/openai/v1"
    assert env["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:8484/r/{run.id}/anthropic"


def test_replay_needs_a_tape(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", "--mode", "replay", "--db", str(tmp_path / "a.db"), "--", "true"]
    )
    assert result.exit_code != 0
    assert "--run" in result.output


def test_replay_rejects_an_unknown_tape(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", "--mode", "replay", "--run", "NOPE", "--db", str(tmp_path / "a.db"), "--", "true"],
    )
    assert result.exit_code != 0
    assert "no such run" in result.output


def test_run_replay_creates_a_child_run_linked_to_the_tape(tmp_path: Path) -> None:
    """The agent is pointed at a fresh replay run, never at the tape itself."""
    db = tmp_path / "a.db"
    with Store.open(db) as store:
        store.create_run(mode="record", run_id="TAPE")

    result = runner.invoke(
        app,
        [
            "run",
            "--mode",
            "replay",
            "--run",
            "TAPE",
            "--db",
            str(db),
            "--",
            sys.executable,
            "-c",
            "import os; assert '/r/' in os.environ['OPENAI_BASE_URL']",
        ],
    )
    assert result.exit_code == 0

    with Store.open(db) as store:
        session = next(r for r in store.list_runs() if r.id != "TAPE")
    assert session.mode == "replay"
    assert session.replay_of == "TAPE"
    assert f"/r/{session.id}" in result.output  # the child is pointed at the replay run
    assert "replaying TAPE" in result.output
    assert "replayed 0 step(s)" in result.output


def test_run_propagates_the_child_exit_code(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    result = runner.invoke(
        app, ["run", "--db", str(db), "--", sys.executable, "-c", "raise SystemExit(3)"]
    )
    assert result.exit_code == 3
    with Store.open(db) as store:
        (run,) = store.list_runs()
    assert run.meta["exit_code"] == 3


def _recorded_run(db: Path) -> str:
    with Store.open(db) as store:
        run = store.create_run(
            mode="record", name="demo", provider="openai", command=["python", "a.py"]
        )
        store.add_step(
            Step(
                run_id=run.id,
                idx=0,
                request={"body": {"model": "llama-3.1-8b", "messages": []}},
                response={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "function": {
                                            "name": "search_flights",
                                            "arguments": '{"from":"SFO"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
                model="llama-3.1-8b",
                usage={"total_tokens": 15},
                latency_ms=120,
                status_code=200,
            )
        )
        store.add_step(
            Step(
                run_id=run.id,
                idx=1,
                request={"body": {}},
                response={"error": {"message": "rate limit"}},
                model="llama-3.1-8b",
                status_code=429,
                latency_ms=8,
            )
        )
        return run.id


def test_runs_lists_recorded_runs(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    run_id = _recorded_run(db)
    result = runner.invoke(app, ["runs", "--db", str(db)])
    assert result.exit_code == 0
    assert run_id in result.stdout
    assert "llama-3.1-8b" in result.stdout
    assert "demo" in result.stdout


def test_runs_is_friendly_when_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["runs", "--db", str(tmp_path / "empty.db")])
    assert result.exit_code == 0
    assert "no runs recorded" in result.stdout


def test_show_renders_the_step_table(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    run_id = _recorded_run(db)
    result = runner.invoke(app, ["show", run_id, "--db", str(db)])
    assert result.exit_code == 0
    assert "search_flights" in result.stdout  # tool call preview
    assert "120ms" in result.stdout
    assert "rate limit" in result.stdout  # the failed step is visible, not hidden
    assert "429" in result.stdout


def test_show_json_is_machine_readable(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    run_id = _recorded_run(db)
    result = runner.invoke(app, ["show", run_id, "--json", "--db", str(db)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["run"]["id"] == run_id
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["has_chunks"] is False


def test_show_reports_an_unknown_run(tmp_path: Path) -> None:
    result = runner.invoke(app, ["show", "NOPE", "--db", str(tmp_path / "a.db")])
    assert result.exit_code == 1
