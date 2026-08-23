from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from typer.testing import CliRunner

from agentvcr import __version__
from agentvcr.cli import app
from agentvcr.core.models import Step, ToolCall
from agentvcr.core.store import Store

runner = CliRunner()

#: Rich's box drawing, and the colour codes around it.
_DECORATION = re.compile(r"\x1b\[[0-9;]*m|[\s\u2500-\u257f]+")


def rendered(result) -> str:
    """CLI output with Rich's panel formatting taken back out.

    Typer renders usage errors and help inside a bordered panel, wrapping the text to
    the terminal width — and *folding* a token that lands on the border itself. Where
    the break falls depends on the width and on the Rich version, so matching the raw
    output is a coin toss that passes locally and fails in CI. Stripping the decoration
    and the whitespace makes the assertion about the message rather than its layout.
    """
    return _DECORATION.sub("", result.output)


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_lists_serve() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "serve" in rendered(result)


def test_serve_rejects_unknown_preset() -> None:
    result = runner.invoke(app, ["serve", "--preset", "nope"])
    assert result.exit_code != 0


def test_help_lists_the_shipped_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("serve", "run", "runs", "show", "rm", "diff", "fork", "ui"):
        assert command in rendered(result)


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
    assert "--run" in rendered(result)


def test_replay_rejects_an_unknown_tape(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", "--mode", "replay", "--run", "NOPE", "--db", str(tmp_path / "a.db"), "--", "true"],
    )
    assert result.exit_code != 0
    assert "nosuchrun" in rendered(result)  # whitespace is stripped; see rendered()
    assert "NOPE" in rendered(result)


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


def _tool_row_run(db: Path) -> str:
    """A run whose timeline has a tool step between its two LLM steps."""
    with Store.open(db) as store:
        run = store.create_run(mode="record", name="flights", provider="openai")
        store.add_step(
            Step(run_id=run.id, idx=0, request={"body": {}}, response={}, status_code=200)
        )
        store.add_step(
            Step(run_id=run.id, idx=1, request={"body": {}}, response={}, status_code=200)
        )
        store.add_tool_call(
            ToolCall(
                run_id=run.id,
                after_step_idx=0,
                tool_name="search_flights",
                args={"origin": "SFO"},
                result={"flights": [{"price": 289}]},
                tool_call_id="call_1",
            )
        )
        return run.id


def test_show_interleaves_the_tool_timeline(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    run_id = _tool_row_run(db)

    result = runner.invoke(app, ["show", run_id, "--db", str(db)])

    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    tool_line = next(line for line in lines if "↳" in line)
    # the tool step sits between the call that asked for it and the next LLM step
    assert lines.index(tool_line) == lines.index(next(x for x in lines if x.startswith("0 "))) + 1
    assert "search_flights" in tool_line
    assert "289" in tool_line  # its result, not its arguments
    assert "tool" in tool_line


def _two_runs(db: Path, *, second_answer: str) -> tuple[str, str]:
    """Two one-step runs that differ only in what the model said."""
    ids = []
    with Store.open(db) as store:
        for answer in ("The cheapest is B6918 at $289.", second_answer):
            run = store.create_run(mode="record", provider="openai")
            store.add_step(
                Step(
                    run_id=run.id,
                    idx=0,
                    request={
                        "body": {"model": "m", "messages": [{"role": "user", "content": "?"}]}
                    },
                    response={"choices": [{"message": {"role": "assistant", "content": answer}}]},
                    fingerprint="same",
                    status_code=200,
                )
            )
            ids.append(run.id)
    return ids[0], ids[1]


def test_diff_of_identical_runs_says_so_and_exits_zero(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    a, b = _two_runs(db, second_answer="The cheapest is B6918 at $289.")

    result = runner.invoke(app, ["diff", a, b, "--db", str(db)])

    assert result.exit_code == 0
    assert "identical" in result.stdout


def test_diff_names_the_diverging_step_and_exits_one(tmp_path: Path) -> None:
    """Exit 1 on a difference is the diff(1) convention — it is what lets a replay in
    CI gate on the runs still matching."""
    db = tmp_path / "a.db"
    a, b = _two_runs(db, second_answer="No flights found.")

    result = runner.invoke(app, ["diff", a, b, "--db", str(db)])

    assert result.exit_code == 1
    assert "runs diverge at step 0 (response text differs)" in result.stdout
    assert "- The cheapest is B6918 at $289." in result.stdout
    assert "+ No flights found." in result.stdout


def test_diff_json_is_machine_readable(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    a, b = _two_runs(db, second_answer="No flights found.")

    result = runner.invoke(app, ["diff", a, b, "--json", "--db", str(db)])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["identical"] is False
    assert payload["steps"][0]["changes"][0]["where"] == "response"


def test_diff_reports_an_unknown_run(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    a, _ = _two_runs(db, second_answer="whatever")

    result = runner.invoke(app, ["diff", a, "NOPE", "--db", str(db)])

    assert result.exit_code == 2
    assert "nosuchrun:NOPE" in rendered(result)  # rendered() drops the whitespace


# --------------------------------------------------------------------------------- fork


def _forkable_run(db: Path) -> str:
    """A one-step recording whose step asks for a tool call, ready to be forked."""
    with Store.open(db) as store:
        run = store.create_run(mode="record", name="flights", provider="openai", command=["agent"])
        store.add_step(
            Step(
                run_id=run.id,
                idx=0,
                request={"body": {"model": "m", "messages": [{"role": "system", "content": "hi"}]}},
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
                status_code=200,
            )
        )
        return run.id


def test_fork_stores_the_edit_and_prints_how_to_re_run_it(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    run_id = _forkable_run(db)
    edit = tmp_path / "flights.json"
    edit.write_text(json.dumps({"flights": [{"price": 289}]}))

    result = runner.invoke(
        app,
        [
            "fork",
            run_id,
            "--at",
            "0",
            "--edit-tool-result",
            f"search_flights={edit}",
            "--db",
            str(db),
        ],
    )
    assert result.exit_code == 0
    assert "search_flights" in result.stdout
    # The re-run command is the deliverable: it is what a Re-run button will spawn.
    assert "--mode fork --run" in result.stdout
    assert "-- agent" in result.stdout

    with Store.open(db) as store:
        fork = next(r for r in store.list_runs() if r.parent_run_id == run_id)
        (stored,) = store.list_edits(fork.id)
    assert fork.fork_step == 0
    assert stored.patch["tool_call_id"] == "c1"
    assert stored.patch["result"] == {"flights": [{"price": 289}]}


def test_fork_explains_an_edit_it_cannot_apply(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    run_id = _forkable_run(db)
    edit = tmp_path / "x.json"
    edit.write_text("{}")

    result = runner.invoke(
        app,
        ["fork", run_id, "--at", "0", "--edit-tool-result", f"book={edit}", "--db", str(db)],
    )
    assert result.exit_code == 1
    assert "search_flights" in rendered(result)  # it names what the step did call


def test_fork_reports_an_unknown_run(tmp_path: Path) -> None:
    result = runner.invoke(app, ["fork", "NOPE", "--at", "0", "--db", str(tmp_path / "a.db")])
    assert result.exit_code == 1
    assert "nosuchrun" in rendered(result)  # whitespace is stripped; see rendered()


def test_show_names_the_fork_point_and_the_edit(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    run_id = _forkable_run(db)
    edit = tmp_path / "flights.json"
    edit.write_text("{}")
    runner.invoke(
        app,
        [
            "fork",
            run_id,
            "--at",
            "0",
            "--edit-tool-result",
            f"search_flights={edit}",
            "--db",
            str(db),
        ],
    )
    with Store.open(db) as store:
        fork = next(r for r in store.list_runs() if r.parent_run_id == run_id)

    result = runner.invoke(app, ["show", fork.id, "--db", str(db)])
    assert result.exit_code == 0
    assert f"forked from {run_id} at step 0" in result.stdout
    assert "edited tool result of search_flights" in result.stdout


def test_run_mode_fork_points_the_agent_at_the_fork_itself(tmp_path: Path) -> None:
    """A fork already exists — that is where its edits live — so nothing is created."""
    db = tmp_path / "a.db"
    run_id = _forkable_run(db)
    runner.invoke(app, ["fork", run_id, "--at", "0", "--db", str(db)])
    with Store.open(db) as store:
        fork_id = next(r for r in store.list_runs() if r.parent_run_id == run_id).id
        before = len(store.list_runs())

    result = runner.invoke(
        app,
        [
            "run",
            "--mode",
            "fork",
            "--run",
            fork_id,
            "--db",
            str(db),
            "--",
            sys.executable,
            "-c",
            f"import os; assert os.environ['AGENTVCR_RUN'] == {fork_id!r}",
        ],
    )
    assert result.exit_code == 0
    assert f"/r/{fork_id}" in result.output
    assert f"branching from {run_id}" in result.output  # the parent, not the fork itself

    with Store.open(db) as store:
        assert len(store.list_runs()) == before  # no new run
        assert store.get_run(fork_id).command[-1].startswith("import os")


def test_run_refuses_to_re_run_a_fork_that_already_branched(tmp_path: Path) -> None:
    """Position on the tape is how many steps the fork has, so a second run would
    resume in the middle of its own branch instead of starting it again."""
    db = tmp_path / "a.db"
    run_id = _forkable_run(db)
    runner.invoke(app, ["fork", run_id, "--at", "0", "--db", str(db)])
    with Store.open(db) as store:
        fork_id = next(r for r in store.list_runs() if r.parent_run_id == run_id).id
        store.add_step(Step(run_id=fork_id, idx=0, request={"body": {}}, status_code=200))

    result = runner.invoke(
        app, ["run", "--mode", "fork", "--run", fork_id, "--db", str(db), "--", "true"]
    )
    assert result.exit_code != 0
    assert "alreadyran" in rendered(result)  # whitespace is stripped; see rendered()


def test_run_mode_fork_needs_a_fork(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", "--mode", "fork", "--db", str(tmp_path / "a.db"), "--", "true"]
    )
    assert result.exit_code != 0
    assert "agentvcrfork" in rendered(result)  # whitespace is stripped; see rendered()


def test_run_mode_fork_rejects_a_plain_recording(tmp_path: Path) -> None:
    db = tmp_path / "a.db"
    run_id = _forkable_run(db)
    result = runner.invoke(
        app, ["run", "--mode", "fork", "--run", run_id, "--db", str(db), "--", "true"]
    )
    assert result.exit_code != 0
    assert "isnotafork" in rendered(result)
