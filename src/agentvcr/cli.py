"""``agentvcr`` command line.

Phases 1–4 ship ``serve``, ``run``, ``runs``, ``show``, ``diff`` and ``fork``; ``ui``
arrives with the phase that gives it something to do.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import typer

from . import __version__
from .config import MISMATCH_POLICIES, MODES, PRESETS, ConfigError, Settings, load_settings
from .core import differ, forker
from .core.models import (
    EDIT_REQUEST_PATCH,
    EDIT_RESPONSE,
    EDIT_TOOL_RESULT,
    STATUS_COMPLETED,
    STATUS_DIVERGED,
    Run,
    Step,
    ToolCall,
)
from .core.store import Store
from .providers import get_provider

app = typer.Typer(
    name="agentvcr",
    help="VCR for AI agents: record, replay, fork and diff agent runs through a proxy.",
    no_args_is_help=True,
    add_completion=False,
)

DB_OPTION = typer.Option(None, "--db", help="SQLite tape path (default .agentvcr/agentvcr.db).")
CONFIG_OPTION = typer.Option(None, "--config", help="Path to an agentvcr.toml.")


def _settings(db: Path | None = None, config: Path | None = None, **overrides: Any) -> Settings:
    try:
        return load_settings(config_path=config, overrides={"db_path": db, **overrides})
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _store(settings: Settings) -> Store:
    return Store.open(settings.db_path)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind address (default 127.0.0.1)."),
    port: int | None = typer.Option(None, "--port", "-p", help="Bind port (default 8484)."),
    db: Path | None = DB_OPTION,
    mode: str | None = typer.Option(None, help=f"Default run mode: {'|'.join(MODES)}."),
    preset: str | None = typer.Option(None, help=f"Upstream preset: {'|'.join(sorted(PRESETS))}."),
    openai_upstream: str | None = typer.Option(
        None, "--openai-upstream", help="Base URL for OpenAI-format calls."
    ),
    anthropic_upstream: str | None = typer.Option(
        None, "--anthropic-upstream", help="Base URL for Anthropic-format calls."
    ),
    mismatch_policy: str | None = typer.Option(
        None, help=f"Replay fingerprint mismatch policy: {'|'.join(MISMATCH_POLICIES)}."
    ),
    config: Path | None = CONFIG_OPTION,
) -> None:
    """Start the proxy server."""
    import uvicorn

    from .server.app import create_app

    upstreams: dict[str, str] = {}
    if preset is not None:
        if preset not in PRESETS:
            raise typer.BadParameter(
                f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}"
            )
        upstreams.update(PRESETS[preset])
    if openai_upstream:
        upstreams["openai"] = openai_upstream
    if anthropic_upstream:
        upstreams["anthropic"] = anthropic_upstream

    settings = _settings(
        db,
        config,
        host=host,
        port=port,
        mode=mode,
        mismatch_policy=mismatch_policy,
        upstreams=upstreams or None,
    )

    typer.echo(
        f"agentvcr {__version__} — mode={settings.mode} db={settings.db_path}\n"
        f"  http://{settings.host}:{settings.port}/openai/v1   -> {settings.upstreams['openai']}\n"
        f"  http://{settings.host}:{settings.port}/anthropic    -> "
        f"{settings.upstreams['anthropic']}"
    )
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Run an agent against the proxy: agentvcr run [options] -- python agent.py",
)
def run(
    ctx: typer.Context,
    name: str | None = typer.Option(None, "--name", help="Human label for the run."),
    mode: str | None = typer.Option(None, "--mode", help=f"Run mode: {'|'.join(MODES)}."),
    tape: str | None = typer.Option(
        None,
        "--run",
        metavar="RUN",
        help="Tape to replay (--mode replay), or the fork to run (--mode fork).",
    ),
    db: Path | None = DB_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    """Create a run, point the child process at the proxy, and record what it does.

    The run id travels in the base URL (``/r/<id>/openai/v1``), so the agent needs no
    header support and no code change beyond reading ``OPENAI_BASE_URL`` — which both
    official SDKs already do.

    ``--mode replay --run <id>`` replays an existing tape instead: the child is pointed
    at a fresh *replay* run linked to that tape, so the replay is recorded in its own
    right and the original recording is never written to.

    ``--mode fork --run <id>`` re-runs a fork made by ``agentvcr fork``. That run
    already exists — it is where the edits live — so this points the agent at it rather
    than creating anything.
    """
    command = list(ctx.args)
    if not command:
        raise typer.BadParameter("no command given; use: agentvcr run -- python agent.py")

    settings = _settings(db, config, mode=mode)
    host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host
    base = f"http://{host}:{settings.port}"

    if settings.mode == "replay" and tape is None:
        raise typer.BadParameter("--mode replay needs a tape: agentvcr run --mode replay --run ID")
    if settings.mode == "fork" and tape is None:
        raise typer.BadParameter(
            "--mode fork needs a fork to run; make one with `agentvcr fork <run> --at N`"
        )
    if tape is not None and settings.mode not in ("replay", "fork"):
        raise typer.BadParameter(f"--run is for replaying or forking; mode is {settings.mode!r}")

    with _store(settings) as store:
        if tape is not None and store.get_run(tape) is None:
            raise typer.BadParameter(f"no such run to replay: {tape}")
        if settings.mode == "fork":
            created = _fork_to_run(store, tape, command=command)
        else:
            created = store.create_run(
                mode=settings.mode, name=name, command=command, replay_of=tape
            )
        env = {
            **os.environ,
            "AGENTVCR_RUN": created.id,
            "AGENTVCR_MODE": settings.mode,
            "OPENAI_BASE_URL": f"{base}/r/{created.id}/openai/v1",
            "ANTHROPIC_BASE_URL": f"{base}/r/{created.id}/anthropic",
        }
        # A replay names the tape it was pointed at; a fork names the run it branched
        # from, which is its parent — not the fork id, which is on the line already.
        origin = created.parent_run_id if settings.mode == "fork" else tape
        preposition = "branching from" if settings.mode == "fork" else "replaying"
        via = f" {preposition} {origin}" if origin else ""
        typer.echo(f"run {created.id} — mode={settings.mode}{via} via {base}/r/{created.id}")
        if not _server_is_up(base):
            typer.secho(
                f"warning: nothing is listening on {base}; start `agentvcr serve` first",
                fg=typer.colors.YELLOW,
                err=True,
            )
        try:
            completed = subprocess.run(command, env=env)  # noqa: S603 - the user's own argv
        except FileNotFoundError as exc:
            store.update_run(created.id, status="failed")
            raise typer.BadParameter(f"cannot run {command[0]!r}: {exc}") from exc

        # Re-read: the proxy may have written to this run while the child was alive
        # (a live-on-miss replay records where it went live), and that must survive.
        final = store.get_run(created.id) or created
        store.update_run(
            created.id,
            status=final.status if final.status == STATUS_DIVERGED else STATUS_COMPLETED,
            meta={**final.meta, "exit_code": completed.returncode},
        )
        steps = store.count_steps(created.id)

    verb = {"replay": "replayed", "fork": "forked"}.get(settings.mode, "recorded")
    typer.echo(f"{verb} {steps} step(s) — agentvcr show {created.id}")
    raise typer.Exit(completed.returncode)


def _fork_to_run(store: Store, fork_id: str, *, command: list[str]) -> Run:
    """The existing fork run to point the agent at, checked over first.

    A fork's steps are its branch, and its position on the tape is how many of them it
    has (DESIGN.md §4) — so re-running one that already ran would resume in the middle
    of its own branch rather than start it again. Fork afresh instead.
    """
    target = store.get_run(fork_id)
    assert target is not None  # the caller looked it up already
    if target.parent_run_id is None:
        raise typer.BadParameter(
            f"{fork_id} is not a fork; branch off it first with "
            f"`agentvcr fork {fork_id} --at <step>`"
        )
    recorded = store.count_steps(fork_id)
    if recorded:
        raise typer.BadParameter(
            f"fork {fork_id} already ran and holds {recorded} step(s); "
            f"make a fresh one with `agentvcr fork {target.parent_run_id} "
            f"--at {target.fork_step}`"
        )
    # The stored argv is what a Re-run button replays (DESIGN.md §6), and the fork was
    # created before anyone knew what would be re-run.
    store.update_run(fork_id, command=command)
    target.command = command
    return target


@app.command(name="runs")
def list_runs(
    limit: int = typer.Option(20, "--limit", "-n", help="How many runs to list."),
    db: Path | None = DB_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    """List recorded runs, newest first."""
    settings = _settings(db, config)
    with _store(settings) as store:
        rows = store.list_runs(limit=limit)
        if not rows:
            typer.echo(f"no runs recorded in {settings.db_path}")
            return
        typer.echo(_row(RUNS_WIDTHS, "RUN", "CREATED", "MODE", "STATUS", "STEPS", "MODEL", "NAME"))
        for item in rows:
            steps = store.list_steps(item.id)
            typer.echo(
                _row(
                    RUNS_WIDTHS,
                    item.id,
                    item.created_at[:19].replace("T", " "),
                    item.mode,
                    item.status,
                    str(len(steps)),
                    _first_model(steps),
                    _run_label(item),
                )
            )


@app.command()
def show(
    run_id: str = typer.Argument(..., metavar="RUN", help="Run id to display."),
    as_json: bool = typer.Option(False, "--json", help="Emit the run as JSON instead."),
    db: Path | None = DB_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    """Show a run's timeline: its LLM steps, and the tool runs between them."""
    settings = _settings(db, config)
    with _store(settings) as store:
        target = store.get_run(run_id)
        if target is None:
            typer.secho(f"no such run: {run_id}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        steps = store.list_steps(target.id)

        if as_json:
            typer.echo(
                json.dumps(
                    {
                        "run": target.as_dict(),
                        "steps": [s.as_dict() for s in steps],
                        "tool_calls": [t.as_dict() for t in store.list_tool_calls(target.id)],
                        "edits": [e.as_dict() for e in store.list_edits(target.id)],
                    },
                    indent=2,
                    default=str,
                )
            )
            return

        typer.echo(f"run {target.id}  mode={target.mode}  status={target.status}")
        typer.echo(f"  created {target.created_at}")
        if target.command:
            typer.echo(f"  command {' '.join(target.command)}")
        if target.parent_run_id:
            typer.echo(f"  forked from {target.parent_run_id} at step {target.fork_step}")
            for edit in store.list_edits(target.id):
                typer.echo(f"  edited {forker.describe(edit)}")
        if target.replay_of:
            live_from = target.meta.get("live_from")
            went_live = f", live from step {live_from}" if live_from is not None else ""
            typer.echo(f"  replay of {target.replay_of}{went_live}")
        if not steps:
            typer.echo("  (no steps recorded)")
            return

        provider = get_provider(target.provider or "openai")
        tools: dict[int, list[ToolCall]] = {}
        for call in store.list_tool_calls(target.id):
            tools.setdefault(call.after_step_idx, []).append(call)

        typer.echo("")
        typer.echo(_row(STEPS_WIDTHS, "STEP", "MODEL", "HTTP", "LATENCY", "TOKENS", "RESPONSE"))
        for step in steps:
            typer.echo(
                _row(
                    STEPS_WIDTHS,
                    str(step.idx),
                    step.model or "-",
                    str(step.status_code or "-"),
                    f"{step.latency_ms}ms" if step.latency_ms is not None else "-",
                    _tokens(step),
                    _preview(provider, step),
                )
            )
            # The tool runs that happened between this LLM call and the next one
            # (DESIGN.md §2) — the proxy never saw them execute, only their results.
            for call in tools.get(step.idx, []):
                typer.echo(
                    _row(
                        STEPS_WIDTHS,
                        "",
                        f"↳ {call.tool_name}",
                        "tool",
                        "-",
                        "-",
                        _compact(call.result)[:70] if call.result is not None else "(no result)",
                    )
                )


@app.command()
def fork(
    run_id: str = typer.Argument(..., metavar="RUN", help="Run to branch away from."),
    at: int = typer.Option(
        ..., "--at", metavar="N", help="Step to branch at, numbered as `agentvcr show` prints it."
    ),
    edit_response: Path | None = typer.Option(
        None,
        "--edit-response",
        metavar="FILE",
        help="Serve this response at step N instead of the recorded one.",
    ),
    edit_tool_result: list[str] = typer.Option(
        None,
        "--edit-tool-result",
        metavar="NAME=FILE",
        help="Give step N's call to tool NAME the result in FILE. Repeatable.",
    ),
    edit_message: list[str] = typer.Option(
        None,
        "--edit-message",
        metavar="I=FILE",
        help="Replace the content of message I of request N — a prompt edit.",
    ),
    name: str | None = typer.Option(None, "--name", help="Human label for the fork."),
    db: Path | None = DB_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    """Branch a recorded run at step N, with an edit, and print how to re-run it.

    The edit is what defines the fork point (DESIGN.md §6). Steps before it replay from
    the tape for free; from the edit onward the agent makes real calls and reacts to
    what you changed — which is the question a fork answers: *would it have gone
    differently?*

    The fork is created empty. It collects its steps when you re-run the agent against
    it with the command this prints.
    """
    settings = _settings(db, config)
    specs = _edit_specs(edit_response, edit_tool_result, edit_message)

    with _store(settings) as store:
        tape = store.get_run(run_id)
        if tape is None:
            typer.secho(f"no such run: {run_id}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        provider = get_provider(tape.provider or "openai")
        if provider is None:
            typer.secho(f"run {run_id} has an unknown format: {tape.provider}", fg="red", err=True)
            raise typer.Exit(1)
        try:
            child = forker.create(
                store, tape=tape, at=at, edits=specs, provider=provider, name=name
            )
        except forker.CannotFork as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc
        edits = store.list_edits(child.id)
        command = tape.command or ["<your agent command>"]

    typer.echo(f"fork {child.id} — from {tape.id} at step {at}")
    for edit in edits:
        typer.echo(f"  edited {forker.describe(edit)}")
    if not edits:
        typer.echo(f"  no edits: steps 0…{at - 1} replay, then the agent runs live")
    typer.echo("")
    typer.echo("re-run it with:")
    typer.echo(f"  agentvcr run --mode fork --run {child.id} -- {' '.join(command)}")


def _edit_specs(
    response: Path | None, tool_results: list[str] | None, messages: list[str] | None
) -> list[forker.EditSpec]:
    """Read the ``--edit-*`` flags off disk into the specs :mod:`forker` resolves."""
    specs: list[forker.EditSpec] = []
    if response is not None:
        specs.append(forker.EditSpec(kind=EDIT_RESPONSE, value=_read_json(response)))
    for kind, raw_values in ((EDIT_TOOL_RESULT, tool_results), (EDIT_REQUEST_PATCH, messages)):
        for raw in raw_values or []:
            target, separator, path = raw.partition("=")
            if not separator:
                raise typer.BadParameter(f"expected TARGET=FILE, got {raw!r}")
            specs.append(forker.EditSpec(kind=kind, value=_read_json(Path(path)), target=target))
    return specs


def _read_json(path: Path) -> Any:
    """An edit's new value. Anything JSON is fine; plain text is taken as a string."""
    try:
        text = path.read_text()
    except OSError as exc:
        raise typer.BadParameter(f"cannot read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except ValueError:
        return text


@app.command()
def diff(
    left_id: str = typer.Argument(..., metavar="RUN_A", help="The run to compare from."),
    right_id: str = typer.Argument(..., metavar="RUN_B", help="The run to compare to."),
    as_json: bool = typer.Option(False, "--json", help="Emit the diff as JSON instead."),
    db: Path | None = DB_OPTION,
    config: Path | None = CONFIG_OPTION,
) -> None:
    """Diff two runs step by step.

    Steps are aligned by fingerprint (DESIGN.md §7), then compared on what the agent actually did:
    request messages, response text, tool calls and tool results. Ids, timestamps and
    token counts differ between any two live calls and are not reported.

    Exits 1 when the runs differ, like ``diff(1)`` — so a replay in CI can gate on it.
    """
    settings = _settings(db, config)
    with _store(settings) as store:
        left = store.get_run(left_id)
        right = store.get_run(right_id)
        for run_id, found in ((left_id, left), (right_id, right)):
            if found is None:
                typer.secho(f"no such run: {run_id}", fg=typer.colors.RED, err=True)
                raise typer.Exit(2)
        assert left is not None and right is not None  # both checked above
        try:
            result = differ.diff_runs(store, left, right)
        except differ.IncomparableRuns as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(2) from exc
        counts = (store.count_steps(left.id), store.count_steps(right.id))

    if as_json:
        typer.echo(json.dumps(result.as_dict(), indent=2, default=str))
        raise typer.Exit(0 if result.identical else 1)

    typer.echo(f"diff {left.id} → {right.id}")
    typer.echo(f"  a  {_diff_side(left, counts[0])}")
    typer.echo(f"  b  {_diff_side(right, counts[1])}")
    typer.echo("")
    for pair in result.pairs:
        if pair.same:
            continue
        if not pair.matched:
            side = "a" if pair.left else "b"
            typer.echo(f"  step {pair.label}  present only in {side}")
            continue
        for change in pair.changes:
            typer.echo(f"  step {pair.label}  {change.detail}")
            if change.left is not None:
                typer.secho(f"    - {change.left}", fg=typer.colors.RED)
            if change.right is not None:
                typer.secho(f"    + {change.right}", fg=typer.colors.GREEN)
    if not result.identical:
        typer.echo("")
    typer.secho(result.summary, fg=typer.colors.GREEN if result.identical else typer.colors.YELLOW)
    raise typer.Exit(0 if result.identical else 1)


@app.command()
def version() -> None:
    """Print the agentvcr version."""
    typer.echo(__version__)


# ------------------------------------------------------------------------------ display

RUNS_WIDTHS = (18, 19, 6, 9, 5, 24)
STEPS_WIDTHS = (4, 22, 4, 8, 6)


def _row(widths: tuple[int, ...], *cells: str) -> str:
    """Left-aligned columns; the last cell runs free so long previews are not clipped."""
    out = [cell.ljust(width) for cell, width in zip(cells, widths, strict=False)]
    out.extend(cells[len(widths) :])
    return "  ".join(out).rstrip()


def _diff_side(item: Run, steps: int) -> str:
    """One line describing a run being diffed: what it is, not what is in it."""
    return f"{item.id:<18}  {item.mode:<10} {item.status:<10} {steps} step(s)  {_run_label(item)}"


def _run_label(item: Run) -> str:
    if item.name:
        return item.name
    if item.parent_run_id:
        return "fork"
    return f"replay of {item.replay_of}" if item.replay_of else "-"


def _first_model(steps: list[Step]) -> str:
    for step in steps:
        if step.model:
            return step.model
    return "-"


def _tokens(step: Step) -> str:
    usage = step.usage or {}
    total = usage.get("total_tokens")
    if total is None and {"input_tokens", "output_tokens"} <= usage.keys():
        total = usage["input_tokens"] + usage["output_tokens"]
    return str(total) if total is not None else "-"


def _preview(provider: Any, step: Step) -> str:
    if not step.ok:
        error = (step.response or {}).get("error")
        message = error.get("message") if isinstance(error, dict) else None
        return f"! {message or 'upstream error'}"[:70]
    if step.response is None:
        return "-"
    try:
        calls = provider.extract_tool_calls(step.response)
        text = provider.assistant_text(step.response)
    except NotImplementedError:  # a provider whose renderer is not written yet
        return "-"
    if calls:
        return "→ " + ", ".join(f"{c['tool_name']}({_compact(c['args'])})" for c in calls)[:68]
    return _compact(text)[:70] if text else "-"


def _compact(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return " ".join(text.split())


def _server_is_up(base: str) -> bool:
    import httpx

    try:
        return httpx.get(f"{base}/healthz", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
