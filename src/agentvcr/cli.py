"""``agentvcr`` command line.

Phases 1–2 ship ``serve``, ``run``, ``runs`` and ``show``; ``fork``, ``diff`` and
``ui`` arrive with the phases that give them something to do.
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
from .core.models import STATUS_COMPLETED, STATUS_DIVERGED, Run, Step
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
        None, "--run", metavar="RUN", help="Tape to replay (required by --mode replay)."
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
    """
    command = list(ctx.args)
    if not command:
        raise typer.BadParameter("no command given; use: agentvcr run -- python agent.py")

    settings = _settings(db, config, mode=mode)
    host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host
    base = f"http://{host}:{settings.port}"

    if settings.mode == "replay" and tape is None:
        raise typer.BadParameter("--mode replay needs a tape: agentvcr run --mode replay --run ID")
    if tape is not None and settings.mode != "replay":
        raise typer.BadParameter(f"--run is for replaying a tape; mode is {settings.mode!r}")

    with _store(settings) as store:
        if tape is not None and store.get_run(tape) is None:
            raise typer.BadParameter(f"no such run to replay: {tape}")
        created = store.create_run(mode=settings.mode, name=name, command=command, replay_of=tape)
        env = {
            **os.environ,
            "AGENTVCR_RUN": created.id,
            "AGENTVCR_MODE": settings.mode,
            "OPENAI_BASE_URL": f"{base}/r/{created.id}/openai/v1",
            "ANTHROPIC_BASE_URL": f"{base}/r/{created.id}/anthropic",
        }
        via = f" replaying {tape}" if tape else ""
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

    verb = "replayed" if settings.mode == "replay" else "recorded"
    typer.echo(f"{verb} {steps} step(s) — agentvcr show {created.id}")
    raise typer.Exit(completed.returncode)


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
    """Show a run's steps."""
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
        if target.replay_of:
            live_from = target.meta.get("live_from")
            went_live = f", live from step {live_from}" if live_from is not None else ""
            typer.echo(f"  replay of {target.replay_of}{went_live}")
        if not steps:
            typer.echo("  (no steps recorded)")
            return

        provider = get_provider(target.provider or "openai")
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
