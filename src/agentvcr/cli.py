"""``agentvcr`` command line.

Phase 0 ships ``serve`` (and ``version``); ``run``, ``runs``, ``show``, ``fork``,
``diff`` and ``ui`` arrive with the phases that give them something to do.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import __version__
from .config import MISMATCH_POLICIES, MODES, PRESETS, ConfigError, load_settings

app = typer.Typer(
    name="agentvcr",
    help="VCR for AI agents: record, replay, fork and diff agent runs through a proxy.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind address (default 127.0.0.1)."),
    port: int | None = typer.Option(None, "--port", "-p", help="Bind port (default 8484)."),
    db: Path | None = typer.Option(None, help="SQLite tape path (default .agentvcr/agentvcr.db)."),
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
    config: Path | None = typer.Option(None, help="Path to an agentvcr.toml."),
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

    try:
        settings = load_settings(
            config_path=config,
            overrides={
                "host": host,
                "port": port,
                "db_path": db,
                "mode": mode,
                "mismatch_policy": mismatch_policy,
                "upstreams": upstreams or None,
            },
        )
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

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


@app.command()
def version() -> None:
    """Print the agentvcr version."""
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
