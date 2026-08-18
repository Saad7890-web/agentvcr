from __future__ import annotations

from typer.testing import CliRunner

from agentvcr import __version__
from agentvcr.cli import app

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
