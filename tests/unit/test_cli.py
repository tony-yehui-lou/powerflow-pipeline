"""The `powerflow` entry point exposes the data-pipeline command group."""

from __future__ import annotations

from typer.testing import CliRunner

from powerflow_pipeline.data.cli import app

runner = CliRunner()


def test_help_lists_the_command_group() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "PowerFlow data pipelines" in result.stdout


def test_bare_invocation_shows_help_rather_than_a_traceback() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2  # conventional "no command given" usage exit
    assert "Usage" in result.stdout
