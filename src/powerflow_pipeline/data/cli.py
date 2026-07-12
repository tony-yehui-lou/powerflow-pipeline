"""Command-line entry point for data pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from powerflow_pipeline.data.core.context import OutputMode, RunContext
from powerflow_pipeline.data.dummy.pipeline import run as run_dummy

app = typer.Typer(help="PowerFlow data pipelines.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Group data-pipeline commands under the ``powerflow`` entry point."""


@app.command()
def dummy(
    input_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    in_place: Annotated[bool, typer.Option("--in-place")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Copy and annotate fixture-style scans for manual pipeline review."""

    if (output is None) == (not in_place):
        raise typer.BadParameter("provide exactly one of --output or --in-place")
    mode = OutputMode.IN_PLACE if in_place else OutputMode.PUBLISH
    context = RunContext.create(
        pipeline="dummy",
        input_root=input_root,
        output_root=output,
        output_mode=mode,
        dry_run=dry_run,
    )
    manifest = run_dummy(context)
    typer.echo(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
