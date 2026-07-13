"""Command-line entry point for data pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from powerflow_pipeline.data.preprocess.config import PreprocessConfig, RotationDirection
from powerflow_pipeline.data.preprocess.flow import preprocess as preprocess_flow

app = typer.Typer(help="PowerFlow data pipelines.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Group data-pipeline commands under the ``powerflow`` entry point.

    Each pipeline adds one command here that builds its config, then invokes its flow.
    """


@app.command()
def preprocess(
    input_root: Annotated[Path, typer.Option("--input", help="Raw capture root, e.g. data/raw.")],
    output_root: Annotated[
        Path, typer.Option("--output", help="Where S1 publishes the portrait streams.")
    ],
    record_root: Annotated[
        Path, typer.Option("--records", help="Where S0 writes record.json and metadata.yaml.")
    ],
    rotation: Annotated[
        RotationDirection,
        typer.Option("--rotation", help="Direction that makes the lifter upright."),
    ] = RotationDirection.CW,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate and plan, but write nothing.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace cameras already published.")
    ] = False,
) -> None:
    """Ingest a raw capture (S0) and rotate every stream to portrait (S1).

    There is no `--in-place` mode: it would rewrite the raw capture, and raw data is
    write-once. Each stage publishes to its own output root instead.
    """

    config = PreprocessConfig(
        raw_root=input_root,
        record_root=record_root,
        output_root=output_root,
        rotation=rotation,
        dry_run=dry_run,
        overwrite=overwrite,
    )
    manifest = preprocess_flow(config)
    typer.echo(
        f"processed {len(manifest.scans)} camera(s), rejected {len(manifest.rejected_scans)}"
    )
