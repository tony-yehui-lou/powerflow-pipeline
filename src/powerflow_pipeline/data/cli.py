"""Command-line entry point for data pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from powerflow_pipeline.data.preprocess.config import PreprocessConfig, RotationDirection
from powerflow_pipeline.data.preprocess.depth_pose_model import DepthBackedPoseModel
from powerflow_pipeline.data.preprocess.flow import preprocess as preprocess_flow
from powerflow_pipeline.data.preprocess.mediapipe_pose_detector import MediaPipePoseDetector
from powerflow_pipeline.data.preprocess.pose_model import PoseModel

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
        Path, typer.Option("--output", help="Where S2 publishes the portrait streams.")
    ],
    record_root: Annotated[
        Path, typer.Option("--records", help="Where S0 writes record.json and metadata.yaml.")
    ],
    cut_root: Annotated[
        Path, typer.Option("--cut", help="Where S1 publishes the time-aligned streams.")
    ],
    retilt_root: Annotated[
        Path, typer.Option("--retilt", help="Where S3 publishes the rectified streams.")
    ],
    crop_root: Annotated[
        Path, typer.Option("--crop", help="Where S4 publishes the cropped streams.")
    ],
    pose_root: Annotated[
        Path | None,
        typer.Option(
            "--pose",
            help="Where S5 publishes detected poses. Omit to skip S5 entirely.",
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            help=(
                "Process only captures whose id matches this glob, e.g. '22 August/Snch/*'. "
                "Must keep a two-camera session whole: excluding its Side camera leaves the "
                "group with no lift window to cut to, and it is rejected."
            ),
        ),
    ] = None,
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
    """Ingest a raw capture (S0), cut it to the lift window (S1), rotate it (S2),
    retilt it level with the floor (S3), crop it to a stable common region (S4), and,
    when `--pose` is given, detect joints (S5).

    Both raw layouts are handled by one invocation: the directory shape is detected per
    capture from where its operator `metadata.yaml` sits, so there is no layout flag to
    get wrong. Output mirrors the raw path exactly, at whatever depth that is.

    There is no `--in-place` mode: it would rewrite the raw capture, and raw data is
    write-once. Each stage publishes to its own output root instead.
    """

    config = PreprocessConfig(
        raw_root=input_root,
        record_root=record_root,
        cut_root=cut_root,
        retilt_root=retilt_root,
        crop_root=crop_root,
        output_root=output_root,
        pose_root=pose_root,
        only=only,
        rotation=rotation,
        dry_run=dry_run,
        overwrite=overwrite,
    )
    # `MediaPipePoseDetector()` loads (downloading on first use) the pretrained model
    # bundle, so it's only constructed when S5 is actually requested.
    pose_model: PoseModel | None = None
    if pose_root is not None:
        pose_model = DepthBackedPoseModel(MediaPipePoseDetector())

    manifest = preprocess_flow(config, pose_model=pose_model)
    typer.echo(
        f"processed {len(manifest.scans)} camera(s), rejected {len(manifest.rejected_scans)}"
    )
