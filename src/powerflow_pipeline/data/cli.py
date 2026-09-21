"""Command-line entry point for data pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from powerflow_pipeline.data.preprocess.config import (
    PoseDetector,
    PreprocessConfig,
    RotationDirection,
    RTMPoseVariant,
)
from powerflow_pipeline.data.preprocess.flow import preprocess as preprocess_flow
from powerflow_pipeline.data.preprocess.mediapipe_pose_detector import MediaPipePoseDetector
from powerflow_pipeline.data.preprocess.pose_model import Detector2D

app = typer.Typer(help="PowerFlow data pipelines.", no_args_is_help=True)


def _build_detector(config: PreprocessConfig) -> Detector2D:
    """The 2D detector `config.pose_detector` selects.

    `rtmpose_detector` is imported lazily: it pulls in `rtmlib`/`onnxruntime`, and a run that
    never asks for RTMPose should not pay for loading them.
    """

    if config.pose_detector is PoseDetector.RTMPOSE:
        from powerflow_pipeline.data.preprocess.rtmpose_detector import RTMPoseDetector

        return RTMPoseDetector(
            config.rtmpose_variant.value,
            det_frequency=config.rtmpose_det_frequency,
            clavicle_shoulder_weight=config.rtmpose_clavicle_shoulder_weight,
            use_depth_for_subject=config.rtmpose_subject_depth,
        )
    return MediaPipePoseDetector()


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
            help=(
                "Where S5 publishes detected 2D poses. Omit to skip S5; give it without "
                "--lift to detect joints without back-projecting them."
            ),
        ),
    ] = None,
    lift_root: Annotated[
        Path | None,
        typer.Option(
            "--lift",
            help=(
                "Where S6 publishes poses lifted to floor-frame metres using LiDAR depth. "
                "Omit to skip S6. Reads S5's published document, so it can lift an earlier "
                "run's detections without re-running the detector."
            ),
        ),
    ] = None,
    skip_detect: Annotated[
        bool,
        typer.Option(
            "--skip-detect",
            help=(
                "Don't run S5; reuse the poses already published under --pose. Use with "
                "--lift to redo the depth back-projection without re-running the detector."
            ),
        ),
    ] = False,
    pose_detector: Annotated[
        PoseDetector,
        typer.Option("--pose-detector", help="Which 2D keypoint model S5 runs."),
    ] = PoseDetector.MEDIAPIPE,
    rtmpose_variant: Annotated[
        RTMPoseVariant,
        typer.Option("--rtmpose-variant", help="RTMPose speed/accuracy tier."),
    ] = RTMPoseVariant.BALANCED,
    rtmpose_det_frequency: Annotated[
        int,
        typer.Option(
            "--rtmpose-det-frequency",
            help="Run RTMPose's person detector every Nth frame, tracking in between.",
        ),
    ] = 1,
    rtmpose_subject_depth: Annotated[
        bool,
        typer.Option(
            "--rtmpose-subject-depth/--no-rtmpose-subject-depth",
            help=(
                "Let S5 read depth to pick which person is the athlete (never to place a "
                "joint). --no-rtmpose-subject-depth makes S5 read no depth at all."
            ),
        ),
    ] = True,
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
        lift_root=lift_root,
        skip_detect=skip_detect,
        pose_detector=pose_detector,
        rtmpose_variant=rtmpose_variant,
        rtmpose_det_frequency=rtmpose_det_frequency,
        rtmpose_subject_depth=rtmpose_subject_depth,
        only=only,
        rotation=rotation,
        dry_run=dry_run,
        overwrite=overwrite,
    )
    # A detector loads (downloading on first use) its pretrained checkpoints, so one is only
    # constructed when S5 is actually requested. S6 needs none: it reads S5's published
    # document, which is what makes `--lift` alone a valid way to re-run just the depth half.
    detector: Detector2D | None = None
    if pose_root is not None and not skip_detect:
        detector = _build_detector(config)

    manifest = preprocess_flow(config, detector=detector)
    typer.echo(
        f"processed {len(manifest.scans)} camera(s), rejected {len(manifest.rejected_scans)}"
    )
