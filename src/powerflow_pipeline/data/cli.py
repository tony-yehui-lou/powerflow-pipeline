"""Command-line entry point for data pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from powerflow_pipeline.data.preprocess.config import (
    BarbellDevice,
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
    """The 2D detector `config.pose_detector` selects; RTMPose unless MediaPipe is asked for.

    `rtmpose_detector` is imported lazily: it pulls in `rtmlib`/`onnxruntime`, and a run that
    never builds a detector should not pay for loading them.
    """

    if config.pose_detector is PoseDetector.MEDIAPIPE:
        return MediaPipePoseDetector()

    from powerflow_pipeline.data.preprocess.rtmpose_detector import RTMPoseDetector

    return RTMPoseDetector(
        config.rtmpose_variant.value,
        det_frequency=config.rtmpose_det_frequency,
        clavicle_shoulder_weight=config.rtmpose_clavicle_shoulder_weight,
        use_depth_for_subject=config.rtmpose_subject_depth,
    )


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
                "Where S6 publishes detected 2D poses. Omit to skip S6; give it without "
                "--lift to detect joints without back-projecting them."
            ),
        ),
    ] = None,
    lift_root: Annotated[
        Path | None,
        typer.Option(
            "--lift",
            help=(
                "Where S7 publishes poses lifted to floor-frame metres using LiDAR depth. "
                "Omit to skip S7. Reads S6's published document, so it can lift an earlier "
                "run's detections without re-running the detector."
            ),
        ),
    ] = None,
    kinematics_root: Annotated[
        Path | None,
        typer.Option(
            "--kinematics",
            help=(
                "Where S8 publishes joint_hva.json: every joint's 3D position and height, solved "
                "from S6's 2D pose, the lifter's body measurements (metadata.yaml athlete:), the "
                "bar (--barbell) and the floor. Needs --pose. Reads no depth."
            ),
        ),
    ] = None,
    phase_root: Annotated[
        Path | None,
        typer.Option(
            "--phase",
            help=(
                "Where S9 publishes phase.json: each rep's events and phase intervals, from S5's "
                "bar, S8's floor, bones and knees (--kinematics) and S6's grip. Needs "
                "--kinematics and --barbell. Reads lift.type from metadata.yaml: snch or CnJ."
            ),
        ),
    ] = None,
    barbell_root: Annotated[
        Path | None,
        typer.Option(
            "--barbell",
            help=(
                "S5's tree (a root mirroring the raw layout, holding barbell.json per "
                "capture): written when --barbell-checkpoint runs S5, read either way. With "
                "--pose, S6 corrects both wrists against the bar the hands hold; without it, "
                "wrists are left exactly as detected."
            ),
        ),
    ] = None,
    barbell_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--barbell-checkpoint",
            help=(
                "A promoted S5 model (its .json card beside it). Runs S5 right after S4, "
                "publishing into --barbell. Omit to read --barbell as already published."
            ),
        ),
    ] = None,
    barbell_device: Annotated[
        BarbellDevice,
        typer.Option("--barbell-device", help="Where S5 runs: auto picks mps, cuda, then cpu."),
    ] = BarbellDevice.AUTO,
    barbell_render_overlay: Annotated[
        bool,
        typer.Option(
            "--barbell-overlay", help="Also publish S5's overlay.mp4 (boxes on every frame)."
        ),
    ] = False,
    skip_detect: Annotated[
        bool,
        typer.Option(
            "--skip-detect",
            help=(
                "Don't run S6; reuse the poses already published under --pose. Use with "
                "--lift to redo the depth back-projection without re-running the detector."
            ),
        ),
    ] = False,
    pose_detector: Annotated[
        PoseDetector,
        typer.Option("--pose-detector", help="Which 2D keypoint model S6 runs."),
    ] = PoseDetector.RTMPOSE,
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
                "Let S6 read depth to pick which person is the athlete (never to place a "
                "joint). --no-rtmpose-subject-depth makes S6 read no depth at all."
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
    when `--barbell-checkpoint` is given, detect the barbell's cap and collar (S5); when
    `--pose` is given, detect joints (S6); `--lift` lifts them with LiDAR depth (S7), and
    `--kinematics` solves them in 3D from the lifter's body measurements instead (S8), and
    `--phase` splits each rep into its lift's phases (S9).

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
        kinematics_root=kinematics_root,
        phase_root=phase_root,
        barbell_root=barbell_root,
        barbell_checkpoint=barbell_checkpoint,
        barbell_device=barbell_device,
        barbell_render_overlay=barbell_render_overlay,
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
    # constructed when S6 is actually requested. S7 needs none: it reads S6's published
    # document, which is what makes `--lift` alone a valid way to re-run just the depth half.
    detector: Detector2D | None = None
    if pose_root is not None and not skip_detect:
        detector = _build_detector(config)

    manifest = preprocess_flow(config, detector=detector)
    typer.echo(
        f"processed {len(manifest.scans)} camera(s), rejected {len(manifest.rejected_scans)}"
    )


@app.command("phase-eval")
def phase_eval_command(
    labels_root: Annotated[
        Path,
        typer.Option("--labels", help="Phase labels, e.g. data/annotations/phase/v1."),
    ],
    phases_root: Annotated[
        Path,
        typer.Option("--phases", help="S9's output, e.g. data/s9_phase_output."),
    ],
) -> None:
    """Score S9's phase.json against hand labels (S9 draft 7 §8.3). Reads only; writes nothing."""

    from powerflow_pipeline.data.preprocess import phase_eval

    report = phase_eval.score(labels_root, phases_root)
    typer.echo(phase_eval.markdown(report, phase_eval.human_floor(labels_root)))
