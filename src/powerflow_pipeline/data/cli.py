"""Command-line entry point for data pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from powerflow_pipeline.data.analyze.config import AnalyzeConfig, AnalyzeStage
from powerflow_pipeline.data.analyze.flow import analyze as analyze_flow
from powerflow_pipeline.data.common.manifest import RunManifest
from powerflow_pipeline.data.detect.config import (
    BarbellDevice,
    DetectConfig,
    PoseDetector,
    RTMPoseVariant,
)
from powerflow_pipeline.data.detect.flow import detect as detect_flow
from powerflow_pipeline.data.detect.mediapipe_pose_detector import MediaPipePoseDetector
from powerflow_pipeline.data.detect.pose_model import Detector2D
from powerflow_pipeline.data.preprocess.config import PreprocessConfig, RotationDirection
from powerflow_pipeline.data.preprocess.flow import preprocess as preprocess_flow

app = typer.Typer(help="PowerFlow data pipelines.", no_args_is_help=True)

# The roots every pipeline after S4 reads, to rebuild each camera from what preprocess published.
Records = Annotated[
    Path, typer.Option("--records", help="Where S0 writes record.json and metadata.yaml.")
]
Cut = Annotated[Path, typer.Option("--cut", help="Where S1 publishes the time-aligned streams.")]
Retilt = Annotated[Path, typer.Option("--retilt", help="Where S3 publishes the rectified streams.")]
Crop = Annotated[Path, typer.Option("--crop", help="Where S4 publishes the cropped streams.")]
Only = Annotated[
    str | None,
    typer.Option(
        "--only",
        help=(
            "Process only captures whose id matches this glob, e.g. '22 August/Snch/*'. "
            "In preprocess it must keep a two-camera session whole: excluding its Side camera "
            "leaves the group with no lift window to cut to, and it is rejected."
        ),
    ),
]
DryRun = Annotated[bool, typer.Option("--dry-run", help="Validate and plan, but write nothing.")]
Overwrite = Annotated[bool, typer.Option("--overwrite", help="Replace cameras already published.")]
BarbellCheckpoint = Annotated[
    Path | None,
    typer.Option(
        "--barbell-checkpoint",
        help=(
            "A promoted S5 model (its .json card beside it). Runs S5, publishing into "
            "--barbell. Omit to read --barbell as already published."
        ),
    ),
]


def _build_detector(config: DetectConfig) -> Detector2D:
    """The 2D detector `config.pose_detector` selects; RTMPose unless MediaPipe is asked for.

    `rtmpose_detector` is imported lazily: it pulls in `rtmlib`/`onnxruntime`, and a run that
    never builds a detector should not pay for loading them.
    """

    if config.pose_detector is PoseDetector.MEDIAPIPE:
        return MediaPipePoseDetector()

    from powerflow_pipeline.data.detect.rtmpose_detector import RTMPoseDetector

    return RTMPoseDetector(
        config.rtmpose_variant.value,
        det_frequency=config.rtmpose_det_frequency,
        clavicle_shoulder_weight=config.rtmpose_clavicle_shoulder_weight,
        use_depth_for_subject=config.rtmpose_subject_depth,
    )


def _echo(manifest: RunManifest) -> None:
    typer.echo(
        f"{manifest.pipeline}: processed {len(manifest.scans)} camera(s), "
        f"rejected {len(manifest.rejected_scans)}"
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
    record_root: Records,
    cut_root: Cut,
    retilt_root: Retilt,
    crop_root: Crop,
    only: Only = None,
    rotation: Annotated[
        RotationDirection,
        typer.Option("--rotation", help="Direction that makes the lifter upright."),
    ] = RotationDirection.CW,
    dry_run: DryRun = False,
    overwrite: Overwrite = False,
) -> None:
    """Ingest a raw capture (S0), cut it to the lift window (S1), rotate it (S2), retilt it
    level with the floor (S3), and crop it to a stable common region (S4).

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
        only=only,
        rotation=rotation,
        dry_run=dry_run,
        overwrite=overwrite,
    )
    _echo(preprocess_flow(config))


@app.command()
def detect(
    record_root: Records,
    cut_root: Cut,
    retilt_root: Retilt,
    crop_root: Crop,
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
                "Where S7 publishes poses lifted to floor-frame metres using LiDAR depth. Off "
                "by default: nothing downstream reads it. Reads S6's published document, so it "
                "can lift an earlier run's detections without re-running the detector."
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
    barbell_checkpoint: BarbellCheckpoint = None,
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
    only: Only = None,
    dry_run: DryRun = False,
    overwrite: Overwrite = False,
) -> None:
    """Detect the barbell's cap and collar (S5, with `--barbell-checkpoint`), the joints (S6,
    with `--pose`), and optionally lift them with LiDAR depth (S7, with `--lift`), on every
    camera preprocess published.
    """

    config = DetectConfig(
        record_root=record_root,
        cut_root=cut_root,
        retilt_root=retilt_root,
        crop_root=crop_root,
        pose_root=pose_root,
        lift_root=lift_root,
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
        dry_run=dry_run,
        overwrite=overwrite,
    )
    # A detector loads (downloading on first use) its pretrained checkpoints, so one is only
    # constructed when S6 is actually requested. S7 needs none: it reads S6's published
    # document, which is what makes `--lift` alone a valid way to re-run just the depth half.
    detector: Detector2D | None = None
    if pose_root is not None and not skip_detect:
        detector = _build_detector(config)
    _echo(detect_flow(config, detector=detector))


@app.command()
def analyze(
    record_root: Records,
    cut_root: Cut,
    retilt_root: Retilt,
    crop_root: Crop,
    pose_root: Annotated[
        Path, typer.Option("--pose", help="S6's published tree: the 2D joints and the grip.")
    ],
    barbell_root: Annotated[
        Path | None,
        typer.Option(
            "--barbell",
            help="S5's published tree. Without it S8 is anchored by the floor alone; S9 needs it.",
        ),
    ] = None,
    kinematics_root: Annotated[
        Path | None,
        typer.Option(
            "--kinematics",
            help=(
                "Where S8 publishes joint_hva.json: every joint's 3D position and height, solved "
                "from S6's 2D pose, the lifter's body measurements (metadata.yaml athlete:), the "
                "bar (--barbell) and the floor. Reads no depth. S9 reads it either way."
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
    bar_hva_root: Annotated[
        Path | None,
        typer.Option(
            "--bar-hva",
            help=(
                "Where S10 publishes bar_hva.json: the bar centre's height, velocity and "
                "acceleration, by box-constrained Tikhonov regularisation of S5's bar track. "
                "Needs --barbell; needs neither --kinematics nor --phase."
            ),
        ),
    ] = None,
    stages: Annotated[
        list[AnalyzeStage] | None,
        typer.Option(
            "--stages",
            help=(
                "Run only these stages (repeatable), e.g. --stages s9 to re-run S9 against the "
                "S8 tree already in --kinematics. Default: s8, s9 and s10."
            ),
        ),
    ] = None,
    only: Only = None,
    dry_run: DryRun = False,
    overwrite: Overwrite = False,
) -> None:
    """Solve the joints in 3D from the lifter's body measurements (S8, with `--kinematics`),
    split each rep into its lift's phases (S9, with `--phase`) and differentiate the bar's
    height (S10, with `--bar-hva`), from what preprocess and detect published. Reads no video
    and loads no detector.
    """

    config = AnalyzeConfig(
        record_root=record_root,
        cut_root=cut_root,
        retilt_root=retilt_root,
        crop_root=crop_root,
        pose_root=pose_root,
        barbell_root=barbell_root,
        kinematics_root=kinematics_root,
        phase_root=phase_root,
        bar_hva_root=bar_hva_root,
        only=only,
        dry_run=dry_run,
        overwrite=overwrite,
    )
    if stages:
        config = config.model_copy(update={"stages": tuple(stages)})
    _echo(analyze_flow(config))


@app.command("all")
def all_pipelines(
    data_root: Annotated[
        Path,
        typer.Option(
            "--data",
            help=(
                "The data repository, e.g. ../data. Reads raw/ and publishes into the standard "
                "sN_<stage>_output folders beside it."
            ),
        ),
    ],
    barbell_checkpoint: BarbellCheckpoint = None,
    only: Only = None,
    overwrite: Overwrite = False,
) -> None:
    """Run preprocess, detect and analyze in order, each reading what the previous published.

    S7 Lift is not run. S5 runs only with `--barbell-checkpoint`; otherwise detect reads the
    S5 tree already published. Every tunable takes its default: run the pipelines one at a
    time to change one. There is no `--dry-run`: detect and analyze read what the previous
    pipeline published, which a dry run never writes.
    """

    record_root = data_root / "s0_ingest_output"
    cut_root = data_root / "s1_cut_output"
    retilt_root = data_root / "s3_retilt_output"
    crop_root = data_root / "s4_crop_output"
    barbell_root = data_root / "s5_barbell_output"
    pose_root = data_root / "s6_pose_output"

    _echo(
        preprocess_flow(
            PreprocessConfig(
                raw_root=data_root / "raw",
                record_root=record_root,
                cut_root=cut_root,
                retilt_root=retilt_root,
                crop_root=crop_root,
                output_root=data_root / "s2_orient_output",
                only=only,
                overwrite=overwrite,
            )
        )
    )
    detect_config = DetectConfig(
        record_root=record_root,
        cut_root=cut_root,
        retilt_root=retilt_root,
        crop_root=crop_root,
        barbell_root=barbell_root,
        barbell_checkpoint=barbell_checkpoint,
        pose_root=pose_root,
        only=only,
        overwrite=overwrite,
    )
    _echo(detect_flow(detect_config, detector=_build_detector(detect_config)))
    _echo(
        analyze_flow(
            AnalyzeConfig(
                record_root=record_root,
                cut_root=cut_root,
                retilt_root=retilt_root,
                crop_root=crop_root,
                pose_root=pose_root,
                barbell_root=barbell_root,
                kinematics_root=data_root / "s8_kinematics_output",
                phase_root=data_root / "s9_phase_output",
                bar_hva_root=data_root / "s10_hva_output",
                only=only,
                overwrite=overwrite,
            )
        )
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
    """Score S9's phase.json against hand labels (S9 draft 8 §8.3). Reads only; writes nothing."""

    from powerflow_pipeline.data.analyze import phase_eval

    report = phase_eval.score(labels_root, phases_root)
    typer.echo(phase_eval.markdown(report, phase_eval.human_floor(labels_root)))
