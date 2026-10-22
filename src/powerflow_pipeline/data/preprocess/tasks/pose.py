"""S6 · Pose detection -> runs an injected 2D detector on S4's cropped RGB and publishes one
`PoseDocument` per camera (docs/spec, GitHub issue #116).

S6 stops in pixel space. The metric lift to floor-frame metres is S7 Lift, a separate stage
reading this stage's output, so the two can be re-run independently: detection is the expensive
half and depth sampling is the half still being revised.

The detector itself is out of scope (issue #118): `detector` is a `Detector2D` this task calls
through, so the stage's plumbing -- reading S4's output, timing the frames, and writing the
result at `pose_storage.py`'s stage-output layout (issue #117) -- is testable against a
stand-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from prefect import task

from powerflow_pipeline.data.common.capture_timing import capture_start_epoch_ms, frame_offsets_ms
from powerflow_pipeline.data.common.models import (
    FileOp,
    Frames,
    JointId,
    PipelineStage,
    PoseDocument,
    StepResult,
)
from powerflow_pipeline.data.common.pose_storage import (
    pose_path,
    skeleton_path,
    write_pose,
    write_sidecar,
    write_skeleton,
)
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess import barbell_input
from powerflow_pipeline.data.preprocess.bar_geometry import BAR_TYPES, solve_track
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.pose_model import (
    Detector2D,
    JointPixelSeries,
    to_pixel_only_series,
)
from powerflow_pipeline.data.preprocess.wrist_refine import Skipped, refine_wrists

# Whose image space `model`'s pixel_position is drawn in -- S6 always reads S4's output.
POSE_SOURCE_STAGE: Final[PipelineStage] = "s4_crop"

# Beside pose.json when the wrists were corrected against the barbell: which frames, and with what.
WRIST_REFINEMENT_FILENAME: Final = "wrist_refinement.json"
WRIST_REFINEMENT_SCHEMA: Final = "1.0.0"


def _refine_wrists(
    record: CameraRecord,
    config: PreprocessConfig,
    joints: dict[JointId, JointPixelSeries],
) -> tuple[dict[JointId, JointPixelSeries], dict[str, Any], list[str], dict[str, Any] | None]:
    """Both wrists corrected against S5's barbell, or unchanged with a warning saying why.

    Never a rejection: a capture whose bar can't be placed still has a valid pose, just one whose
    far wrist is as unreliable overhead as the detector made it.
    """

    assert config.barbell_root is not None
    path = barbell_input.detections_path(config.barbell_root, record.relative)

    def unchanged(
        reason: str,
    ) -> tuple[dict[JointId, JointPixelSeries], dict[str, Any], list[str], None]:
        return joints, {}, [f"wrists not corrected: {reason}"], None

    if not path.is_file():
        return unchanged(f"no barbell detections at {path}")
    bar_type = barbell_input.read_bar_type(record.metadata_path)
    if bar_type not in BAR_TYPES:
        return unchanged(
            f"barbell type {bar_type!r} in {record.metadata_path} is not one of {sorted(BAR_TYPES)}"
        )
    try:
        detections = barbell_input.read_detections(
            path,
            n_frames=record.n_frames,
            frame_size=(record.rgb_width, record.rgb_height),
            min_score=config.barbell_min_score,
        )
    except ValueError as error:
        return unchanged(str(error))

    bar = BAR_TYPES[bar_type]
    bars = solve_track(detections.caps, detections.collars, record.intrinsics, bar)
    result = refine_wrists(joints["leftWrist"], joints["rightWrist"], bars, record.intrinsics, bar)
    if isinstance(result, Skipped):
        return unchanged(result.reason)

    derived = {
        "bar_frames_placed": sum(1 for pose in bars if pose is not None),
        "wrist_far": result.far,
        "wrist_half_grip_mm": round(result.half_grip_mm, 1),
        "wrist_frames_on_bar": result.frames_on_bar,
        "wrist_far_corrected": len(result.far_corrected),
        "wrist_near_corrected": len(result.near_corrected),
    }
    sidecar = {
        "schemaVersion": WRIST_REFINEMENT_SCHEMA,
        "captureId": record.capture_id,
        "barbellDetections": str(path),
        "barbellCheckpointSha256": detections.checkpoint_sha256,
        "barbellType": bar_type,
        "minScore": config.barbell_min_score,
        "near": result.near,
        "far": result.far,
        "halfGripMm": round(result.half_grip_mm, 1),
        "residualPx": [round(v, 1) for v in result.residual_px],
        "barFramesPlaced": derived["bar_frames_placed"],
        "framesOnBar": result.frames_on_bar,
        "farCorrected": list(result.far_corrected),
        "nearCorrected": list(result.near_corrected),
        "method": "docs/specs/preprocessing/S6-wrist-refinement.md",
    }
    corrected = {**joints, "leftWrist": result.left, "rightWrist": result.right}
    return corrected, derived, [], sidecar


@task(retries=1)
def detect_pose_camera(
    record: CameraRecord, config: PreprocessConfig, detector: Detector2D
) -> tuple[CameraRecord, StepResult]:
    """Run `detector` on one camera's S4 output and publish the resulting 2D `PoseDocument`.

    `record` is S4's output record; like S4 Crop, everything S6 reads lives in the prior
    stage's own output tree, except the capture's epoch anchor, which only S1 recorded. No
    depth is read here at all -- that is S7 Lift.
    """

    assert config.pose_root is not None  # the caller checks this before looping cameras
    source = record.source
    destination = pose_path(config.pose_root, record.relative.parent, record.relative.name)
    log_task_paths(source, destination)

    file_ops = [
        FileOp(op="write", src=source / "rgb.mp4", dst=destination),
        FileOp(op="publish", src=source, dst=destination),
    ]
    sidecar_path = destination.parent / WRIST_REFINEMENT_FILENAME
    if config.barbell_root is not None:
        detections = barbell_input.detections_path(config.barbell_root, record.relative)
        file_ops.append(FileOp(op="write", src=detections, dst=sidecar_path))
    if config.dry_run:
        return record, StepResult(file_ops=file_ops)

    detected = detector.detect(source / "rgb.mp4", record.n_frames)
    derived: dict[str, Any] = {"n_frames": record.n_frames}
    warnings: list[str] = []
    sidecar: dict[str, Any] | None = None
    if config.barbell_root is not None:
        detected, wrist_derived, warnings, sidecar = _refine_wrists(record, config, detected)
        derived.update(wrist_derived)
    joints = to_pixel_only_series(detected, record.n_frames)
    frames = Frames(
        count=record.n_frames,
        fps=record.fps,
        start_epoch_ms=capture_start_epoch_ms(config.cut_root, record.relative),
        t_ms=tuple(frame_offsets_ms(source / "rgb.mp4")),
    )
    document = PoseDocument(
        capture_id=record.capture_id,
        role=record.role,
        stage=POSE_SOURCE_STAGE,
        frames=frames,
        joints=joints,
    )

    if config.overwrite:
        for stale in (destination, sidecar_path):
            if stale.exists():
                stale.unlink()
    write_pose(destination, document)
    if sidecar is not None:
        write_sidecar(sidecar_path, sidecar)

    return record, StepResult(derived=derived, warnings=warnings, file_ops=file_ops)


def ensure_skeleton_published(pose_root: Path, *, overwrite: bool) -> Path:
    """Publish `skeleton.human-v2.json` once per run; a rerun leaves it alone unless `overwrite`.

    Unlike a per-camera pose document, the topology doesn't vary by scan, so this is called
    once for the whole run rather than per camera (see `flow.py`).
    """

    path = skeleton_path(pose_root)
    if path.exists():
        if not overwrite:
            return path
        path.unlink()
    return write_skeleton(pose_root)
