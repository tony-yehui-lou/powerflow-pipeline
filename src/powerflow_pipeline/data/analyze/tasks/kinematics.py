"""S8 · Kinematics -> every joint's 3D position and height from S6's 2D pose, the lifter's body
measurements, the bar and the floor (docs/specs/preprocessing/S8-joint-hva-draft3.md).

Everything read here is another stage's published output: S6's `pose.json` and
`wrist_refinement.json`, S5's barbell detections, S3's floor offset, S0's intrinsics off the
crop record, and the capture's raw `metadata.yaml` (bar type and `athlete:` block). No depth is
read. The document is assembled by `joint_hva.build`; this task is the I/O around it.

A capture this stage cannot solve is never rejected: S0-S4 and S6 already published, and the
reason is a manifest warning (plus null arrays with a reason, where a document can be written).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from prefect import task

from powerflow_pipeline.data.analyze import athlete_input, joint_hva
from powerflow_pipeline.data.analyze.config import AnalyzeConfig
from powerflow_pipeline.data.common import barbell_input
from powerflow_pipeline.data.common.bar_geometry import BAR_TYPES, BarPose, solve_track
from powerflow_pipeline.data.common.capture_models import CameraRecord
from powerflow_pipeline.data.common.models import FileOp, StepResult
from powerflow_pipeline.data.common.pose_storage import (
    WRIST_REFINEMENT_FILENAME,
    pose_path,
    read_pose,
    write_sidecar,
)
from powerflow_pipeline.data.common.task_logging import log_task_paths

KINEMATICS_FILENAME: Final = "joint_hva.json"


def kinematics_path(kinematics_root: Path, record: CameraRecord) -> Path:
    return kinematics_root / record.relative / KINEMATICS_FILENAME


def bar_track(
    record: CameraRecord, config: AnalyzeConfig
) -> tuple[list[BarPose | None], list[str]]:
    """S5's bar on every frame it can be placed, or no bar with a warning saying why."""

    none: list[BarPose | None] = [None] * record.n_frames
    if config.barbell_root is None:
        return none, ["no --barbell: joints are anchored by the floor alone"]
    path = barbell_input.detections_path(config.barbell_root, record.relative)
    if not path.is_file():
        return none, [f"no barbell detections at {path}: joints anchored by the floor alone"]
    bar_type = barbell_input.read_bar_type(record.metadata_path)
    if bar_type not in BAR_TYPES:
        return none, [f"barbell type {bar_type!r} is not one of {sorted(BAR_TYPES)}: no bar"]
    try:
        detections = barbell_input.read_detections(
            path,
            n_frames=record.n_frames,
            frame_size=(record.rgb_width, record.rgb_height),
            min_score=config.barbell_min_score,
        )
    except ValueError as error:
        return none, [f"{error}: no bar"]
    bar = BAR_TYPES[bar_type]
    return solve_track(detections.caps, detections.collars, record.intrinsics, bar), []


def read_grip(sidecar_path: Path) -> tuple[frozenset[int], float | None, Any]:
    """S6's hands-on-bar frames, half grip and near side, from `wrist_refinement.json`.

    `farCorrected` is exactly the hands-on-bar frames: S6 replaces the far wrist on every one of
    them (S6-wrist-refinement.md §5).
    """

    if not sidecar_path.is_file():
        return frozenset(), None, None
    sidecar = json.loads(sidecar_path.read_text())
    near = sidecar.get("near")
    side = "left" if near == "leftWrist" else "right" if near == "rightWrist" else None
    return frozenset(sidecar.get("farCorrected", [])), sidecar.get("halfGripMm"), side


def floor_offset_mm(config: AnalyzeConfig, record: CameraRecord) -> float | None:
    """S3's floor offset in mm, the floor anchor's fallback; None without S3's sidecar."""

    path = config.retilt_root / record.relative / "retilt_sidecar.json"
    if not path.is_file():
        return None
    offset: float = json.loads(path.read_text())["floor_offset_m"]
    return offset * 1000.0


@task(retries=1)
def solve_kinematics_camera(
    record: CameraRecord, config: AnalyzeConfig
) -> tuple[CameraRecord, StepResult]:
    """Solve one camera's joints in 3D and publish `joint_hva.json` to `config.kinematics_root`.

    `record` is S4's output record, the one S6 was given: it carries the intrinsics and the image
    size the barbell detections must match.
    """

    assert config.pose_root is not None  # the flow checks both before looping cameras
    assert config.kinematics_root is not None

    source = pose_path(config.pose_root, record.relative.parent, record.relative.name)
    destination = kinematics_path(config.kinematics_root, record)
    log_task_paths(source, destination)
    file_ops = [FileOp(op="write", src=source, dst=destination)]
    if config.dry_run:
        return record, StepResult(file_ops=file_ops)
    if not source.is_file():
        return record, StepResult(warnings=[f"kinematics not solved: no S6 pose at {source}"])

    pose = read_pose(source)
    bars, warnings = bar_track(record, config)
    hands_on_bar, half_grip, near_side = read_grip(source.parent / WRIST_REFINEMENT_FILENAME)
    if config.barbell_root is not None and half_grip is None:
        warnings.append("no wrist_refinement.json from S6: the hands are not placed on the bar")

    built = joint_hva.build(
        joint_hva.Inputs(
            pose=pose,
            k=record.intrinsics,
            bars=bars,
            hands_on_bar=hands_on_bar,
            half_grip_mm=half_grip,
            near_side=near_side,
            floor_offset_mm=floor_offset_mm(config, record),
            athlete=athlete_input.read_athlete(record.metadata_path),
            clavicle_shoulder_weight=config.rtmpose_clavicle_shoulder_weight,
        )
    )

    if config.overwrite and destination.exists():
        destination.unlink()
    write_sidecar(destination, built.document)
    return record, StepResult(
        derived=built.derived, warnings=warnings + built.warnings, file_ops=file_ops
    )
