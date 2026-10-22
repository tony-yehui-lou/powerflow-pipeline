"""S9 · Phase -> each rep's events and phase intervals from S5's bar, S8's floor, bones and knees,
and S6's grip (docs/specs/preprocessing/S9-phase-intervals-draft7.md).

Everything read here is another stage's published output: S8's `joint_hva.json`, S5's barbell
detections (the bar track is solved again, as S8 solves it), S6's `wrist_refinement.json`, and
the capture's raw `metadata.yaml` (`lift.type`, `lift.result`). The rules are
`phase_events.detect`; this task is the I/O around it.

A capture this stage cannot segment is never rejected: S0-S8 already published, and the reason
is a manifest warning (and, where a document can be written, `absent` events).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import yaml
from prefect import task

from powerflow_pipeline.data.common.models import FileOp, StepResult
from powerflow_pipeline.data.common.pose_storage import pose_path, write_sidecar
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess import phase_events as pe
from powerflow_pipeline.data.preprocess.bar_geometry import BarPose
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.tasks.kinematics import (
    bar_track,
    kinematics_path,
    read_grip,
)
from powerflow_pipeline.data.preprocess.tasks.pose import WRIST_REFINEMENT_FILENAME

PHASE_FILENAME: Final = "phase.json"


def phase_path(phase_root: Path, record: CameraRecord) -> Path:
    return phase_root / record.relative / PHASE_FILENAME


def read_lift(metadata_path: Path) -> tuple[object, object]:
    """`lift.type` and `lift.result` from the capture's raw operator metadata, as written."""

    if not metadata_path.is_file():
        return None, None
    loaded = yaml.safe_load(metadata_path.read_text())
    lift = loaded.get("lift") if isinstance(loaded, dict) else None
    if not isinstance(lift, dict):
        return None, None
    return lift.get("type"), lift.get("result")


def bar_heights(
    record: CameraRecord, config: PreprocessConfig, y_floor_mm: float | None
) -> tuple[list[float | None], list[str]]:
    """S5's bar centre height above S8's floor on every frame it can be placed (§1.1)."""

    bars: list[BarPose | None]
    bars, warnings = bar_track(record, config)
    if y_floor_mm is None:
        return [None] * record.n_frames, [*warnings, "no floor in joint_hva.json: no bar heights"]
    return [None if b is None else y_floor_mm - b.centre_mm[1] for b in bars], warnings


def _knee(document: dict[str, Any], sigma_max_mm: float) -> list[float | None]:
    def side(name: str) -> pe.Side:
        joint = document["joints"][name]
        mm = [None if h is None else h * 1000.0 for h in joint["heightM"]]
        sigma = [None if s is None else s * 1000.0 for s in joint["heightSigmaM"]]
        return pe.Side(mm, sigma, joint["flags"])

    return pe.pair_height(side("leftKnee"), side("rightKnee"), sigma_max_mm=sigma_max_mm)


@task(retries=1)
def detect_phases_camera(
    record: CameraRecord, config: PreprocessConfig
) -> tuple[CameraRecord, StepResult]:
    """Find one camera's lift events and publish `phase.json` to `config.phase_root`.

    `record` is S4's output record, the one S8 was given: it carries the intrinsics and the image
    size S5's detections must match.
    """

    assert config.phase_root is not None  # the flow checks these before looping cameras
    assert config.kinematics_root is not None and config.pose_root is not None

    source = kinematics_path(config.kinematics_root, record)
    destination = phase_path(config.phase_root, record)
    log_task_paths(source, destination)
    file_ops = [FileOp(op="write", src=source, dst=destination)]
    if config.dry_run:
        return record, StepResult(file_ops=file_ops)
    if not source.is_file():
        return record, StepResult(warnings=[f"phases not found: no S8 kinematics at {source}"])

    lift_type, result = read_lift(record.metadata_path)
    lift = pe.LIFT_TYPES.get(lift_type) if isinstance(lift_type, str) else None
    if lift is None:
        reason = f"lift.type {lift_type!r} is not one of {sorted(pe.LIFT_TYPES)}"
        return record, StepResult(warnings=[f"phases not found: {reason}"])

    th = config.phase
    document = json.loads(source.read_text())
    y_floor_m = (document.get("floor") or {}).get("yFloorM")
    bar, warnings = bar_heights(record, config, None if y_floor_m is None else y_floor_m * 1000.0)
    t_ms = document["frames"]["tMs"]
    signals = pe.Signals(t_ms=t_ms, bar_mm=bar, knee_mm=_knee(document, th.joint_sigma_max_mm))

    grip_path = pose_path(config.pose_root, record.relative.parent, record.relative.name).parent
    _, half_grip, _ = read_grip(grip_path / WRIST_REFINEMENT_FILENAME)
    body = pe.body_from_bones(
        document.get("bones") or {}, (document.get("ankleHeight") or {}).get("usedM"), half_grip
    )
    bands = None if body is None else pe.bands(body, thresholds=th)
    if body is None:
        warnings.append("no bone lengths in joint_hva.json: no height bands, gated events absent")

    detection = pe.detect(signals, lift, th, bands, result_override=result)
    payload = pe.to_document(
        detection,
        capture_id=record.capture_id,
        t_ms=t_ms,
        start_epoch_ms=document["frames"]["startEpochMs"],
        body_bands=bands,
        thresholds=th,
        inputs={
            "jointHva": str(source),
            "wristRefinement": str(grip_path / WRIST_REFINEMENT_FILENAME),
        },
    )

    if config.overwrite and destination.exists():
        destination.unlink()
    write_sidecar(destination, payload)
    found = sum(e.frame is not None for e in detection.events.values())
    derived = {
        "phase_lift": lift,
        "phase_result": detection.result[0],
        "phase_events_found": f"{found}/{len(detection.events)}",
        "phase_flags": detection.flags,
    }
    return record, StepResult(derived=derived, warnings=warnings, file_ops=file_ops)
