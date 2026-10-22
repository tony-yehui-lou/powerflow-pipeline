"""The detect flow: S5 barbell, S6 pose, S7 lift (optional, off by default).

Ordered orchestration only.

Runs per camera, on what preprocess published: each camera's record is rebuilt from S0-S4's
output (`common/camera_records.py`), so detect can be re-run -- a new checkpoint, a new pose
model -- without re-encoding anything. Nothing here is cross-camera.
"""

from __future__ import annotations

from pathlib import Path

from prefect import flow

from powerflow_pipeline.data.common.camera_records import discover_published_cameras
from powerflow_pipeline.data.common.context import OutputMode, RunContext
from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.manifest import RunManifest, emit_manifest, scan_outcome
from powerflow_pipeline.data.common.models import RejectedScan, StepResult
from powerflow_pipeline.data.detect.barbell_model import (
    BarbellDetector,
    build_barbell_detector,
    read_card,
)
from powerflow_pipeline.data.detect.config import DetectConfig
from powerflow_pipeline.data.detect.pose_model import Detector2D
from powerflow_pipeline.data.detect.tasks.barbell import detect_barbell_camera
from powerflow_pipeline.data.detect.tasks.lift import lift_pose_camera
from powerflow_pipeline.data.detect.tasks.pose import (
    detect_pose_camera,
    ensure_skeleton_published,
)

PIPELINE = "detect"


@flow(name=PIPELINE)
def detect(
    config: DetectConfig,
    detector: Detector2D | None = None,
    barbell_detector: BarbellDetector | None = None,
) -> RunManifest:
    """Detect the barbell and the joints on every camera preprocess published. Or reject.

    S5 Barbell runs when `config.barbell_checkpoint` is set (or a `barbell_detector` is
    injected), before S6 so S6 reads what it just published. The flow builds S5's detector
    itself from config -- once, and never in a dry run -- so a Prefect deployment, which can
    pass nothing but parameters, can run it.

    S6 Pose runs only when `detector` is given. S7 Lift runs only when `config.lift_root` is
    set, independently of S6: the two stages communicate through S6's published document.
    """

    if detector is not None and config.pose_root is None:
        raise ValueError("detector was given but config.pose_root is unset")
    if config.lift_root is not None and config.pose_root is None:
        raise ValueError("config.lift_root is set but config.pose_root is unset: nothing to lift")
    run_barbell = barbell_detector is not None or config.barbell_checkpoint is not None
    if run_barbell and config.barbell_root is None:
        raise ValueError(
            "S5 Barbell would run but config.barbell_root is unset: nowhere to publish"
        )
    if not (run_barbell or detector is not None or config.lift_root is not None):
        raise ValueError(
            "nothing to run: give a barbell checkpoint (S5), a pose detector (S6) or a lift root"
            " (S7)"
        )
    if barbell_detector is None and config.barbell_checkpoint is not None:
        read_card(config.barbell_checkpoint)  # refuse an unreproducible checkpoint before any work
        if not config.dry_run:
            barbell_detector = build_barbell_detector(config)

    context = RunContext.create(
        pipeline=PIPELINE,
        input_root=config.crop_root,
        output_root=_manifest_root(config, run_barbell=run_barbell, run_pose=detector is not None),
        output_mode=OutputMode.PUBLISH,
        dry_run=config.dry_run,
    )
    manifest = RunManifest(
        pipeline=context.pipeline,
        input_root=context.input_root,
        output_root=context.output_root,
        output_mode=context.output_mode,
        dry_run=context.dry_run,
    )

    if not config.dry_run:
        # Each pose tree is self-describing: joint names mean nothing without the topology
        # they were captured against, so every root that holds a pose document gets one.
        if detector is not None:
            assert config.pose_root is not None  # checked above
            ensure_skeleton_published(config.pose_root, overwrite=config.overwrite)
        if config.lift_root is not None:
            ensure_skeleton_published(config.lift_root, overwrite=config.overwrite)

    found = discover_published_cameras(
        config.record_root, config.cut_root, config.crop_root, only=config.only
    )
    manifest.rejected_scans.extend(found.rejected)

    for record in found.records:
        ran: list[tuple[str, StepResult]] = []
        try:
            if run_barbell:
                ran.append(("barbell", detect_barbell_camera(record, config, barbell_detector)[1]))
            if detector is not None:
                ran.append(("pose", detect_pose_camera(record, config, detector)[1]))
            if config.lift_root is not None:
                ran.append(("lift", lift_pose_camera(record, config)[1]))
        except ScanRejected as rejection:
            manifest.rejected_scans.append(
                RejectedScan(scan_id=record.capture_id, source=record.source, reason=str(rejection))
            )
            continue
        manifest.scans.append(
            scan_outcome(record.capture_id, record.source, ran, dry_run=config.dry_run)
        )

    manifest_path = None if config.dry_run else _manifest_path(context.output_root)
    return emit_manifest(manifest, manifest_path)


def _manifest_root(config: DetectConfig, *, run_barbell: bool, run_pose: bool) -> Path:
    """The run's manifest sits in the tree of the last stage it ran: S7, else S6, else S5."""

    if config.lift_root is not None:
        return config.lift_root
    if run_pose:
        assert config.pose_root is not None
        return config.pose_root
    assert run_barbell and config.barbell_root is not None
    return config.barbell_root


def _manifest_path(root: Path | None) -> Path:
    assert root is not None  # publish mode always has an output root
    root.mkdir(parents=True, exist_ok=True)
    return root / "manifest.json"
