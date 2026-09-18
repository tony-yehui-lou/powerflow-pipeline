"""The preprocess flow: S0 ingest, S1 cut, S2 orient, S3 retilt, S4 crop, S5 pose.

Ordered orchestration only.

Cut needs one capture to define the shared epoch window, so the loop is per *group*: ingest
every capture, derive the cut interval from the group's interval owner, then cut and orient
each member against that shared interval. A group is whatever shares one lift window --
a two-camera session, or a single-camera capture on its own -- and discovery decides which.
A group without a usable owner, or whose cut interval is invalid, is rejected whole: the
invariant is cross-camera, not per-camera.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from prefect import flow

from powerflow_pipeline.data.common.context import OutputMode, RunContext
from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.manifest import RunManifest, emit_manifest
from powerflow_pipeline.data.common.models import RejectedScan, ScanOutcome
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import (
    CameraRecord,
    CaptureLayout,
    CaptureUnit,
    CutInterval,
)
from powerflow_pipeline.data.preprocess.pose_model import PoseModel
from powerflow_pipeline.data.preprocess.retilt import assess_plane_consistency
from powerflow_pipeline.data.preprocess.tasks.crop import crop_camera
from powerflow_pipeline.data.preprocess.tasks.cut import cut_camera, resolve_cut_interval
from powerflow_pipeline.data.preprocess.tasks.discover import discover_captures
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from powerflow_pipeline.data.preprocess.tasks.metadata import write_group_metadata
from powerflow_pipeline.data.preprocess.tasks.orient import orient_camera
from powerflow_pipeline.data.preprocess.tasks.pose import (
    detect_pose_camera,
    ensure_skeleton_published,
)
from powerflow_pipeline.data.preprocess.tasks.retilt import retilt_camera

PIPELINE = "preprocess"

# Below this, a capture day's median is not a reference worth comparing against: with two
# captures both deviate from their own midpoint equally, so a disagreement accuses both.
MIN_CAPTURES_FOR_CONSISTENCY = 3


@flow(name=PIPELINE)
def preprocess(config: PreprocessConfig, pose_model: PoseModel | None = None) -> RunManifest:
    """Ingest, cut to the shared lift window, and rotate to portrait. Or reject, with a reason.

    S5 Pose runs after S4 Crop for every surviving camera, but only when `pose_model` is
    given -- a real model is issue #118, still open, so every existing caller that doesn't
    pass one gets exactly today's S0-S4 behaviour.
    """

    if pose_model is not None and config.pose_root is None:
        raise ValueError("pose_model was given but config.pose_root is unset")

    context = RunContext.create(
        pipeline=PIPELINE,
        input_root=config.raw_root,
        output_root=config.output_root,
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

    if pose_model is not None and not config.dry_run:
        assert config.pose_root is not None  # checked above
        ensure_skeleton_published(config.pose_root, overwrite=config.overwrite)

    discovery = discover_captures(config.raw_root)
    manifest.rejected_scans.extend(discovery.rejected)
    captures = [capture for capture in discovery.captures if config.selects(capture.capture_id)]
    published: dict[str, list[CameraRecord]] = defaultdict(list)

    # S0 Ingest -- per capture, independent.
    ingested: dict[str, CameraRecord] = {}
    for capture in captures:
        try:
            ingested[capture.capture_id] = ingest_camera(capture, config)
        except ScanRejected as rejection:
            manifest.rejected_scans.append(
                RejectedScan(
                    scan_id=capture.capture_id, source=capture.source, reason=str(rejection)
                )
            )

    # S1 Cut -- per group (the interval owner defines the window), then per capture.
    group_intervals: dict[str, CutInterval] = {}
    group_rejected: set[str] = set()
    outcomes: dict[str, ScanOutcome] = {}
    for group_id, members in _group_captures(captures, ingested):
        interval = _group_cut_interval(config, group_id, members, ingested, manifest)
        if interval is None:
            group_rejected.add(group_id)  # rejection already recorded in the manifest
            continue
        group_intervals[group_id] = interval

        for capture in members:
            record = ingested[capture.capture_id]
            try:
                cut_record, cut_step = cut_camera(record, interval, config)
                orient_record, orient_step = orient_camera(cut_record, config)
                # A camera rejected here has already published S2 output -- the same
                # shape as a camera rejected at S2 having already published S1 output.
                retilt_record, retilt_step = retilt_camera(orient_record, config)
                crop_record, crop_step = crop_camera(retilt_record, config)

                steps = ["ingest", "cut", "orient", "retilt", "crop"]
                derived = {
                    **cut_step.derived,
                    **orient_step.derived,
                    **retilt_step.derived,
                    **crop_step.derived,
                }
                warnings = (
                    cut_step.warnings
                    + orient_step.warnings
                    + retilt_step.warnings
                    + crop_step.warnings
                )
                file_ops = (
                    cut_step.file_ops
                    + orient_step.file_ops
                    + retilt_step.file_ops
                    + crop_step.file_ops
                )

                if pose_model is not None:
                    _, pose_step = detect_pose_camera(crop_record, config, pose_model)
                    steps.append("pose")
                    derived = {**derived, **pose_step.derived}
                    warnings = warnings + pose_step.warnings
                    file_ops = file_ops + pose_step.file_ops
            except ScanRejected as rejection:
                manifest.rejected_scans.append(
                    RejectedScan(
                        scan_id=capture.capture_id,
                        source=capture.source,
                        reason=str(rejection),
                    )
                )
                continue

            published[group_id].append(cut_record)
            outcome = ScanOutcome(
                scan_id=capture.capture_id,
                source=capture.source,
                status="planned" if config.dry_run else "published",
                steps=steps,
                derived=derived,
                warnings=warnings,
                file_ops=file_ops,
            )
            outcomes[capture.capture_id] = outcome
            manifest.scans.append(outcome)

    # S3's day-level cross-check, which needs every capture of a day fitted before it can run.
    _flag_plane_disagreement(captures, ingested, outcomes, config)

    # Per-group metadata, then the groups that lost every capture along the way.
    for group_id, group_records in published.items():
        write_group_metadata(
            config.raw_root / Path(group_id).parts[0] / "meta.yaml",
            group_records,
            config,
            group_intervals.get(group_id),
        )

    for group_id in _groups_without_a_capture(captures, set(published), group_rejected):
        manifest.rejected_scans.append(
            RejectedScan(
                scan_id=group_id,
                source=config.raw_root / group_id,
                reason="session has no usable camera",
            )
        )

    manifest_path = None if config.dry_run else _manifest_path(config)
    return emit_manifest(manifest, manifest_path)


def _group_captures(
    captures: list[CaptureUnit], ingested: dict[str, CameraRecord]
) -> list[tuple[str, list[CaptureUnit]]]:
    """Return `group_id, [CaptureUnit]` for every group with all its captures ingested."""

    by_group: dict[str, list[CaptureUnit]] = defaultdict(list)
    for capture in captures:
        if capture.capture_id in ingested:
            by_group[capture.group_id].append(capture)
    return sorted(by_group.items())


def _interval_owner(members: list[CaptureUnit]) -> CaptureUnit | None:
    """The capture whose lift window the whole group is cut to, chosen by role.

    A two-camera session is cut to its `side` camera's window; a single-camera capture is
    its own owner. Role, never the literal directory name -- no August capture is called
    `Side`, and a July one being called `Side` is an artefact of how it was foldered.
    """

    for capture in members:
        if capture.role == "single":
            return capture
    for capture in members:
        if capture.role == "side":
            return capture
    return None


def _group_cut_interval(
    config: PreprocessConfig,
    group_id: str,
    members: list[CaptureUnit],
    ingested: dict[str, CameraRecord],
    manifest: RunManifest,
) -> CutInterval | None:
    """Derive the group's shared cut interval from its owner, or reject the group."""

    owner = _interval_owner(members)
    if owner is None:
        manifest.rejected_scans.append(
            RejectedScan(
                scan_id=group_id,
                source=config.raw_root / group_id,
                reason="session has no Side camera",
            )
        )
        return None

    try:
        return resolve_cut_interval(owner.metadata_path, ingested[owner.capture_id])
    except ScanRejected as rejection:
        manifest.rejected_scans.append(
            RejectedScan(
                scan_id=group_id,
                source=config.raw_root / group_id,
                reason=str(rejection),
            )
        )
        return None


def _flag_plane_disagreement(
    captures: list[CaptureUnit],
    ingested: dict[str, CameraRecord],
    outcomes: dict[str, ScanOutcome],
    config: PreprocessConfig,
) -> None:
    """Warn on any single-camera capture whose floor plane disagrees with its day's median.

    Warn-only, and deliberately not written into `retilt_sidecar.json`: the median is not
    knowable when that sidecar is written, and rewriting a published sidecar afterwards
    would break the staging-then-publish rule every stage here follows. The manifest is the
    audit record for a run, which is exactly what this is.
    """

    single_camera = {
        capture.capture_id for capture in captures if capture.layout is CaptureLayout.SINGLE_CAMERA
    }
    by_date: dict[str, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for capture_id, outcome in outcomes.items():
        if capture_id not in single_camera:
            continue
        derived = outcome.derived
        if not {"tilt_deg", "roll_deg", "floor_offset_m"} <= derived.keys():
            continue  # a dry run plans the stage without fitting anything
        by_date[ingested[capture_id].date][capture_id] = (
            derived["tilt_deg"],
            derived["roll_deg"],
            derived["floor_offset_m"],
        )

    for fits in by_date.values():
        if len(fits) < MIN_CAPTURES_FOR_CONSISTENCY:
            continue
        consistency = assess_plane_consistency(
            fits,
            tilt_tolerance_deg=config.retilt_group_tilt_tolerance_deg,
            height_tolerance_m=config.retilt_group_height_tolerance_m,
        )
        for capture_id in fits:
            outcome = outcomes[capture_id]
            outcome.derived["group_tilt_median_deg"] = consistency.tilt_median_deg
            outcome.derived["group_height_median_m"] = consistency.height_median_m
            outcome.warnings.extend(consistency.warnings[capture_id])


def _groups_without_a_capture(
    captures: list[CaptureUnit], surviving: set[str], group_rejected: set[str]
) -> list[str]:
    """Groups that published nothing and were not already rejected with a reason of their own.

    Membership is read off the discovered captures, never inferred from how many segments a
    `scan_id` has: both layouts produce three, so the old count-based inference cannot tell
    a group from a capture.

    A single-camera group *is* its capture, so the capture's own rejection already says
    everything this rollup would -- under the same `scan_id`, which would read as two
    unrelated failures. Only a group that could have lost *some* of its cameras gets the
    rollup; that is what it was for.
    """

    groups = {
        capture.group_id
        for capture in captures
        if capture.layout is not CaptureLayout.SINGLE_CAMERA
    }
    return sorted(groups - surviving - group_rejected)


def _manifest_path(config: PreprocessConfig) -> Path:
    config.record_root.mkdir(parents=True, exist_ok=True)
    return config.record_root / "manifest.json"
