"""The preprocess flow: S0 ingest, S1 cut, S2 orient. Ordered orchestration only.

Cut needs the Side camera to define the shared epoch window, so the loop is per-session:
ingest every camera, derive the cut interval from the Side record, then cut and orient
each camera against that shared interval. A session without a usable Side or whose cut
interval is invalid is rejected whole — the invariant is cross-camera, not per-camera.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from prefect import flow

from powerflow_pipeline.data.common.context import OutputMode, RunContext
from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.manifest import RunManifest, emit_manifest
from powerflow_pipeline.data.common.models import RejectedScan, ScanOutcome
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraDir, CameraRecord, CutInterval
from powerflow_pipeline.data.preprocess.tasks.cut import cut_camera, resolve_cut_interval
from powerflow_pipeline.data.preprocess.tasks.discover import discover_sessions
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from powerflow_pipeline.data.preprocess.tasks.metadata import write_session_metadata
from powerflow_pipeline.data.preprocess.tasks.orient import orient_camera

PIPELINE = "preprocess"


@flow(name=PIPELINE)
def preprocess(config: PreprocessConfig) -> RunManifest:
    """Ingest, cut to the shared lift window, and rotate to portrait. Or reject, with a reason."""

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

    cameras = list(discover_sessions(config.raw_root))
    sessions: dict[tuple[str, str], list[CameraRecord]] = defaultdict(list)

    # S0 Ingest — per camera, optional.
    ingested: dict[tuple[str, str, str], CameraRecord] = {}
    for camera in cameras:
        try:
            ingested[(camera.date, camera.session, camera.camera)] = ingest_camera(camera, config)
        except ScanRejected as rejection:
            manifest.rejected_scans.append(
                RejectedScan(scan_id=camera.camera_id, source=camera.source, reason=str(rejection))
            )

    # S1 Cut — per session (Side defines the window), then per camera.
    session_intervals: dict[tuple[str, str], CutInterval] = {}
    for (date, session), session_cameras in _group_cameras(cameras, ingested):
        interval = _session_cut_interval(config, date, session, ingested, manifest)
        if interval is None:
            continue  # rejection already recorded in the manifest
        session_intervals[(date, session)] = interval

        for camera in session_cameras:
            record = ingested[(camera.date, camera.session, camera.camera)]
            try:
                cut_record, cut_step = cut_camera(record, interval, config)
                orient_step = orient_camera(cut_record, config)
            except ScanRejected as rejection:
                manifest.rejected_scans.append(
                    RejectedScan(
                        scan_id=camera.camera_id, source=camera.source, reason=str(rejection)
                    )
                )
                continue

            sessions[(date, session)].append(cut_record)
            manifest.scans.append(
                ScanOutcome(
                    scan_id=camera.camera_id,
                    source=camera.source,
                    status="planned" if config.dry_run else "published",
                    steps=["ingest", "cut", "orient"],
                    derived={**cut_step.derived, **orient_step.derived},
                    warnings=cut_step.warnings + orient_step.warnings,
                    file_ops=cut_step.file_ops + orient_step.file_ops,
                )
            )

    # Per-session metadata + zero-camera session rejection.
    for (date, _), session_records in sessions.items():
        interval = session_intervals.get((date, _))
        write_session_metadata(
            config.raw_root / date / "meta.yaml", session_records, config, interval
        )

    for date, session_name in _sessions_without_a_camera(manifest, set(sessions)):
        manifest.rejected_scans.append(
            RejectedScan(
                scan_id=f"{date}/{session_name}",
                source=config.raw_root / date / session_name,
                reason="session has no usable camera",
            )
        )

    manifest_path = None if config.dry_run else _manifest_path(config)
    return emit_manifest(manifest, manifest_path)


def _group_cameras(
    cameras: Sequence[CameraDir],
    ingested: dict[tuple[str, str, str], CameraRecord],
) -> list[tuple[tuple[str, str], list[CameraDir]]]:
    """Return `(date, session), [CameraDir]` for every session with all its cameras ingested."""

    from collections import defaultdict

    ingested_ids = {(k[0], k[1]) for k in ingested}
    by_session: dict[tuple[str, str], list[CameraDir]] = defaultdict(list)
    for camera in cameras:
        key = (camera.date, camera.session)
        if key in ingested_ids:
            by_session[key].append(camera)
    return sorted(by_session.items())


def _session_cut_interval(
    config: PreprocessConfig,
    date: str,
    session: str,
    ingested: dict[tuple[str, str, str], CameraRecord],
    manifest: RunManifest,
) -> CutInterval | None:
    """Derive the shared cut interval from the Side record, or reject the session."""

    side_key = (date, session, "Side")
    if side_key not in ingested:
        manifest.rejected_scans.append(
            RejectedScan(
                scan_id=f"{date}/{session}",
                source=config.raw_root / date / session,
                reason="session has no Side camera",
            )
        )
        return None

    try:
        return resolve_cut_interval(config.raw_root, date, session, ingested[side_key])
    except ScanRejected as rejection:
        manifest.rejected_scans.append(
            RejectedScan(
                scan_id=f"{date}/{session}",
                source=config.raw_root / date / session,
                reason=str(rejection),
            )
        )
        return None


def _sessions_without_a_camera(
    manifest: RunManifest, surviving: set[tuple[str, str]]
) -> list[tuple[str, str]]:
    rejected_per_camera = {
        (parts[0], parts[1])
        for rejected_scan in manifest.rejected_scans
        if len(parts := rejected_scan.scan_id.split("/")) == 3
    }
    rejected_per_session = {
        (parts[0], parts[1])
        for rejected_scan in manifest.rejected_scans
        if len(parts := rejected_scan.scan_id.split("/")) == 2
    }
    return sorted((rejected_per_camera - surviving) - rejected_per_session)


def _manifest_path(config: PreprocessConfig) -> Path:
    config.record_root.mkdir(parents=True, exist_ok=True)
    return config.record_root / "manifest.json"
