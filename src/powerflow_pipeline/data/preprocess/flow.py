"""The preprocess flow: S0 ingest, then S1 orient. Ordered orchestration only."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from prefect import flow

from powerflow_pipeline.data.common.context import OutputMode, RunContext
from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.manifest import RunManifest, emit_manifest
from powerflow_pipeline.data.common.models import RejectedScan, ScanOutcome
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.tasks.discover import discover_sessions
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from powerflow_pipeline.data.preprocess.tasks.metadata import write_session_metadata
from powerflow_pipeline.data.preprocess.tasks.orient import orient_camera

PIPELINE = "preprocess"


@flow(name=PIPELINE)
def preprocess(config: PreprocessConfig) -> RunManifest:
    """Rotate every camera to portrait, or reject it with a reason. Never both."""

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

    records: dict[tuple[str, str], list[CameraRecord]] = defaultdict(list)
    for camera in discover_sessions(config.raw_root):
        try:
            record = ingest_camera(camera, config)
            step = orient_camera(record, config)
        except ScanRejected as rejection:
            manifest.rejected_scans.append(
                RejectedScan(scan_id=camera.camera_id, source=camera.source, reason=str(rejection))
            )
            continue

        records[(camera.date, camera.session)].append(record)
        manifest.scans.append(
            ScanOutcome(
                scan_id=camera.camera_id,
                source=camera.source,
                status="planned" if config.dry_run else "published",
                steps=["ingest", "orient"],
                derived=step.derived,
                warnings=step.warnings,
                file_ops=step.file_ops,
            )
        )

    for (date, _), session_records in records.items():
        write_session_metadata(config.raw_root / date / "meta.yaml", session_records, config)

    # A session whose every camera was rejected is a rejected session; S2 decides whether a
    # one-camera session is usable, but a zero-camera one never is.
    for date, session in _sessions_without_a_camera(manifest, set(records)):
        manifest.rejected_scans.append(
            RejectedScan(
                scan_id=f"{date}/{session}",
                source=config.raw_root / date / session,
                reason="session has no usable camera",
            )
        )

    manifest_path = None if config.dry_run else _manifest_path(config)
    return emit_manifest(manifest, manifest_path)


def _sessions_without_a_camera(
    manifest: RunManifest, surviving: set[tuple[str, str]]
) -> list[tuple[str, str]]:
    rejected = {
        (parts[0], parts[1])
        for rejected_scan in manifest.rejected_scans
        if len(parts := rejected_scan.scan_id.split("/")) == 3
    }
    return sorted(rejected - surviving)


def _manifest_path(config: PreprocessConfig) -> Path:
    config.record_root.mkdir(parents=True, exist_ok=True)
    return config.record_root / "manifest.json"
