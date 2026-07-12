"""Ordered orchestration for the dummy review pipeline."""

from __future__ import annotations

from pathlib import Path

from powerflow_pipeline.data.core.context import RunContext
from powerflow_pipeline.data.core.manifest import RunManifest
from powerflow_pipeline.data.core.models import Scan, ScanOutcome, StepResult
from powerflow_pipeline.data.core.runner import PipelineRunner
from powerflow_pipeline.data.dummy.steps.copy import copy_scan
from powerflow_pipeline.data.dummy.steps.sidecar import write_sidecar
from powerflow_pipeline.data.dummy.steps.validate import validate_metadata


def process_scan(scan: Scan, staging: Path | None, target: Path, dry_run: bool) -> ScanOutcome:
    """Validate, stage, and annotate one scan in the documented order."""

    metadata, validation = validate_metadata(scan)
    copy = copy_scan(scan, staging, target, dry_run)
    sidecar = write_sidecar(scan, metadata, staging, target, dry_run)
    results: tuple[StepResult, ...] = (validation, copy, sidecar)
    return ScanOutcome(
        scan_id=scan.scan_id,
        source=scan.source,
        status="planned" if dry_run else "published",
        steps=["validate_metadata", "copy_scan", "write_sidecar"],
        derived={key: value for result in results for key, value in result.derived.items()},
        warnings=[warning for result in results for warning in result.warnings],
        file_ops=[operation for result in results for operation in result.file_ops],
    )


def validate_staged_scan(path: Path) -> None:
    """Ensure every required copied and derived artifact exists before publishing."""

    if not (path / "meta.json").is_file():
        raise RuntimeError(f"staged scan is missing meta.json: {path}")
    if not (path / "frames").is_dir() or not any((path / "frames").rglob("*")):
        raise RuntimeError(f"staged scan is missing frames: {path}")
    if not (path / "dummy.sidecar.json").is_file():
        raise RuntimeError(f"staged scan is missing dummy sidecar: {path}")


def run(context: RunContext) -> RunManifest:
    """Execute the dummy pipeline with the shared transactional runner."""

    return PipelineRunner("dummy", process_scan, validate_staged_scan).run(context)
