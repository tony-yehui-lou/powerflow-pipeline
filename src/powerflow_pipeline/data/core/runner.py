"""Generic staged runner used by concrete data pipelines."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from powerflow_pipeline.data.core.context import OutputMode, RunContext
from powerflow_pipeline.data.core.discovery import discover_scans
from powerflow_pipeline.data.core.errors import ScanRejected
from powerflow_pipeline.data.core.filesystem import (
    cleanup,
    commit_in_place,
    create_staging_root,
    publish_staging,
)
from powerflow_pipeline.data.core.manifest import RunManifest
from powerflow_pipeline.data.core.models import RejectedScan, Scan, ScanOutcome

ProcessScan = Callable[[Scan, Path | None, Path, bool], ScanOutcome]
ValidateStagedScan = Callable[[Path], None]


class PipelineRunner:
    """Run scans in staging and publish only after every staged result validates."""

    def __init__(
        self, pipeline: str, process_scan: ProcessScan, validate_staged_scan: ValidateStagedScan
    ):
        self.pipeline = pipeline
        self.process_scan = process_scan
        self.validate_staged_scan = validate_staged_scan

    def run(self, context: RunContext) -> RunManifest:
        """Execute the configured pipeline and return its complete audit manifest."""

        discovered = discover_scans(context.input_root)
        staging_root: Path | None = None
        if not context.dry_run:
            anchor = (
                context.output_root
                if context.output_mode is OutputMode.PUBLISH
                else context.input_root
            )
            if anchor is None:
                raise RuntimeError("publish mode requires output_root")
            staging_root = create_staging_root(anchor)

        outcomes: list[ScanOutcome] = []
        rejected = list(discovered.rejected_scans)
        try:
            for scan in discovered.scans:
                target = context.destination_for(scan.scan_id)
                scan_staging = staging_root / scan.scan_id if staging_root is not None else None
                try:
                    outcomes.append(self.process_scan(scan, scan_staging, target, context.dry_run))
                except ScanRejected as error:
                    rejected.append(
                        RejectedScan(scan_id=scan.scan_id, source=scan.source, reason=str(error))
                    )

            manifest = RunManifest(
                pipeline=self.pipeline,
                input_root=context.input_root,
                output_root=context.output_root,
                output_mode=context.output_mode,
                dry_run=context.dry_run,
                scans=outcomes,
                rejected_scans=rejected,
            )
            if context.dry_run:
                return manifest
            if staging_root is None:
                raise RuntimeError("non-dry-run requires staging")

            for outcome in outcomes:
                self.validate_staged_scan(staging_root / outcome.scan_id)
            manifest.write(staging_root / "manifest.json")

            if context.output_mode is OutputMode.PUBLISH:
                if context.output_root is None:
                    raise RuntimeError("publish mode requires output_root")
                publish_staging(staging_root, context.output_root)
                staging_root = None
            else:
                replacements = [
                    (staging_root / outcome.scan_id, context.input_root / outcome.scan_id)
                    for outcome in outcomes
                ]
                replacements.append(
                    (staging_root / "manifest.json", context.input_root / "manifest.json")
                )
                commit_in_place(replacements)
            return manifest
        finally:
            if staging_root is not None:
                cleanup(staging_root)
