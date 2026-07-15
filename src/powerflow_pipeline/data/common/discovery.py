"""Deterministic scan discovery independent of pipeline-specific metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from prefect import task

from powerflow_pipeline.data.common.filesystem import INTERNAL_PREFIX
from powerflow_pipeline.data.common.models import RejectedScan, Scan
from powerflow_pipeline.data.common.task_logging import log_task_paths


@dataclass(slots=True)
class DiscoveryResult:
    """Scans ready for pipeline validation plus structurally rejected entries."""

    scans: list[Scan] = field(default_factory=list)
    rejected_scans: list[RejectedScan] = field(default_factory=list)


@task
def discover_scans(input_root: Path) -> DiscoveryResult:
    """Find immediate-child scan directories in stable order."""

    log_task_paths(input_root, None)
    result = DiscoveryResult()
    for candidate in sorted(input_root.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir() or candidate.name.startswith(INTERNAL_PREFIX):
            continue
        meta_path = candidate / "meta.json"
        frames_path = candidate / "frames"
        if not meta_path.is_file():
            result.rejected_scans.append(
                RejectedScan(
                    scan_id=candidate.name,
                    source=candidate,
                    reason="missing required meta.json",
                )
            )
            continue
        frame_files = (
            [path for path in frames_path.rglob("*") if path.is_file()]
            if frames_path.is_dir()
            else []
        )
        if not frame_files:
            result.rejected_scans.append(
                RejectedScan(
                    scan_id=candidate.name,
                    source=candidate,
                    reason="missing or empty required frames directory",
                )
            )
            continue
        files = tuple(
            sorted(path.relative_to(candidate) for path in candidate.rglob("*") if path.is_file())
        )
        result.scans.append(Scan(scan_id=candidate.name, source=candidate, files=files))
    return result
