"""Create downstream metadata needed to review the dummy output."""

from __future__ import annotations

from pathlib import Path

from powerflow_pipeline.data.core.filesystem import write_json
from powerflow_pipeline.data.core.models import CropBounds, FileOp, Scan, StepResult
from powerflow_pipeline.data.dummy.config import DummyScanMetadata


def write_sidecar(
    scan: Scan,
    metadata: DummyScanMetadata,
    staging: Path | None,
    target: Path,
    dry_run: bool,
) -> StepResult:
    """Persist the deterministic scale, crop, and provenance sidecar."""

    bounds = CropBounds(x0=0, y0=0, x1=metadata.width, y1=metadata.height)
    px_per_mm = metadata.marker_px / metadata.marker_mm
    sidecar = {
        "crop_bounds": bounds.model_dump(),
        "pipeline": "dummy",
        "provenance": {"metadata_file": "meta.json", "source": str(scan.source)},
        "px_per_mm": px_per_mm,
        "scan_id": scan.scan_id,
    }
    target_sidecar = target / "dummy.sidecar.json"
    if not dry_run:
        if staging is None:
            raise RuntimeError("staging path is required outside dry-run mode")
        write_json(staging / "dummy.sidecar.json", sidecar)
    return StepResult(
        derived={"crop_bounds": bounds.model_dump(), "px_per_mm": px_per_mm},
        file_ops=[FileOp(op="write", src=scan.source / "meta.json", dst=target_sidecar)],
    )
