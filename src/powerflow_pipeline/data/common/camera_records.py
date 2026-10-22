"""Rebuild S4's camera record from what S0-S4 published, for the pipelines that run after them.

`preprocess` hands each camera's record from stage to stage in memory. `detect` and `analyze`
run separately (docs/superpowers/specs/2026-10-22-pipeline-phase-split-design.md §2.3), so
they rebuild the same `CameraRecord` S4 returned, field for field, from the published files:

- S0's `record.json`: everything S1-S4 never change;
- S1's `cut_sidecar.json`: the aligned frame count and the retained stream counts;
- S4's `crop_sidecar.json` and `camera_matrix.csv`: the cropped sizes and intrinsics.

Nothing is recomputed. A camera is discovered by its S4 output: a camera rejected anywhere in
S0-S4 published no `crop_sidecar.json`, so it never reaches the later pipelines.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Final

from prefect import task

from powerflow_pipeline.data.common.capture_models import CameraRecord, Intrinsics, StreamCounts
from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.filesystem import INTERNAL_PREFIX
from powerflow_pipeline.data.common.models import RejectedScan
from powerflow_pipeline.data.common.task_logging import log_task_paths

RECORD_FILENAME: Final = "record.json"
CUT_SIDECAR_FILENAME: Final = "cut_sidecar.json"
CROP_SIDECAR_FILENAME: Final = "crop_sidecar.json"
CAMERA_MATRIX_FILENAME: Final = "camera_matrix.csv"


@dataclass(slots=True)
class PublishedCameras:
    """Every camera S4 published, rebuilt, plus the ones whose upstream files are incomplete."""

    records: list[CameraRecord] = field(default_factory=list)
    rejected: list[RejectedScan] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ScanRejected(f"no {path.name} published at {path}")
    return json.loads(path.read_text())


def _read_intrinsics(path: Path) -> Intrinsics:
    """S4's `K`, already rotated to portrait by S2 and shifted by the crop."""

    if not path.is_file():
        raise ScanRejected(f"no {path.name} published at {path}")
    with path.open(newline="") as handle:
        rows = [[float(cell) for cell in row] for row in csv.reader(handle) if row]
    return Intrinsics.from_matrix(rows, frame="portrait")


def read_published_record(
    relative: Path, *, record_root: Path, cut_root: Path, crop_root: Path
) -> CameraRecord:
    """The `CameraRecord` S4 returned for this camera. `ScanRejected` if a file is missing."""

    record = CameraRecord.model_validate(_read_json(record_root / relative / RECORD_FILENAME))
    cut = _read_json(cut_root / relative / CUT_SIDECAR_FILENAME)
    crop = _read_json(crop_root / relative / CROP_SIDECAR_FILENAME)
    retained = cut["retained"]
    rgb_size, depth_size = crop["rgb_output_size"], crop["depth_output_size"]
    return record.model_copy(
        update={
            "source": crop_root / relative,
            "counts": StreamCounts(**{name: retained[name]["count"] for name in retained}),
            "n_frames": cut["n_frames"],
            "rgb_width": rgb_size[0],
            "rgb_height": rgb_size[1],
            "depth_width": depth_size[0],
            "depth_height": depth_size[1],
            "intrinsics": _read_intrinsics(crop_root / relative / CAMERA_MATRIX_FILENAME),
        }
    )


@task
def discover_published_cameras(
    record_root: Path, cut_root: Path, crop_root: Path, *, only: str | None
) -> PublishedCameras:
    """Every camera with S4 output under `crop_root` matching `only`, rebuilt, in id order."""

    log_task_paths(crop_root, record_root)
    found = PublishedCameras()
    if not crop_root.is_dir():
        return found
    for sidecar in sorted(crop_root.rglob(CROP_SIDECAR_FILENAME)):
        relative = sidecar.parent.relative_to(crop_root)
        if any(part.startswith(INTERNAL_PREFIX) for part in relative.parts):
            continue  # a stage's staging directory, never published output
        if only is not None and not fnmatch(str(relative), only):
            continue
        try:
            found.records.append(
                read_published_record(
                    relative, record_root=record_root, cut_root=cut_root, crop_root=crop_root
                )
            )
        except ScanRejected as rejection:
            found.rejected.append(
                RejectedScan(scan_id=str(relative), source=sidecar.parent, reason=str(rejection))
            )
    return found
