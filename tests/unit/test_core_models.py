"""Contracts shared by every pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from powerflow_pipeline.data.core.context import OutputMode, RunContext
from powerflow_pipeline.data.core.models import CropBounds, FileOp, Scan, StepResult


def test_crop_bounds_exposes_width_and_height() -> None:
    bounds = CropBounds(x0=32, y0=32, x1=1888, y1=1048)
    assert bounds.width == 1856
    assert bounds.height == 1016


@pytest.mark.parametrize(
    ("x0", "y0", "x1", "y1"),
    [
        (32, 32, 32, 1048),  # collapsed in x
        (32, 32, 1888, 32),  # collapsed in y
        (100, 32, 50, 1048),  # inverted in x
        (32, 100, 1888, 50),  # inverted in y
    ],
)
def test_crop_bounds_rejects_non_positive_extent(x0: int, y0: int, x1: int, y1: int) -> None:
    with pytest.raises(ValidationError):
        CropBounds(x0=x0, y0=y0, x1=x1, y1=y1)


def test_crop_bounds_rejects_negative_origin() -> None:
    with pytest.raises(ValidationError):
        CropBounds(x0=-1, y0=0, x1=100, y1=100)


def test_scan_is_frozen() -> None:
    scan = Scan(scan_id="scan_0001", source=Path("/tmp/scan_0001"), files=(Path("meta.json"),))
    with pytest.raises(ValidationError):
        scan.scan_id = "other"  # type: ignore[misc]


def test_step_result_defaults_are_empty() -> None:
    result = StepResult()
    assert result.derived == {}
    assert result.warnings == []
    assert result.file_ops == []


def test_file_op_round_trips_through_json() -> None:
    op = FileOp(op="publish", src=Path("/tmp/staging/a"), dst=Path("/out/a"))
    assert FileOp.model_validate_json(op.model_dump_json()) == op


def test_publish_context_requires_an_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_root"):
        RunContext.create(
            pipeline="dummy",
            input_root=tmp_path,
            output_root=None,
            output_mode=OutputMode.PUBLISH,
        )


def test_in_place_context_rejects_an_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not"):
        RunContext.create(
            pipeline="dummy",
            input_root=tmp_path,
            output_root=tmp_path / "output",
            output_mode=OutputMode.IN_PLACE,
        )
