"""S3 Retilt's models: `FloorRegion`, `PlaneFit`, `RetiltResult`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from powerflow_pipeline.data.common.models import CropBounds
from powerflow_pipeline.data.preprocess.models import FloorRegion, PlaneFit, RetiltResult


def test_floor_region_rejects_x_degenerate() -> None:
    with pytest.raises(ValidationError):
        FloorRegion(x0=0.5, y0=0.2, x1=0.5, y1=0.1)


def test_floor_region_rejects_y_degenerate() -> None:
    # y1 must be strictly less than y0 (top-right above bottom-left) -- the CORRECTED rule.
    with pytest.raises(ValidationError):
        FloorRegion(x0=0.0, y0=0.5, x1=1.0, y1=0.6)


def test_floor_region_accepts_real_capture_values() -> None:
    # data/raw/11 July/30kg_Set1/metadata.yaml, side region.
    # (0, round(1920*0.64503), round(1440*0.40268), round(1920*1.0)) = (0, 1238, 580, 1920).
    region = FloorRegion(x0=0.0, y0=1.0, x1=0.40268, y1=0.64503)
    assert region.pixel_bounds(1440, 1920) == (0, 1238, 580, 1920)


def test_floor_region_rejects_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        FloorRegion(x0=-0.1, y0=1.0, x1=0.5, y1=0.5)


def test_plane_fit_round_trips_through_json() -> None:
    fit = PlaneFit(
        normal=[0.01, -0.999, 0.02],
        rms_residual_m=0.004,
        n_points=1200,
        n_frames_sampled=8,
        confidence_mode="conf2_only",
    )
    assert PlaneFit.model_validate_json(fit.model_dump_json()) == fit


def test_retilt_result_round_trips_through_json() -> None:
    result = RetiltResult(
        tilt_deg=7.0,
        roll_deg=-3.0,
        plane=PlaneFit(
            normal=[0.0, -1.0, 0.0],
            rms_residual_m=0.003,
            n_points=900,
            n_frames_sampled=6,
            confidence_mode="conf1_and_2",
        ),
        gravity_agreement_deg=1.2,
        homography_rgb=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        homography_depth=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        valid_bounds_px=CropBounds(x0=0, y0=0, x1=100, y1=200),
        translation_span_m=0.01,
    )
    assert RetiltResult.model_validate_json(result.model_dump_json()) == result


def test_retilt_result_allows_no_gravity_agreement() -> None:
    result = RetiltResult(
        tilt_deg=0.0,
        roll_deg=0.0,
        plane=PlaneFit(
            normal=[0.0, -1.0, 0.0],
            rms_residual_m=0.001,
            n_points=600,
            n_frames_sampled=4,
            confidence_mode="conf2_only",
        ),
        gravity_agreement_deg=None,
        homography_rgb=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        homography_depth=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        valid_bounds_px=CropBounds(x0=0, y0=0, x1=100, y1=200),
        translation_span_m=0.0,
    )
    assert result.gravity_agreement_deg is None
