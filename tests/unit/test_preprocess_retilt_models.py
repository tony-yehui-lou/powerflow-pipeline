"""S3 Retilt's models: `PlaneFit`, `RetiltResult`."""

from __future__ import annotations

from powerflow_pipeline.data.common.capture_models import PlaneFit, RetiltResult
from powerflow_pipeline.data.common.models import CropBounds


def test_plane_fit_round_trips_through_json() -> None:
    fit = PlaneFit(
        normal=[0.01, -0.999, 0.02],
        rms_residual_m=0.004,
        n_points=1200,
        n_frames_sampled=8,
        confidence_mode="conf2_only",
        floor_offset_m=1.5,
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
            floor_offset_m=1.2,
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
            floor_offset_m=1.0,
        ),
        gravity_agreement_deg=None,
        homography_rgb=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        homography_depth=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        valid_bounds_px=CropBounds(x0=0, y0=0, x1=100, y1=200),
        translation_span_m=0.0,
    )
    assert result.gravity_agreement_deg is None
