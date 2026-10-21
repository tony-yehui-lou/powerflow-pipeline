"""Draft 1's body-plane solve, S8's initial guess: closed form, checked against 3D truth."""

from __future__ import annotations

import math

import pytest

from powerflow_pipeline.data.preprocess import body_plane as bp
from powerflow_pipeline.data.preprocess.bar_geometry import project
from powerflow_pipeline.data.preprocess.models import Intrinsics

K = Intrinsics(fx=1336.9, fy=1336.9, cx=705.0, cy=802.6, frame="portrait")
YAW = 38.6
AXIS = (math.sin(math.radians(YAW)), 0.0, math.cos(math.radians(YAW)))
CENTRE = (-150.0, 400.0, 2777.0)


def at(offset: float, dy: float = 0.0, ds: float = 0.0) -> tuple[float, float, float]:
    """A point `offset` along the bar from its centre, `dy` lower and `ds` along the sagittal."""

    n = (AXIS[2], 0.0, -AXIS[0])
    return tuple(  # type: ignore[return-value]
        c + offset * a + ds * s + (dy if i == 1 else 0.0)
        for i, (c, a, s) in enumerate(zip(CENTRE, AXIS, n, strict=True))
    )


@pytest.mark.parametrize("offset", [-460.0, -90.0, 0.0, 220.0])
def test_place_on_plane_recovers_a_point_on_its_plane(offset: float) -> None:
    truth = at(offset, dy=350.0, ds=120.0)

    got = bp.place_on_plane(project(truth, K), K, CENTRE, AXIS, offset)

    assert got is not None
    assert got == pytest.approx(truth, abs=1e-6)


def test_a_ray_along_the_plane_is_ill_conditioned() -> None:
    # Looking straight down the plane: the axis perpendicular to the optical axis.
    side_axis = (1.0, 0.0, 0.0)
    assert bp.place_on_plane((K.cx, K.cy), K, (0.0, 0.0, 3000.0), side_axis, 0.0) is None


def test_place_on_floor_puts_the_point_at_the_given_height() -> None:
    truth = (200.0, 1300.0, 3100.0)  # 1300 mm below the camera

    got = bp.place_on_floor(project(truth, K), K, y_mm=1300.0)

    assert got == pytest.approx(truth, abs=1e-6)
    assert bp.place_on_floor((K.cx, K.cy - 50.0), K, y_mm=1300.0) is None  # above the horizon


def test_lateral_offsets_put_the_near_side_towards_the_cap() -> None:
    offsets = bp.lateral_offsets(
        near_side="right", half_grip_mm=460.0, shoulder_width_mm=400.0, hip_width_mm=180.0
    )

    assert offsets["rightWrist"] == -460.0
    assert offsets["leftWrist"] == 460.0
    assert offsets["leftShoulder"] == 200.0
    assert offsets["rightElbow"] == pytest.approx(-(460.0 + 200.0) / 2)
    assert offsets["leftAnkle"] == offsets["leftHip"] == 90.0
    assert offsets["head"] == 0.0
