"""The 90 degree rotation is exact: every output pixel is an input pixel, moved."""

from __future__ import annotations

import numpy as np
import pytest

from powerflow_pipeline.data.preprocess.config import RotationDirection
from powerflow_pipeline.data.preprocess.geometry import (
    rotate_frame,
    rotate_intrinsics,
    rotated_size,
)
from powerflow_pipeline.data.preprocess.models import Intrinsics

DEPTH_W, DEPTH_H = 256, 192
MARKER_U, MARKER_V = 5, 3

# The averaged per-frame intrinsics measured from data/raw/9 July/cnj_45kg_Set1/Front.
AVERAGED_K = Intrinsics(fx=1346.9054, fy=1346.9054, cx=968.3401, cy=719.6459, frame="landscape")


def marker_depth() -> np.ndarray:
    """A landscape depth map whose single hot pixel is asymmetric in both axes."""

    depth = np.zeros((DEPTH_H, DEPTH_W), dtype=np.uint16)
    depth[MARKER_V, MARKER_U] = 1234
    return depth


def test_rotated_size_swaps_the_axes() -> None:
    assert rotated_size(DEPTH_W, DEPTH_H) == (DEPTH_H, DEPTH_W)


def test_cw_rotation_maps_pixels_exactly() -> None:
    rotated = rotate_frame(marker_depth(), RotationDirection.CW)

    assert rotated.shape == (DEPTH_W, DEPTH_H)  # (H', W') = (256, 192): portrait
    # CW: u' = H - 1 - v, v' = u
    assert np.argwhere(rotated == 1234).tolist() == [[MARKER_U, DEPTH_H - 1 - MARKER_V]]
    assert (rotated[MARKER_U, 188], MARKER_U, 188) == (1234, 5, 188)


def test_ccw_rotation_maps_pixels_exactly() -> None:
    rotated = rotate_frame(marker_depth(), RotationDirection.CCW)

    assert rotated.shape == (DEPTH_W, DEPTH_H)
    # CCW: u' = v, v' = W - 1 - u
    assert np.argwhere(rotated == 1234).tolist() == [[DEPTH_W - 1 - MARKER_U, MARKER_V]]
    assert (rotated[250, MARKER_V], 250, MARKER_V) == (1234, 250, 3)


def test_rotation_invents_no_values_and_preserves_dtype() -> None:
    """Nearest-neighbour by construction: the output value set equals the input's."""

    source = np.random.default_rng(0).integers(0, 4096, size=(DEPTH_H, DEPTH_W), dtype=np.uint16)

    rotated = rotate_frame(source, RotationDirection.CW)

    assert rotated.dtype == np.uint16
    assert np.array_equal(np.unique(rotated), np.unique(source))
    assert sorted(rotated.ravel().tolist()) == sorted(source.ravel().tolist())


def test_confidence_rotation_never_invents_a_middle_value() -> None:
    """Interpolating {0, 2} would fabricate a 1 the sensor never reported."""

    confidence = np.zeros((DEPTH_H, DEPTH_W), dtype=np.uint8)
    confidence[:, DEPTH_W // 2 :] = 2

    rotated = rotate_frame(confidence, RotationDirection.CW)

    assert rotated.dtype == np.uint8
    assert set(np.unique(rotated).tolist()) == {0, 2}


def test_colour_frames_rotate_channel_wise() -> None:
    source = np.zeros((DEPTH_H, DEPTH_W, 3), dtype=np.uint8)
    source[MARKER_V, MARKER_U] = (10, 20, 30)

    rotated = rotate_frame(source, RotationDirection.CW)

    assert rotated.shape == (DEPTH_W, DEPTH_H, 3)
    assert rotated[MARKER_U, DEPTH_H - 1 - MARKER_V].tolist() == [10, 20, 30]


def test_cw_then_ccw_is_the_identity() -> None:
    source = marker_depth()

    assert np.array_equal(
        rotate_frame(rotate_frame(source, RotationDirection.CW), RotationDirection.CCW), source
    )


def test_cw_intrinsics_land_on_the_portrait_centre() -> None:
    """ISSUE-01: K is landscape. Rotated CW it must describe the portrait frame."""

    rotated = rotate_intrinsics(AVERAGED_K, width=1920, height=1440, rotation=RotationDirection.CW)

    assert rotated.fx == pytest.approx(1346.9054)  # fx' = fy
    assert rotated.fy == pytest.approx(1346.9054)  # fy' = fx
    assert rotated.cx == pytest.approx(1440 - 1 - 719.6459)  # 719.3541
    assert rotated.cy == pytest.approx(968.3401)
    assert rotated.frame == "portrait"

    # The point of the rewrite: it now sits on the portrait centre, not 240 px away.
    assert abs(rotated.cx - 1440 / 2) < 1.0
    assert abs(rotated.cy - 1920 / 2) < 10.0


def test_ccw_intrinsics_use_the_other_reflection() -> None:
    rotated = rotate_intrinsics(AVERAGED_K, width=1920, height=1440, rotation=RotationDirection.CCW)

    assert rotated.cx == pytest.approx(719.6459)
    assert rotated.cy == pytest.approx(1920 - 1 - 968.3401)


def test_rotating_intrinsics_twice_returns_the_original_frame() -> None:
    rotated = rotate_intrinsics(AVERAGED_K, width=1920, height=1440, rotation=RotationDirection.CW)
    restored = rotate_intrinsics(rotated, width=1440, height=1920, rotation=RotationDirection.CCW)

    assert restored.fx == pytest.approx(AVERAGED_K.fx)
    assert restored.cx == pytest.approx(AVERAGED_K.cx)
    assert restored.cy == pytest.approx(AVERAGED_K.cy)
    assert restored.frame == "landscape"
