"""Pure maths for S4 Crop: residual translation, guard band, intersection, even-extent fix.

No I/O, no Prefect -- numeric behavior is checked against hand-worked values and the four
real cameras' measured crop rectangles (never accepted merely because it ran, per CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.models import CropBounds
from powerflow_pipeline.data.preprocess.crop import (
    crop_fractions,
    crop_intrinsics,
    depth_bounds,
    guard_depth_m,
    intersect_crop,
    max_translation_m,
    motion_bounds_px,
    read_valid_bounds,
    reference_displacements,
    shrink_to_even,
)
from powerflow_pipeline.data.preprocess.models import Intrinsics

# --- read_valid_bounds ------------------------------------------------------------------


def test_read_valid_bounds_parses_the_flat_list() -> None:
    sidecar = {"valid_bounds_px": [0, 238, 1440, 1920]}
    bounds = read_valid_bounds(sidecar)
    assert (bounds.x0, bounds.y0, bounds.x1, bounds.y1) == (0, 238, 1440, 1920)


def test_read_valid_bounds_rejects_missing_key() -> None:
    with pytest.raises(ScanRejected, match="valid_bounds_px"):
        read_valid_bounds({})


def test_read_valid_bounds_rejects_wrong_length() -> None:
    with pytest.raises(ScanRejected, match="valid_bounds_px"):
        read_valid_bounds({"valid_bounds_px": [0, 238, 1440]})


def test_read_valid_bounds_rejects_degenerate() -> None:
    with pytest.raises(ScanRejected, match="degenerate"):
        read_valid_bounds({"valid_bounds_px": [100, 238, 50, 1920]})  # x0 > x1


# --- reference_displacements -------------------------------------------------------------


def test_reference_displacements_zero_motion_is_zero() -> None:
    positions = np.zeros((5, 3))
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (5, 1))  # identity
    displacements = reference_displacements(positions, quaternions)
    assert displacements == pytest.approx(np.zeros((5, 3)))


def test_reference_displacements_expresses_world_motion_in_frame_zero() -> None:
    # Frame 0's camera-to-world rotation is 90 deg about Y: world X <- camera Z.
    from scipy.spatial.transform import Rotation

    r0 = Rotation.from_euler("y", 90, degrees=True)
    q0 = r0.as_quat()  # (x, y, z, w)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # world moves +1 in X
    quaternions = np.array([q0, q0])

    displacements = reference_displacements(positions, quaternions)

    # R_0^T maps world +X to camera +Z for this rotation.
    assert displacements[1] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


def test_reference_displacements_rejects_non_finite_pose() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]])
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    with pytest.raises(ScanRejected, match="finite"):
        reference_displacements(positions, quaternions)


# --- max_translation_m / guard_depth_m ----------------------------------------------------


def test_max_translation_m_is_the_largest_norm() -> None:
    displacements = np.array([[0.001, 0.0, 0.0], [0.0, 0.0, 0.02], [0.0, 0.0, 0.0]])
    assert max_translation_m(displacements) == pytest.approx(0.02)


def test_guard_depth_m_is_the_lower_quantile() -> None:
    samples = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    z = guard_depth_m(samples, quantile=0.0)
    assert z == pytest.approx(1.0)


def test_guard_depth_m_rejects_empty_sample() -> None:
    with pytest.raises(ScanRejected, match="depth"):
        guard_depth_m(np.array([]), quantile=0.05)


# --- motion_bounds_px: hand-worked against 30kg_Set1/Front (real capture) -----------------


def test_motion_bounds_px_matches_real_camera_30kg_front() -> None:
    # camera_matrix.csv: fx=1343.002, cx=724.818, cy=968.286; valid_bounds_px [0,238,1440,1920]
    # -- the real 30kg_Set1/Front camera's numbers, per the S4 calibration notes. Two rows,
    # each isolating one axis (dZ=0 throughout, so r_x/r_y never enter this check): the first
    # pushes m_x to just under 1.5 px (ceil -> 2), the second pushes m_y to just under 0.6 px
    # (ceil -> 1) -- reproducing the measured (m_x, m_y) = (2, 1) px pre-safety-margin.
    k = Intrinsics(fx=1343.002, fy=1343.002, cx=724.818, cy=968.286, frame="portrait")
    valid = CropBounds(x0=0, y0=238, x1=1440, y1=1920)
    displacements = np.array([[0.0015, 0.0, 0.0], [0.0, 0.0006, 0.0]])

    left, top, right, bottom = motion_bounds_px(
        displacements, k, valid, z_guard_m=1.40, safety_px=8, size=(1440, 1920)
    )

    assert (left, right) == (10, 10)  # ceil(~1.44) + 8
    assert (top, bottom) == (9, 9)  # ceil(~0.58) + 8


def test_motion_bounds_px_zero_displacement_is_exactly_safety_px() -> None:
    k = Intrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, frame="portrait")
    valid = CropBounds(x0=0, y0=0, x1=1000, y1=1000)
    displacements = np.zeros((3, 3))

    left, top, right, bottom = motion_bounds_px(
        displacements, k, valid, z_guard_m=1.5, safety_px=8, size=(1000, 1000)
    )

    assert (left, top, right, bottom) == (8, 8, 8, 8)


# --- intersect_crop: real-camera reproduction ---------------------------------------------


def test_intersect_crop_matches_real_camera_30kg_front() -> None:
    valid = CropBounds(x0=0, y0=238, x1=1440, y1=1920)
    bounds, source = intersect_crop(valid, (10, 9, 10, 9), size=(1440, 1920))

    assert (bounds.x0, bounds.y0, bounds.x1, bounds.y1) == (10, 238, 1430, 1911)
    assert source == {"left": "motion", "top": "valid", "right": "motion", "bottom": "motion"}


def test_intersect_crop_rejects_empty_overlap() -> None:
    valid = CropBounds(x0=0, y0=0, x1=100, y1=100)
    with pytest.raises(ScanRejected, match="overlap"):
        intersect_crop(valid, (60, 0, 60, 0), size=(100, 100))  # left+right >= width


# --- shrink_to_even: the encoder-parity fix -------------------------------------------------


def test_shrink_to_even_trims_an_odd_height() -> None:
    bounds = CropBounds(x0=10, y0=238, x1=1430, y1=1911)  # 1420 x 1673
    shrunk = shrink_to_even(bounds)
    assert (shrunk.width, shrunk.height) == (1420, 1672)
    assert (shrunk.x0, shrunk.y0, shrunk.x1) == (10, 238, 1430)  # only y1 moved


def test_shrink_to_even_leaves_already_even_bounds_alone() -> None:
    bounds = CropBounds(x0=18, y0=268, x1=1422, y1=1908)  # 1404 x 1640, both even
    assert shrink_to_even(bounds) == bounds


# --- depth_bounds: real-camera rescale ------------------------------------------------------


def test_depth_bounds_matches_real_camera_30kg_front() -> None:
    crop = CropBounds(x0=10, y0=238, x1=1430, y1=1910)  # post-even-shrink
    bounds = depth_bounds(crop, rgb_size=(1440, 1920), depth_size=(192, 256))
    assert (bounds.x0, bounds.y0, bounds.x1, bounds.y1) == (2, 32, 190, 254)


def test_depth_bounds_rejects_degenerate_result() -> None:
    crop = CropBounds(x0=0, y0=0, x1=2, y1=1920)  # sub-one-depth-pixel wide
    with pytest.raises(ScanRejected, match="degenerate"):
        depth_bounds(crop, rgb_size=(1440, 1920), depth_size=(192, 256))


# --- crop_intrinsics --------------------------------------------------------------------


def test_crop_intrinsics_shifts_the_principal_point_only() -> None:
    k = Intrinsics(fx=1343.002, fy=1343.002, cx=724.818, cy=968.286, frame="portrait")
    bounds = CropBounds(x0=10, y0=238, x1=1430, y1=1910)

    cropped = crop_intrinsics(k, bounds)

    assert cropped.fx == k.fx
    assert cropped.fy == k.fy
    assert cropped.cx == pytest.approx(714.818)
    assert cropped.cy == pytest.approx(730.286)
    assert cropped.frame == "portrait"


# --- crop_fractions ----------------------------------------------------------------------


def test_crop_fractions_matches_real_camera_30kg_front() -> None:
    bounds = CropBounds(x0=10, y0=238, x1=1430, y1=1910)
    frac_w, frac_h = crop_fractions(bounds, size=(1440, 1920))
    assert frac_w == pytest.approx(1.0 - 1420 / 1440)
    assert frac_h == pytest.approx(1.0 - 1672 / 1920)
