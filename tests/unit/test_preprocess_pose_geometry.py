"""Pure maths for S5 Pose's depth-based metric lift.

No I/O, no fixtures beyond synthetic numpy arrays -- checked numerically against known-good
geometry, never merely "it ran" (CLAUDE.md), same discipline as `test_preprocess_retilt_math.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.pose_geometry import (
    backproject_pixel,
    lift_pixel_to_floor_frame,
    lookup_patch_depth_m,
    rgb_pixel_to_depth_pixel,
    to_floor_frame,
)

# --- rgb_pixel_to_depth_pixel ------------------------------------------------------------


def test_rgb_pixel_to_depth_pixel_scales_proportionally() -> None:
    # Same 0.13333 scale factor 4-retilt.md §2 confirms from real capture resolutions.
    row, col = rgb_pixel_to_depth_pixel(960, 720, rgb_size=(1440, 1920), depth_size=(192, 256))
    assert (row, col) == (128, 96)


def test_rgb_pixel_to_depth_pixel_is_identity_at_equal_resolution() -> None:
    assert rgb_pixel_to_depth_pixel(10, 20, rgb_size=(100, 100), depth_size=(100, 100)) == (10, 20)


# --- lookup_patch_depth_m ------------------------------------------------------------------


def test_lookup_patch_depth_m_prefers_confidence_2() -> None:
    depth = np.full((5, 5), 2000, dtype=np.uint16)
    depth[2, 2] = 5000  # a confidence-1 outlier at the exact centre pixel
    confidence = np.full((5, 5), 2, dtype=np.uint8)
    confidence[2, 2] = 1

    result = lookup_patch_depth_m(depth, confidence, row=2, col=2, radius=2)

    assert result == pytest.approx(2.0)  # metres; the conf-1 outlier is excluded


def test_lookup_patch_depth_m_falls_back_to_confidence_1_when_no_confidence_2() -> None:
    depth = np.full((5, 5), 3000, dtype=np.uint16)
    confidence = np.full((5, 5), 1, dtype=np.uint8)

    result = lookup_patch_depth_m(depth, confidence, row=2, col=2, radius=1)

    assert result == pytest.approx(3.0)


def test_lookup_patch_depth_m_never_selects_zero_depth() -> None:
    depth = np.zeros((3, 3), dtype=np.uint16)
    confidence = np.full((3, 3), 2, dtype=np.uint8)

    assert lookup_patch_depth_m(depth, confidence, row=1, col=1, radius=1) is None


def test_lookup_patch_depth_m_falls_back_to_confidence_0_rather_than_giving_up() -> None:
    # Confidence 0 doesn't mean garbage, only that ARKit won't vouch for it -- a real depth
    # reading with no confident (1/2) alternative nearby is still used, not dropped.
    depth = np.full((3, 3), 1500, dtype=np.uint16)
    confidence = np.zeros((3, 3), dtype=np.uint8)

    assert lookup_patch_depth_m(depth, confidence, row=1, col=1, radius=1) == pytest.approx(1.5)


def test_lookup_patch_depth_m_prefers_confidence_1_and_2_over_0_when_both_present() -> None:
    depth = np.full((3, 3), 9000, dtype=np.uint16)  # confidence-0 depth: should be ignored
    depth[1, 1] = 1500  # the one confidence-2 pixel
    confidence = np.zeros((3, 3), dtype=np.uint8)
    confidence[1, 1] = 2

    result = lookup_patch_depth_m(depth, confidence, row=1, col=1, radius=1)

    assert result == pytest.approx(1.5)


def test_lookup_patch_depth_m_clamps_to_array_bounds() -> None:
    depth = np.full((4, 4), 1200, dtype=np.uint16)
    confidence = np.full((4, 4), 2, dtype=np.uint8)

    # A corner pixel's box would run off the array on two sides -- must not raise/wrap.
    result = lookup_patch_depth_m(depth, confidence, row=0, col=0, radius=2)

    assert result == pytest.approx(1.2)


# --- backproject_pixel -----------------------------------------------------------------


def test_backproject_pixel_recovers_a_known_point() -> None:
    k_d = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, frame="portrait")

    # A pixel 10 px right and 20 px down from the principal point, 2 m away.
    x, y, z = backproject_pixel(row=70, col=60, depth_m=2.0, k_d=k_d)

    assert (x, y, z) == pytest.approx((0.2, 0.4, 2.0))


def test_backproject_pixel_at_principal_point_has_zero_xy() -> None:
    k_d = Intrinsics(fx=200.0, fy=200.0, cx=50.0, cy=50.0, frame="portrait")

    x, y, z = backproject_pixel(row=50, col=50, depth_m=1.5, k_d=k_d)

    assert (x, y) == pytest.approx((0.0, 0.0))
    assert z == pytest.approx(1.5)


# --- to_floor_frame ----------------------------------------------------------------------


def test_to_floor_frame_a_point_on_the_floor_has_zero_height() -> None:
    # A point directly below the camera, at exactly the fitted floor's own depth reading,
    # sits at Y_cam == floor_offset_m -- its floor-frame height must be exactly zero.
    x, y, z = to_floor_frame((0.3, 1.8, 2.5), floor_offset_m=1.8)
    assert (x, y, z) == pytest.approx((0.3, 0.0, 2.5))


def test_to_floor_frame_a_point_above_camera_level_is_positive_height() -> None:
    # Smaller Y_cam (higher in the image, CV Y-down convention) than the floor offset ->
    # positive floor-frame height, e.g. a lifter's head well above the floor.
    _, y, _ = to_floor_frame((0.0, 0.2, 2.0), floor_offset_m=1.8)
    assert y == pytest.approx(1.6)


def test_to_floor_frame_x_and_z_pass_through_unchanged() -> None:
    x, _, z = to_floor_frame((0.42, 1.0, 3.14), floor_offset_m=2.0)
    assert (x, z) == pytest.approx((0.42, 3.14))


# --- lift_pixel_to_floor_frame (end-to-end composition) -----------------------------------


def test_lift_pixel_to_floor_frame_end_to_end() -> None:
    depth = np.full((100, 100), 2000, dtype=np.uint16)  # 2 m everywhere
    confidence = np.full((100, 100), 2, dtype=np.uint8)
    k_d = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, frame="portrait")

    # Equal-resolution RGB/depth so the pixel remap is the identity, isolating the
    # back-projection + floor-frame composition being tested here.
    result = lift_pixel_to_floor_frame(
        row=50,
        col=50,
        depth=depth,
        confidence=confidence,
        rgb_size=(100, 100),
        depth_size=(100, 100),
        k_d=k_d,
        floor_offset_m=2.0,
    )

    assert result is not None
    # Pixel is the principal point, so camera-frame (x, y) = (0, 0) -- exactly camera height,
    # which is `floor_offset_m` above the floor.
    assert result == pytest.approx((0.0, 2.0, 2.0))


def test_lift_pixel_to_floor_frame_returns_none_when_depth_is_unusable() -> None:
    depth = np.zeros((10, 10), dtype=np.uint16)
    confidence = np.zeros((10, 10), dtype=np.uint8)
    k_d = Intrinsics(fx=50.0, fy=50.0, cx=5.0, cy=5.0, frame="portrait")

    result = lift_pixel_to_floor_frame(
        row=5,
        col=5,
        depth=depth,
        confidence=confidence,
        rgb_size=(10, 10),
        depth_size=(10, 10),
        k_d=k_d,
        floor_offset_m=1.5,
    )

    assert result is None
