"""Pure maths for S3 Retilt: region parsing, back-projection, plane fit, rectification.

No I/O, no Prefect, no fixtures beyond synthetic numpy arrays -- this module is checked
numerically against known-good geometry, never merely "it ran" (CLAUDE.md).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.retilt import (
    backproject,
    depth_intrinsics,
    depth_scale_map,
    fit_plane,
    gravity_agreement_deg,
    gravity_down_camera,
    homography,
    parse_region_point,
    read_floor_region,
    rectifying_rotation,
    select_floor_pixels,
    tilt_roll_from_normal,
    translation_span_m,
    valid_bounds,
)

# --- test helpers, defined against this module's own verified geometry ----------------


def _unit_normal(a: float, b: float) -> np.ndarray:
    """The unit normal of the plane `Y = a*X + b*Z + c`, oriented with `n_y < 0`."""

    n = np.array([a, -1.0, b])
    return n / np.linalg.norm(n)


def _normal_from_tilt_roll(tilt_deg: float, roll_deg: float) -> np.ndarray:
    """Inverse of `rectifying_rotation`: the normal that rectifies to exactly this tilt/roll."""

    r = rectifying_rotation(tilt_deg, roll_deg)
    return r.T @ np.array([0.0, -1.0, 0.0])


def _synthetic_plane_points(
    a: float, b: float, c: float, n: int, noise_m: float, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(0.5, 3.0, n)
    y = a * x + b * z + c + rng.normal(0.0, noise_m, n)
    return np.stack([x, y, z], axis=-1)


# --- parse_region_point / read_floor_region --------------------------------------------


def test_parse_region_point_real_capture_value() -> None:
    assert parse_region_point("(0.40268, 0.64503)") == pytest.approx((0.40268, 0.64503))


def test_parse_region_point_rejects_unparseable() -> None:
    with pytest.raises(ValueError):
        parse_region_point("0.4, 0.6")


def test_read_floor_region_picks_the_role_prefix() -> None:
    metadata = {
        "video": {
            "front_floor_region_bottom_left_in_pixels": "(0, 1)",
            "front_floor_region_top_right_in_pixels": "(1, 0.75093)",
            "side_floor_region_bottom_left_in_pixels": "(0, 1)",
            "side_floor_region_top_right_in_pixels": "(0.40268, 0.64503)",
        }
    }
    region = read_floor_region(metadata, "side")
    assert (region.x0, region.y0, region.x1, region.y1) == pytest.approx(
        (0.0, 1.0, 0.40268, 0.64503)
    )


def test_read_floor_region_rejects_missing_field() -> None:
    with pytest.raises(ScanRejected):
        read_floor_region({"video": {}}, "front")


def test_read_floor_region_rejects_unparseable_field() -> None:
    metadata = {
        "video": {
            "front_floor_region_bottom_left_in_pixels": "not-a-point",
            "front_floor_region_top_right_in_pixels": "(1, 0.75093)",
        }
    }
    with pytest.raises(ScanRejected):
        read_floor_region(metadata, "front")


def test_read_floor_region_rejects_degenerate_region() -> None:
    metadata = {
        "video": {
            "front_floor_region_bottom_left_in_pixels": "(0.5, 0.2)",
            "front_floor_region_top_right_in_pixels": "(0.5, 0.1)",
        }
    }
    with pytest.raises(ScanRejected):
        read_floor_region(metadata, "front")


def test_read_floor_region_reads_unprefixed_keys_for_a_single_camera() -> None:
    """One camera governed by its own metadata has nothing to tell apart, so no prefix."""

    metadata = {
        "video": {
            "floor_region_bottom_left_in_pixels": "(0.20, 0.78)",
            "floor_region_top_right_in_pixels": "(0.55, 0.68)",
        }
    }

    region = read_floor_region(metadata, "single")

    assert (region.x0, region.y0, region.x1, region.y1) == pytest.approx((0.20, 0.78, 0.55, 0.68))


def test_read_floor_region_accepts_the_side_alias_for_a_single_camera() -> None:
    """Its lift window is already spelled `..._side_in_ms`; either spelling should work."""

    metadata = {
        "video": {
            "side_floor_region_bottom_left_in_pixels": "(0.20, 0.78)",
            "side_floor_region_top_right_in_pixels": "(0.55, 0.68)",
        }
    }

    region = read_floor_region(metadata, "single")

    assert (region.x0, region.x1) == pytest.approx((0.20, 0.55))


def test_read_floor_region_does_not_fall_back_to_a_front_prefix_for_a_single_camera() -> None:
    """The real August files pair a `front_` bottom-left with an unprefixed top-right.

    That is a half-corrected annotation, not an alternate spelling. Reading it would accept
    a rectangle nobody finished checking; the rejection names the unprefixed field instead.
    """

    metadata = {
        "video": {
            "front_floor_region_bottom_left_in_pixels": "(0.65204, 0.66118)",
            "floor_region_top_right_in_pixels": "(0.79080, 0.86117)",
        }
    }

    with pytest.raises(ScanRejected, match="floor_region_bottom_left_in_pixels"):
        read_floor_region(metadata, "single")


def test_read_floor_region_rejects_an_inverted_y_region_rather_than_swapping() -> None:
    """`y` runs top->bottom, so bottom-left's y must be the larger one.

    Every real August region is written the other way round. Silently swapping would make
    the pipeline accept an annotation whose author had the convention backwards, which is
    exactly the state their rectangle turned out to be in.
    """

    metadata = {
        "video": {
            "floor_region_bottom_left_in_pixels": "(0.65204, 0.66118)",
            "floor_region_top_right_in_pixels": "(0.79080, 0.86117)",
        }
    }

    with pytest.raises(ScanRejected, match="degenerate on y"):
        read_floor_region(metadata, "single")


def test_read_floor_region_rejects_when_the_video_block_is_absent() -> None:
    with pytest.raises(ScanRejected, match="no video block"):
        read_floor_region({}, "single")


# --- depth_intrinsics --------------------------------------------------------------------


def test_depth_intrinsics_pixel_centre_scaling() -> None:
    k = Intrinsics(fx=1343.0, fy=1343.0, cx=724.8, cy=968.3, frame="portrait")
    k_d = depth_intrinsics(k, rgb_size=(1440, 1920), depth_size=(192, 256))
    s = 192 / 1440
    assert k_d.cx == pytest.approx((724.8 + 0.5) * s - 0.5)
    assert k_d.cy == pytest.approx((968.3 + 0.5) * s - 0.5)
    assert k_d.fx == pytest.approx(1343.0 * s)
    assert k_d.fy == pytest.approx(1343.0 * s)


# --- select_floor_pixels ------------------------------------------------------------------


def test_select_floor_pixels_conf2_dominant_uses_conf2_only() -> None:
    depth = np.full((20, 20), 1500, dtype=np.uint16)
    confidence = np.zeros((20, 20), dtype=np.uint8)
    confidence[2:18, 2:18] = 2  # a large connected conf-2 block: 16x16 = 256 of 400 pixels
    bounds = (0, 0, 20, 20)

    rows, cols, mode = select_floor_pixels(depth, confidence, bounds)

    assert mode == "conf2_only"
    assert len(rows) == 256
    assert set(confidence[rows, cols].tolist()) == {2}


def test_select_floor_pixels_conf2_sparse_falls_back_to_conf1_and_2() -> None:
    depth = np.full((20, 20), 1500, dtype=np.uint16)
    confidence = np.zeros((20, 20), dtype=np.uint8)
    confidence[0:20, 0:10] = 1  # half the region is confidence-1
    confidence[2:4, 2:4] = 2  # a tiny conf-2 patch: 4 of 400 pixels, well under 1/6
    bounds = (0, 0, 20, 20)

    rows, cols, mode = select_floor_pixels(depth, confidence, bounds)

    assert mode == "conf1_and_2"
    assert set(confidence[rows, cols].tolist()) <= {1, 2}
    assert 0 not in confidence[rows, cols].tolist()


def test_select_floor_pixels_conf1_and_2_reads_the_whole_region_not_the_largest_component() -> None:
    """§1: the conf-1-and-2 fallback reads from the annotated region, not the conf-2 component.

    Conf-1 pixels sit outside the (tiny) largest conf-2 component but inside the region; they
    must still be included once the conf2-only threshold is not met.
    """

    depth = np.full((20, 20), 1500, dtype=np.uint16)
    confidence = np.zeros((20, 20), dtype=np.uint8)
    confidence[0, 0] = 2  # a single-pixel conf-2 component: far under the 1/6 threshold
    confidence[10:15, 10:15] = 1  # conf-1 pixels elsewhere in the region, outside that component
    bounds = (0, 0, 20, 20)

    rows, cols, mode = select_floor_pixels(depth, confidence, bounds)

    assert mode == "conf1_and_2"
    selected = set(zip(rows.tolist(), cols.tolist(), strict=True))
    assert (0, 0) in selected
    assert any(10 <= r < 15 and 10 <= c < 15 for r, c in selected)


def test_select_floor_pixels_never_selects_zero_depth() -> None:
    depth = np.full((20, 20), 1500, dtype=np.uint16)
    confidence = np.full((20, 20), 2, dtype=np.uint8)
    depth[5, 5] = 0
    bounds = (0, 0, 20, 20)

    rows, cols, _ = select_floor_pixels(depth, confidence, bounds)

    assert not any((r, c) == (5, 5) for r, c in zip(rows.tolist(), cols.tolist(), strict=True))


# --- backproject --------------------------------------------------------------------------


def test_backproject_recovers_a_known_point() -> None:
    k_d = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, frame="portrait")
    rows = np.array([40])  # v == cy -> Y == 0
    cols = np.array([150])  # u - cx == 100 -> X == Z at fx == 100
    depth_mm = np.array([2000])  # Z == 2.0 m

    points = backproject(rows, cols, depth_mm, k_d)

    assert points.shape == (1, 3)
    assert points[0] == pytest.approx([2.0, 0.0, 2.0])


# --- fit_plane --------------------------------------------------------------------------


def test_fit_plane_recovers_known_normal() -> None:
    points = _synthetic_plane_points(a=0.05, b=0.02, c=1.5, n=2000, noise_m=0.001)

    fit = fit_plane(points)

    expected = _unit_normal(0.05, 0.02)
    assert np.dot(fit.normal, expected) > 0.9999
    assert fit.rms_residual_m < 0.01
    assert fit.n_points == 2000


def test_fit_plane_orients_normal_toward_camera() -> None:
    points = _synthetic_plane_points(a=-0.03, b=0.04, c=1.2, n=1000, noise_m=0.001)

    fit = fit_plane(points)

    assert fit.normal[1] < 0


def test_fit_plane_recovers_floor_offset() -> None:
    # A point on the plane's own Y-axis intercept (X=Z=0) sits at exactly `c` metres --
    # `floor_offset_m` is the perpendicular (not axis-aligned) distance, so it's `c`
    # divided by the plane normal's un-normalized length, per the module's own formula.
    a, b, c = 0.05, 0.02, 1.5
    points = _synthetic_plane_points(a=a, b=b, c=c, n=4000, noise_m=0.0005)

    fit = fit_plane(points)

    expected = c / math.sqrt(a**2 + 1 + b**2)
    assert fit.floor_offset_m == pytest.approx(expected, rel=1e-2)


def test_fit_plane_floor_offset_is_exactly_c_for_a_level_floor() -> None:
    # No tilt/roll (a = b = 0): the perpendicular distance collapses to `c` itself.
    points = _synthetic_plane_points(a=0.0, b=0.0, c=2.0, n=2000, noise_m=0.0005)

    fit = fit_plane(points)

    assert fit.floor_offset_m == pytest.approx(2.0, rel=1e-2)


# --- tilt_roll_from_normal / rectifying_rotation ------------------------------------------


def test_tilt_roll_recovers_baked_in_angles() -> None:
    n = _normal_from_tilt_roll(tilt_deg=7.0, roll_deg=-3.0)

    tilt, roll = tilt_roll_from_normal(n)

    assert tilt == pytest.approx(7.0, abs=1e-3)
    assert roll == pytest.approx(-3.0, abs=1e-3)


def test_tilt_roll_is_zero_for_a_level_floor() -> None:
    n = np.array([0.0, -1.0, 0.0])
    tilt, roll = tilt_roll_from_normal(n)
    assert tilt == pytest.approx(0.0, abs=1e-9)
    assert roll == pytest.approx(0.0, abs=1e-9)


def test_rectifying_rotation_levels_the_normal() -> None:
    n = _normal_from_tilt_roll(tilt_deg=12.0, roll_deg=4.0)

    r = rectifying_rotation(*tilt_roll_from_normal(n))

    assert r @ n == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)


def test_rectifying_rotation_is_identity_at_zero_angles() -> None:
    r = rectifying_rotation(0.0, 0.0)
    assert r == pytest.approx(np.eye(3))


# --- homography ---------------------------------------------------------------------------


def test_homography_identity_at_zero_rotation() -> None:
    k = Intrinsics(fx=100, fy=100, cx=50, cy=50, frame="portrait")
    assert homography(k, np.eye(3)) == pytest.approx(np.eye(3))


# --- valid_bounds ---------------------------------------------------------------------------


def test_valid_bounds_shrinks_under_a_roll() -> None:
    """A tilt (about X) keystones the image and shrinks its valid extent; a pure roll
    (about Z, the optical axis) is an in-plane rotation and does not -- its diamond-shaped
    overlap with the original frame still touches all four edges. Exercise both."""

    k = Intrinsics(fx=200.0, fy=200.0, cx=100.0, cy=100.0, frame="portrait")
    size = (200, 200)

    h_identity = homography(k, np.eye(3))
    h_rolled = homography(k, rectifying_rotation(0.0, 15.0))
    h_tilted = homography(k, rectifying_rotation(15.0, 0.0))

    assert valid_bounds(h_rolled, size).width == valid_bounds(h_identity, size).width
    assert valid_bounds(h_tilted, size).height < valid_bounds(h_identity, size).height


# --- gravity_down_camera / gravity_agreement_deg ------------------------------------------


def test_gravity_down_camera_identity_quaternion() -> None:
    g = gravity_down_camera(0.0, 0.0, 0.0, 1.0)

    # Hand-worked from the four-line chain at identity (R = I):
    # g_world = (0,-1,0) -> g_arkit = (0,-1,0) -> g_cv = diag(1,-1,-1) @ g_arkit = (0,1,0)
    # -> g_portrait = R_z(90 deg) @ (0,1,0) = (-1,0,0).
    assert g == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)


def test_gravity_agreement_deg_is_zero_when_normal_matches_gravity() -> None:
    normal = np.array([0.0, -1.0, 0.0])
    g_camera = np.array([0.0, 1.0, 0.0])  # "down" opposite the floor normal
    assert gravity_agreement_deg(normal, g_camera) == pytest.approx(0.0, abs=1e-9)


def test_gravity_agreement_deg_is_ninety_when_orthogonal() -> None:
    normal = np.array([0.0, -1.0, 0.0])
    g_camera = np.array([1.0, 0.0, 0.0])
    assert gravity_agreement_deg(normal, g_camera) == pytest.approx(90.0, abs=1e-9)


# --- depth_scale_map -------------------------------------------------------------------------


def test_depth_scale_map_matches_hand_worked_point() -> None:
    k_d = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, frame="portrait")
    r = rectifying_rotation(10.0, 5.0)
    size = (100, 80)

    scale_map = depth_scale_map(k_d, r, size)

    u, v = 30, 60
    k_inv = np.linalg.inv(np.array(k_d.to_matrix()))
    expected = r[2, :] @ (k_inv @ np.array([u, v, 1.0]))
    assert scale_map.shape == (80, 100)
    assert scale_map[v, u] == pytest.approx(expected)


def test_depth_scale_map_is_one_at_zero_rotation() -> None:
    k_d = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, frame="portrait")
    scale_map = depth_scale_map(k_d, np.eye(3), (100, 80))
    assert scale_map == pytest.approx(np.ones((80, 100)))


# --- translation_span_m -----------------------------------------------------------------------


def test_translation_span_m_max_pairwise_distance() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, 1.0, 0.0]])
    assert translation_span_m(xyz) == pytest.approx(5.0)  # (0,0,0) to (3,4,0), a 3-4-5 triangle


def test_translation_span_m_zero_for_a_single_frame() -> None:
    xyz = np.array([[1.0, 2.0, 3.0]])
    assert translation_span_m(xyz) == pytest.approx(0.0)


def test_rotation_matrices_are_self_consistent_across_random_normals() -> None:
    """The published spec text (`R = R_x(tilt) @ R_z(roll)`) does not level the normal;
    numeric verification (CLAUDE.md) shows `R_z(-roll)` is required -- see the corrected
    §4 formula. This test guards that correction across many random floor orientations.
    """

    rng = np.random.default_rng(1)
    for _ in range(200):
        n = rng.normal(size=3)
        n[1] = -abs(n[1]) - 0.05
        n = n / np.linalg.norm(n)
        if abs(math.degrees(math.acos(np.clip(-n[1], -1, 1)))) >= 44:
            continue  # stay within the plausible tilt/roll range this stage assumes
        tilt, roll = tilt_roll_from_normal(n)
        r = rectifying_rotation(tilt, roll)
        assert r @ n == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)
