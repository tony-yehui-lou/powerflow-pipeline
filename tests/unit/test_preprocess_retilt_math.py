"""Pure maths for S3 Retilt: floor detection, back-projection, plane fit, rectification.

No I/O, no Prefect, no fixtures beyond synthetic numpy arrays -- this module is checked
numerically against known-good geometry, never merely "it ran" (CLAUDE.md).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from powerflow_pipeline.data.common.capture_models import Intrinsics
from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.preprocess.retilt import (
    backproject,
    depth_intrinsics,
    depth_scale_map,
    fit_plane,
    gravity_agreement_deg,
    gravity_down_camera,
    gravity_up_camera,
    height_peaks,
    homography,
    rectifying_rotation,
    refine_floor,
    select_confident_pixels,
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


# --- depth_intrinsics --------------------------------------------------------------------


def test_depth_intrinsics_pixel_centre_scaling() -> None:
    k = Intrinsics(fx=1343.0, fy=1343.0, cx=724.8, cy=968.3, frame="portrait")
    k_d = depth_intrinsics(k, rgb_size=(1440, 1920), depth_size=(192, 256))
    s = 192 / 1440
    assert k_d.cx == pytest.approx((724.8 + 0.5) * s - 0.5)
    assert k_d.cy == pytest.approx((968.3 + 0.5) * s - 0.5)
    assert k_d.fx == pytest.approx(1343.0 * s)
    assert k_d.fy == pytest.approx(1343.0 * s)


# --- select_confident_pixels --------------------------------------------------------------


def test_select_confident_pixels_conf2_reads_the_whole_frame() -> None:
    depth = np.full((20, 20), 1500, dtype=np.uint16)
    confidence = np.zeros((20, 20), dtype=np.uint8)
    confidence[0:5, :] = 1
    confidence[10:20, :] = 2

    rows, cols = select_confident_pixels(depth, confidence, min_confidence=2)

    assert len(rows) == 200
    assert set(confidence[rows, cols].tolist()) == {2}


def test_select_confident_pixels_conf1_admits_both_levels_never_zero() -> None:
    depth = np.full((20, 20), 1500, dtype=np.uint16)
    confidence = np.zeros((20, 20), dtype=np.uint8)
    confidence[0:5, :] = 1
    confidence[10:20, :] = 2

    rows, cols = select_confident_pixels(depth, confidence, min_confidence=1)

    assert len(rows) == 300
    assert set(confidence[rows, cols].tolist()) == {1, 2}


def test_select_confident_pixels_never_selects_zero_depth() -> None:
    depth = np.full((20, 20), 1500, dtype=np.uint16)
    confidence = np.full((20, 20), 2, dtype=np.uint8)
    depth[5, 5] = 0

    rows, cols = select_confident_pixels(depth, confidence, min_confidence=2)

    assert len(rows) == 399
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

    # Hand-worked from the chain at identity (R = I): g_world = (0,-1,0) -> camera axes
    # (already CV: X right, Y down, Z forward -- no ARKit flip) = (0,-1,0)
    # -> g_portrait = R_z(90 deg) @ (0,-1,0) = (1,0,0).
    assert g == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


# Real first-frame odometry quaternions and the floor normals fitted on the same captures
# from an operator-annotated rectangle (`../data/s2_orient_output`, `s3_retilt_output`,
# 2026-10-20). Two independent sources of "up": with the ARKit->CV flip the old chain
# applied, they disagreed by 10.5 and 15.1 deg; without it, by 1.1 and 1.5 deg.
_REAL_GRAVITY_CASES = [
    pytest.param(
        (0.7096546, -0.702781, 0.034170967, 0.03635269),
        (0.012776706015365609, -0.9966208191150993, -0.08113999439083903),
        id="10 September/30kg2",
    ),
    pytest.param(
        (0.694247, -0.7172322, 0.02781919, 0.0531521),
        (-0.03762359449513068, -0.9895225608297499, -0.1393899807956081),
        id="11 July/30kg_Set1/Front",
    ),
]


@pytest.mark.parametrize(("quat", "fitted_normal"), _REAL_GRAVITY_CASES)
def test_gravity_up_camera_agrees_with_real_fitted_floors(
    quat: tuple[float, float, float, float], fitted_normal: tuple[float, float, float]
) -> None:
    up = gravity_up_camera(np.array([quat]))

    angle = math.degrees(math.acos(float(np.clip(up @ np.array(fitted_normal), -1.0, 1.0))))
    assert angle < 2.0


def test_gravity_up_camera_is_unit_and_opposes_gravity() -> None:
    quat = (0.7096546, -0.702781, 0.034170967, 0.03635269)

    up = gravity_up_camera(np.array([quat]))

    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert up == pytest.approx(-gravity_down_camera(*quat))


def test_gravity_up_camera_averages_sampled_rows() -> None:
    from scipy.spatial.transform import Rotation

    base = Rotation.from_quat([0.7096546, -0.702781, 0.034170967, 0.03635269])
    wobble = Rotation.from_euler("x", 1.0, degrees=True)
    rows = np.array([(base * wobble).as_quat(), (base * wobble.inv()).as_quat()])

    up = gravity_up_camera(rows)

    assert up == pytest.approx(gravity_up_camera(np.array([base.as_quat()])), abs=1e-4)


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


# --- height_peaks -------------------------------------------------------------------------


def _cluster(height: float, n: int, spread: float = 0.002, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(height - spread, height + spread, n)


def test_height_peaks_finds_a_single_floor() -> None:
    heights = np.concatenate([_cluster(-1.40, 800), np.linspace(-1.3, 1.0, 200)])

    peaks = height_peaks(heights, bin_m=0.01, min_fraction=0.15)

    assert len(peaks) == 1
    assert peaks[0][0] == pytest.approx(-1.40, abs=0.01)
    assert peaks[0][1] >= 0.8


def test_height_peaks_are_lowest_first_even_when_a_higher_one_is_larger() -> None:
    """A table filling more of the frame than the floor is still not the floor."""

    heights = np.concatenate([_cluster(-1.40, 300), _cluster(-0.70, 500, seed=1)])

    peaks = height_peaks(heights, bin_m=0.01, min_fraction=0.15)

    assert [h for h, _ in peaks] == pytest.approx([-1.40, -0.70], abs=0.01)
    assert peaks[0][1] < peaks[1][1]


def test_height_peaks_ignore_a_sparse_cluster_below_the_floor() -> None:
    """Specular multipath off a shiny floor puts a few points below it; never 15%."""

    heights = np.concatenate([_cluster(-1.60, 100), _cluster(-1.40, 900, seed=1)])

    peaks = height_peaks(heights, bin_m=0.01, min_fraction=0.15)

    assert len(peaks) == 1
    assert peaks[0][0] == pytest.approx(-1.40, abs=0.01)


def test_height_peaks_find_nothing_in_evenly_spread_heights() -> None:
    """A wall seen square-on spreads over its whole height: no horizontal surface."""

    heights = np.linspace(-1.5, 1.5, 3000)

    assert height_peaks(heights, bin_m=0.01, min_fraction=0.15) == []


def test_height_peaks_handle_a_perfectly_flat_cloud() -> None:
    """Noise-free synthetic floors put every point in one bin."""

    peaks = height_peaks(np.full(500, -1.5), bin_m=0.01, min_fraction=0.15)

    assert len(peaks) == 1
    assert peaks[0][0] == pytest.approx(-1.5, abs=0.01)
    assert peaks[0][1] == pytest.approx(1.0)


def test_height_peaks_reject_an_empty_cloud() -> None:
    assert height_peaks(np.empty(0), bin_m=0.01, min_fraction=0.15) == []


# --- refine_floor ---------------------------------------------------------------------------


def _floor_cloud(normal: np.ndarray, offset_m: float, n: int, seed: int = 0) -> np.ndarray:
    """Points on the plane `p . normal = -offset_m`, spread like a real floor in view."""

    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, n)
    z = rng.uniform(0.8, 5.0, n)
    # Solve normal . (x, y, z) = -offset for y.
    y = (-offset_m - normal[0] * x - normal[2] * z) / normal[1]
    return np.stack([x, y, z], axis=-1)


def _tilt_about_x(v: np.ndarray, degrees: float) -> np.ndarray:
    t = math.radians(degrees)
    r = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    out: np.ndarray = r @ v
    return out


@pytest.mark.parametrize("gravity_error_deg", [0.0, 2.0, 4.0])
def test_refine_floor_recovers_the_plane_from_a_biased_gravity_seed(
    gravity_error_deg: float,
) -> None:
    """The property the design rests on: gravity only has to be roughly right."""

    normal = _normal_from_tilt_roll(8.0, -2.0)
    offset = 1.35
    points = _floor_cloud(normal, offset, 20_000)
    up = _tilt_about_x(normal, gravity_error_deg)
    h0 = float(np.median(points @ up))

    plane, iterations = refine_floor(
        points, up, h0, seed_band_m=0.03, inlier_band_m=0.02, max_iterations=3
    )

    angle = math.degrees(math.acos(min(1.0, float(np.array(plane.normal) @ normal))))
    assert angle < 0.01
    assert plane.floor_offset_m == pytest.approx(offset, abs=1e-3)
    assert plane.n_points == 20_000
    assert 1 <= iterations <= 3


def test_refine_floor_ignores_a_lifter_and_a_wall() -> None:
    normal = _normal_from_tilt_roll(8.0, -2.0)
    offset = 1.35
    floor = _floor_cloud(normal, offset, 20_000)
    rng = np.random.default_rng(3)
    lifter = np.stack(  # a body-sized box standing on the floor, 3 m out
        [rng.uniform(-0.3, 0.3, 4000), rng.uniform(-0.5, 1.3, 4000), rng.uniform(2.8, 3.2, 4000)],
        axis=-1,
    )
    wall = np.stack(  # a back wall, square to the camera
        [rng.uniform(-2, 2, 6000), rng.uniform(-2, 1.4, 6000), np.full(6000, 6.0)], axis=-1
    )
    points = np.concatenate([floor, lifter, wall])
    up = _tilt_about_x(normal, 3.0)
    peaks = height_peaks(points @ up, bin_m=0.01, min_fraction=0.15)

    plane, _ = refine_floor(
        points, up, peaks[0][0], seed_band_m=0.03, inlier_band_m=0.02, max_iterations=3
    )

    angle = math.degrees(math.acos(min(1.0, float(np.array(plane.normal) @ normal))))
    assert angle < 0.1
    assert plane.floor_offset_m == pytest.approx(offset, abs=0.005)


def test_refine_floor_stops_early_once_the_inliers_settle() -> None:
    normal = _normal_from_tilt_roll(0.0, 0.0)
    points = _floor_cloud(normal, 1.5, 2000)

    _, iterations = refine_floor(
        points, normal, -1.5, seed_band_m=0.03, inlier_band_m=0.02, max_iterations=10
    )

    assert iterations < 10


def test_refine_floor_rejects_a_seed_with_no_points_near_it() -> None:
    normal = _normal_from_tilt_roll(0.0, 0.0)
    points = _floor_cloud(normal, 1.5, 2000)

    with pytest.raises(ScanRejected, match="floor"):
        refine_floor(points, normal, 3.0, seed_band_m=0.03, inlier_band_m=0.02, max_iterations=3)


def test_refine_floor_rejects_a_fit_that_keeps_none_of_its_points() -> None:
    """A checkerboard of heights 2.5 cm either side of flat: inside the 3 cm seed band, but
    the least-squares plane through it is flat and every point sits outside 2 cm of it."""

    y = 0.025
    points = np.array([[-1, y, 2], [1, -y, 2], [-1, -y, 4], [1, y, 4]], dtype=np.float64)

    with pytest.raises(ScanRejected, match="lost its inliers"):
        refine_floor(
            points,
            np.array([0.0, -1.0, 0.0]),
            0.0,
            seed_band_m=0.03,
            inlier_band_m=0.02,
            max_iterations=3,
        )
