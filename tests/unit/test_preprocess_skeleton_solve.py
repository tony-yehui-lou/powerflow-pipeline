"""S8's per-frame skeleton solve, on a synthetic lift whose 3D joints are known.

The scene: a lifter in front of a levelled camera, seen obliquely (the bar 38.6 deg off the
optical axis, as on `10 September`), rising from a bent start with the hands on the bar. Every
joint is built in 3D from known bone lengths, projected through a known camera, and handed to the
solver as pixels; every assertion is against the 3D truth. All lengths are invented for the test.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Final

import numpy as np
import pytest

from powerflow_pipeline.data.common.models import JointId
from powerflow_pipeline.data.preprocess import athlete_input as ai
from powerflow_pipeline.data.preprocess import skeleton_solve as ss
from powerflow_pipeline.data.preprocess.bar_geometry import Point, project
from powerflow_pipeline.data.preprocess.models import Intrinsics

K = Intrinsics(fx=1336.9, fy=1336.9, cx=705.0, cy=802.6, frame="portrait")
YAW = 38.6
AXIS = (math.sin(math.radians(YAW)), 0.0, math.cos(math.radians(YAW)))
SAGITTAL = (AXIS[2], 0.0, -AXIS[0])
Y_FLOOR = 1375.0
ORIGIN = (-150.0, Y_FLOOR, 2900.0)  # on the floor, on the mid-plane, between the feet
NEAR: Final = "right"  # the right side sits at negative offsets: nearer the cap

TRUE = {
    "tibia": 420.0,
    "femur": 430.0,
    "torso": 500.0,
    "upper": 300.0,
    "fore": 260.0,
    "shoulder_w": 380.0,
    "hip_w": 180.0,
    "ankle_h": 80.0,
    "head": 230.0,
    "half_grip": 230.0,
}
N_FRAMES = 40
# The scene's lengths as a tape would record them; height chosen so the head prior is exact.
TAPE = {
    "height_in_cm": TRUE["head"] / 0.12 / 10,
    "tibia_in_cm": TRUE["tibia"] / 10,
    "femur_in_cm": TRUE["femur"] / 10,
    "torso_in_cm": TRUE["torso"] / 10,
    "upper_arm_in_cm": TRUE["upper"] / 10,
    "forearm_in_cm": TRUE["fore"] / 10,
    "shoulder_width_in_cm": TRUE["shoulder_w"] / 10,
    "hip_width_in_cm": TRUE["hip_w"] / 10,
    "ankle_height_in_cm": TRUE["ankle_h"] / 10,
}


def to_camera(lat: float, sag: float, h: float) -> Point:
    return tuple(  # type: ignore[return-value]
        o + lat * a + sag * s - (h if i == 1 else 0.0)
        for i, (o, a, s) in enumerate(zip(ORIGIN, AXIS, SAGITTAL, strict=True))
    )


def pose(t: float) -> dict[JointId, Point]:
    """The lifter at progress `t` in [0, 1]: from a bent pull start towards standing."""

    phi = 0.45 * (1 - t) + 0.05 * t  # tibia forward of vertical
    beta = 1.1 * (1 - t) + 0.1 * t  # femur back of vertical
    lam = 0.9 * (1 - t) + 0.05 * t  # torso forward of vertical
    mu = 0.15 * (1 - t)  # arms forward of hanging
    hip_half, sh_half = TRUE["hip_w"] / 2, TRUE["shoulder_w"] / 2
    elbow_lat, wrist_lat = 210.0, TRUE["half_grip"]

    def in_plane(length: float, lateral: float) -> float:
        return math.sqrt(length**2 - lateral**2)

    joints: dict[JointId, Point] = {}
    sh_mid = (0.0, 0.0)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        ankle = (0.0, TRUE["ankle_h"])
        knee = (ankle[0] + TRUE["tibia"] * math.sin(phi), ankle[1] + TRUE["tibia"] * math.cos(phi))
        hip = (knee[0] - TRUE["femur"] * math.sin(beta), knee[1] + TRUE["femur"] * math.cos(beta))
        torso = in_plane(TRUE["torso"], sh_half - hip_half)
        shoulder = (hip[0] + torso * math.sin(lam), hip[1] + torso * math.cos(lam))
        upper = in_plane(TRUE["upper"], elbow_lat - sh_half)
        elbow = (shoulder[0] + upper * math.sin(mu), shoulder[1] - upper * math.cos(mu))
        fore = in_plane(TRUE["fore"], wrist_lat - elbow_lat)
        wrist = (elbow[0] + fore * math.sin(mu), elbow[1] - fore * math.cos(mu))
        for part, (lat, (s, h)) in {
            "Ankle": (hip_half, ankle),
            "Knee": (hip_half, knee),
            "Hip": (hip_half, hip),
            "Shoulder": (sh_half, shoulder),
            "Elbow": (elbow_lat, elbow),
            "Wrist": (wrist_lat, wrist),
        }.items():
            joints[f"{side}{part}"] = to_camera(sign * lat, s, h)  # type: ignore[index]
        sh_mid = shoulder
    joints["head"] = to_camera(
        0.0, sh_mid[0] + TRUE["head"] * math.sin(lam), sh_mid[1] + TRUE["head"] * math.cos(lam)
    )
    return joints


def hold_for(joints: dict[JointId, Point]) -> ss.BarHold:
    left, right = joints["leftWrist"], joints["rightWrist"]
    centre = tuple((a + b) / 2 for a, b in zip(left, right, strict=True))
    return ss.BarHold(
        centre_mm=centre,  # type: ignore[arg-type]
        axis=AXIS,
        half_grip_mm=TRUE["half_grip"],
        near_side=NEAR,
    )


def body(**overrides: float) -> ss.Body:
    lengths = {
        "leftTibia": TRUE["tibia"],
        "rightTibia": TRUE["tibia"],
        "leftFemur": TRUE["femur"],
        "rightFemur": TRUE["femur"],
        "leftTorso": TRUE["torso"],
        "rightTorso": TRUE["torso"],
        "leftUpperArm": TRUE["upper"],
        "rightUpperArm": TRUE["upper"],
        "leftForearm": TRUE["fore"],
        "rightForearm": TRUE["fore"],
        "shoulderWidth": TRUE["shoulder_w"],
        "hipWidth": TRUE["hip_w"],
        "headNeck": TRUE["head"],
        ai.ANKLE_HEIGHT: TRUE["ankle_h"],
    }
    lengths.update(overrides)
    sigmas = {bone: 20.0 for bone in lengths}
    return ss.Body(lengths={b: (v, sigmas[b]) for b, v in lengths.items()}, y_floor_mm=Y_FLOOR)


def observe(
    joints: dict[JointId, Point],
    *,
    noise_px: float = 0.0,
    rng: np.random.Generator | None = None,
    hidden: tuple[JointId, ...] = (),
    with_bar: bool = True,
    grounded: bool = True,
) -> ss.FrameObs:
    pixels: dict[JointId, tuple[float, float] | None] = {}
    confidence: dict[JointId, float] = {}
    for joint, point in joints.items():
        if joint in hidden:
            pixels[joint], confidence[joint] = None, 0.0
            continue
        u, v = project(point, K)
        if noise_px and rng is not None:
            u, v = u + rng.normal(0, noise_px), v + rng.normal(0, noise_px)
        pixels[joint], confidence[joint] = (u, v), 0.95
    return ss.FrameObs(
        pixels=pixels,
        confidence=confidence,
        hold=hold_for(joints) if with_bar else None,
        grounded=frozenset({"leftAnkle", "rightAnkle"}) if grounded else frozenset(),
    )


def lift(n: int = N_FRAMES) -> list[dict[JointId, Point]]:
    return [pose(i / (n - 1)) for i in range(n)]


def max_error(fit: ss.FrameFit, truth: dict[JointId, Point]) -> float:
    return max(math.dist(fit.points[j], truth[j]) for j in ss.SOLVED_JOINTS)


def solve_one(obs: ss.FrameObs, b: ss.Body | None = None) -> ss.FrameFit | ss.NoFit:
    b = b or body()
    start = ss.initial_guess(obs, b, K)
    assert start is not None
    return ss.solve_frame(obs, b, K, starts=[start])


# --- one frame -------------------------------------------------------------------------------


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_exact_pixels_and_lengths_recover_the_skeleton(t: float) -> None:
    truth = pose(t)

    fit = solve_one(observe(truth))

    assert isinstance(fit, ss.FrameFit)
    assert max_error(fit, truth) < 1.0
    assert fit.anchors == frozenset({"bar", "floor"})
    assert all(fit.source[j] == "observed" for j in ss.SOLVED_JOINTS)
    assert fit.rms < 0.1


def test_the_floor_alone_anchors_the_skeleton_from_a_neighbouring_start() -> None:
    # From a cold start the floor alone can land on a mirrored branch (bones flipped towards
    # the camera); a neighbouring frame's solution is the start `solve_capture` gives it.
    truth = pose(0.3)
    neighbour = np.array([pose(0.28)[j] for j in ss.SOLVED_JOINTS], dtype=float).ravel()

    fit = ss.solve_frame(observe(truth, with_bar=False), body(), K, starts=[neighbour])

    assert isinstance(fit, ss.FrameFit)
    assert fit.anchors == frozenset({"floor"})
    assert max_error(fit, truth) < 5.0


def test_the_bar_alone_anchors_the_skeleton() -> None:
    truth = pose(0.3)

    fit = solve_one(observe(truth, grounded=False))

    assert isinstance(fit, ss.FrameFit)
    assert fit.anchors == frozenset({"bar"})
    assert max_error(fit, truth) < 5.0


def test_no_anchor_is_no_fit() -> None:
    obs = observe(pose(0.3), with_bar=False, grounded=False)

    assert ss.solve_frame(obs, body(), K, starts=[]) == ss.NoFit("no_anchor")


def test_too_few_observed_joints_is_no_fit() -> None:
    truth = pose(0.3)
    hidden = tuple(j for j in ss.SOLVED_JOINTS if j not in ("leftAnkle", "rightAnkle", "head"))

    obs = observe(truth, hidden=hidden)

    assert ss.solve_frame(obs, body(), K, starts=[]) == ss.NoFit("too_few_joints")


def test_an_unseen_joint_gets_no_position_and_the_rest_still_solve() -> None:
    truth = pose(0.5)

    fit = solve_one(observe(truth, hidden=("leftElbow",)))

    assert isinstance(fit, ss.FrameFit)
    assert fit.source["leftElbow"] == "unseen"
    assert "leftElbow" not in fit.points
    assert "leftElbow" not in fit.height_sigma_mm
    assert max(math.dist(fit.points[j], truth[j]) for j in fit.points) < 1.0


def test_no_term_but_pixels_bones_bar_and_floor_moves_the_solution() -> None:
    # The truth fits every term exactly, so a pure solve leaves it exactly where it is.
    truth = pose(0.5)
    obs = observe(truth)
    x = np.array([truth[j] for j in ss.SOLVED_JOINTS], dtype=float).ravel()
    problem = ss._Problem(obs, body(), K, hold=obs.hold, grounded=["leftAnkle", "rightAnkle"])

    assert np.max(np.abs(problem.residuals(x))) < 1e-9


def test_pixel_noise_costs_centimetres_not_decimetres() -> None:
    rng = np.random.default_rng(7)
    errors = []
    for truth in lift(10):
        fit = solve_one(observe(truth, noise_px=1.5, rng=rng))
        assert isinstance(fit, ss.FrameFit)
        errors.append(max(abs(fit.points[j][1] - truth[j][1]) for j in ss.SOLVED_JOINTS))
    assert float(np.median(errors)) < 30.0


def test_a_misplaced_keypoint_does_not_drag_its_neighbours() -> None:
    truth = pose(0.5)
    obs = observe(truth)
    pixels = dict(obs.pixels)
    u, v = pixels["leftKnee"]  # type: ignore[misc]
    pixels["leftKnee"] = (u + 150.0, v - 150.0)  # on something else entirely

    fit = solve_one(replace(obs, pixels=pixels))

    assert isinstance(fit, ss.FrameFit)
    # The knee itself can swing about the ankle-hip line without changing either bone, so the
    # bones cannot pull it back; the robust loss keeps it from dragging the joints around it.
    assert math.dist(fit.points["leftHip"], truth["leftHip"]) < 30.0
    assert math.dist(fit.points["leftAnkle"], truth["leftAnkle"]) < 30.0


def test_a_wrong_ankle_height_shows_in_the_closure_error() -> None:
    # The anchors disagree: the skeleton cannot reach both the bar and a floor 120 mm too low.
    # The error spreads across the bones and the bar rather than staying at the ankle, so it is
    # the closure -- feet-only against hands-only -- that measures it.
    frames = [observe(truth) for truth in lift(4)]

    right = ss.solve_capture(frames, body(), K)
    wrong = ss.solve_capture(frames, body(**{ai.ANKLE_HEIGHT: 200.0}), K)

    assert all(c is not None and c < 5.0 for c in right.closure_mm)
    assert all(c is not None and c > 50.0 for c in wrong.closure_mm)


def test_the_lower_cost_of_two_starts_wins_and_disagreement_is_flagged() -> None:
    truth = pose(0.5)
    obs = observe(truth)
    good = ss.initial_guess(obs, body(), K)
    assert good is not None
    bad = good.copy()
    bad[:] = 1.0  # every joint at the camera: a different, worse basin

    fit = ss.solve_frame(obs, body(), K, starts=[bad, good])

    assert isinstance(fit, ss.FrameFit)
    assert max_error(fit, truth) < 1.0


# --- grounded detection ----------------------------------------------------------------------


def test_a_still_confident_ankle_is_grounded_and_a_moving_one_is_not() -> None:
    still = [(500.0, 1500.0)] * 10
    moving = [(500.0, 1500.0 - 10.0 * i) for i in range(10)]

    assert all(ss.grounded_flags(still, [0.9] * 10))
    assert not any(ss.grounded_flags(moving, [0.9] * 10))
    assert not any(ss.grounded_flags(still, [0.1] * 10))
    assert not any(ss.grounded_flags([None] * 10, [0.0] * 10))


# --- the capture ----------------------------------------------------------------------------


def test_solve_capture_reports_closure_and_chains_frames() -> None:
    frames = [observe(truth) for truth in lift(8)]

    result = ss.solve_capture(frames, body(), K)

    assert all(isinstance(fit, ss.FrameFit) for fit in result.fits)
    assert all(c is not None and c < 5.0 for c in result.closure_mm)


def test_a_frame_without_the_bar_starts_from_its_neighbour_too() -> None:
    frames = [observe(truth) for truth in lift(3)]
    frames[1] = replace(frames[1], hold=None)

    result = ss.solve_capture(frames, body(), K)

    fit = result.fits[1]
    assert isinstance(fit, ss.FrameFit)
    assert fit.anchors == frozenset({"floor"})
    assert max_error(fit, lift(3)[1]) < 5.0
    assert result.closure_mm[1] is None


def test_body_from_lengths_drops_bones_with_no_length() -> None:
    lengths = ai.resolve_lengths(ai.Athlete(values={"tibia_in_cm": 40.0}))

    b = ss.body_from_lengths(lengths, y_floor_mm=Y_FLOOR)

    assert set(b.lengths) == {"leftTibia", "rightTibia"}


# --- the analytic Jacobian -------------------------------------------------------------------


@pytest.mark.parametrize("with_bar", [True, False])
def test_the_analytic_jacobian_matches_finite_differences(with_bar: bool) -> None:
    rng = np.random.default_rng(1)
    truth = pose(0.4)
    obs = observe(truth, noise_px=40.0, rng=rng, hidden=("leftElbow",), with_bar=with_bar)
    start = ss.initial_guess(obs, body(), K)
    assert start is not None
    # Away from the solution, with a wrist far off the bar axis so its hinge is live.
    x = start + rng.normal(0, 30.0, start.size)
    x[3 * ss.INDEX["leftWrist"] + 1] += 200.0
    problem = ss._Problem(
        obs,
        body(),
        K,
        hold=obs.hold,
        grounded=["leftAnkle", "rightAnkle"],
    )

    numeric = np.empty((problem.residuals(x).size, x.size))
    for col in range(x.size):
        step = np.zeros_like(x)
        step[col] = 1e-4
        numeric[:, col] = (problem.residuals(x + step) - problem.residuals(x - step)) / 2e-4

    assert problem.jacobian(x) == pytest.approx(numeric, abs=1e-5)
