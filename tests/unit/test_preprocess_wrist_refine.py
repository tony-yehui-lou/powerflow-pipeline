"""S6's wrist correction against the barbell, on a synthetic lift with a known truth.

The scene: a level bar rests, rises overhead and holds; the hands grip it at a fixed width; the
wrist joint sits off the bar by a 3D offset that flips sides overhead, as it does on a real lift.
The detector's far wrist is made to jump the way occlusion makes it jump, its near wrist to glitch
for two frames, and at the end the hands leave the bar.
"""

from __future__ import annotations

import math

import pytest

from powerflow_pipeline.data.preprocess import bar_geometry as bg
from powerflow_pipeline.data.preprocess import wrist_refine as wr
from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.pose_model import JointPixelSeries

K = Intrinsics(fx=1336.9, fy=1336.9, cx=705.0, cy=802.6, frame="portrait")
BAR = bg.BAR_TYPES["male IWF"]
HALF_GRIP = 460.0
YAW = 38.6
N_REST, N_RISE, N_HOLD, N_RELEASE = 30, 40, 40, 15
N = N_REST + N_RISE + N_HOLD + N_RELEASE
JUMPS = range(80, 90)  # far wrist snaps elsewhere, as it does under occlusion overhead
GLITCH = (95, 96)  # near wrist briefly on something else


def project(p: tuple[float, float, float]) -> tuple[float, float]:
    return (K.fx * p[0] / p[2] + K.cx, K.fy * p[1] / p[2] + K.cy)


def centre_y(i: int) -> float:
    if i < N_REST:
        return 1150.0
    if i < N_REST + N_RISE:
        return 1150.0 - 1650.0 * (i - N_REST + 1) / N_RISE
    return -500.0


def scene() -> dict[str, list]:
    a = (math.sin(math.radians(YAW)), 0.0, math.cos(math.radians(YAW)))
    caps, collars, near_true, far_true = [], [], [], []

    def along(c: tuple[float, float, float], d: float) -> tuple[float, float, float]:
        return (c[0] + d * a[0], c[1] + d * a[1], c[2] + d * a[2])

    for i in range(N):
        c = (-150.0, centre_y(i), 2777.0)
        at = lambda d, c=c: along(c, d)  # noqa: E731
        caps.append(project(at(-BAR.cap_to_centre_mm)))
        collars.append(project(at(BAR.cap_to_collar_mm - BAR.cap_to_centre_mm)))
        # Wrist above the bar at rest, below it overhead.
        dy = -40.0 if centre_y(i) > 300.0 else 40.0
        for out, d in ((near_true, -HALF_GRIP), (far_true, HALF_GRIP)):
            hand = at(d)
            out.append(project((hand[0], hand[1] + dy, hand[2])))
    return {"caps": caps, "collars": collars, "near": near_true, "far": far_true}


def series(points: list, confidence: float) -> JointPixelSeries:
    return JointPixelSeries(
        pixel_position=tuple((round(p[0]), round(p[1])) for p in points),
        confidence=tuple(confidence for _ in points),
    )


def detections(truth: dict[str, list]) -> tuple[JointPixelSeries, JointPixelSeries]:
    near = list(truth["near"])
    far = list(truth["far"])
    for i in JUMPS:
        far[i] = (far[i][0] - 250.0, far[i][1] + 300.0)
    for i in GLITCH:
        near[i] = (near[i][0] - 400.0, near[i][1] + 500.0)
    for i in range(N - N_RELEASE, N):  # hands off the bar: both wrists well below it
        near[i] = (near[i][0], near[i][1] + 400.0)
        far[i] = (far[i][0], far[i][1] + 400.0)
    # As on real lifts: the far wrist is seen confidently low down, and barely overhead.
    far_series = series(far, 0.85)
    far_series = JointPixelSeries(
        pixel_position=far_series.pixel_position,
        confidence=tuple(0.85 if centre_y(i) > 300.0 else 0.4 for i in range(N)),
    )
    return series(near, 0.9), far_series


@pytest.fixture
def lift() -> tuple[dict[str, list], list[bg.BarPose | None]]:
    truth = scene()
    bars = bg.solve_track(truth["caps"], truth["collars"], K, BAR)
    return truth, bars


def refine(
    lift: tuple[dict[str, list], list[bg.BarPose | None]], swap: bool = False
) -> tuple[wr.WristRefinement, JointPixelSeries, JointPixelSeries]:
    truth, bars = lift
    near, far = detections(truth)
    left, right = (far, near) if not swap else (near, far)
    result = wr.refine_wrists(left, right, bars, K, BAR)
    assert isinstance(result, wr.WristRefinement), result
    return result, near, far


class TestRefineWrists:
    def test_identifies_the_far_wrist_as_the_one_nearer_the_collar(self, lift) -> None:  # type: ignore[no-untyped-def]
        assert refine(lift)[0].far == "leftWrist"
        assert refine(lift, swap=True)[0].far == "rightWrist"

    def test_measures_the_grip(self, lift) -> None:  # type: ignore[no-untyped-def]
        result, _, _ = refine(lift)
        assert result.half_grip_mm == pytest.approx(HALF_GRIP, abs=15.0)

    def test_far_wrist_is_placed_on_the_bar_through_jumps_and_the_offset_flip(self, lift) -> None:  # type: ignore[no-untyped-def]
        truth, _ = lift
        result, _, _ = refine(lift)
        # Not exact: the near wrist is placed at the depth of its nearest bar point, which a wrist
        # sitting 40 mm off the bar is not quite at. That costs up to ~5 px mid-rise here.
        for i in range(N - N_RELEASE):
            got = result.left.pixel_position[i]
            assert got is not None
            assert math.dist(got, truth["far"][i]) <= 5.0, i

    def test_a_near_wrist_glitch_is_corrected_too(self, lift) -> None:  # type: ignore[no-untyped-def]
        truth, _ = lift
        result, _, _ = refine(lift)
        for i in GLITCH:
            got = result.right.pixel_position[i]
            assert got is not None
            assert math.dist(got, truth["near"][i]) <= 4.0
        assert result.near_corrected == GLITCH

    def test_hands_off_the_bar_are_left_alone(self, lift) -> None:  # type: ignore[no-untyped-def]
        result, near, far = refine(lift)
        release = range(N - N_RELEASE + wr.ON_BAR_WINDOW + 1, N)
        for i in release:
            assert result.left.pixel_position[i] == far.pixel_position[i]
            assert result.right.pixel_position[i] == near.pixel_position[i]

    def test_corrected_points_carry_the_near_wrists_confidence(self, lift) -> None:  # type: ignore[no-untyped-def]
        result, _, _ = refine(lift)
        assert result.left.confidence[JUMPS[0]] == pytest.approx(0.9)
        assert JUMPS[0] in result.far_corrected

    def test_series_lengths_and_contract_are_kept(self, lift) -> None:  # type: ignore[no-untyped-def]
        result, _, _ = refine(lift)
        for s in (result.left, result.right):
            assert len(s.pixel_position) == N
            assert len(s.confidence) == N
            assert all(p is None or isinstance(p[0], int) for p in s.pixel_position)

    def test_without_a_bar_there_is_nothing_to_refine(self, lift) -> None:  # type: ignore[no-untyped-def]
        truth, _ = lift
        near, far = detections(truth)
        skipped = wr.refine_wrists(far, near, [None] * N, K, BAR)
        assert isinstance(skipped, wr.Skipped)
        assert "grip" in skipped.reason
