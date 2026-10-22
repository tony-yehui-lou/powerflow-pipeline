"""S8's joint angles (`S8-joint-angles-draft3.md`), on hand-built poses whose angles are known.

Every pose is built in the lifter's own axes -- up, forward and left-to-right -- and placed in the
levelled camera frame (X right, Y down, Z forward) at a chosen yaw, so the same lifter can be
turned to face any way and must read the same angles.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from powerflow_pipeline.data.analyze import joint_angles as ja
from powerflow_pipeline.data.common.bar_geometry import Point
from powerflow_pipeline.data.common.models import JointId

UP = np.array([0.0, -1.0, 0.0])
APEX = np.array([-150.0, 400.0, 3000.0])
YAWS = (0.0, 45.0, 90.0, 180.0, -135.0)


def right_axis(yaw: float) -> Point:
    """The lifter's left-to-right axis, horizontal, at `yaw` degrees."""

    return (math.sin(math.radians(yaw)), 0.0, math.cos(math.radians(yaw)))


def forward(m: Point) -> np.ndarray:
    return np.cross(UP, np.array(m))


def toward(deg_from_up: float, m: Point) -> np.ndarray:
    """A unit direction in the sagittal plane, `deg_from_up` from up towards forward."""

    r = math.radians(deg_from_up)
    return math.cos(r) * UP + math.sin(r) * forward(m)


def pt(v: np.ndarray) -> Point:
    return (float(v[0]), float(v[1]), float(v[2]))


def hinge(
    apex: JointId,
    a: JointId,
    a_deg: float,
    c: JointId,
    c_deg: float,
    m: Point,
    length: float = 400.0,
) -> dict[JointId, Point]:
    """`apex` with neighbours `a` and `c`, each `length` away at the given directions."""

    return {
        apex: pt(APEX),
        a: pt(APEX + length * toward(a_deg, m)),
        c: pt(APEX + length * toward(c_deg, m)),
    }


def knee(angle: float, m: Point) -> dict[JointId, Point]:
    # Thigh up from the knee; flexion swings the shank back.
    return hinge("leftKnee", "leftHip", 0.0, "leftAnkle", -angle, m)


def hip(angle: float, m: Point) -> dict[JointId, Point]:
    # Trunk up from the hip; flexion swings the thigh forward.
    return hinge("leftHip", "leftShoulder", 0.0, "leftKnee", angle, m)


def elbow(angle: float, m: Point) -> dict[JointId, Point]:
    # Upper arm up from the elbow; flexion swings the forearm forward.
    return hinge("leftElbow", "leftShoulder", 0.0, "leftWrist", angle, m, length=280.0)


def shoulder(angle: float, m: Point) -> dict[JointId, Point]:
    # Trunk down from the shoulder; 0 is the arm hanging along it, forward is positive.
    return hinge("leftShoulder", "leftHip", 180.0, "leftElbow", 180.0 - angle, m)


def solve(
    points: dict[JointId, Point],
    m: Point | None,
    *,
    sources: dict[JointId, str] | None = None,
    flags: dict[JointId, list[str]] | None = None,
    branch_ambiguous: bool = False,
    floor_only: bool = False,
    conflict: bool = False,
) -> ja.FrameAngles:
    return ja.frame_angles(
        points,
        sources if sources is not None else dict.fromkeys(points, "observed"),
        flags or {},
        ja.Axis(vector=m, source=None if m is None else "bar", conflict=conflict),
        branch_ambiguous=branch_ambiguous,
        floor_only=floor_only,
    )


# -- the included angle ---------------------------------------------------------------------


@pytest.mark.parametrize("deg", [0.0, 30.0, 90.0, 179.9, 180.0])
def test_included_angle_is_exact_including_straight(deg: float) -> None:
    u = (1.0, 0.0, 0.0)
    v = (math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0.0)

    assert ja.included_deg(u, v) == pytest.approx(deg, abs=1e-9)


# -- each joint, signed, at every yaw -------------------------------------------------------


@pytest.mark.parametrize("yaw", YAWS)
@pytest.mark.parametrize(
    ("build", "joint", "angle"),
    [
        (knee, "leftKnee", 90.0),
        (knee, "leftKnee", 45.0),
        (knee, "leftKnee", 180.0),
        (knee, "leftKnee", 185.0),
        (hip, "leftHip", 90.0),
        (hip, "leftHip", 190.0),
        (elbow, "leftElbow", 60.0),
        (elbow, "leftElbow", 185.0),
        (shoulder, "leftShoulder", -5.0),
        (shoulder, "leftShoulder", 0.0),
        (shoulder, "leftShoulder", 90.0),
        (shoulder, "leftShoulder", 180.0),
        (shoulder, "leftShoulder", 190.0),
    ],
)
def test_each_joint_reads_its_built_angle_whichever_way_the_lifter_faces(
    yaw: float, build: object, joint: JointId, angle: float
) -> None:
    m = right_axis(yaw)
    points = build(angle, m)  # type: ignore[operator]

    got = solve(points, m).joints[joint]

    assert got.deg == pytest.approx(angle, abs=0.05)
    assert got.source == "observed"
    assert got.flags == ()


def test_a_flexed_joint_reads_below_straight_on_both_sides() -> None:
    m = right_axis(30.0)
    left = knee(100.0, m)
    right = hinge("rightKnee", "rightHip", 0.0, "rightAnkle", -100.0, m)

    got = solve({**left, **right}, m).joints

    assert got["leftKnee"].deg == pytest.approx(100.0, abs=0.05)
    assert got["rightKnee"].deg == pytest.approx(100.0, abs=0.05)


def test_translation_changes_no_angle() -> None:
    m = right_axis(10.0)
    points = hip(120.0, m)
    moved = {j: (p[0] + 500.0, p[1] - 200.0, p[2] + 900.0) for j, p in points.items()}

    assert solve(moved, m).joints["leftHip"].deg == pytest.approx(120.0, abs=0.05)


def test_the_angle_is_continuous_through_straight() -> None:
    m = right_axis(45.0)
    sweep = [170.0 + 0.5 * i for i in range(41)]  # 170 .. 190

    got = [solve(knee(a, m), m).joints["leftKnee"].deg for a in sweep]

    assert got == pytest.approx(sweep, abs=0.05)
    assert all(a is not None and b is not None and b > a for a, b in itertools.pairwise(got))


# -- nulls and flags ------------------------------------------------------------------------


def test_a_missing_joint_nulls_the_angle() -> None:
    m = right_axis(0.0)
    points = knee(90.0, m)
    del points["leftAnkle"]

    got = solve(points, m).joints["leftKnee"]

    assert got == ja.Angle(deg=None, source=None, flags=())


def test_an_unseen_joint_nulls_the_angle() -> None:
    m = right_axis(0.0)
    points = knee(90.0, m)
    sources = dict.fromkeys(points, "observed") | {"leftHip": "unseen"}

    assert solve(points, m, sources=sources).joints["leftKnee"].deg is None


def test_a_degenerate_segment_nulls_the_angle() -> None:
    m = right_axis(0.0)
    points = hinge("leftKnee", "leftHip", 0.0, "leftAnkle", -90.0, m, length=ja.MIN_SEGMENT_MM / 2)

    assert solve(points, m).joints["leftKnee"].deg is None


def test_with_no_axis_the_angle_is_unsigned_and_flagged() -> None:
    m = right_axis(0.0)

    got = solve(hip(190.0, m), None).joints["leftHip"]

    assert got.deg == pytest.approx(170.0, abs=0.05)
    assert got.flags == ("unsigned",)


def test_a_weak_joint_makes_the_angle_weak() -> None:
    m = right_axis(0.0)
    points = knee(90.0, m)
    sources = dict.fromkeys(points, "observed") | {"leftAnkle": "weak"}

    assert solve(points, m, sources=sources).joints["leftKnee"].source == "weak"


def test_flags_are_inherited_from_the_joints_and_the_frame() -> None:
    m = right_axis(0.0)
    points = knee(90.0, m)
    flags: dict[JointId, list[str]] = {
        "leftHip": ["uncertain"],
        "leftAnkle": ["ground_conflict", "uncertain"],
    }

    got = solve(points, m, flags=flags, branch_ambiguous=True, conflict=True).joints["leftKnee"]

    assert got.flags == ("uncertain", "ground_conflict", "branch_ambiguous", "axis_conflict")


def test_a_floor_only_frame_keeps_its_angles_flagged() -> None:
    m = right_axis(0.0)
    points = {**knee(90.0, m), **lower_body(m)}

    got = solve(points, m, floor_only=True)

    assert got.joints["leftKnee"].deg is not None
    assert "floor_only" in got.joints["leftKnee"].flags
    assert "floor_only" in got.segments["trunk"].flags


def test_a_knee_bending_sideways_is_off_sagittal() -> None:
    m = right_axis(0.0)
    up = UP
    side = np.array(m)
    points: dict[JointId, Point] = {
        "leftKnee": pt(APEX),
        "leftHip": pt(APEX + 400.0 * up),
        "leftAnkle": pt(APEX + 400.0 * (-math.cos(1.0) * up + math.sin(1.0) * side)),
    }

    got = solve(points, m).joints["leftKnee"]

    assert got.deg is not None
    assert "off_sagittal" in got.flags


def test_a_nearly_straight_joint_is_never_off_sagittal() -> None:
    m = right_axis(0.0)
    side = np.array(m)
    points: dict[JointId, Point] = {
        "leftKnee": pt(APEX),
        "leftHip": pt(APEX + 400.0 * UP),
        "leftAnkle": pt(APEX + 400.0 * (-math.cos(0.05) * UP + math.sin(0.05) * side)),
    }

    assert solve(points, m).joints["leftKnee"].flags == ()


def test_joints_with_no_angle_are_not_reported() -> None:
    m = right_axis(0.0)

    got = solve(knee(90.0, m), m)

    assert set(got.joints) == set(ja.JOINT_ANGLES)
    assert "leftAnkle" not in got.joints


# -- the left-to-right axis -----------------------------------------------------------------

LEFT_HIP: Point = (0.0, 0.0, 0.0)
RIGHT_HIP: Point = (200.0, -30.0, 0.0)  # tilted: the axis must come out horizontal


def test_near_left_keeps_the_bar_axis() -> None:
    got = ja.left_right_axis((1.0, 0.0, 0.0), "left", LEFT_HIP, RIGHT_HIP)

    assert got == ja.Axis(vector=(1.0, 0.0, 0.0), source="bar", conflict=False)


def test_near_right_reverses_the_bar_axis() -> None:
    got = ja.left_right_axis((-1.0, 0.0, 0.0), "right", LEFT_HIP, RIGHT_HIP)

    assert got.vector == pytest.approx((1.0, 0.0, 0.0))
    assert got.source == "bar"
    assert not got.conflict


def test_a_near_side_the_hips_contradict_keeps_the_bar_and_flags_it() -> None:
    got = ja.left_right_axis((-1.0, 0.0, 0.0), "left", LEFT_HIP, RIGHT_HIP)

    assert got.vector == pytest.approx((-1.0, 0.0, 0.0))
    assert got.conflict


def test_with_no_near_side_the_hips_sign_the_bar() -> None:
    got = ja.left_right_axis((-1.0, 0.0, 0.0), None, LEFT_HIP, RIGHT_HIP)

    assert got.vector == pytest.approx((1.0, 0.0, 0.0))
    assert got.source == "bar"


def test_with_no_near_side_and_no_hips_the_bar_is_skipped() -> None:
    assert ja.left_right_axis((1.0, 0.0, 0.0), None, None, RIGHT_HIP) == ja.Axis(None, None)


def test_with_no_bar_the_hips_are_used_levelled() -> None:
    got = ja.left_right_axis(None, "left", LEFT_HIP, RIGHT_HIP)

    assert got.vector == pytest.approx((1.0, 0.0, 0.0))
    assert got.source == "hips"


def test_with_nothing_there_is_no_axis() -> None:
    assert ja.left_right_axis(None, "left", None, None) == ja.Axis(None, None)


def test_vertical_hips_give_no_axis() -> None:
    assert ja.left_right_axis(None, None, (0.0, 0.0, 0.0), (0.0, -200.0, 0.0)).vector is None


# -- segment angles to vertical -------------------------------------------------------------


def lower_body(m: Point) -> dict[JointId, Point]:
    """Knees 20 deg ahead of the ankles, hips 30 deg behind the knees, trunk 40 deg forward."""

    points: dict[JointId, Point] = {}
    for side, sign in (("left", -1.0), ("right", 1.0)):
        lateral = sign * 100.0 * np.array(m)
        ankle = APEX + lateral
        knee_ = ankle + 420.0 * toward(20.0, m)
        hip_ = knee_ + 430.0 * toward(-30.0, m)
        points[f"{side}Ankle"] = pt(ankle)  # type: ignore[index]
        points[f"{side}Knee"] = pt(knee_)  # type: ignore[index]
        points[f"{side}Hip"] = pt(hip_)  # type: ignore[index]
        points[f"{side}Shoulder"] = pt(hip_ + 500.0 * toward(40.0, m))  # type: ignore[index]
    return points


@pytest.mark.parametrize("yaw", YAWS)
def test_segments_read_their_lean_from_vertical(yaw: float) -> None:
    m = right_axis(yaw)

    got = solve(lower_body(m), m).segments

    assert got["leftShin"].deg == pytest.approx(20.0, abs=0.05)
    assert got["rightShin"].deg == pytest.approx(20.0, abs=0.05)
    assert got["leftThigh"].deg == pytest.approx(-30.0, abs=0.05)
    assert got["trunk"].deg == pytest.approx(40.0, abs=0.05)
    assert got["trunk"].source == "observed"


def test_with_no_axis_segments_are_unsigned() -> None:
    m = right_axis(0.0)

    got = solve(lower_body(m), None).segments["leftThigh"]

    assert got.deg == pytest.approx(30.0, abs=0.05)
    assert got.flags == ("unsigned",)


def test_segments_never_flag_off_sagittal() -> None:
    m = right_axis(0.0)
    points = lower_body(m)
    points["leftKnee"] = pt(np.array(points["leftAnkle"]) + 420.0 * np.array(m))

    assert "off_sagittal" not in solve(points, m).segments["leftShin"].flags


def test_the_trunk_needs_both_shoulders_and_both_hips() -> None:
    m = right_axis(0.0)
    points = lower_body(m)
    del points["rightShoulder"]

    got = solve(points, m).segments

    assert got["trunk"].deg is None
    assert got["leftShin"].deg is not None


def test_a_null_frame_has_every_angle_null() -> None:
    got = ja.NULL_FRAME

    assert set(got.joints) == set(ja.JOINT_ANGLES)
    assert set(got.segments) == set(ja.SEGMENTS)
    assert all(a.deg is None for a in [*got.joints.values(), *got.segments.values()])
