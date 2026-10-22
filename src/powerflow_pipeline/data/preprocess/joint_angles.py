"""S8 · Joint angles: each joint's included angle and each segment's lean, from S8's 3D points.

`docs/specs/preprocessing/S8-joint-angles-draft3.md`. Pure: one frame's solved points in, its
angles out, with no smoothing across frames and nothing read off disk (`joint_hva` assembles the
document). Units are millimetres in S4's levelled camera frame (X right, Y down, Z forward), so
up is exactly -Y; the frame has no forward axis of the athlete's, so a sign needs the lifter's
left-to-right axis (`left_right_axis`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from powerflow_pipeline.data.common.models import JointId
from powerflow_pipeline.data.preprocess.bar_geometry import Point
from powerflow_pipeline.data.preprocess.body_plane import Side

Vector = NDArray[np.float64]
AxisSource = Literal["bar", "hips"]
AngleSource = Literal["observed", "weak"]
AngleKind = Literal["hinge", "shoulder"]

UP: Final = np.array([0.0, -1.0, 0.0])
# A segment shorter than this is a broken solve, not a bone (§2.6).
MIN_SEGMENT_MM: Final = 50.0
# Below this included angle the plane of the angle is defined well enough to test (§2.7) ...
OFF_SAGITTAL_BELOW_DEG: Final = 170.0
# ... and it is off-sagittal when its normal is further than this from the left-right axis.
OFF_SAGITTAL_MAX_DEG: Final = 45.0
SHOULDER_MIN_DEG: Final = -90.0  # the shoulder reads over (-90, 270] (§2.4)
INHERITED_FLAGS: Final = ("uncertain", "ground_conflict")


@dataclass(frozen=True)
class JointAngle:
    """The angle at `apex` between the segments to `a` and to `c` (§2.3, §2.4)."""

    a: JointId
    apex: JointId
    c: JointId
    kind: AngleKind
    flexion_sign: float  # s_j: chosen so anatomical flexion gives sigma >= 0


# Spec §2.3's table. `u = a - apex`, `v = c - apex`.
JOINT_ANGLES: Final[dict[JointId, JointAngle]] = {
    "leftKnee": JointAngle("leftHip", "leftKnee", "leftAnkle", "hinge", 1.0),
    "rightKnee": JointAngle("rightHip", "rightKnee", "rightAnkle", "hinge", 1.0),
    "leftHip": JointAngle("leftShoulder", "leftHip", "leftKnee", "hinge", -1.0),
    "rightHip": JointAngle("rightShoulder", "rightHip", "rightKnee", "hinge", -1.0),
    "leftElbow": JointAngle("leftShoulder", "leftElbow", "leftWrist", "hinge", -1.0),
    "rightElbow": JointAngle("rightShoulder", "rightElbow", "rightWrist", "hinge", -1.0),
    "leftShoulder": JointAngle("leftHip", "leftShoulder", "leftElbow", "shoulder", 1.0),
    "rightShoulder": JointAngle("rightHip", "rightShoulder", "rightElbow", "shoulder", 1.0),
}

# Spec §3: segment -> (proximal, distal); `trunk` runs between the two midpoints.
SEGMENTS: Final[dict[str, tuple[tuple[JointId, ...], tuple[JointId, ...]]]] = {
    "leftShin": (("leftKnee",), ("leftAnkle",)),
    "rightShin": (("rightKnee",), ("rightAnkle",)),
    "leftThigh": (("leftHip",), ("leftKnee",)),
    "rightThigh": (("rightHip",), ("rightKnee",)),
    "trunk": (("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")),
}


@dataclass(frozen=True)
class Axis:
    """The lifter's left-to-right axis on one frame: horizontal and unit, or none (§2.1)."""

    vector: Point | None
    source: AxisSource | None
    conflict: bool = False


@dataclass(frozen=True)
class FrameFlags:
    """What the whole frame's solve says about every angle on it (§2.7, §2.8)."""

    branch_ambiguous: bool = False
    floor_only: bool = False


@dataclass(frozen=True)
class Angle:
    deg: float | None
    source: AngleSource | None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameAngles:
    joints: dict[JointId, Angle]
    segments: dict[str, Angle]


NO_ANGLE: Final = Angle(deg=None, source=None)
NULL_FRAME: Final = FrameAngles(
    joints=dict.fromkeys(JOINT_ANGLES, NO_ANGLE), segments=dict.fromkeys(SEGMENTS, NO_ANGLE)
)


def included_deg(u: Point | Vector, v: Point | Vector) -> float:
    """The angle between `u` and `v` in [0, 180]. atan2, so it stays exact near 0 and 180."""

    a, b = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b)))


def _level(vector: Point | Vector) -> Vector | None:
    """`vector` with its vertical part removed and made unit; None when nothing is left."""

    v = np.asarray(vector, dtype=float)
    h = v - (v @ UP) * UP
    n = float(np.linalg.norm(h))
    return None if n < 1e-9 else h / n


def _point(v: Vector) -> Point:
    return (float(v[0]), float(v[1]), float(v[2]))


def left_right_axis(
    bar_axis: Point | None,
    near_side: Side | None,
    left_hip: Point | None,
    right_hip: Point | None,
) -> Axis:
    """§2.1: the bar signed by S6's near side, else by the hips; else the hips alone; else none.

    S5's axis runs cap -> collar, the near cap to the far collar, so it points from the near side
    to the far side. The hips only check it: a near side they contradict keeps the bar's sign
    and is flagged as a conflict.
    """

    hips = None
    if left_hip is not None and right_hip is not None:
        hips = _level(np.subtract(right_hip, left_hip))
    bar = None if bar_axis is None else _level(bar_axis)
    if bar is not None:
        if near_side is not None:
            m = bar if near_side == "left" else -bar
            return Axis(_point(m), "bar", conflict=hips is not None and float(m @ hips) < 0)
        if hips is not None:
            return Axis(_point(bar if float(bar @ hips) >= 0 else -bar), "bar")
    if hips is not None:
        return Axis(_point(hips), "hips")
    return Axis(None, None)


def _source(joints: Sequence[JointId], sources: Mapping[JointId, str]) -> AngleSource:
    return "observed" if all(sources.get(j) == "observed" for j in joints) else "weak"


def _flags(
    joints: Sequence[JointId],
    joint_flags: Mapping[JointId, Sequence[str]],
    axis: Axis,
    frame: FrameFlags,
) -> list[str]:
    flags: list[str] = []
    for joint in joints:
        for flag in joint_flags.get(joint, ()):
            if flag in INHERITED_FLAGS and flag not in flags:
                flags.append(flag)
    if frame.branch_ambiguous:
        flags.append("branch_ambiguous")
    if frame.floor_only:
        flags.append("floor_only")
    if axis.vector is None:
        flags.append("unsigned")
    if axis.conflict:
        flags.append("axis_conflict")
    return flags


def _seen(
    joints: Sequence[JointId], points: Mapping[JointId, Point], sources: Mapping[JointId, str]
) -> bool:
    return all(j in points and sources.get(j) not in (None, "unseen") for j in joints)


def _joint(
    spec: JointAngle,
    points: Mapping[JointId, Point],
    sources: Mapping[JointId, str],
    joint_flags: Mapping[JointId, Sequence[str]],
    axis: Axis,
    frame: FrameFlags,
) -> Angle:
    joints = (spec.a, spec.apex, spec.c)
    if not _seen(joints, points, sources):
        return NO_ANGLE
    apex = np.asarray(points[spec.apex], dtype=float)
    u = np.asarray(points[spec.a], dtype=float) - apex
    v = np.asarray(points[spec.c], dtype=float) - apex
    if min(float(np.linalg.norm(u)), float(np.linalg.norm(v))) < MIN_SEGMENT_MM:
        return NO_ANGLE

    theta = included_deg(u, v)
    flags = _flags(joints, joint_flags, axis, frame)
    if axis.vector is None:
        deg = theta
    else:
        m = np.asarray(axis.vector, dtype=float)
        normal = np.cross(u, v)
        flexion_side = spec.flexion_sign * float(normal @ m) >= 0
        if spec.kind == "hinge":
            deg = theta if flexion_side else 360.0 - theta
        else:
            deg = theta if flexion_side else -theta
            if deg <= SHOULDER_MIN_DEG:
                deg += 360.0
        n = float(np.linalg.norm(normal))
        if (
            theta < OFF_SAGITTAL_BELOW_DEG
            and n > 0
            and abs(float(normal @ m)) / n < math.cos(math.radians(OFF_SAGITTAL_MAX_DEG))
        ):
            flags.append("off_sagittal")
    return Angle(deg=deg, source=_source(joints, sources), flags=tuple(flags))


def _segment(
    proximal: Sequence[JointId],
    distal: Sequence[JointId],
    points: Mapping[JointId, Point],
    sources: Mapping[JointId, str],
    joint_flags: Mapping[JointId, Sequence[str]],
    axis: Axis,
    frame: FrameFlags,
) -> Angle:
    joints = (*proximal, *distal)
    if not _seen(joints, points, sources):
        return NO_ANGLE
    p = np.mean([points[j] for j in proximal], axis=0)
    d = p - np.mean([points[j] for j in distal], axis=0)
    if float(np.linalg.norm(d)) < MIN_SEGMENT_MM:
        return NO_ANGLE
    up = float(d @ UP)
    if axis.vector is None:
        deg = math.degrees(math.atan2(float(np.linalg.norm(d - up * UP)), up))
    else:
        fwd = np.cross(UP, np.asarray(axis.vector, dtype=float))
        deg = math.degrees(math.atan2(float(d @ fwd), up))
    flags = _flags(joints, joint_flags, axis, frame)
    return Angle(deg=deg, source=_source(joints, sources), flags=tuple(flags))


def frame_angles(
    points: Mapping[JointId, Point],
    sources: Mapping[JointId, str],
    joint_flags: Mapping[JointId, Sequence[str]],
    axis: Axis,
    *,
    branch_ambiguous: bool = False,
    floor_only: bool = False,
) -> FrameAngles:
    """One solved frame's joint angles (§2) and segment angles to vertical (§3).

    `floor_only`: the frame's solve had no bar anchor, so its angles may be S8's mirror pair
    (§2.8). They are kept, and flagged.
    """

    frame = FrameFlags(branch_ambiguous=branch_ambiguous, floor_only=floor_only)
    return FrameAngles(
        joints={
            joint: _joint(spec, points, sources, joint_flags, axis, frame)
            for joint, spec in JOINT_ANGLES.items()
        },
        segments={
            name: _segment(p, d, points, sources, joint_flags, axis, frame)
            for name, (p, d) in SEGMENTS.items()
        },
    )
