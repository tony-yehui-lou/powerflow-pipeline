"""S8 · Joint HVA: one capture's `joint_hva.json`, assembled from S6's 2D pose and the solve.

`docs/specs/preprocessing/S8-joint-hva-draft3.md` §6. Pure: every input arrives already read
(`tasks/kinematics.py` does the I/O), so the whole document is testable against a synthetic
lift. Height only: each value is the solved position's height above the floor on that frame,
with no smoothing, regularisation or adjustment. Joint and segment angles come from the same
solved points (`joint_angles`, `S8-joint-angles-draft3.md` §4), on `constrained` frames only.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from powerflow_pipeline.data.analyze import athlete_input as ai
from powerflow_pipeline.data.analyze import joint_angles as ja
from powerflow_pipeline.data.analyze import skeleton_solve as ss
from powerflow_pipeline.data.analyze.body_plane import Side, lateral_offsets, place_on_plane
from powerflow_pipeline.data.common.bar_geometry import (
    BarPose,
    FloorAnchor,
    Point,
    floor_anchor,
)
from powerflow_pipeline.data.common.capture_models import Intrinsics
from powerflow_pipeline.data.common.models import HUMAN_SKELETON, JointId, PoseDocument

# 1.1.0 added the joint angles, `segments`, `angleConvention` and `frameStatus.angleAxis`.
SCHEMA_VERSION: Final = "1.1.0"
METHOD_SPEC: Final = "docs/specs/preprocessing/S8-joint-hva-draft3.md"

FrameMethod = Literal["constrained", "bar_plane"]

# A height whose posterior sigma exceeds this is kept but flagged `uncertain` (draft 3 §3.7).
MAX_HEIGHT_SIGMA_MM: Final = 50.0

ANGLE_CONVENTION: Final = {
    "joint": "included; 180 = straight; > 180 = past straight (knee, hip, elbow)",
    "shoulder": "included trunk-to-upper-arm; (-90, 270]; 0 = arm along trunk, positive = forward",
    "segment": "inclination from vertical; positive = proximal end ahead of distal end",
}

CLAVICLES: Final[dict[JointId, JointId]] = {
    "leftClavicle": "leftShoulder",
    "rightClavicle": "rightShoulder",
}


@dataclass(frozen=True)
class Inputs:
    """Everything one capture's document is built from, already read off disk."""

    pose: PoseDocument
    k: Intrinsics
    bars: Sequence[BarPose | None]  # S5's bar per frame, `pose.frames.count` long
    hands_on_bar: frozenset[int]  # S6's hands-on-bar frames (`wrist_refinement.json`)
    half_grip_mm: float | None
    near_side: Side | None
    floor_offset_mm: float | None  # S3's floor, the fallback anchor and the cross-check
    athlete: ai.Athlete
    clavicle_shoulder_weight: float = 0.5  # S6's own interpolation weight


@dataclass(frozen=True)
class Built:
    document: dict[str, Any]
    derived: dict[str, Any]
    warnings: list[str]


def _m(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value / 1000.0, digits)


def _holds(inputs: Inputs) -> list[ss.BarHold | None]:
    if inputs.half_grip_mm is None or inputs.near_side is None:
        return [None] * len(inputs.bars)
    holds: list[ss.BarHold | None] = []
    for i, bar in enumerate(inputs.bars):
        if bar is None or i not in inputs.hands_on_bar:
            holds.append(None)
            continue
        holds.append(
            ss.BarHold(
                centre_mm=bar.centre_mm,
                axis=bar.axis,
                half_grip_mm=inputs.half_grip_mm,
                near_side=inputs.near_side,
            )
        )
    return holds


def observations(inputs: Inputs) -> list[ss.FrameObs]:
    """S6's pixels per frame, with the bar where the hands hold it and which ankles are down."""

    joints = inputs.pose.joints
    n = inputs.pose.frames.count
    grounded = {
        ankle: ss.grounded_flags(
            [
                None if p is None else (float(p[0]), float(p[1]))
                for p in joints[ankle].pixel_position
            ],
            joints[ankle].confidence,
        )
        for ankle in ss.ANKLES
    }
    holds = _holds(inputs)
    frames = []
    for i in range(n):
        pixels = {}
        confidence = {}
        for joint in ss.SOLVED_JOINTS:
            px = joints[joint].pixel_position[i]
            pixels[joint] = None if px is None else (float(px[0]), float(px[1]))
            confidence[joint] = joints[joint].confidence[i]
        frames.append(
            ss.FrameObs(
                pixels=pixels,
                confidence=confidence,
                hold=holds[i],
                grounded=frozenset(a for a in ss.ANKLES if grounded[a][i]),
            )
        )
    return frames


def _bar_plane(obs: ss.FrameObs, k: Intrinsics, body: ss.Body) -> ss.FrameFit | ss.NoFit:
    """Draft 1's body-plane solve as the whole answer, for a capture with no measurements."""

    hold = obs.hold
    if hold is None:
        return ss.NoFit("no_anchor")
    offsets = lateral_offsets(
        near_side=hold.near_side,
        half_grip_mm=hold.half_grip_mm,
        shoulder_width_mm=ss.DEFAULT_SHOULDER_WIDTH_MM,
        hip_width_mm=ss.DEFAULT_HIP_WIDTH_MM,
    )
    points: dict[JointId, Point] = {}
    for joint in ss.SOLVED_JOINTS:
        px = obs.pixels.get(joint)
        if px is None or obs.confidence.get(joint, 0.0) < ss.MIN_CONF:
            continue
        point = place_on_plane(px, k, hold.centre_mm, hold.axis, offsets[joint])
        if point is not None:
            points[joint] = point
    if not points:
        return ss.NoFit("no_fit")
    return ss.FrameFit(
        points=points,
        height_sigma_mm={},
        source={j: ss.joint_source(obs, j) for j in points},
        anchors=frozenset({"bar"}),
        rms=math.nan,
    )


def _length_entry(length: ai.Length) -> dict[str, Any]:
    return {
        "field": length.field,
        "measuredM": _m(length.measured_mm),
        "populationM": _m(length.population_mm),
        "usedM": _m(length.used_mm),
        "source": length.source,
        "sigmaM": _m(length.sigma_mm),
    }


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def build(inputs: Inputs) -> Built:
    """The capture's `joint_hva.json`, its manifest values and its warnings."""

    n = inputs.pose.frames.count
    warnings = list(inputs.athlete.warnings)
    anchor: FloorAnchor = floor_anchor(inputs.bars, floor_offset_mm=inputs.floor_offset_mm)
    frames = observations(inputs)

    lengths = ai.resolve_lengths(inputs.athlete)
    for length in lengths.values():
        warnings.extend(length.warnings)
    fits: list[ss.FrameFit | ss.NoFit]
    closure: list[float | None] = [None] * n
    reasons: list[str | None]

    if anchor.y_floor_mm is None:
        warnings.append("no floor: no rest-plate anchor from the bar and no S3 floor plane")
        fits = [ss.NoFit("no_anchor")] * n
        method: FrameMethod | None = None
    elif not inputs.athlete.usable:
        warnings.append(
            "no usable athlete measurement in metadata.yaml: body-plane solve only "
            "(draft 1), heights while the hands hold the bar"
        )
        body = ss.Body(lengths={}, y_floor_mm=anchor.y_floor_mm)
        fits = [_bar_plane(obs, inputs.k, body) for obs in frames]
        method = "bar_plane"
    else:
        body = ss.body_from_lengths(lengths, y_floor_mm=anchor.y_floor_mm)
        solved = ss.solve_capture(frames, body, inputs.k)
        fits, closure = solved.fits, solved.closure_mm
        method = "constrained"

    reasons = [fit.reason if isinstance(fit, ss.NoFit) else None for fit in fits]
    y_floor = anchor.y_floor_mm
    joints_out: dict[str, dict[str, Any]] = {}
    w = inputs.clavicle_shoulder_weight
    for joint in HUMAN_SKELETON.joints:
        position: list[list[float] | None] = []
        height: list[float | None] = []
        sigma: list[float | None] = []
        source: list[str | None] = []
        flags: list[list[str]] = []
        for fit in fits:
            point, s, src = _joint_at(fit, joint, w)
            if point is None or y_floor is None:
                position.append(None)
                height.append(None)
                sigma.append(None)
                source.append(None)
                flags.append([])
                continue
            assert isinstance(fit, ss.FrameFit)
            position.append([round(c / 1000.0, 4) for c in point])
            height.append(_m(y_floor - point[1]))
            sigma.append(_m(s))
            source.append(src)
            frame_flags = []
            if s is not None and s > MAX_HEIGHT_SIGMA_MM:
                frame_flags.append("uncertain")
            if joint in fit.ground_conflict:
                frame_flags.append("ground_conflict")
            flags.append(frame_flags)
        joints_out[joint] = {
            "positionM": position,
            "heightM": height,
            "heightSigmaM": sigma,
            "source": source,
            "flags": flags,
        }

    angles, axes = _angles(inputs, fits, method, joints_out)
    for joint in HUMAN_SKELETON.joints:
        series = [frame.joints[joint] for frame in angles] if joint in ja.JOINT_ANGLES else None
        joints_out[joint].update(
            {
                "angleDeg": None if series is None else [_deg(a.deg) for a in series],
                "angleSource": None if series is None else [a.source for a in series],
                "angleFlags": None if series is None else [list(a.flags) for a in series],
            }
        )
    segments = {
        name: {
            "inclinationDeg": [_deg(frame.segments[name].deg) for frame in angles],
            "source": [frame.segments[name].source for frame in angles],
            "flags": [list(frame.segments[name].flags) for frame in angles],
        }
        for name in ja.SEGMENTS
    }
    axis_conflicts = sum(1 for axis in axes if axis is not None and axis.conflict)
    floor_only = sum(
        1
        for fit, axis in zip(fits, axes, strict=True)
        if axis is not None and isinstance(fit, ss.FrameFit) and "bar" not in fit.anchors
    )
    if floor_only:
        warnings.append(
            f"angles: {floor_only} frames were solved from the floor alone (floor_only): their "
            "angles may be S8's mirror solution, so read them against bar-anchored frames"
        )
    if axis_conflicts:
        warnings.append(
            f"angles: S6's near side and the hips disagree on left and right on {axis_conflicts} "
            "frames (axis_conflict): the near side or S6's left/right hips may be swapped"
        )

    solved_fits = [fit for fit in fits if isinstance(fit, ss.FrameFit)]
    by_anchor = Counter(
        "+".join(sorted(fit.anchors)) if isinstance(fit, ss.FrameFit) else fit.reason
        for fit in fits
    )
    closures = [c for c in closure if c is not None]
    used_mm = {bone_id: length.used_mm for bone_id, length in lengths.items()}
    height_cm = inputs.athlete.values.get("height_in_cm")
    armspan_cm = inputs.athlete.values.get("armspan_in_cm")
    armspan = ai.armspan_check(
        used_mm,
        height_mm=None if height_cm is None else height_cm * 10.0,
        armspan_mm=None if armspan_cm is None else armspan_cm * 10.0,
    )
    if armspan is not None and armspan.warn:
        warnings.append(
            f"armspan check: the arm lengths imply {armspan.implied_mm / 10:.1f} cm against a "
            f"measured {armspan.measured_mm / 10:.1f} cm"
        )

    document: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "captureId": inputs.pose.capture_id,
        "method": METHOD_SPEC,
        "frames": inputs.pose.frames.model_dump(by_alias=True),
        "floor": {
            "anchor": anchor.method,
            "yFloorM": _m(anchor.y_floor_mm),
            "restFrames": anchor.rest_frames,
            "s3FloorPlaneYM": _m(anchor.floor_plane_y_mm),
        },
        "bones": {
            bone_id: _length_entry(length)
            for bone_id, length in lengths.items()
            if bone_id != ai.ANKLE_HEIGHT
        },
        "ankleHeight": _length_entry(lengths[ai.ANKLE_HEIGHT]),
        "armspanCheck": None
        if armspan is None
        else {
            "measuredM": _m(armspan.measured_mm),
            "impliedM": _m(armspan.implied_mm),
            "warn": armspan.warn,
        },
        "frameStatus": {
            "method": [method if isinstance(fit, ss.FrameFit) else None for fit in fits],
            "reason": reasons,
            "anchors": [
                sorted(fit.anchors) if isinstance(fit, ss.FrameFit) else [] for fit in fits
            ],
            "closureErrorM": [_m(c) for c in closure],
            "branchAmbiguous": [
                isinstance(fit, ss.FrameFit) and fit.branch_ambiguous for fit in fits
            ],
            "rms": [
                round(fit.rms, 3)
                if isinstance(fit, ss.FrameFit) and not math.isnan(fit.rms)
                else None
                for fit in fits
            ],
            "angleAxis": [None if axis is None else axis.source for axis in axes],
        },
        "angleConvention": dict(ANGLE_CONVENTION),
        "joints": joints_out,
        "segments": segments,
    }
    derived: dict[str, Any] = {
        "kinematics_method": method,
        "kinematics_floor_anchor": anchor.method,
        "kinematics_frames_solved": len(solved_fits),
        "kinematics_frames_by_anchor": dict(sorted(by_anchor.items())),
        "kinematics_closure_median_mm": None
        if not closures
        else round(statistics.median(closures), 1),
        "kinematics_closure_p95_mm": None
        if not closures
        else round(_percentile(closures, 0.95) or 0.0, 1),
        "kinematics_branch_ambiguous": sum(1 for fit in solved_fits if fit.branch_ambiguous),
        "kinematics_ground_conflict": sum(1 for fit in solved_fits if fit.ground_conflict),
        "kinematics_bone_sources": {b: length.source for b, length in lengths.items()},
        "angles_frames_by_joint": {
            joint: sum(1 for frame in angles if frame.joints[joint].deg is not None)
            for joint in ja.JOINT_ANGLES
        },
        "angles_axis": dict(
            sorted(Counter("none" if a is None else a.source or "none" for a in axes).items())
        ),
        "angles_off_sagittal": sum(
            1 for frame in angles for a in frame.joints.values() if "off_sagittal" in a.flags
        ),
        "angles_axis_conflict": axis_conflicts,
        "angles_floor_only": floor_only,
    }
    return Built(document=document, derived=derived, warnings=warnings)


def _deg(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _angles(
    inputs: Inputs,
    fits: Sequence[ss.FrameFit | ss.NoFit],
    method: FrameMethod | None,
    joints_out: dict[str, dict[str, Any]],
) -> tuple[list[ja.FrameAngles], list[ja.Axis | None]]:
    """Each frame's angles and left-right axis; `bar_plane` and unsolved frames get none (§2.6).

    A `bar_plane` frame's points sit on one plane at fixed lateral offsets, so their angle would
    be a projected one, which the spec rules out.
    """

    angles: list[ja.FrameAngles] = []
    axes: list[ja.Axis | None] = []
    for i, fit in enumerate(fits):
        if method != "constrained" or not isinstance(fit, ss.FrameFit):
            angles.append(ja.NULL_FRAME)
            axes.append(None)
            continue
        bar = inputs.bars[i]
        axis = ja.left_right_axis(
            None if bar is None else bar.axis,
            inputs.near_side,
            fit.points.get("leftHip"),
            fit.points.get("rightHip"),
        )
        flags = {joint: joints_out[joint]["flags"][i] for joint in fit.points}
        angles.append(
            ja.frame_angles(
                fit.points,
                fit.source,
                flags,
                axis,
                branch_ambiguous=fit.branch_ambiguous,
                floor_only="bar" not in fit.anchors,
            )
        )
        axes.append(axis)
    return angles, axes


def _joint_at(
    fit: ss.FrameFit | ss.NoFit, joint: JointId, shoulder_weight: float
) -> tuple[Point | None, float | None, str | None]:
    """A joint's point, height sigma and source on one frame; clavicles derived from shoulders."""

    if not isinstance(fit, ss.FrameFit):
        return None, None, None
    if joint in CLAVICLES:
        shoulder = fit.points.get(CLAVICLES[joint])
        left, right = fit.points.get("leftShoulder"), fit.points.get("rightShoulder")
        if shoulder is None or left is None or right is None:
            return None, None, None
        neck = [(a + b) / 2 for a, b in zip(left, right, strict=True)]
        x, y, z = (
            (1 - shoulder_weight) * c + shoulder_weight * s
            for c, s in zip(neck, shoulder, strict=True)
        )
        return (x, y, z), fit.height_sigma_mm.get(CLAVICLES[joint]), "derived"
    found = fit.points.get(joint)
    if found is None:
        return None, None, None
    return found, fit.height_sigma_mm.get(joint), fit.source.get(joint)
