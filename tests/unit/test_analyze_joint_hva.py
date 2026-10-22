"""S8's `joint_hva.json`, built from a synthetic lift's 2D pose and checked against its 3D truth."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from powerflow_pipeline.data.analyze import athlete_input as ai
from powerflow_pipeline.data.analyze import joint_angles as ja
from powerflow_pipeline.data.analyze import joint_hva as jh
from powerflow_pipeline.data.common.bar_geometry import BarPose, project
from powerflow_pipeline.data.common.models import (
    HUMAN_SKELETON,
    Frames,
    JointId,
    JointSeries,
    PoseDocument,
)
from tests.unit.test_analyze_skeleton_solve import (
    AXIS,
    NEAR,
    SAGITTAL,
    TAPE,
    TRUE,
    Y_FLOOR,
    YAW,
    K,
    hold_for,
    lift,
)

N = 35


def pose_document(truths: list[dict[JointId, tuple[float, float, float]]]) -> PoseDocument:
    joints = {}
    for joint in HUMAN_SKELETON.joints:
        pixels = []
        for truth in truths:
            # S6's clavicles are interpolated; the scene has none, so any pixel will do.
            point = truth.get(joint, truth["head"])
            u, v = project(point, K)
            pixels.append((round(u), round(v)))
        joints[joint] = JointSeries(
            position=None, pixel_position=tuple(pixels), confidence=tuple(0.95 for _ in pixels)
        )
    return PoseDocument(
        capture_id="10 September/test1",
        role="single",
        stage="s4_crop",
        frames=Frames(
            count=len(truths),
            fps=60.0,
            start_epoch_ms=0,
            t_ms=tuple(round(i * 1000 / 60) for i in range(len(truths))),
        ),
        joints=joints,
    )


def bars(truths: list[dict[JointId, tuple[float, float, float]]]) -> list[BarPose]:
    out = []
    for truth in truths:
        hold = hold_for(truth)
        c = hold.centre_mm
        cap = tuple(ci - 1100.0 * a for ci, a in zip(c, AXIS, strict=True))
        collar = tuple(ci + 655.0 * a for ci, a in zip(c, AXIS, strict=True))
        # Collar-only frames never set the rest-plate anchor, so the floor is S3's, which is
        # the scene's true floor -- the synthetic bar never rests on plates.
        out.append(BarPose(cap, collar, c, YAW, "collar-only"))  # type: ignore[arg-type]
    return out


def inputs(**overrides: object) -> jh.Inputs:
    truths = lift(N)
    base = {
        "pose": pose_document(truths),
        "k": K,
        "bars": bars(truths),
        "hands_on_bar": frozenset(range(N)),
        "half_grip_mm": TRUE["half_grip"],
        "near_side": NEAR,
        "floor_offset_mm": Y_FLOOR,
        "athlete": ai.Athlete(values=dict(TAPE)),
    }
    base.update(overrides)
    return jh.Inputs(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def built() -> jh.Built:
    return jh.build(inputs())


def test_heights_match_the_truth_to_rounding(built: jh.Built) -> None:
    truths = lift(N)
    joints = built.document["joints"]
    for joint in ("leftAnkle", "rightKnee", "leftHip", "rightShoulder", "leftWrist", "head"):
        for i in range(N):
            want = (Y_FLOOR - truths[i][joint][1]) / 1000.0
            assert joints[joint]["heightM"][i] == pytest.approx(want, abs=0.015), (joint, i)


def test_every_frame_is_constrained_with_both_anchors(built: jh.Built) -> None:
    status = built.document["frameStatus"]
    assert set(status["method"]) == {"constrained"}
    assert all(anchors == ["bar", "floor"] for anchors in status["anchors"])
    assert all(c is not None and c < 0.02 for c in status["closureErrorM"])
    assert built.derived["kinematics_frames_solved"] == N
    assert built.derived["kinematics_frames_by_anchor"] == {"bar+floor": N}


def test_clavicles_are_derived_between_neck_and_shoulder(built: jh.Built) -> None:
    joints = built.document["joints"]
    left = joints["leftClavicle"]
    assert set(left["source"]) == {"derived"}
    sh, other = joints["leftShoulder"]["positionM"][0], joints["rightShoulder"]["positionM"][0]
    neck = [(a + b) / 2 for a, b in zip(sh, other, strict=True)]
    want = [(n + s) / 2 for n, s in zip(neck, sh, strict=True)]
    assert left["positionM"][0] == pytest.approx(want, abs=1e-3)


def test_bones_report_the_length_each_was_solved_with(built: jh.Built) -> None:
    bones = built.document["bones"]
    tibia = bones["leftTibia"]
    assert tibia["field"] == "tibia_in_cm"
    assert tibia["source"] == "measured"
    assert tibia["usedM"] == pytest.approx(TRUE["tibia"] / 1000)
    assert bones["headNeck"]["source"] == "population"
    assert built.document["ankleHeight"]["usedM"] == pytest.approx(TRUE["ankle_h"] / 1000)
    assert built.document["floor"]["anchor"] == "floor-plane"
    assert built.document["armspanCheck"] is None  # the tape has no armspan


def test_the_document_is_json(built: jh.Built) -> None:
    json.dumps(built.document)


def test_without_measurements_the_body_plane_solve_stands_in() -> None:
    result = jh.build(inputs(athlete=ai.Athlete(values={})))

    status = result.document["frameStatus"]
    assert set(status["method"]) == {"bar_plane"}
    assert any("body-plane" in w for w in result.warnings)
    hip = result.document["joints"]["leftHip"]["heightM"]
    assert all(h is not None for h in hip)
    assert result.document["joints"]["leftHip"]["heightSigmaM"][0] is None


def test_without_a_floor_every_value_is_null_with_a_reason() -> None:
    result = jh.build(inputs(floor_offset_mm=None, bars=[None] * N))

    assert all(h is None for h in result.document["joints"]["head"]["heightM"])
    assert set(result.document["frameStatus"]["reason"]) == {"no_anchor"}
    assert any("no floor" in w for w in result.warnings)


def test_off_bar_frames_stand_on_the_floor_alone() -> None:
    result = jh.build(inputs(hands_on_bar=frozenset(range(10, N))))

    anchors = result.document["frameStatus"]["anchors"]
    assert anchors[0] == ["floor"]
    assert anchors[20] == ["bar", "floor"]
    assert result.document["frameStatus"]["closureErrorM"][0] is None


def test_an_inconsistent_armspan_warns() -> None:
    values = dict(TAPE) | {"armspan_in_cm": 120.0}

    result = jh.build(inputs(athlete=ai.Athlete(values=values)))

    assert result.document["armspanCheck"]["warn"] is True
    assert any("armspan" in w for w in result.warnings)


def test_an_unseen_joint_has_no_height() -> None:
    doc = pose_document(lift(N))
    elbow = doc.joints["leftElbow"]
    hidden = JointSeries(
        position=None,
        pixel_position=tuple(None for _ in elbow.pixel_position),
        confidence=tuple(0.0 for _ in elbow.confidence),
    )
    doc = doc.model_copy(update={"joints": {**doc.joints, "leftElbow": hidden}})

    result = jh.build(inputs(pose=doc))

    elbow_out = result.document["joints"]["leftElbow"]
    assert elbow_out["heightM"] == [None] * N
    assert elbow_out["positionM"] == [None] * N
    assert result.document["joints"]["rightElbow"]["heightM"][0] is not None


# -- joint angles (S8-joint-angles-draft3.md) ----------------------------------------------


def true_included(truth: dict[JointId, tuple[float, float, float]], joint: JointId) -> float:
    spec = ja.JOINT_ANGLES[joint]
    apex = np.array(truth[spec.apex])
    u, v = np.array(truth[spec.a]) - apex, np.array(truth[spec.c]) - apex
    return math.degrees(math.acos(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))


def test_knee_and_hip_angles_match_the_truth_and_read_flexed(built: jh.Built) -> None:
    truths = lift(N)
    joints = built.document["joints"]
    for joint in ("leftKnee", "rightKnee", "leftHip", "rightHip"):
        for i in range(N):
            got = joints[joint]["angleDeg"][i]
            # The lift is a bent start rising to standing: every knee and hip is flexed.
            assert got == pytest.approx(true_included(truths[i], joint), abs=1.0), (joint, i)
            assert got < 180.0
        assert set(joints[joint]["angleSource"]) == {"observed"}


# The angle maths is exact on true points (test_analyze_joint_angles.py); what is left here is
# S8's own position error. Measured on this lift: the elbows are its least certain joints (up to
# 20 mm off in 3D, against 3 mm at the knees), which turns a near-straight arm by up to 8 deg.
ARM_TOLERANCE_DEG: dict[JointId, float] = {
    "leftShoulder": 1.0,
    "rightShoulder": 4.0,
    "leftElbow": 4.0,
    "rightElbow": 9.0,
}


def test_the_elbow_and_shoulder_angles_match_the_truth(built: jh.Built) -> None:
    truths = lift(N)
    joints = built.document["joints"]
    for joint, tolerance in ARM_TOLERANCE_DEG.items():
        for i in range(N):
            got = joints[joint]["angleDeg"][i]
            # The arms hang nearly straight, so either side of straight (or of 0) is possible.
            included = abs(got) if got <= 180.0 else 360.0 - got
            want = true_included(truths[i], joint)
            assert included == pytest.approx(want, abs=tolerance), (joint, i)


def test_joints_with_no_angle_publish_null_as_a_whole(built: jh.Built) -> None:
    joints = built.document["joints"]
    for joint in ("leftAnkle", "rightWrist", "leftClavicle", "head"):
        assert joints[joint]["angleDeg"] is None
        assert joints[joint]["angleSource"] is None
        assert joints[joint]["angleFlags"] is None


def test_segments_match_the_truth(built: jh.Built) -> None:
    truths = lift(N)
    up = np.array([0.0, -1.0, 0.0])
    shin = built.document["segments"]["leftShin"]["inclinationDeg"]
    for i in range(N):
        d = np.array(truths[i]["leftKnee"]) - np.array(truths[i]["leftAnkle"])
        want = math.degrees(math.atan2(d @ np.array(SAGITTAL), d @ up))
        assert shin[i] == pytest.approx(want, abs=1.0), i
    # The torso leans forward throughout the pull.
    assert all(t > 0 for t in built.document["segments"]["trunk"]["inclinationDeg"])


def test_the_axis_comes_from_the_bar_and_agrees_with_the_hips(built: jh.Built) -> None:
    assert set(built.document["frameStatus"]["angleAxis"]) == {"bar"}
    assert built.derived["angles_axis"] == {"bar": N}
    assert built.derived["angles_axis_conflict"] == 0
    assert built.derived["angles_frames_by_joint"]["leftKnee"] == N
    assert not any("axis" in w for w in built.warnings)


def test_the_document_states_its_angle_convention(built: jh.Built) -> None:
    assert built.document["schemaVersion"] == "1.1.0"
    assert set(built.document["angleConvention"]) == {"joint", "shoulder", "segment"}


def test_angles_are_rounded_to_a_tenth_of_a_degree(built: jh.Built) -> None:
    knee = built.document["joints"]["leftKnee"]["angleDeg"]
    assert all(round(a, 1) == a for a in knee)


def test_with_no_bar_the_hips_give_the_axis() -> None:
    result = jh.build(inputs(bars=[None] * N, hands_on_bar=frozenset()))

    assert set(result.document["frameStatus"]["angleAxis"]) == {"hips"}
    # Floor-only frames are S8's mirror-pair case: most land on the wrong branch, and their
    # angles carry the frame's flag rather than a value anyone should read.
    status = result.document["frameStatus"]["branchAmbiguous"]
    flags = result.document["joints"]["leftKnee"]["angleFlags"]
    assert any(status)
    assert all(
        "branch_ambiguous" in f for f, ambiguous in zip(flags, status, strict=True) if ambiguous
    )


def test_a_wrong_near_side_is_flagged_and_warned(monkeypatch: pytest.MonkeyPatch) -> None:
    # A wrong near side would also mislead the solve, so the conflict is injected at the axis.
    original = ja.left_right_axis

    def conflicting(*args: object) -> ja.Axis:
        return dataclasses.replace(original(*args), conflict=True)  # type: ignore[arg-type]

    monkeypatch.setattr(ja, "left_right_axis", conflicting)
    result = jh.build(inputs())

    assert result.derived["angles_axis_conflict"] == N
    assert any("axis_conflict" in w for w in result.warnings)
    assert "axis_conflict" in result.document["joints"]["leftKnee"]["angleFlags"][0]


def test_body_plane_frames_get_no_angle() -> None:
    result = jh.build(inputs(athlete=ai.Athlete(values={})))

    assert result.document["joints"]["leftKnee"]["angleDeg"] == [None] * N
    assert result.document["segments"]["trunk"]["inclinationDeg"] == [None] * N
    assert result.document["frameStatus"]["angleAxis"] == [None] * N
    assert result.derived["angles_frames_by_joint"]["leftKnee"] == 0


def test_off_bar_frames_keep_their_angles_flagged_floor_only() -> None:
    result = jh.build(inputs(hands_on_bar=frozenset(range(10, N))))

    knee = result.document["joints"]["leftKnee"]
    shin = result.document["segments"]["leftShin"]
    for i in range(N):
        floor_only = i < 10
        assert knee["angleDeg"][i] is not None
        assert ("floor_only" in knee["angleFlags"][i]) is floor_only, i
        assert ("floor_only" in shin["flags"][i]) is floor_only, i
    assert result.derived["angles_floor_only"] == 10
    assert any("floor_only" in w for w in result.warnings)


def test_bar_anchored_frames_raise_no_floor_only_warning(built: jh.Built) -> None:
    assert built.derived["angles_floor_only"] == 0
    assert not any("floor_only" in w for w in built.warnings)
