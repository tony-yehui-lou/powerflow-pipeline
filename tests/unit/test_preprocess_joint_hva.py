"""S8's `joint_hva.json`, built from a synthetic lift's 2D pose and checked against its 3D truth."""

from __future__ import annotations

import json

import pytest

from powerflow_pipeline.data.common.models import (
    HUMAN_SKELETON,
    Frames,
    JointId,
    JointSeries,
    PoseDocument,
)
from powerflow_pipeline.data.preprocess import athlete_input as ai
from powerflow_pipeline.data.preprocess import joint_hva as jh
from powerflow_pipeline.data.preprocess.bar_geometry import BarPose, project
from tests.unit.test_preprocess_skeleton_solve import (
    AXIS,
    NEAR,
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
