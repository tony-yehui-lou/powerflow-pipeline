"""Contracts shared by every pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from powerflow_pipeline.data.common.context import OutputMode, RunContext
from powerflow_pipeline.data.common.models import (
    HUMAN_SKELETON,
    Bone,
    CropBounds,
    FileOp,
    Joint,
    Position3D,
    Scan,
    Skeleton,
    StepResult,
)


def test_crop_bounds_exposes_width_and_height() -> None:
    bounds = CropBounds(x0=32, y0=32, x1=1888, y1=1048)
    assert bounds.width == 1856
    assert bounds.height == 1016


@pytest.mark.parametrize(
    ("x0", "y0", "x1", "y1"),
    [
        (32, 32, 32, 1048),  # collapsed in x
        (32, 32, 1888, 32),  # collapsed in y
        (100, 32, 50, 1048),  # inverted in x
        (32, 100, 1888, 50),  # inverted in y
    ],
)
def test_crop_bounds_rejects_non_positive_extent(x0: int, y0: int, x1: int, y1: int) -> None:
    with pytest.raises(ValidationError):
        CropBounds(x0=x0, y0=y0, x1=x1, y1=y1)


def test_crop_bounds_rejects_negative_origin() -> None:
    with pytest.raises(ValidationError):
        CropBounds(x0=-1, y0=0, x1=100, y1=100)


def test_scan_is_frozen() -> None:
    scan = Scan(scan_id="scan_0001", source=Path("/tmp/scan_0001"), files=(Path("meta.json"),))
    with pytest.raises(ValidationError):
        scan.scan_id = "other"


def test_step_result_defaults_are_empty() -> None:
    result = StepResult()
    assert result.derived == {}
    assert result.warnings == []
    assert result.file_ops == []


def test_file_op_round_trips_through_json() -> None:
    op = FileOp(op="publish", src=Path("/tmp/staging/a"), dst=Path("/out/a"))
    assert FileOp.model_validate_json(op.model_dump_json()) == op


def test_publish_context_requires_an_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_root"):
        RunContext.create(
            pipeline="preprocess",
            input_root=tmp_path,
            output_root=None,
            output_mode=OutputMode.PUBLISH,
        )


def test_in_place_context_rejects_an_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not"):
        RunContext.create(
            pipeline="preprocess",
            input_root=tmp_path,
            output_root=tmp_path / "output",
            output_mode=OutputMode.IN_PLACE,
        )


def test_context_rejects_a_missing_input_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        RunContext.create(
            pipeline="preprocess",
            input_root=tmp_path / "absent",
            output_root=tmp_path / "output",
            output_mode=OutputMode.PUBLISH,
        )


def test_position3d_round_trips_through_json() -> None:
    position = Position3D(x=1.0, y=-2.5, z=0.0)
    assert Position3D.model_validate_json(position.model_dump_json()) == position


def test_position3d_is_frozen() -> None:
    position = Position3D(x=0.0, y=0.0, z=0.0)
    with pytest.raises(ValidationError):
        position.x = 1.0


def test_joint_holds_an_id_and_a_3d_position() -> None:
    joint = Joint(id="leftKnee", position=Position3D(x=0.1, y=0.2, z=0.3))
    assert joint.id == "leftKnee"
    assert joint.position == Position3D(x=0.1, y=0.2, z=0.3)


def test_joint_rejects_an_unknown_id() -> None:
    with pytest.raises(ValidationError):
        Joint(id="leftPinky", position=Position3D(x=0.0, y=0.0, z=0.0))  # type: ignore[arg-type]


def test_joint_is_frozen() -> None:
    joint = Joint(id="head", position=Position3D(x=0.0, y=0.0, z=0.0))
    with pytest.raises(ValidationError):
        joint.id = "leftAnkle"  # type: ignore[assignment]


def test_bone_rejects_a_self_loop() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        Bone(from_joint="head", to_joint="head")


def test_bone_is_frozen() -> None:
    bone = Bone(from_joint="leftClavicle", to_joint="head")
    with pytest.raises(ValidationError):
        bone.to_joint = "rightClavicle"  # type: ignore[assignment]


def test_skeleton_rejects_a_bone_referencing_an_unlisted_joint() -> None:
    with pytest.raises(ValidationError, match="not in joints"):
        Skeleton(
            joints=("head", "leftClavicle"),
            bones=(Bone(from_joint="leftClavicle", to_joint="rightClavicle"),),
        )


def test_skeleton_rejects_duplicate_joints() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        Skeleton(joints=("head", "head"), bones=())


def test_skeleton_is_frozen() -> None:
    skeleton = Skeleton(joints=("head",), bones=())
    with pytest.raises(ValidationError):
        skeleton.joints = ("head", "leftAnkle")  # type: ignore[assignment]


def test_human_skeleton_has_fifteen_joints_and_fifteen_bones() -> None:
    assert len(HUMAN_SKELETON.joints) == 15
    assert len(HUMAN_SKELETON.bones) == 15
    assert len(set(HUMAN_SKELETON.joints)) == 15  # no duplicates


def test_human_skeleton_contains_head_and_the_seven_paired_joints() -> None:
    assert set(HUMAN_SKELETON.joints) == {
        "leftAnkle",
        "rightAnkle",
        "leftKnee",
        "rightKnee",
        "leftHip",
        "rightHip",
        "leftShoulder",
        "rightShoulder",
        "leftElbow",
        "rightElbow",
        "leftWrist",
        "rightWrist",
        "leftClavicle",
        "rightClavicle",
        "head",
    }


def test_human_skeleton_hip_bone_crosses_the_body_and_is_not_mirrored() -> None:
    pairs = {(bone.from_joint, bone.to_joint) for bone in HUMAN_SKELETON.bones}
    assert ("leftHip", "rightHip") in pairs
    assert ("rightHip", "leftHip") not in pairs


def test_human_skeleton_both_clavicles_connect_to_the_single_head_joint() -> None:
    pairs = {(bone.from_joint, bone.to_joint) for bone in HUMAN_SKELETON.bones}
    assert ("leftClavicle", "head") in pairs
    assert ("rightClavicle", "head") in pairs


def test_human_skeleton_round_trips_through_json() -> None:
    assert Skeleton.model_validate_json(HUMAN_SKELETON.model_dump_json()) == HUMAN_SKELETON


def test_destination_for_follows_the_output_mode(tmp_path: Path) -> None:
    published = RunContext.create(
        pipeline="preprocess",
        input_root=tmp_path,
        output_root=tmp_path / "output",
        output_mode=OutputMode.PUBLISH,
    )
    assert published.destination_for("scan_0001") == (tmp_path / "output" / "scan_0001").resolve()

    in_place = RunContext.create(
        pipeline="preprocess",
        input_root=tmp_path,
        output_root=None,
        output_mode=OutputMode.IN_PLACE,
    )
    assert in_place.destination_for("scan_0001") == (tmp_path / "scan_0001").resolve()
