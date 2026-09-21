"""The stored form of the human body model: per-frame pose time series."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from powerflow_pipeline.data.common.models import (
    HUMAN_SKELETON,
    POSE_SCHEMA_VERSION,
    SKELETON_ID,
    Frames,
    JointId,
    JointSeries,
    PoseDocument,
    Position3D,
    skeleton_document,
)

EPOCH_MS = 1783749250270


def _frames(count: int = 3) -> Frames:
    return Frames(
        count=count,
        fps=60.0,
        start_epoch_ms=EPOCH_MS,
        t_ms=tuple(17 * index for index in range(count)),
    )


def _series(count: int = 3) -> JointSeries:
    return JointSeries(
        position=tuple((0.1 * index, 0.2 * index, 0.3 * index) for index in range(count)),
        pixel_position=tuple((index, 2 * index) for index in range(count)),
        confidence=tuple(0.9 for _ in range(count)),
    )


def _pixel_only_series(count: int = 3) -> JointSeries:
    return JointSeries(
        position=None,
        pixel_position=tuple((index, 2 * index) for index in range(count)),
        confidence=tuple(0.9 for _ in range(count)),
    )


def _joints(count: int = 3) -> dict[JointId, JointSeries]:
    return {joint: _series(count) for joint in HUMAN_SKELETON.joints}


def _pixel_only_joints(count: int = 3) -> dict[JointId, JointSeries]:
    return {joint: _pixel_only_series(count) for joint in HUMAN_SKELETON.joints}


def _document(count: int = 3) -> PoseDocument:
    return PoseDocument(
        capture_id="11 July/30kg_Set1/Side",
        role="side",
        stage="s4_crop",
        frames=_frames(count),
        joints=_joints(count),
    )


# --- Frames ------------------------------------------------------------------


def test_frames_rejects_a_t_ms_length_that_disagrees_with_count() -> None:
    with pytest.raises(ValidationError):
        Frames(count=3, fps=60.0, start_epoch_ms=EPOCH_MS, t_ms=(0, 17))


def test_frames_rejects_an_empty_capture() -> None:
    with pytest.raises(ValidationError):
        Frames(count=0, fps=60.0, start_epoch_ms=EPOCH_MS, t_ms=())


def test_frames_rejects_a_non_positive_frame_rate() -> None:
    with pytest.raises(ValidationError):
        Frames(count=1, fps=0.0, start_epoch_ms=EPOCH_MS, t_ms=(0,))


def test_frames_rejects_a_negative_offset() -> None:
    with pytest.raises(ValidationError):
        Frames(count=2, fps=60.0, start_epoch_ms=EPOCH_MS, t_ms=(-1, 0))


def test_frames_rejects_offsets_that_travel_backwards() -> None:
    with pytest.raises(ValidationError):
        Frames(count=3, fps=60.0, start_epoch_ms=EPOCH_MS, t_ms=(0, 33, 17))


def test_frames_is_frozen() -> None:
    frames = _frames()
    with pytest.raises(ValidationError):
        frames.count = 9


# --- JointSeries -------------------------------------------------------------


def test_joint_series_rejects_ragged_arrays() -> None:
    with pytest.raises(ValidationError):
        JointSeries(
            position=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pixel_position=((0, 0),),
            confidence=(0.9, 0.9),
        )


def test_joint_series_rejects_confidence_outside_the_unit_interval() -> None:
    with pytest.raises(ValidationError):
        JointSeries(position=((0.0, 0.0, 0.0),), pixel_position=((0, 0),), confidence=(1.4,))


def test_joint_series_drops_position_and_pixel_position_out_together() -> None:
    with pytest.raises(ValidationError):
        JointSeries(position=(None,), pixel_position=((0, 0),), confidence=(0.0,))


def test_joint_series_requires_zero_confidence_for_an_occluded_frame() -> None:
    with pytest.raises(ValidationError):
        JointSeries(position=(None,), pixel_position=(None,), confidence=(0.8,))


def test_joint_series_accepts_a_fully_occluded_frame() -> None:
    series = JointSeries(position=(None,), pixel_position=(None,), confidence=(0.0,))
    assert series.position_at(0) is None


def test_joint_series_position_at_rebuilds_a_position3d() -> None:
    series = _series()
    assert series.position_at(1) == Position3D(x=0.1, y=0.2, z=0.3)


# --- JointSeries, 2D-only (no metric lift) -----------------------------------


def test_joint_series_accepts_a_pixel_only_series() -> None:
    """A capture run with the 3D lift switched off carries pixels but no positions at all.

    `position=None` is not the same as a tuple of `None`s: the latter says every frame dropped
    out, this says the stage never computed positions. Only the second is true here.
    """

    series = _pixel_only_series()

    assert series.position is None
    assert series.pixel_position == ((0, 0), (1, 2), (2, 4))
    assert len(series) == 3


def test_joint_series_position_at_is_none_throughout_a_pixel_only_series() -> None:
    assert _pixel_only_series().position_at(1) is None


def test_joint_series_still_pairs_pixel_and_confidence_without_positions() -> None:
    with pytest.raises(ValidationError):
        JointSeries(position=None, pixel_position=(None,), confidence=(0.8,))


def test_joint_series_rejects_ragged_arrays_without_positions() -> None:
    with pytest.raises(ValidationError):
        JointSeries(position=None, pixel_position=((0, 0),), confidence=(0.9, 0.9))


def test_joint_series_rejects_confidence_outside_the_unit_interval_without_positions() -> None:
    with pytest.raises(ValidationError):
        JointSeries(position=None, pixel_position=((0, 0),), confidence=(1.4,))


# --- PoseDocument ------------------------------------------------------------


def test_pose_document_requires_every_skeleton_joint() -> None:
    joints = _joints()
    del joints["head"]
    with pytest.raises(ValidationError):
        PoseDocument(
            capture_id="11 July/30kg_Set1/Side",
            role="side",
            stage="s4_crop",
            frames=_frames(),
            joints=joints,
        )


def test_pose_document_rejects_a_series_that_disagrees_with_the_frame_count() -> None:
    joints = _joints()
    joints["leftKnee"] = _series(2)
    with pytest.raises(ValidationError):
        PoseDocument(
            capture_id="11 July/30kg_Set1/Side",
            role="side",
            stage="s4_crop",
            frames=_frames(3),
            joints=joints,
        )


def test_pose_document_rejects_an_unknown_pipeline_stage() -> None:
    with pytest.raises(ValidationError):
        PoseDocument(
            capture_id="11 July/30kg_Set1/Side",
            role="side",
            stage="s9_invent",  # type: ignore[arg-type]
            frames=_frames(),
            joints=_joints(),
        )


def test_pose_document_defaults_to_the_current_schema_and_skeleton_ids() -> None:
    document = _document()
    assert document.schema_version == POSE_SCHEMA_VERSION
    assert document.skeleton_id == SKELETON_ID


def test_pose_document_serialises_the_camel_case_names_the_document_family_uses() -> None:
    payload = _document().to_document()
    assert payload["schemaVersion"] == POSE_SCHEMA_VERSION
    assert payload["skeletonId"] == SKELETON_ID
    assert payload["frames"]["startEpochMs"] == EPOCH_MS
    assert payload["frames"]["tMs"] == [0, 17, 34]
    assert payload["joints"]["leftKnee"]["pixelPosition"] == [[0, 0], [1, 2], [2, 4]]


def test_pose_document_stores_positions_as_compact_triples_not_objects() -> None:
    payload = _document().to_document()
    assert payload["joints"]["head"]["position"][0] == [0.0, 0.0, 0.0]


def test_pose_document_round_trips_through_json() -> None:
    document = _document()
    assert PoseDocument.from_document(json.loads(json.dumps(document.to_document()))) == document


def test_pose_document_is_frozen() -> None:
    document = _document()
    with pytest.raises(ValidationError):
        document.camera = "Front"


def test_pose_document_accepts_a_pixel_only_document() -> None:
    document = PoseDocument(
        capture_id="11 July/30kg_Set1/Side",
        role="side",
        stage="s4_crop",
        frames=_frames(),
        joints=_pixel_only_joints(),
    )

    assert document.has_positions is False


def test_pose_document_reports_positions_when_the_metric_lift_ran() -> None:
    assert _document().has_positions is True


def test_pose_document_rejects_a_mix_of_lifted_and_pixel_only_joints() -> None:
    """Either the metric lift ran for this capture or it did not; per-joint is meaningless."""

    joints = _joints()
    joints["leftKnee"] = _pixel_only_series()

    with pytest.raises(ValidationError):
        PoseDocument(
            capture_id="11 July/30kg_Set1/Side",
            role="side",
            stage="s4_crop",
            frames=_frames(),
            joints=joints,
        )


def test_pose_document_rejects_a_pixel_only_series_disagreeing_with_the_frame_count() -> None:
    joints = _pixel_only_joints(3)
    joints["leftKnee"] = _pixel_only_series(2)

    with pytest.raises(ValidationError):
        PoseDocument(
            capture_id="11 July/30kg_Set1/Side",
            role="side",
            stage="s4_crop",
            frames=_frames(3),
            joints=joints,
        )


def test_pixel_only_document_writes_a_null_position_not_an_array_of_nulls() -> None:
    payload = PoseDocument(
        capture_id="11 July/30kg_Set1/Side",
        role="side",
        stage="s4_crop",
        frames=_frames(),
        joints=_pixel_only_joints(),
    ).to_document()

    assert payload["joints"]["head"]["position"] is None
    assert payload["joints"]["head"]["pixelPosition"] == [[0, 0], [1, 2], [2, 4]]


def test_pixel_only_document_round_trips_through_json() -> None:
    document = PoseDocument(
        capture_id="11 July/30kg_Set1/Side",
        role="side",
        stage="s4_crop",
        frames=_frames(),
        joints=_pixel_only_joints(),
    )

    assert PoseDocument.from_document(json.loads(json.dumps(document.to_document()))) == document


# --- skeleton document -------------------------------------------------------


def test_skeleton_document_is_generated_from_the_human_skeleton_constant() -> None:
    payload = skeleton_document()
    assert payload["id"] == SKELETON_ID
    assert payload["joints"] == list(HUMAN_SKELETON.joints)


def test_skeleton_document_lists_each_bone_as_a_pair_of_joint_names() -> None:
    payload = skeleton_document()
    assert payload["bones"][0] == ["leftAnkle", "leftKnee"]
    assert len(payload["bones"]) == len(HUMAN_SKELETON.bones)


def test_pose_document_names_a_single_camera_capture() -> None:
    """A `Literal["Side", "Front"]` camera name could not name one at all."""

    joints = {joint: _series() for joint in HUMAN_SKELETON.joints}

    document = PoseDocument(
        capture_id="22 August/Snch/110kgSnch1",
        role="single",
        stage="s4_crop",
        frames=_frames(),
        joints=joints,
    )

    assert document.capture_id == "22 August/Snch/110kgSnch1"
    assert document.role == "single"
    assert document.to_document()["captureId"] == "22 August/Snch/110kgSnch1"
