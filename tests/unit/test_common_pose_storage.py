"""Where stored human body model data lands, and the write-once rules it obeys."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from powerflow_pipeline.data.common.errors import PublishError
from powerflow_pipeline.data.common.models import (
    HUMAN_SKELETON,
    SKELETON_ID,
    Frames,
    JointId,
    JointSeries,
    PoseDocument,
)
from powerflow_pipeline.data.common.pose_storage import (
    POSE_FILENAME,
    POSE_OUTPUT_DIR,
    dumps,
    pose_path,
    read_pose,
    skeleton_path,
    write_pose,
    write_skeleton,
)


def _document(count: int = 2) -> PoseDocument:
    series = JointSeries(
        position=tuple((0.5, 1.0 * index, 2.0) for index in range(count)),
        pixel_position=tuple((10 * index, 20) for index in range(count)),
        confidence=tuple(0.8 for _ in range(count)),
    )
    joints: dict[JointId, JointSeries] = {joint: series for joint in HUMAN_SKELETON.joints}
    return PoseDocument(
        capture_id="11 July/30kg_Set1/Side",
        role="side",
        stage="s4_crop",
        frames=Frames(
            count=count,
            fps=60.0,
            start_epoch_ms=1783749250270,
            t_ms=tuple(17 * index for index in range(count)),
        ),
        joints=joints,
    )


# --- path layout -------------------------------------------------------------


def test_pose_path_follows_the_stage_output_layout(tmp_path: Path) -> None:
    path = pose_path(tmp_path, "11 July/30kg_Set1", "Side")
    assert path == tmp_path / "11 July/30kg_Set1" / "Side" / POSE_FILENAME


def test_pose_output_dir_names_the_new_stage() -> None:
    assert POSE_OUTPUT_DIR == "s5_pose_output"


def test_pose_path_accepts_a_session_path_of_any_depth(tmp_path: Path) -> None:
    path = pose_path(tmp_path, "22 August/CnJ/82kgCnJ1", "Front")
    assert path == tmp_path / "22 August/CnJ/82kgCnJ1" / "Front" / POSE_FILENAME


def test_pose_path_rejects_an_absolute_session_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative"):
        pose_path(tmp_path, "/etc", "Side")


def test_pose_path_rejects_a_session_path_that_escapes_the_stage_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escape"):
        pose_path(tmp_path, "11 July/../../elsewhere", "Side")


def test_skeleton_path_uses_the_versioned_id(tmp_path: Path) -> None:
    assert skeleton_path(tmp_path) == tmp_path / f"skeleton.{SKELETON_ID}.json"


# --- writing and reading -----------------------------------------------------


def test_write_pose_creates_the_camera_directory(tmp_path: Path) -> None:
    path = pose_path(tmp_path, "11 July/30kg_Set1", "Side")
    write_pose(path, _document())
    assert path.is_file()


def test_write_pose_round_trips_through_read_pose(tmp_path: Path) -> None:
    path = pose_path(tmp_path, "11 July/30kg_Set1", "Side")
    document = _document()
    write_pose(path, document)
    assert read_pose(path) == document


def test_write_pose_refuses_to_overwrite_an_existing_document(tmp_path: Path) -> None:
    path = pose_path(tmp_path, "11 July/30kg_Set1", "Side")
    write_pose(path, _document())
    with pytest.raises(PublishError, match="already exists"):
        write_pose(path, _document())


def test_write_pose_emits_sorted_reviewable_json(tmp_path: Path) -> None:
    path = pose_path(tmp_path, "11 July/30kg_Set1", "Side")
    write_pose(path, _document())
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert list(payload) == sorted(payload)
    assert payload["skeletonId"] == SKELETON_ID


def test_read_pose_rejects_a_document_that_breaks_an_invariant(tmp_path: Path) -> None:
    path = pose_path(tmp_path, "11 July/30kg_Set1", "Side")
    write_pose(path, _document())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["joints"]["head"]["confidence"] = [0.8]  # now shorter than the frame axis
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        read_pose(path)


# --- skeleton document -------------------------------------------------------


def test_write_skeleton_renders_the_generated_topology(tmp_path: Path) -> None:
    path = write_skeleton(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["id"] == SKELETON_ID
    assert payload["joints"] == list(HUMAN_SKELETON.joints)
    assert len(payload["bones"]) == len(HUMAN_SKELETON.bones)


def test_write_skeleton_refuses_to_overwrite_an_existing_document(tmp_path: Path) -> None:
    write_skeleton(tmp_path)
    with pytest.raises(PublishError, match="already exists"):
        write_skeleton(tmp_path)


# --- document formatting -----------------------------------------------------


def test_dumps_renders_an_empty_object() -> None:
    assert dumps({}) == "{}"


def test_dumps_keeps_a_per_frame_array_on_one_line() -> None:
    assert dumps({"tMs": [0, 17, 33]}) == '{\n  "tMs": [0, 17, 33]\n}'


def test_dumps_keeps_an_array_of_arrays_on_one_line() -> None:
    assert dumps({"pixelPosition": [[1, 2], None, [3, 4]]}) == (
        '{\n  "pixelPosition": [[1, 2], null, [3, 4]]\n}'
    )


def test_dumps_indents_objects_nested_inside_an_array() -> None:
    assert dumps([{"a": 1}, {"b": 2}]) == '[\n  {\n    "a": 1\n  },\n  {\n    "b": 2\n  }\n]'


def test_dumps_sorts_object_keys_for_a_stable_diff() -> None:
    assert dumps({"b": 1, "a": 2}) == '{\n  "a": 2,\n  "b": 1\n}'
