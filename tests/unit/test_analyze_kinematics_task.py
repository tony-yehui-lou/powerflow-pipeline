"""S8 · Kinematics as a task: reads every input off another stage's published output, writes
`joint_hva.json`, and never rejects a capture. The geometry is `test_analyze_skeleton_solve`'s
and `test_analyze_joint_hva`'s; this covers the plumbing on the synthetic S0-S4 capture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from powerflow_pipeline.data.analyze.config import AnalyzeConfig
from powerflow_pipeline.data.analyze.tasks.kinematics import (
    kinematics_path,
    solve_kinematics_camera,
)
from powerflow_pipeline.data.common.capture_models import CameraRecord
from powerflow_pipeline.data.common.errors import PublishError
from powerflow_pipeline.data.common.models import HUMAN_SKELETON, Frames, JointSeries, PoseDocument
from powerflow_pipeline.data.common.pose_storage import pose_path, write_pose, write_sidecar
from tests.conftest import MakeCamera, MakeSessionMetadata
from tests.unit.test_detect_pose import (
    _bar_scene,
    _build_crop_record,
    _declare_bar,
    _make_pose_session,
    _write_detections,
)
from tests.unit.test_detect_pose import (
    make_config as _detect_make_config,
)


def make_config(tmp_path: Path, **overrides: Any) -> AnalyzeConfig:
    """An analyze run over the trees `_build_crop_record` and S6 publish into."""

    detect = _detect_make_config(tmp_path)
    assert detect.pose_root is not None
    return AnalyzeConfig(
        record_root=detect.record_root,
        cut_root=detect.cut_root,
        retilt_root=detect.retilt_root,
        crop_root=detect.crop_root,
        pose_root=detect.pose_root,
        **overrides,
    )


ATHLETE = (
    "athlete:\n"
    "  height_in_cm: 175.0\n"
    "  tibia_in_cm: 42.0\n"
    "  femur_in_cm: 43.0\n"
    "  torso_in_cm: 50.0\n"
    "  shoulder_width_in_cm: 38.0\n"
)


@pytest.fixture
def capture(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> tuple[AnalyzeConfig, CameraRecord]:
    raw = _make_pose_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(
        tmp_path,
        barbell_root=tmp_path / "s5",
        kinematics_root=tmp_path / "s8_kinematics_output",
    )
    return config, _build_crop_record(raw, config)


def _publish_pose(config: AnalyzeConfig, record: CameraRecord) -> Path:
    """A 2D pose for `record`: every joint still, near the bar scene's hands."""

    assert config.pose_root is not None
    scene = _bar_scene(record, record.n_frames)
    near = scene["near"][0]
    pixels = tuple((round(near[0]), round(near[1])) for _ in range(record.n_frames))
    series = JointSeries(position=None, pixel_position=pixels, confidence=(0.9,) * len(pixels))
    document = PoseDocument(
        capture_id=record.capture_id,
        role=record.role,
        stage="s4_crop",
        frames=Frames(
            count=record.n_frames,
            fps=record.fps,
            start_epoch_ms=0,
            t_ms=tuple(17 * i for i in range(record.n_frames)),
        ),
        joints={joint: series for joint in HUMAN_SKELETON.joints},
    )
    path = pose_path(config.pose_root, record.relative.parent, record.relative.name)
    write_pose(path, document)
    return path


def _add_athlete(record: CameraRecord) -> None:
    text = record.metadata_path.read_text()
    record.metadata_path.write_text(text.rstrip("\n").removesuffix("...") + "\n" + ATHLETE)


def test_dry_run_plans_the_write_and_writes_nothing(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture
    config = config.model_copy(update={"dry_run": True})

    _, step = solve_kinematics_camera.fn(record, config)

    assert config.kinematics_root is not None
    assert [op.op for op in step.file_ops] == ["write"]
    assert not kinematics_path(config.kinematics_root, record).exists()


def test_without_an_s6_pose_it_warns_and_writes_nothing(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture

    _, step = solve_kinematics_camera.fn(record, config)

    assert any("no S6 pose" in w for w in step.warnings)
    assert config.kinematics_root is not None
    assert not kinematics_path(config.kinematics_root, record).exists()


def test_publishes_joint_hva_with_every_joint_and_reports_to_the_manifest(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture
    _publish_pose(config, record)
    _add_athlete(record)

    _, step = solve_kinematics_camera.fn(record, config)

    assert config.kinematics_root is not None
    destination = kinematics_path(config.kinematics_root, record)
    document = json.loads(destination.read_text())
    assert document["captureId"] == record.capture_id
    assert set(document["joints"]) == set(HUMAN_SKELETON.joints)
    assert document["bones"]["leftTibia"]["source"] == "measured"
    assert document["bones"]["leftUpperArm"]["source"] == "population"
    assert document["floor"]["anchor"] == "floor-plane"  # S3's; no bar at rest here
    assert step.derived["kinematics_method"] == "constrained"
    # No barbell detections were written: the floor alone anchors, and the manifest says so.
    assert any("no barbell detections" in w for w in step.warnings)


def test_reads_the_bar_and_the_grip_when_s5_and_s6_published_them(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture
    pose = _publish_pose(config, record)
    _add_athlete(record)
    _declare_bar(record)
    _write_detections(config, record, _bar_scene(record, record.n_frames))
    write_sidecar(
        pose.parent / "wrist_refinement.json",
        {"near": "rightWrist", "halfGripMm": 460.0, "farCorrected": list(range(record.n_frames))},
    )

    _, step = solve_kinematics_camera.fn(record, config)

    assert not any("barbell" in w for w in step.warnings)
    assert not any("wrist_refinement" in w for w in step.warnings)


def test_an_undeclared_bar_is_not_guessed(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture
    _publish_pose(config, record)
    _write_detections(config, record, _bar_scene(record, record.n_frames))

    _, step = solve_kinematics_camera.fn(record, config)

    assert any("barbell type" in w for w in step.warnings)
    assert any("wrist_refinement" in w for w in step.warnings)


def test_without_barbell_root_the_floor_alone_anchors(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture
    config = config.model_copy(update={"barbell_root": None})
    _publish_pose(config, record)

    _, step = solve_kinematics_camera.fn(record, config)

    assert any("no --barbell" in w for w in step.warnings)


def test_refuses_to_overwrite_unless_asked(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture
    _publish_pose(config, record)
    solve_kinematics_camera.fn(record, config)

    with pytest.raises(PublishError):
        solve_kinematics_camera.fn(record, config)
    solve_kinematics_camera.fn(record, config.model_copy(update={"overwrite": True}))
