"""S5 · Pose detection -> runs an injected model on S4's cropped RGB and publishes one
`PoseDocument` per camera. The model itself (issue #118) is stubbed here; this covers the
plumbing that connects S4's output to `pose_storage.py`'s stage-output layout (issue #117).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from powerflow_pipeline.data.common.errors import PublishError
from powerflow_pipeline.data.common.models import HUMAN_SKELETON, JointId
from powerflow_pipeline.data.common.pose_storage import pose_path, read_pose
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.pose_model import JointPixelSeries
from powerflow_pipeline.data.preprocess.tasks.crop import crop_camera
from powerflow_pipeline.data.preprocess.tasks.cut import cut_camera, resolve_cut_interval
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from powerflow_pipeline.data.preprocess.tasks.orient import orient_camera
from powerflow_pipeline.data.preprocess.tasks.pose import (
    detect_pose_camera,
    ensure_skeleton_published,
)
from powerflow_pipeline.data.preprocess.tasks.retilt import retilt_camera
from tests.conftest import MakeCamera, MakeSessionMetadata, sole_capture
from tests.unit.test_preprocess_ingest import make_config as _base_make_config


def make_config(tmp_path: Path, **overrides: object) -> PreprocessConfig:
    # Sized for the synthetic fixture's tiny frame, same as the crop tests.
    overrides.setdefault("crop_safety_px", 1)
    overrides.setdefault("pose_root", tmp_path / "s5_pose_output")
    return _base_make_config(tmp_path, **overrides)


class _StubModel:
    """A fixed, valid 2D detection for every frame -- stands in for a real `Detector2D`."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, int]] = []

    def detect(self, rgb_path: Path, n_frames: int) -> dict[JointId, JointPixelSeries]:
        self.calls.append((rgb_path, n_frames))
        series = JointPixelSeries(
            pixel_position=tuple((0, 0) for _ in range(n_frames)),
            confidence=tuple(0.9 for _ in range(n_frames)),
        )
        return {joint: series for joint in HUMAN_SKELETON.joints}


def _make_pose_session(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> Path:
    raw = tmp_path / "raw"
    make_camera(
        raw,
        camera="Side",
        rgb_frames=5,
        depth_frames=6,
        render_floor_plane=True,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    make_session_metadata(raw, lift_start_ms=16, lift_end_ms=80, floor_regions="full_frame")
    return raw


def _build_crop_record(
    raw_root: Path, config: PreprocessConfig, camera: str = "Side"
) -> CameraRecord:
    """Run S0 -> S1 -> S2 -> S3 -> S4 on the one synthetic camera named `camera`."""

    camera_dir = sole_capture(raw_root, camera)
    ingested = ingest_camera.fn(camera_dir, config)
    interval = resolve_cut_interval.fn(camera_dir.metadata_path, ingested)
    cut_record, _ = cut_camera.fn(ingested, interval, config)
    orient_record, _ = orient_camera.fn(cut_record, config)
    retilt_record, _ = retilt_camera.fn(orient_record, config)
    crop_record, _ = crop_camera.fn(retilt_record, config)
    return crop_record


# --- the happy path ------------------------------------------------------------------


def test_detect_pose_camera_publishes_a_pose_document(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_pose_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    crop_record = _build_crop_record(raw, config)
    model = _StubModel()

    record, step = detect_pose_camera.fn(crop_record, config, model)

    assert record == crop_record  # the task doesn't mutate the record
    destination = pose_path(
        config.pose_root, crop_record.relative.parent, crop_record.relative.name
    )
    assert destination.is_file()
    document = read_pose(destination)
    assert document.capture_id == "9 July/cnj_45kg_Set1/Side"
    assert document.role == "side"
    assert document.stage == "s4_crop"
    assert document.frames.count == crop_record.n_frames
    assert step.derived["n_frames"] == crop_record.n_frames


def test_detect_pose_camera_calls_the_model_with_the_s4_rgb(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_pose_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    crop_record = _build_crop_record(raw, config)
    model = _StubModel()

    detect_pose_camera.fn(crop_record, config, model)

    [(rgb_path, n_frames)] = model.calls
    assert rgb_path == crop_record.source / "rgb.mp4"
    assert n_frames == crop_record.n_frames


def test_detect_pose_camera_orders_frame_offsets_from_zero(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_pose_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    crop_record = _build_crop_record(raw, config)

    _, _ = detect_pose_camera.fn(crop_record, config, _StubModel())

    destination = pose_path(
        config.pose_root, crop_record.relative.parent, crop_record.relative.name
    )
    document = read_pose(destination)
    assert document.frames.t_ms[0] == 0
    assert list(document.frames.t_ms) == sorted(document.frames.t_ms)


# --- dry run ---------------------------------------------------------------------------


def test_detect_pose_camera_dry_run_writes_nothing(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_pose_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10, dry_run=True)
    crop_record = _build_crop_record(raw, config)
    model = _StubModel()

    _, step = detect_pose_camera.fn(crop_record, config, model)

    assert not model.calls
    assert not config.pose_root.exists()
    assert step.file_ops


# --- write-once --------------------------------------------------------------------------


def test_detect_pose_camera_refuses_to_overwrite_by_default(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_pose_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    crop_record = _build_crop_record(raw, config)
    detect_pose_camera.fn(crop_record, config, _StubModel())

    with pytest.raises(PublishError, match="already exists"):
        detect_pose_camera.fn(crop_record, config, _StubModel())


def test_detect_pose_camera_overwrites_when_configured(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_pose_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10, overwrite=True)
    crop_record = _build_crop_record(raw, config)
    detect_pose_camera.fn(crop_record, config, _StubModel())

    detect_pose_camera.fn(crop_record, config, _StubModel())  # does not raise


# --- skeleton --------------------------------------------------------------------------


def test_ensure_skeleton_published_writes_once(tmp_path: Path) -> None:
    pose_root = tmp_path / "s5"
    path = ensure_skeleton_published(pose_root, overwrite=False)
    assert path.is_file()


def test_ensure_skeleton_published_is_idempotent_without_overwrite(tmp_path: Path) -> None:
    pose_root = tmp_path / "s5"
    first = ensure_skeleton_published(pose_root, overwrite=False)
    second = ensure_skeleton_published(pose_root, overwrite=False)  # does not raise
    assert first == second


def test_ensure_skeleton_published_rewrites_when_overwrite_is_set(tmp_path: Path) -> None:
    pose_root = tmp_path / "s5"
    ensure_skeleton_published(pose_root, overwrite=False)
    ensure_skeleton_published(pose_root, overwrite=True)  # does not raise
