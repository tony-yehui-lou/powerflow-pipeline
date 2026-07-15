"""S1 Cut: trims every time-indexed stream to the shared epoch interval, or rejects why not."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import av
import cv2
import pytest
import yaml

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.preprocess.models import CameraRecord, CutInterval, Intrinsics
from powerflow_pipeline.data.preprocess.tasks.cut import (
    cut_camera,
    read_lift_window,
    resolve_cut_interval,
)
from powerflow_pipeline.data.preprocess.tasks.discover import discover_sessions
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from tests.conftest import CREATION_TIME, MakeCamera
from tests.unit.test_preprocess_ingest import make_config

CREATED_EPOCH_MS = 1783591455000  # creation_time_to_epoch_ms(CREATION_TIME)


def only_camera(raw: Path) -> Any:
    (camera,) = discover_sessions.fn(raw)
    return camera


def build_record(
    tmp_path: Path, make_camera: MakeCamera, *, camera: str = "Front", **camera_kwargs: Any
) -> CameraRecord:
    raw = tmp_path / "raw"
    make_camera(raw, camera=camera, **camera_kwargs)
    return ingest_camera.fn(only_camera(raw), make_config(tmp_path))


def make_interval(*, start_ms: int, end_ms: int, **overrides: Any) -> CutInterval:
    defaults: dict[str, Any] = {
        "cut_start_epoch_ms": start_ms,
        "cut_end_epoch_ms": end_ms,
        "side_creation_time": CREATION_TIME,
        "side_created_epoch_ms": CREATED_EPOCH_MS,
        "lift_start_time_side_in_ms": start_ms - CREATED_EPOCH_MS,
        "lift_end_time_side_in_ms": end_ms - CREATED_EPOCH_MS,
    }
    defaults.update(overrides)
    return CutInterval(**defaults)


def bare_record(**overrides: Any) -> CameraRecord:
    """A minimal, file-free `CameraRecord` for testing checks that fire before any I/O."""

    defaults: dict[str, Any] = {
        "date": "9 July",
        "session": "cnj_45kg_Set1",
        "camera": "Front",
        "source": Path("/nonexistent"),
        "rgb_width": 64,
        "rgb_height": 48,
        "fps": 60.0,
        "depth_width": 16,
        "depth_height": 12,
        "counts": {"rgb": 4, "depth": 5, "confidence": 5, "odometry": 5, "imu": 8},
        "n_frames": 4,
        "intrinsics": Intrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, frame="landscape"),
        "static_intrinsics": Intrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, frame="landscape"),
        "odometry_intrinsics_drift": 0.0,
        "creation_time": CREATION_TIME,
        "stopwatch_legible": None,
    }
    defaults.update(overrides)
    return CameraRecord(**defaults)


# --- cut_camera: checks that fire before any file I/O -------------------------------------


def test_missing_creation_time_is_rejected(tmp_path: Path) -> None:
    record = bare_record(creation_time=None)
    interval = make_interval(start_ms=CREATED_EPOCH_MS, end_ms=CREATED_EPOCH_MS + 100)

    with pytest.raises(ScanRejected, match="rgb creation time missing for camera Front"):
        cut_camera.fn(record, interval, make_config(tmp_path))


def test_unconvertible_creation_time_is_rejected(tmp_path: Path) -> None:
    record = bare_record(creation_time="not-a-timestamp")
    interval = make_interval(start_ms=CREATED_EPOCH_MS, end_ms=CREATED_EPOCH_MS + 100)

    with pytest.raises(ScanRejected, match="rgb creation time not convertible to epoch ms"):
        cut_camera.fn(record, interval, make_config(tmp_path))


# --- cut_camera: checks that need a real camera directory ---------------------------------


def test_no_rgb_frames_in_interval_is_rejected(tmp_path: Path, make_camera: MakeCamera) -> None:
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9)
    # Odometry/depth index 8 lands at 133ms, but RGB ends at index 7.
    interval = make_interval(start_ms=CREATED_EPOCH_MS + 133, end_ms=CREATED_EPOCH_MS + 133)

    with pytest.raises(ScanRejected, match="no rgb frames in cut interval for camera Front"):
        cut_camera.fn(record, interval, make_config(tmp_path))


def test_empty_odometry_in_interval_is_rejected(tmp_path: Path, make_camera: MakeCamera) -> None:
    # rgb spans well past 100ms; odometry, sampled at 1kHz, ends at ~8ms. An interval in
    # between leaves rgb non-empty but odometry empty.
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9, odometry_hz=1000.0)
    interval = make_interval(start_ms=CREATED_EPOCH_MS + 50, end_ms=CREATED_EPOCH_MS + 100)

    with pytest.raises(ScanRejected, match=r"empty odometry\.csv in cut interval"):
        cut_camera.fn(record, interval, make_config(tmp_path))


def test_empty_imu_in_interval_is_rejected(tmp_path: Path, make_camera: MakeCamera) -> None:
    # imu (3 rows, 1/125s apart) ends at 16ms; rgb+odometry both cover well past that.
    record = build_record(tmp_path, make_camera, rgb_frames=20, depth_frames=21, imu_rows=3)
    interval = make_interval(start_ms=CREATED_EPOCH_MS + 100, end_ms=CREATED_EPOCH_MS + 150)

    with pytest.raises(ScanRejected, match=r"empty imu\.csv in cut interval"):
        cut_camera.fn(record, interval, make_config(tmp_path))


def test_timestamps_unavailable_for_odometry_is_rejected(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9)
    odometry_path = record.source / "odometry.csv"
    lines = odometry_path.read_text().splitlines(keepends=True)
    odometry_path.write_text(lines[0].replace("timestamp", "not_a_timestamp") + "".join(lines[1:]))
    interval = make_interval(start_ms=CREATED_EPOCH_MS, end_ms=CREATED_EPOCH_MS + 100)

    with pytest.raises(ScanRejected, match=r"timestamps unavailable for odometry\.csv"):
        cut_camera.fn(record, interval, make_config(tmp_path))


def test_depth_confidence_mismatch_before_cut_is_rejected(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9)
    next(iter((record.source / "confidence").glob("*.png"))).unlink()
    interval = make_interval(start_ms=CREATED_EPOCH_MS, end_ms=CREATED_EPOCH_MS + 100)

    with pytest.raises(ScanRejected, match="depth/confidence frame count mismatch before cut"):
        cut_camera.fn(record, interval, make_config(tmp_path))


def test_rgb_uses_odometry_timestamps_instead_of_mp4_pts(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    # MP4 PTS would keep indices 2..5, while odometry timestamps keep indices 2..6.
    # Odometry is authoritative, so every frame-paired stream must keep the latter range.
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9, imu_rows=20)
    interval = make_interval(start_ms=CREATED_EPOCH_MS + 33, end_ms=CREATED_EPOCH_MS + 100)

    cut_record, step = cut_camera.fn(record, interval, make_config(tmp_path, dry_run=True))

    assert cut_record.counts.rgb == 5
    assert cut_record.counts.depth == 5
    assert cut_record.counts.confidence == 5
    assert cut_record.counts.odometry == 5
    assert cut_record.n_frames == 5
    assert step.derived["frame_timestamp_source"] == "odometry.csv:timestamp"


def test_cut_rgb_timeline_starts_at_zero_and_follows_odometry(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9, imu_rows=20)
    interval = make_interval(start_ms=CREATED_EPOCH_MS + 33, end_ms=CREATED_EPOCH_MS + 100)

    cut_record, _ = cut_camera.fn(record, interval, make_config(tmp_path))

    times: list[float] = []
    with av.open(str(cut_record.source / "rgb.mp4")) as container:
        for frame in container.decode(container.streams.video[0]):
            assert frame.time is not None
            times.append(float(frame.time))

    assert times == pytest.approx([0.0, 0.017, 0.034, 0.050, 0.067], abs=1e-5)


# --- cut_camera: the happy path, verified numerically --------------------------------------


def test_trims_every_stream_to_the_shared_interval(tmp_path: Path, make_camera: MakeCamera) -> None:
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9, imu_rows=20)
    interval = make_interval(start_ms=CREATED_EPOCH_MS + 33, end_ms=CREATED_EPOCH_MS + 100)

    cut_record, step = cut_camera.fn(record, interval, make_config(tmp_path))

    # Odometry is the authoritative frame clock. Its offsets are
    # [0,17,33,50,67,83,100,117,133], so [33,100] keeps paired indices 2..6.
    assert cut_record.counts.rgb == 5
    assert cut_record.counts.depth == 5
    assert cut_record.counts.confidence == 5
    assert cut_record.counts.odometry == 5
    assert cut_record.n_frames == 5
    assert step.derived["rgb_kept"] == 5
    assert step.derived["depth_kept"] == 5
    assert step.derived["frame_timestamp_source"] == "odometry.csv:timestamp"
    assert step.derived["cut_start_epoch_ms"] == CREATED_EPOCH_MS + 33
    assert step.derived["cut_end_epoch_ms"] == CREATED_EPOCH_MS + 100
    assert {op.op for op in step.file_ops} == {"copy", "write", "publish"}

    destination = cut_record.source
    assert destination == make_config(tmp_path).cut_root / record.relative

    with av.open(str(destination / "rgb.mp4")) as container:
        decoded = sum(1 for _ in container.decode(container.streams.video[0]))
    assert decoded == 5

    depth_files = sorted((destination / "depth").glob("*.png"))
    confidence_files = sorted((destination / "confidence").glob("*.png"))
    assert [path.name for path in depth_files] == [f"{i:06d}.png" for i in range(5)]
    assert [path.name for path in confidence_files] == [f"{i:06d}.png" for i in range(5)]
    # Depth values survive a lossless copy untouched.
    depth_array = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
    assert depth_array is not None
    assert depth_array.dtype.name == "uint16"

    with (destination / "odometry.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    assert [row["frame"] for row in rows] == [f"{i:06d}" for i in range(2, 7)]

    # camera_matrix.csv is passed through byte-for-byte, never rewritten.
    source_hash = hashlib.sha256((record.source / "camera_matrix.csv").read_bytes()).hexdigest()
    dest_hash = hashlib.sha256((destination / "camera_matrix.csv").read_bytes()).hexdigest()
    assert source_hash == dest_hash

    sidecar = yaml.safe_load((destination / "cut_sidecar.json").read_text())
    assert sidecar["cut_start_epoch_ms"] == CREATED_EPOCH_MS + 33
    assert sidecar["frame_timestamp_source"] == "odometry.csv:timestamp"
    assert sidecar["retained"]["depth"]["count"] == 5
    assert sidecar["retained"]["imu"]["count"] > 0

    assert not list(destination.parent.rglob(".powerflow-*"))


def test_dry_run_writes_nothing(tmp_path: Path, make_camera: MakeCamera) -> None:
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9, imu_rows=20)
    interval = make_interval(start_ms=CREATED_EPOCH_MS + 33, end_ms=CREATED_EPOCH_MS + 100)
    config = make_config(tmp_path, dry_run=True)

    cut_record, step = cut_camera.fn(record, interval, config)

    assert cut_record.n_frames == 5
    assert step.file_ops
    assert not config.cut_root.exists()


def test_write_once_refuses_to_overwrite(tmp_path: Path, make_camera: MakeCamera) -> None:
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9, imu_rows=20)
    interval = make_interval(start_ms=CREATED_EPOCH_MS + 33, end_ms=CREATED_EPOCH_MS + 100)
    config = make_config(tmp_path)

    cut_camera.fn(record, interval, config)
    with pytest.raises(Exception, match="already exists"):
        cut_camera.fn(record, interval, config)


def test_overwrite_replaces_the_published_camera(tmp_path: Path, make_camera: MakeCamera) -> None:
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9, imu_rows=20)
    interval = make_interval(start_ms=CREATED_EPOCH_MS + 33, end_ms=CREATED_EPOCH_MS + 100)
    config = make_config(tmp_path)

    cut_camera.fn(record, interval, config)
    cut_record, _ = cut_camera.fn(record, interval, make_config(tmp_path, overwrite=True))
    assert cut_record.n_frames == 5


# --- read_lift_window ------------------------------------------------------------------


def test_read_lift_window_parses_valid_values(tmp_path: Path, make_session_metadata: Any) -> None:
    path = make_session_metadata(tmp_path, lift_start_ms=30090, lift_end_ms=90780)
    assert read_lift_window(path) == (30090, 90780)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"omit": ["lift_start_time_side_in_ms"]}, "missing lift window field"),
        ({"lift_start_ms": "abc"}, "non-numeric lift window field"),
        ({"lift_start_ms": -10}, "negative lift window field"),
        ({"lift_start_ms": 900, "lift_end_ms": 100}, "lift window out of order"),
        ({"body": "not: [valid, yaml: at all"}, "missing lift window field"),
    ],
)
def test_read_lift_window_rejects(
    tmp_path: Path, make_session_metadata: Any, kwargs: dict[str, Any], match: str
) -> None:
    if "body" in kwargs:
        with pytest.raises(yaml.YAMLError):
            path = make_session_metadata(tmp_path, **kwargs)
            read_lift_window(path)
        return
    path = make_session_metadata(tmp_path, **kwargs)
    with pytest.raises(ScanRejected, match=match):
        read_lift_window(path)


def test_read_lift_window_missing_file_reads_as_missing_field(tmp_path: Path) -> None:
    with pytest.raises(ScanRejected, match="missing lift window field"):
        read_lift_window(tmp_path / "9 July" / "cnj_45kg_Set1" / "metadata.yaml")


# --- resolve_cut_interval ----------------------------------------------------------------


def test_resolve_cut_interval_derives_the_shared_bounds(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, camera="Side", rgb_frames=5, depth_frames=6, odometry_hz=5.0)
    make_session_metadata(raw, lift_start_ms=100, lift_end_ms=900)
    side_record = ingest_camera.fn(only_camera(raw), make_config(tmp_path))

    interval = resolve_cut_interval.fn(raw, "9 July", "cnj_45kg_Set1", side_record)

    assert interval.cut_start_epoch_ms == CREATED_EPOCH_MS + 100
    assert interval.cut_end_epoch_ms == CREATED_EPOCH_MS + 900
    assert interval.side_creation_time == CREATION_TIME
    assert interval.lift_start_time_side_in_ms == 100
    assert interval.lift_end_time_side_in_ms == 900


def test_resolve_cut_interval_rejects_side_creation_time_missing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "9 July" / "cnj_45kg_Set1").mkdir(parents=True)
    (raw / "9 July" / "cnj_45kg_Set1" / "metadata.yaml").write_text(
        "lift:\n  lift_start_time_side_in_ms: 100\n  lift_end_time_side_in_ms: 900\n"
    )
    side_record = bare_record(creation_time=None)

    with pytest.raises(ScanRejected, match="rgb creation time missing"):
        resolve_cut_interval.fn(raw, "9 July", "cnj_45kg_Set1", side_record)


def test_resolve_cut_interval_rejects_interval_outside_capture(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, camera="Side", rgb_frames=4, depth_frames=5)
    make_session_metadata(raw, lift_start_ms=100_000, lift_end_ms=200_000)
    side_record = ingest_camera.fn(only_camera(raw), make_config(tmp_path))

    with pytest.raises(ScanRejected, match="cut interval outside side capture"):
        resolve_cut_interval.fn(raw, "9 July", "cnj_45kg_Set1", side_record)
