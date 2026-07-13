"""S0 gates S1: every rejection carries an exact, actionable reason."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraDir
from powerflow_pipeline.data.preprocess.tasks.discover import discover_sessions
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from tests.conftest import ODOMETRY_CX, ODOMETRY_CY, STATIC_FX, MakeCamera


def make_config(tmp_path: Path, **overrides: Any) -> PreprocessConfig:
    return PreprocessConfig(
        raw_root=tmp_path / "raw",
        record_root=tmp_path / "s0",
        output_root=tmp_path / "s1",
        **overrides,
    )


def only_camera(raw: Path) -> CameraDir:
    (camera,) = discover_sessions.fn(raw)
    return camera


def reason_for(tmp_path: Path, make_camera: MakeCamera, **camera_kwargs: Any) -> str:
    """Build a camera, ingest it, and return the rejection reason it produced."""

    raw = tmp_path / "raw"
    make_camera(raw, **camera_kwargs)
    config = make_config(tmp_path, require_stopwatch_attestation=False)

    with pytest.raises(ScanRejected) as rejection:
        ingest_camera.fn(only_camera(raw), config)
    return str(rejection.value)


# --- the happy path ------------------------------------------------------------------


def test_a_valid_camera_produces_a_record(tmp_path: Path, make_camera: MakeCamera) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, rgb_frames=4, depth_frames=5)

    record = ingest_camera.fn(only_camera(raw), make_config(tmp_path))

    assert record.camera_id == "9 July/cnj_45kg_Set1/Front"
    assert (record.rgb_width, record.rgb_height) == (64, 48)
    assert (record.depth_width, record.depth_height) == (16, 12)
    assert record.counts.model_dump() == {
        "rgb": 4,
        "depth": 5,
        "confidence": 5,
        "odometry": 5,
        "imu": 8,
    }
    assert record.creation_time is not None


def test_n_frames_is_the_rgb_count_the_excess_depth_frame_is_dropped(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    """ISSUE-03: RGB is systematically one frame short; the trailing depth frame is excess."""

    raw = tmp_path / "raw"
    make_camera(raw, rgb_frames=4, depth_frames=5)

    record = ingest_camera.fn(only_camera(raw), make_config(tmp_path))

    assert record.n_frames == 4
    assert record.counts.depth == 5


def test_intrinsics_are_the_average_of_the_per_frame_odometry_values(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    """ISSUE-02: odometry's drifting per-frame K is authoritative, averaged, not the static file."""

    raw = tmp_path / "raw"
    make_camera(raw, rgb_frames=4, depth_frames=4)  # fx alternates 44.0, 46.0

    record = ingest_camera.fn(only_camera(raw), make_config(tmp_path))

    assert record.intrinsics.fx == pytest.approx(45.0)
    assert record.intrinsics.fy == pytest.approx(45.0)
    assert record.intrinsics.cx == pytest.approx(ODOMETRY_CX)
    assert record.intrinsics.cy == pytest.approx(ODOMETRY_CY)
    assert record.intrinsics.frame == "landscape"  # ISSUE-01
    assert record.intrinsics.distortion is None  # the capture app exports none

    # The static matrix is kept only as provenance; the drift is why it is not trusted.
    assert record.static_intrinsics.fx == pytest.approx(STATIC_FX)
    assert record.odometry_intrinsics_drift == pytest.approx(STATIC_FX - 44.0)


def test_rgb_short_by_one_is_accepted(tmp_path: Path, make_camera: MakeCamera) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, rgb_frames=4, depth_frames=5)

    record = ingest_camera.fn(only_camera(raw), make_config(tmp_path))

    assert record.n_frames == 4


def test_rgb_short_by_two_is_rejected(tmp_path: Path, make_camera: MakeCamera) -> None:
    reason = reason_for(tmp_path, make_camera, rgb_frames=4, depth_frames=6)

    assert reason == "rgb frame count outside tolerance: rgb=4 depth=6 tolerance=1"


def test_the_record_is_written_to_the_record_root(tmp_path: Path, make_camera: MakeCamera) -> None:
    raw = tmp_path / "raw"
    make_camera(raw)
    config = make_config(tmp_path)

    record = ingest_camera.fn(only_camera(raw), config)

    path = config.record_root / "9 July" / "cnj_45kg_Set1" / "Front" / "record.json"
    assert json.loads(path.read_text())["n_frames"] == record.n_frames


def test_a_dry_run_writes_no_record(tmp_path: Path, make_camera: MakeCamera) -> None:
    raw = tmp_path / "raw"
    make_camera(raw)
    config = make_config(tmp_path, dry_run=True)

    ingest_camera.fn(only_camera(raw), config)

    assert not config.record_root.exists()


# --- V1..V11: one test per rule, asserting the exact reason string ---------------------


@pytest.mark.parametrize(
    ("omit", "name"),
    [
        ("rgb", "rgb.mp4"),
        ("depth", "depth"),
        ("confidence", "confidence"),
        ("camera_matrix", "camera_matrix.csv"),
        ("odometry", "odometry.csv"),
        ("imu", "imu.csv"),
    ],
)
def test_v1_missing_stream(tmp_path: Path, make_camera: MakeCamera, omit: str, name: str) -> None:
    assert reason_for(tmp_path, make_camera, omit=[omit]) == f"missing required stream: {name}"


@pytest.mark.parametrize("stream", ["depth", "confidence"])
def test_v2_empty_stream(tmp_path: Path, make_camera: MakeCamera, stream: str) -> None:
    counts = {"depth_frames": 0} if stream == "depth" else {"confidence_frames": 0}

    assert reason_for(tmp_path, make_camera, **counts) == f"empty required stream: {stream}"


def test_v3_depth_confidence_count_mismatch(tmp_path: Path, make_camera: MakeCamera) -> None:
    reason = reason_for(tmp_path, make_camera, depth_frames=5, confidence_frames=4)

    assert reason == "depth/confidence frame count mismatch: 5 vs 4"


def test_v4_depth_confidence_resolution_mismatch(tmp_path: Path, make_camera: MakeCamera) -> None:
    reason = reason_for(tmp_path, make_camera, confidence_size=(32, 24))

    assert reason == "depth/confidence resolution mismatch: (12, 16) vs (24, 32)"


def test_v5_odometry_depth_count_mismatch(tmp_path: Path, make_camera: MakeCamera) -> None:
    reason = reason_for(tmp_path, make_camera, depth_frames=5, odometry_rows=4)

    assert reason == "odometry/depth frame count mismatch: 4 vs 5"


def test_v6_rgb_longer_than_depth_is_rejected(tmp_path: Path, make_camera: MakeCamera) -> None:
    """The tolerance is one-sided: depth may lead RGB, never trail it."""

    reason = reason_for(tmp_path, make_camera, rgb_frames=6, depth_frames=5)

    assert reason == "rgb frame count outside tolerance: rgb=6 depth=5 tolerance=1"


@pytest.mark.parametrize(
    "matrix",
    ["not,a,matrix", "46.7411, 0.0, 32.279\n0.0, 46.7411, 23.9895", ""],
    ids=["unparseable", "not-3x3", "empty"],
)
def test_v7_malformed_camera_matrix(tmp_path: Path, make_camera: MakeCamera, matrix: str) -> None:
    reason = reason_for(tmp_path, make_camera, camera_matrix=matrix)

    assert reason == "intrinsics absent or malformed"


def test_v8_non_positive_focal_length(tmp_path: Path, make_camera: MakeCamera) -> None:
    reason = reason_for(
        tmp_path, make_camera, camera_matrix="0.0, 0.0, 32.279\n0.0, 0.0, 23.9895\n0,0,1"
    )

    assert reason == "intrinsics absent or malformed"


def test_v9_confidence_outside_the_reported_set(tmp_path: Path, make_camera: MakeCamera) -> None:
    reason = reason_for(tmp_path, make_camera, confidence_values=(0, 7))

    assert reason == "confidence values outside {0,1,2}: [7]"


def test_v10_unexpected_depth_dtype(tmp_path: Path, make_camera: MakeCamera) -> None:
    reason = reason_for(tmp_path, make_camera, depth_dtype=np.uint8)

    assert reason == "unexpected depth dtype: uint8"


def test_v11_stopwatch_not_attested(tmp_path: Path, make_camera: MakeCamera) -> None:
    raw = tmp_path / "raw"
    make_camera(raw)
    config = make_config(tmp_path, require_stopwatch_attestation=True)

    with pytest.raises(ScanRejected) as rejection:
        ingest_camera.fn(only_camera(raw), config, stopwatch_legible=None)

    assert str(rejection.value) == "stopwatch not attested legible for camera Front"


def test_v11_passes_when_attested(tmp_path: Path, make_camera: MakeCamera) -> None:
    raw = tmp_path / "raw"
    make_camera(raw)
    config = make_config(tmp_path, require_stopwatch_attestation=True)

    record = ingest_camera.fn(only_camera(raw), config, stopwatch_legible=True)

    assert record.stopwatch_legible is True


def test_non_contiguous_depth_indices_are_rejected(tmp_path: Path, make_camera: MakeCamera) -> None:
    """A gap in the numbering means frame k is not frame k; the index join would silently lie."""

    reason = reason_for(tmp_path, make_camera, depth_frames=5, skip_depth_index=2)

    assert reason == "non-contiguous depth frame indices"
