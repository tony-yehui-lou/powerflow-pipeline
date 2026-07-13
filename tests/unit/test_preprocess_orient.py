"""S1 establishes I1: portrait pixels, in the frame the emitted intrinsics describe."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from powerflow_pipeline.data.preprocess.config import RotationDirection
from powerflow_pipeline.data.preprocess.tasks.discover import discover_sessions
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from powerflow_pipeline.data.preprocess.tasks.orient import orient_camera
from tests.conftest import (
    MARKER_DEPTH,
    MARKER_UV,
    ODOMETRY_CX,
    ODOMETRY_CY,
    STATIC_FX,
    MakeCamera,
)
from tests.unit.test_preprocess_ingest import make_config


def orient(tmp_path: Path, make_camera: MakeCamera, **overrides: object) -> Path:
    """Run S0 then S1 on one synthetic camera; return the published output directory."""

    raw = tmp_path / "raw"
    make_camera(raw, rgb_frames=4, depth_frames=5)
    config = make_config(tmp_path, **overrides)
    (camera,) = discover_sessions.fn(raw)
    record = ingest_camera.fn(camera, config)

    orient_camera.fn(record, config)
    return config.output_root / "9 July" / "cnj_45kg_Set1" / "Front"


def read_depth(path: Path) -> np.ndarray:
    """Read a stream frame exactly as stored: no depth scaling, no colour mapping."""

    frame: np.ndarray | None = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert frame is not None
    return frame


def test_every_stream_leaves_portrait(tmp_path: Path, make_camera: MakeCamera) -> None:
    out = orient(tmp_path, make_camera)

    depth = read_depth(out / "depth" / "000000.png")
    confidence = read_depth(out / "confidence" / "000000.png")
    with av.open(str(out / "rgb.mp4")) as container:
        stream = container.streams.video[0]
        rgb_height, rgb_width = stream.codec_context.height, stream.codec_context.width

    assert (rgb_height, rgb_width) == (64, 48)  # 64x48 landscape -> 48x64 portrait
    assert rgb_height > rgb_width
    assert depth.shape == (16, 12) and depth.shape[0] > depth.shape[1]
    assert confidence.shape == (16, 12)


def test_depth_pixels_move_exactly_where_cw_says(tmp_path: Path, make_camera: MakeCamera) -> None:
    """The marker is asymmetric: a flip, a transpose, or the wrong direction each miss."""

    out = orient(tmp_path, make_camera)

    depth = read_depth(out / "depth" / "000000.png")
    source_u, source_v = MARKER_UV
    source_height = 12

    assert np.argwhere(depth == MARKER_DEPTH).tolist() == [
        [source_u, source_height - 1 - source_v]  # CW: u' = H - 1 - v, v' = u
    ]


def test_depth_keeps_its_dtype_and_invents_no_values(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, rgb_frames=4, depth_frames=5)
    config = make_config(tmp_path)
    (camera,) = discover_sessions.fn(raw)
    record = ingest_camera.fn(camera, config)
    orient_camera.fn(record, config)

    source = read_depth(raw / "9 July" / "cnj_45kg_Set1" / "Front" / "depth" / "000000.png")
    rotated = read_depth(
        config.output_root / "9 July" / "cnj_45kg_Set1" / "Front" / "depth" / "000000.png"
    )

    assert rotated.dtype == np.uint16
    assert np.array_equal(np.unique(rotated), np.unique(source))


def test_confidence_stays_inside_the_reported_set(tmp_path: Path, make_camera: MakeCamera) -> None:
    out = orient(tmp_path, make_camera)

    confidence = read_depth(out / "confidence" / "000000.png")

    assert confidence.dtype == np.uint8
    assert set(np.unique(confidence).tolist()) <= {0, 1, 2}
    assert 1 not in np.unique(confidence)  # nothing interpolated between 0 and 2


def test_only_the_aligned_frames_are_emitted(tmp_path: Path, make_camera: MakeCamera) -> None:
    """ISSUE-03: the excess trailing depth/confidence frame is dropped, not carried."""

    out = orient(tmp_path, make_camera)

    assert len(list((out / "depth").glob("*.png"))) == 4  # from 5
    assert len(list((out / "confidence").glob("*.png"))) == 4
    with av.open(str(out / "rgb.mp4")) as container:
        assert sum(1 for _ in container.decode(video=0)) == 4


def test_the_camera_matrix_is_rewritten_rotated_and_averaged(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    """ISSUE-01 + ISSUE-02: K is landscape, so it rotates too, from the averaged odometry."""

    out = orient(tmp_path, make_camera)

    with (out / "camera_matrix.csv").open(newline="") as handle:
        matrix = [[float(cell) for cell in row] for row in csv.reader(handle) if row]

    fx, cx = matrix[0][0], matrix[0][2]
    fy, cy = matrix[1][1], matrix[1][2]
    # Averaged odometry K over 5 rows alternating 44.0/46.0: fx = fy = 44.8.
    # Rotated CW about a 64x48 source: cx' = H - 1 - cy, cy' = cx.
    assert (fx, fy) == pytest.approx((44.8, 44.8))
    assert cx == pytest.approx(48 - 1 - ODOMETRY_CY)
    assert cy == pytest.approx(ODOMETRY_CX)
    assert matrix[2] == [0.0, 0.0, 1.0]


def test_the_rotated_principal_point_lands_on_the_portrait_centre(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    """The point of ISSUE-01: rotating K puts (cx, cy) back on the centre of the image."""

    out = orient(tmp_path, make_camera)
    sidecar = json.loads((out / "sidecar.json").read_text())

    offset_x, offset_y = sidecar["preflight"]["offset_from_portrait_centre_px"]

    assert abs(offset_x) < 1.5  # portrait width 48 -> centre 24
    assert abs(offset_y) < 1.5  # portrait height 64 -> centre 32


def test_the_source_camera_matrix_survives_byte_for_byte(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    raw = tmp_path / "raw"
    out = orient(tmp_path, make_camera)

    source = (raw / "9 July" / "cnj_45kg_Set1" / "Front" / "camera_matrix.csv").read_bytes()
    copied = (out / "camera_matrix_source.csv").read_bytes()

    assert hashlib.sha256(copied).hexdigest() == hashlib.sha256(source).hexdigest()


def test_odometry_is_truncated_and_imu_is_verbatim(tmp_path: Path, make_camera: MakeCamera) -> None:
    raw = tmp_path / "raw"
    out = orient(tmp_path, make_camera)
    source = raw / "9 July" / "cnj_45kg_Set1" / "Front"

    odometry = (out / "odometry.csv").read_text().splitlines()
    assert len(odometry) == 1 + 4  # header + n_frames rows; the excess row is gone
    assert odometry[0] == (source / "odometry.csv").read_text().splitlines()[0]

    assert (out / "imu.csv").read_bytes() == (source / "imu.csv").read_bytes()


def test_the_sidecar_attests_what_was_done(tmp_path: Path, make_camera: MakeCamera) -> None:
    out = orient(tmp_path, make_camera)

    sidecar = json.loads((out / "sidecar.json").read_text())

    assert sidecar["rotation"] == "cw"
    assert sidecar["k_rewritten"] is True
    assert sidecar["k_source"] == "odometry.csv (per-frame mean)"
    assert sidecar["n_frames"] == 4
    assert sidecar["dropped_frames"] == 1
    assert sidecar["rgb"] == {"input": [64, 48], "output": [48, 64]}
    assert sidecar["depth"] == {"input": [16, 12], "output": [12, 16]}
    assert sidecar["intrinsics"]["rotated"]["frame"] == "portrait"
    assert sidecar["intrinsics"]["static"]["fx"] == pytest.approx(STATIC_FX)
    assert sidecar["preflight"]["principal_point_inside_image"] is True
    assert "odometry.csv" in sidecar["notes"][0]


def test_ccw_rotates_every_stream_the_other_way(tmp_path: Path, make_camera: MakeCamera) -> None:
    """All three streams share one direction; rotating them apart is the silent failure."""

    out = orient(tmp_path, make_camera, rotation=RotationDirection.CCW)

    depth = read_depth(out / "depth" / "000000.png")
    source_u, source_v = MARKER_UV
    source_width = 16

    assert np.argwhere(depth == MARKER_DEPTH).tolist() == [
        [source_width - 1 - source_u, source_v]  # CCW: u' = v, v' = W - 1 - u
    ]


def test_the_rgb_block_lands_where_the_rotation_predicts(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    """RGB rotates with the same handedness as depth: the block moves top-left -> top-right."""

    out = orient(tmp_path, make_camera)

    with av.open(str(out / "rgb.mp4")) as container:
        frame = next(container.decode(video=0)).to_ndarray(format="gray")

    height, width = frame.shape
    quadrants = {
        "top_left": frame[: height // 2, : width // 2].mean(),
        "top_right": frame[: height // 2, width // 2 :].mean(),
        "bottom_left": frame[height // 2 :, : width // 2].mean(),
        "bottom_right": frame[height // 2 :, width // 2 :].mean(),
    }

    assert max(quadrants, key=lambda key: quadrants[key]) == "top_right"


def test_a_variable_rate_clip_survives_the_re_encode(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    """The capture drops frames, so its timestamps are irregular.

    Re-encoding irregular timing with B-frames makes libx264 derive decode timestamps that
    collide unless the output timebase is fine enough to separate them, and the muxer then
    rejects the packet outright. It needs enough frames for B-frames to appear at all.
    """

    raw = tmp_path / "raw"
    make_camera(raw, rgb_frames=48, depth_frames=49, imu_rows=96)
    config = make_config(tmp_path)
    (camera,) = discover_sessions.fn(raw)
    record = ingest_camera.fn(camera, config)

    orient_camera.fn(record, config)

    published = config.output_root / "9 July" / "cnj_45kg_Set1" / "Front" / "rgb.mp4"
    with av.open(str(published)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        pts = [frame.pts for frame in frames if frame.pts is not None]
        time_base = stream.time_base
        assert time_base is not None
        span = float((pts[-1] - pts[0]) * time_base)

    assert len(frames) == 48  # not one frame lost to a rejected packet
    assert len(pts) == 48
    # The source advances 2 ticks every 4th frame at 1/60: 47 gaps, 11 of them doubled.
    assert span == pytest.approx((47 + 11) / 60)


def test_a_dry_run_publishes_nothing(tmp_path: Path, make_camera: MakeCamera) -> None:
    out = orient(tmp_path, make_camera, dry_run=True)

    assert not out.exists()
    assert not any((tmp_path / "s1").glob("**/.powerflow-*"))


def test_publishing_twice_refuses_to_overwrite(tmp_path: Path, make_camera: MakeCamera) -> None:
    """Write-once: a second run must not silently half-replace a published camera."""

    raw = tmp_path / "raw"
    make_camera(raw, rgb_frames=4, depth_frames=5)
    config = make_config(tmp_path)
    (camera,) = discover_sessions.fn(raw)
    record = ingest_camera.fn(camera, config)
    orient_camera.fn(record, config)

    with pytest.raises(Exception, match="output path already exists"):
        orient_camera.fn(record, config)


def test_overwrite_replaces_a_published_camera(tmp_path: Path, make_camera: MakeCamera) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, rgb_frames=4, depth_frames=5)
    config = make_config(tmp_path)
    (camera,) = discover_sessions.fn(raw)
    record = ingest_camera.fn(camera, config)
    orient_camera.fn(record, config)

    result = orient_camera.fn(record, make_config(tmp_path, overwrite=True))

    published = config.output_root / "9 July" / "cnj_45kg_Set1" / "Front"
    assert result.derived["n_frames"] == 4
    assert len(list((published / "depth").glob("*.png"))) == 4


def test_the_step_result_reports_the_file_operations(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, rgb_frames=4, depth_frames=5)
    config = make_config(tmp_path)
    (camera,) = discover_sessions.fn(raw)
    record = ingest_camera.fn(camera, config)

    result = orient_camera.fn(record, config)

    assert result.derived["rotation"] == "cw"
    assert result.derived["k_rewritten"] is True
    assert {op.op for op in result.file_ops} == {"copy", "write", "publish"}
