"""Shared fixtures for the pipeline test suite."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Protocol

import av
import cv2
import numpy as np
import pytest

from powerflow_pipeline.data.common.context import OutputMode, RunContext
from powerflow_pipeline.data.preprocess.config import RotationDirection
from powerflow_pipeline.data.preprocess.geometry import rotate_intrinsics, rotated_size
from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.retilt import depth_intrinsics, rectifying_rotation

VALID_META: dict[str, Any] = {
    "scan_id": "scan_0001",
    "width": 1920,
    "height": 1080,
    "marker_px": 200.0,
    "marker_mm": 50.0,
}

# A synthetic capture, shaped like data/raw but tiny. Both streams keep the real 4:3
# frustum, and the depth marker is asymmetric in both axes so that a wrong rotation
# direction, a bare transpose, and a flip each produce a *different* wrong answer.
RGB_SIZE = (64, 48)  # (W, H) landscape
DEPTH_SIZE = (16, 12)  # (W, H) landscape
MARKER_UV = (5, 3)  # (u, v) in the depth frame
MARKER_DEPTH = 1234
CREATION_TIME = "2026-07-09T10:04:15.000000Z"  # the tag the real rgb.mp4 carries
SOURCE_TIME_BASE = Fraction(1, 60)  # the capture's timebase
SOURCE_RATE = 60

# The real camera's intrinsics, scaled 1/30 with the image (1920x1440 -> 64x48), so the
# fixture is a faithful miniature: a principal point that only makes sense against the
# full-size frame would make S1's preflight check meaningless here.
STATIC_FX = 46.7411  # 1402.2333 / 30
STATIC_CX, STATIC_CY = 32.279, 23.9895  # 968.371 / 30, 719.6846 / 30
STATIC_K = f"{STATIC_FX}, 0.0, {STATIC_CX}\n0.0, {STATIC_FX}, {STATIC_CY}\n0.0, 0.0, 1.0"
# Per-frame rows whose fx straddles the static value and drifts, as the real ones do.
ODOMETRY_FX = (44.0, 46.0)
ODOMETRY_CX, ODOMETRY_CY = 32.28, 23.99


class MakeCamera(Protocol):
    def __call__(
        self,
        raw_root: Path,
        *,
        date: str = ...,
        session: str = ...,
        camera: str = ...,
        rgb_frames: int = ...,
        depth_frames: int = ...,
        confidence_frames: int | None = ...,
        odometry_rows: int | None = ...,
        imu_rows: int = ...,
        depth_dtype: Any = ...,
        depth_size: tuple[int, int] = ...,
        confidence_size: tuple[int, int] | None = ...,
        confidence_values: Sequence[int] = ...,
        camera_matrix: str | None = ...,
        omit: Sequence[str] = ...,
        skip_depth_index: int | None = ...,
        creation_time: str = ...,
        uptime_base: float = ...,
        odometry_hz: float = ...,
        imu_hz: float = ...,
        render_floor_plane: bool = ...,
        floor_tilt_deg: float = ...,
        floor_roll_deg: float = ...,
        floor_distance_m: float = ...,
        odometry_translation: Callable[[int], tuple[float, float, float]] | None = ...,
        rgb_size: tuple[int, int] = ...,
    ) -> Path: ...


def _render_floor_plane(
    raw_depth_size: tuple[int, int],
    rgb_size: tuple[int, int],
    landscape_fx: float,
    landscape_cx: float,
    landscape_cy: float,
    floor_tilt_deg: float,
    floor_roll_deg: float,
    floor_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Render `(depth_mm uint16, confidence uint8)` in the RAW landscape frame such that,
    after S1 Orient's exact CW pixel permutation (it relocates depth pixels; it never
    resamples their values -- see `orient.rotate_png_stream`), the resulting PORTRAIT
    frame shows a real floor plane whose fitted normal has exactly `(floor_tilt_deg,
    floor_roll_deg)`.

    `landscape_fx/cx/cy` must be the *averaged odometry* intrinsics S0 Ingest will
    actually compute for this camera (ISSUE-02) -- not the static `camera_matrix.csv`
    value -- so the geometry used to render depth here matches, bit for bit, the `K`
    S3 Retilt will later back-project through.

    Computed directly in the portrait frame, where the requested normal is close to
    "up" and the maths is well-conditioned, then permuted back through the inverse of
    S1's CW mapping. A pixel whose implied floor point is not in front of the camera
    (e.g. above the horizon a shallow tilt puts within frame) gets depth 0 / confidence
    0 -- a real depth sensor reports no return there too, and this is exactly `select_
    floor_pixels`'s definition of "not floor", so no separate narrow region is needed:
    a `full_frame`-annotated region and this confidence mask cooperate correctly.
    """

    w_raw, h_raw = raw_depth_size
    rgb_w_raw, rgb_h_raw = rgb_size
    landscape_k = Intrinsics(
        fx=landscape_fx, fy=landscape_fx, cx=landscape_cx, cy=landscape_cy, frame="landscape"
    )
    rotated_static_k = rotate_intrinsics(
        landscape_k, width=rgb_w_raw, height=rgb_h_raw, rotation=RotationDirection.CW
    )
    port_rgb_size = rotated_size(rgb_w_raw, rgb_h_raw)
    port_depth_size = rotated_size(w_raw, h_raw)
    k_d_port = depth_intrinsics(
        rotated_static_k, rgb_size=port_rgb_size, depth_size=port_depth_size
    )

    n_portrait = rectifying_rotation(floor_tilt_deg, floor_roll_deg).T @ np.array([0.0, -1.0, 0.0])
    w_port, h_port = port_depth_size
    cols, rows = np.meshgrid(
        np.arange(w_port, dtype=np.float64), np.arange(h_port, dtype=np.float64)
    )
    denom = (
        n_portrait[0] * (cols - k_d_port.cx) / k_d_port.fx
        + n_portrait[1] * (rows - k_d_port.cy) / k_d_port.fy
        + n_portrait[2]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z_port = -floor_distance_m / denom  # a plane at perpendicular distance floor_distance_m
    valid = (z_port > 0.2 * floor_distance_m) & (z_port < 10.0 * floor_distance_m)

    depth_port = np.where(valid, np.clip(z_port * 1000.0, 1, 65535), 0).astype(np.uint16)
    confidence_port = np.where(valid, 2, 0).astype(np.uint8)

    # Invert S1 Orient's exact CW mapping (raw (u, v) -> portrait (h_raw - 1 - v, u)):
    # raw row v_raw pulls portrait column (h_raw - 1 - v_raw), across every raw column.
    depth_raw = np.zeros((h_raw, w_raw), dtype=np.uint16)
    confidence_raw = np.zeros((h_raw, w_raw), dtype=np.uint8)
    for v_raw in range(h_raw):
        u_port = h_raw - 1 - v_raw
        depth_raw[v_raw, :] = depth_port[:, u_port]
        confidence_raw[v_raw, :] = confidence_port[:, u_port]
    return depth_raw, confidence_raw


def _write_rgb(path: Path, frames: int, size: tuple[int, int], creation_time: str) -> None:
    """Encode a landscape clip with a bright block in the top-left quadrant.

    Variable frame rate, like the real capture: the device drops frames, so presentation
    timestamps advance by one tick or two. A uniform clip would hide the DTS collisions a
    re-encode can produce from irregular timing.
    """

    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[: height // 3, : width // 3] = 255

    with av.open(str(path), "w") as container:
        container.metadata["creation_time"] = creation_time
        stream = container.add_stream("libx264", rate=SOURCE_RATE)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.time_base = SOURCE_TIME_BASE
        stream.codec_context.time_base = SOURCE_TIME_BASE

        pts = 0
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = pts
            frame.time_base = SOURCE_TIME_BASE
            pts += 2 if i % 4 == 3 else 1  # every fourth interval is a dropped frame
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def make_camera() -> MakeCamera:
    """Build one `<date>/<session>/<camera>` directory, with a knob per rejection rule."""

    def _make(
        raw_root: Path,
        *,
        date: str = "9 July",
        session: str = "cnj_45kg_Set1",
        camera: str = "Front",
        rgb_frames: int = 4,
        depth_frames: int = 5,  # the real capture is always one depth frame long
        confidence_frames: int | None = None,
        odometry_rows: int | None = None,
        imu_rows: int = 8,
        depth_dtype: Any = np.uint16,
        depth_size: tuple[int, int] = DEPTH_SIZE,
        confidence_size: tuple[int, int] | None = None,
        confidence_values: Sequence[int] = (0, 2),
        camera_matrix: str | None = None,
        omit: Sequence[str] = (),
        skip_depth_index: int | None = None,
        creation_time: str = CREATION_TIME,
        uptime_base: float = 370169.0,
        odometry_hz: float = 60.0,
        imu_hz: float = 125.0,
        render_floor_plane: bool = False,
        floor_tilt_deg: float = 0.0,
        floor_roll_deg: float = 0.0,
        floor_distance_m: float = 1.5,
        odometry_translation: Callable[[int], tuple[float, float, float]] | None = None,
        rgb_size: tuple[int, int] = RGB_SIZE,
    ) -> Path:
        confidence_frames = depth_frames if confidence_frames is None else confidence_frames
        odometry_rows = depth_frames if odometry_rows is None else odometry_rows
        confidence_size = depth_size if confidence_size is None else confidence_size

        camera_dir = raw_root / date / session / camera
        camera_dir.mkdir(parents=True)

        if "rgb" not in omit:
            _write_rgb(camera_dir / "rgb.mp4", rgb_frames, rgb_size, creation_time)

        # ODOMETRY_FX/CX/CY are tuned to RGB_SIZE exactly (centred, correct FOV); a caller
        # that overrides rgb_size gets them scaled proportionally so the principal point
        # stays centred and the FOV stays constant -- identity when rgb_size == RGB_SIZE, so
        # every existing caller (which never overrides it) is unaffected byte-for-byte.
        scale_x = rgb_size[0] / RGB_SIZE[0]
        scale_y = rgb_size[1] / RGB_SIZE[1]
        scaled_cx, scaled_cy = ODOMETRY_CX * scale_x, ODOMETRY_CY * scale_y

        floor_depth: np.ndarray | None = None
        floor_confidence: np.ndarray | None = None
        if render_floor_plane:
            # The averaged-odometry K that S0 Ingest will actually compute (ISSUE-02) --
            # rendering against anything else would bias the fitted normal.
            avg_fx = float(
                np.mean([ODOMETRY_FX[i % len(ODOMETRY_FX)] for i in range(odometry_rows)])
            )
            floor_depth, floor_confidence = _render_floor_plane(
                depth_size,
                rgb_size,
                avg_fx * scale_x,
                scaled_cx,
                scaled_cy,
                floor_tilt_deg,
                floor_roll_deg,
                floor_distance_m,
            )

        if "depth" not in omit:
            (camera_dir / "depth").mkdir()
            width, height = depth_size
            for i in range(depth_frames):
                if i == skip_depth_index:
                    continue
                if floor_depth is not None:
                    depth = floor_depth.astype(depth_dtype)
                else:
                    depth = np.zeros((height, width), dtype=depth_dtype)
                    depth[MARKER_UV[1], MARKER_UV[0]] = min(
                        MARKER_DEPTH, int(np.iinfo(depth_dtype).max)
                    )
                cv2.imwrite(str(camera_dir / "depth" / f"{i:06d}.png"), depth)

        if "confidence" not in omit:
            (camera_dir / "confidence").mkdir()
            width, height = confidence_size
            for i in range(confidence_frames):
                if floor_confidence is not None:
                    confidence = floor_confidence
                else:
                    confidence = np.full((height, width), confidence_values[0], dtype=np.uint8)
                    confidence[MARKER_UV[1], MARKER_UV[0]] = confidence_values[-1]
                cv2.imwrite(str(camera_dir / "confidence" / f"{i:06d}.png"), confidence)

        if "camera_matrix" not in omit:
            body = STATIC_K if camera_matrix is None else camera_matrix
            (camera_dir / "camera_matrix.csv").write_text(body)

        if "odometry" not in omit:
            rows = [
                "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy,"
                " distortion_center_x, distortion_center_y"
            ]
            for i in range(odometry_rows):
                fx = ODOMETRY_FX[i % len(ODOMETRY_FX)] * scale_x
                fy = ODOMETRY_FX[i % len(ODOMETRY_FX)] * scale_y
                x, y, z = odometry_translation(i) if odometry_translation else (0.0, 0.0, 0.0)
                rows.append(
                    f"{uptime_base + i / odometry_hz:.6f}, {i:06d}, {x}, {y}, {z}, 0.0, 0.0, 0.0,"
                    f" 1.0, {fx}, {fy}, {scaled_cx}, {scaled_cy}, , "
                )
            (camera_dir / "odometry.csv").write_text("\n".join(rows) + "\n")

        if "imu" not in omit:
            rows = ["timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z"]
            rows += [
                f"{uptime_base + i / imu_hz:.6f}, 0.0, -0.98, 0.1, 0.0, 0.0, 0.0"
                for i in range(imu_rows)
            ]
            (camera_dir / "imu.csv").write_text("\n".join(rows) + "\n")

        return camera_dir

    return _make


META_TEMPLATE = """---
lift:
  dateTime_epoch: long
  weight_in_kg: float
  type: [cnj,snch]
  result: [success, fail]
athlete:
  name: string
  measureDate: ISO 8601 date
  height_in_cm: float
  weight_in_kg: float
  tibia_in_cm: float
  femur_in_cm: float
  torso_in_cm: float
  armspan_in_cm: float
...
"""


@pytest.fixture
def make_meta_template() -> Callable[..., Path]:
    """Write the date-level `meta.yaml`, verbatim from data/raw/9 July."""

    def _make(raw_root: Path, *, date: str = "9 July", body: str = META_TEMPLATE) -> Path:
        date_dir = raw_root / date
        date_dir.mkdir(parents=True, exist_ok=True)
        path = date_dir / "meta.yaml"
        path.write_text(body)
        return path

    return _make


class MakeSessionMetadata(Protocol):
    def __call__(
        self,
        raw_root: Path,
        *,
        date: str = ...,
        session: str = ...,
        lift_start_ms: Any = ...,
        lift_end_ms: Any = ...,
        omit: Sequence[str] = ...,
        body: str | None = ...,
        floor_regions: Literal["full_frame"] | dict[str, str] | None = ...,
    ) -> Path: ...


@pytest.fixture
def make_session_metadata() -> MakeSessionMetadata:
    """Write the operator-authored, raw `metadata.yaml` a session ships with.

    Defaults mirror `data/raw/9 July/cnj_45kg_Set1/metadata.yaml`. Pass a field as a
    non-numeric value or a negative one, swap `lift_start_ms`/`lift_end_ms`, or list a
    field in `omit` to drive one of the Cut stage's rejection cases; pass `body` to
    replace the file verbatim (e.g. malformed YAML).
    """

    def _make(
        raw_root: Path,
        *,
        date: str = "9 July",
        session: str = "cnj_45kg_Set1",
        lift_start_ms: Any = 100,
        lift_end_ms: Any = 900,
        omit: Sequence[str] = (),
        body: str | None = None,
        floor_regions: Literal["full_frame"] | dict[str, str] | None = None,
    ) -> Path:
        session_dir = raw_root / date / session
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / "metadata.yaml"

        if body is not None:
            path.write_text(body)
            return path

        lines = ["lift:"]
        if "lift_start_time_side_in_ms" not in omit:
            lines.append(f"  lift_start_time_side_in_ms: {lift_start_ms}")
        if "lift_end_time_side_in_ms" not in omit:
            lines.append(f"  lift_end_time_side_in_ms: {lift_end_ms}")

        if floor_regions == "full_frame":
            lines += [
                "video:",
                "  front_floor_region_bottom_left_in_pixels: (0, 1)",
                "  front_floor_region_top_right_in_pixels: (1, 0)",
                "  side_floor_region_bottom_left_in_pixels: (0, 1)",
                "  side_floor_region_top_right_in_pixels: (1, 0)",
            ]
        elif isinstance(floor_regions, dict):
            lines.append("video:")
            for key, value in floor_regions.items():
                if value == "omit":  # drop the field entirely, mirroring the lift-window knob
                    continue
                lines.append(f"  {key}: {value}")

        path.write_text("\n".join(lines) + "\n")
        return path

    return _make


class MakeScan(Protocol):
    def __call__(
        self,
        root: Path,
        scan_id: str,
        *,
        meta: dict[str, Any] | str | None = None,
        frames: int = 2,
    ) -> Path: ...


@pytest.fixture
def make_scan() -> MakeScan:
    """Build a scan directory: `meta.json` plus inert placeholder frames.

    `meta` overrides fields of the default valid meta, or replaces the file body
    entirely when a raw string (used for the malformed-JSON rejection case).
    """

    def _make(
        root: Path,
        scan_id: str,
        *,
        meta: dict[str, Any] | str | None = None,
        frames: int = 2,
    ) -> Path:
        scan_dir = root / scan_id
        (scan_dir / "frames").mkdir(parents=True)
        for i in range(frames):
            (scan_dir / "frames" / f"{i:03d}.bin").write_bytes(bytes([i]) * 8)

        if isinstance(meta, str):
            body = meta
        else:
            payload = {**VALID_META, "scan_id": scan_id, **(meta or {})}
            body = json.dumps(payload)
        (scan_dir / "meta.json").write_text(body)
        return scan_dir

    return _make


@pytest.fixture
def make_context() -> Callable[..., RunContext]:
    """Build a RunContext for tests; defaults to publish mode, no progress bar."""

    def _make(
        input_root: Path,
        *,
        output_root: Path | None = None,
        mode: OutputMode = OutputMode.PUBLISH,
        dry_run: bool = False,
        pipeline: str = "preprocess",
    ) -> RunContext:
        return RunContext.create(
            pipeline=pipeline,
            input_root=input_root,
            output_root=output_root,
            output_mode=mode,
            dry_run=dry_run,
        )

    return _make
