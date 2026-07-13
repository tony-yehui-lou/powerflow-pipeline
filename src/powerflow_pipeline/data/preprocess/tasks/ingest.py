"""S0 · Ingest, validate, record.

Establishes no invariant: it is the gate that lets S1 assume its inputs exist. Every
rejection is per camera, carries an exact reason, and leaves the other cameras alone.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from prefect import task

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.filesystem import write_json
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import (
    CameraDir,
    CameraRecord,
    Intrinsics,
    StreamCounts,
)

CONFIDENCE_VALUES = {0, 1, 2}
REQUIRED_STREAMS = (
    "rgb.mp4",
    "depth",
    "confidence",
    "camera_matrix.csv",
    "odometry.csv",
    "imu.csv",
)


@dataclass(frozen=True, slots=True)
class RgbProbe:
    """What one full decode pass of `rgb.mp4` tells us. Cached so S1 never repeats it."""

    width: int
    height: int
    fps: float
    n_frames: int
    creation_time: str | None


@dataclass(frozen=True, slots=True)
class OdometryProbe:
    """Row count plus the averaged per-frame intrinsics (ISSUE-02)."""

    n_rows: int
    intrinsics: Intrinsics
    fx_drift: float


def probe_rgb(path: Path) -> RgbProbe:
    """Decode `rgb.mp4` end to end: the header's frame count is not the truth.

    The real capture's container advertises one more frame than it can decode, so the
    count must come from the decoder.
    """

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        rate = stream.average_rate
        n_frames = sum(1 for _ in container.decode(stream))
        creation_time = container.metadata.get("creation_time")

    return RgbProbe(
        width=width,
        height=height,
        fps=float(rate) if rate else 0.0,
        n_frames=n_frames,
        creation_time=creation_time,
    )


def frame_paths(directory: Path) -> list[Path]:
    """Return the PNG frames of a stream in index order."""

    return sorted(directory.glob("*.png"))


def is_contiguous(paths: list[Path]) -> bool:
    """True when the filenames are the zero-padded range `000000 .. N-1`."""

    return [path.stem for path in paths] == [f"{i:06d}" for i in range(len(paths))]


def read_camera_matrix(path: Path) -> Intrinsics:
    """Parse the bare 3x3 `K` the capture app ships. Raises `ScanRejected` if malformed."""

    try:
        with path.open(newline="") as handle:
            rows = [[float(cell) for cell in row] for row in csv.reader(handle) if row]
    except ValueError as error:
        raise ScanRejected("intrinsics absent or malformed") from error

    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ScanRejected("intrinsics absent or malformed")

    intrinsics = Intrinsics.from_matrix(rows, frame="landscape")
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ScanRejected("intrinsics absent or malformed")
    return intrinsics


def probe_odometry(path: Path, static_fx: float) -> OdometryProbe:
    """Average the per-frame `fx, fy, cx, cy`, and measure their drift from the static `K`.

    ISSUE-01 and ISSUE-02 in force: the averaged odometry matrix is authoritative, and it
    describes the *landscape* buffer — S1 rotates it.
    """

    frame = pd.read_csv(path, skipinitialspace=True)
    missing = {"fx", "fy", "cx", "cy"} - set(frame.columns)
    if missing:
        raise ScanRejected("intrinsics absent or malformed")

    intrinsics = Intrinsics(
        fx=float(frame["fx"].mean()),
        fy=float(frame["fy"].mean()),
        cx=float(frame["cx"].mean()),
        cy=float(frame["cy"].mean()),
        frame="landscape",
    )
    drift = float((frame["fx"] - static_fx).abs().max()) if len(frame) else 0.0
    return OdometryProbe(n_rows=len(frame), intrinsics=intrinsics, fx_drift=drift)


def count_csv_rows(path: Path) -> int:
    """Count data rows, excluding the header."""

    with path.open() as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_frame(path: Path) -> np.ndarray:
    """Read one stream frame exactly as stored: no depth conversion, no colour mapping."""

    frame: np.ndarray | None = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ScanRejected(f"unreadable frame: {path.name}")
    return frame


def _check_streams_exist(source: Path) -> None:
    for name in REQUIRED_STREAMS:
        if not (source / name).exists():
            raise ScanRejected(f"missing required stream: {name}")


def _check_image_streams(depth: list[Path], confidence: list[Path]) -> tuple[np.ndarray, ...]:
    """Validate the depth/confidence pair and return the frames read while doing so."""

    for name, paths in (("depth", depth), ("confidence", confidence)):
        if not paths:
            raise ScanRejected(f"empty required stream: {name}")
        if not is_contiguous(paths):
            raise ScanRejected(f"non-contiguous {name} frame indices")

    if len(depth) != len(confidence):
        raise ScanRejected(
            f"depth/confidence frame count mismatch: {len(depth)} vs {len(confidence)}"
        )

    # First and last frame of each: a stream that changes shape or dtype midway is a
    # different capture, and validating only frame 0 would wave it through.
    frames = tuple(
        read_frame(path) for path in (depth[0], depth[-1], confidence[0], confidence[-1])
    )
    for depth_frame, confidence_frame in ((frames[0], frames[2]), (frames[1], frames[3])):
        if depth_frame.shape[:2] != confidence_frame.shape[:2]:
            raise ScanRejected(
                "depth/confidence resolution mismatch: "
                f"{depth_frame.shape[:2]} vs {confidence_frame.shape[:2]}"
            )
        if depth_frame.dtype != np.uint16:
            raise ScanRejected(f"unexpected depth dtype: {depth_frame.dtype}")
        outside = sorted(set(np.unique(confidence_frame).tolist()) - CONFIDENCE_VALUES)
        if outside:
            raise ScanRejected(f"confidence values outside {{0,1,2}}: {outside}")

    return frames


@task(retries=1)
def ingest_camera(
    camera: CameraDir, config: PreprocessConfig, stopwatch_legible: bool | None = None
) -> CameraRecord:
    """Validate one camera and emit its `record.json`. Raises `ScanRejected` on any failure."""

    source = camera.source
    _check_streams_exist(source)

    depth_paths = frame_paths(source / "depth")
    confidence_paths = frame_paths(source / "confidence")
    depth_frame = _check_image_streams(depth_paths, confidence_paths)[0]

    static = read_camera_matrix(source / "camera_matrix.csv")
    odometry = probe_odometry(source / "odometry.csv", static.fx)
    if odometry.n_rows != len(depth_paths):
        raise ScanRejected(
            f"odometry/depth frame count mismatch: {odometry.n_rows} vs {len(depth_paths)}"
        )

    rgb = probe_rgb(source / "rgb.mp4")
    shortfall = len(depth_paths) - rgb.n_frames
    if not 0 <= shortfall <= config.frame_count_tolerance:
        raise ScanRejected(
            f"rgb frame count outside tolerance: rgb={rgb.n_frames} "
            f"depth={len(depth_paths)} tolerance={config.frame_count_tolerance}"
        )

    if config.require_stopwatch_attestation and stopwatch_legible is not True:
        raise ScanRejected(f"stopwatch not attested legible for camera {camera.camera}")

    record = CameraRecord(
        date=camera.date,
        session=camera.session,
        camera=camera.camera,
        source=source,
        rgb_width=rgb.width,
        rgb_height=rgb.height,
        fps=rgb.fps,
        depth_width=int(depth_frame.shape[1]),
        depth_height=int(depth_frame.shape[0]),
        counts=StreamCounts(
            rgb=rgb.n_frames,
            depth=len(depth_paths),
            confidence=len(confidence_paths),
            odometry=odometry.n_rows,
            imu=count_csv_rows(source / "imu.csv"),
        ),
        n_frames=rgb.n_frames,  # ISSUE-03: the trailing depth/confidence/odometry entry is excess
        intrinsics=odometry.intrinsics,
        static_intrinsics=static,
        odometry_intrinsics_drift=odometry.fx_drift,
        creation_time=rgb.creation_time,
        stopwatch_legible=stopwatch_legible,
    )

    if not config.dry_run:
        destination = config.record_root / camera.relative
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "record.json", record.model_dump(mode="json"))

    return record
