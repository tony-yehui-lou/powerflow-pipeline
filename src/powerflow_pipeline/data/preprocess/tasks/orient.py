"""S1 · Orientation repair → establishes I1.

The capture app writes every image stream landscape. This stage rotates all three the
same way, and rotates `K` with them (ISSUE-01: the matrix describes the landscape buffer,
not the portrait frame the source spec assumed). Rotating RGB one way and depth the other
would leave them mirror-misaligned while each looked individually correct.
"""

from __future__ import annotations

import csv
import shutil
from fractions import Fraction
from pathlib import Path

import av
import av.filter
import cv2
import numpy as np
from prefect import task

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.filesystem import (
    cleanup,
    create_staging_root,
    publish_staging,
    write_json,
)
from powerflow_pipeline.data.common.models import FileOp, StepResult
from powerflow_pipeline.data.preprocess.config import PreprocessConfig, RotationDirection
from powerflow_pipeline.data.preprocess.geometry import (
    rotate_frame,
    rotate_intrinsics,
    rotated_size,
    transpose_filter,
)
from powerflow_pipeline.data.preprocess.models import CameraRecord, Intrinsics
from powerflow_pipeline.data.preprocess.tasks.ingest import CONFIDENCE_VALUES, read_frame

PASSTHROUGH = ("imu.csv",)
OUTPUT_TIME_BASE = Fraction(1, 90000)  # fine enough to keep B-frame DTS distinct (see rotate_rgb)
ODOMETRY_NOTE = (
    "odometry.csv keeps its per-frame fx/fy/cx/cy columns in the landscape frame; "
    "camera_matrix.csv is the authoritative portrait intrinsics."
)


def rotate_rgb(
    source: Path, destination: Path, rotation: RotationDirection, n_frames: int, crf: int
) -> None:
    """Rotate the clip with ffmpeg's own `transpose` filter, re-encoding once with libx264.

    Streams frame by frame: a session is thousands of 1440x1920 frames and must never be
    held in memory. Presentation timestamps flow through the filter untouched, and are
    re-expressed in `OUTPUT_TIME_BASE` on the way out.
    """

    with av.open(str(source)) as inp, av.open(str(destination), "w") as out:
        in_stream = inp.streams.video[0]
        in_stream.thread_type = "AUTO"

        graph = av.filter.Graph()
        buffer = graph.add_buffer(template=in_stream)
        transpose = graph.add("transpose", transpose_filter(rotation))
        sink = graph.add("buffersink")
        buffer.link_to(transpose)
        transpose.link_to(sink)
        graph.configure()

        width, height = rotated_size(in_stream.codec_context.width, in_stream.codec_context.height)
        out_stream = out.add_stream(
            "libx264", rate=in_stream.average_rate, options={"crf": str(crf)}
        )
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"
        # The capture drops frames, so its timestamps are irregular. At the source's coarse
        # 1/60 timebase, libx264's B-frame decode timestamps collide and the muxer rejects
        # the packet; a fine timebase keeps them strictly increasing. The stream's timebase
        # does not reach the encoder on its own — the codec context needs it too.
        out_stream.time_base = OUTPUT_TIME_BASE
        out_stream.codec_context.time_base = OUTPUT_TIME_BASE
        if (creation_time := inp.metadata.get("creation_time")) is not None:
            out.metadata["creation_time"] = creation_time

        written = 0
        for frame in inp.decode(in_stream):
            if written >= n_frames:
                break
            graph.push(frame)
            while True:
                try:
                    rotated = graph.pull()
                except (av.error.BlockingIOError, av.error.EOFError):
                    break
                assert isinstance(rotated, av.VideoFrame)
                for packet in out_stream.encode(rotated):
                    out.mux(packet)
                written += 1

        for packet in out_stream.encode():
            out.mux(packet)


def rotate_png_stream(
    source: Path, destination: Path, rotation: RotationDirection, n_frames: int
) -> None:
    """Rotate `n_frames` PNG frames losslessly, dropping the excess trailing ones (ISSUE-03)."""

    destination.mkdir(parents=True)
    for path in sorted(source.glob("*.png"))[:n_frames]:
        cv2.imwrite(str(destination / path.name), rotate_frame(read_frame(path), rotation))


def truncate_csv(source: Path, destination: Path, rows: int) -> None:
    """Copy a header plus the first `rows` data rows, so every stream shares one index space."""

    lines = source.read_text().splitlines(keepends=True)
    destination.write_text("".join(lines[: rows + 1]))


def write_camera_matrix(path: Path, intrinsics: Intrinsics) -> None:
    """Write `K` in the same bare 3x3 shape the capture app ships."""

    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(intrinsics.to_matrix())


def preflight(intrinsics: Intrinsics, width: int, height: int) -> dict[str, object]:
    """Record where the rotated principal point sits. Recorded, never silently trusted.

    Against the un-rotated `K` this check failed by 240 px, which is what exposed ISSUE-01.
    """

    return {
        "principal_point_inside_image": bool(
            0 <= intrinsics.cx < width and 0 <= intrinsics.cy < height
        ),
        "offset_from_portrait_centre_px": [
            round(intrinsics.cx - width / 2, 3),
            round(intrinsics.cy - height / 2, 3),
        ],
    }


@task(retries=1)
def orient_camera(record: CameraRecord, config: PreprocessConfig) -> StepResult:
    """Rotate one camera's three image streams to portrait and publish it whole."""

    source = record.source
    destination = config.output_root / record.relative
    rotation = config.rotation

    rgb_size = rotated_size(record.rgb_width, record.rgb_height)
    depth_size = rotated_size(record.depth_width, record.depth_height)
    rotated = rotate_intrinsics(
        record.intrinsics, width=record.rgb_width, height=record.rgb_height, rotation=rotation
    )
    verdict = preflight(rotated, *rgb_size)

    sidecar = {
        "rotation": rotation.value,
        "k_rewritten": True,  # ISSUE-01: K described the landscape buffer
        "k_source": "odometry.csv (per-frame mean)",  # ISSUE-02
        "n_frames": record.n_frames,
        "dropped_frames": record.counts.depth - record.n_frames,
        "rgb": {"input": [record.rgb_width, record.rgb_height], "output": list(rgb_size)},
        "depth": {"input": [record.depth_width, record.depth_height], "output": list(depth_size)},
        "intrinsics": {
            "static": record.static_intrinsics.model_dump(),
            "averaged": record.intrinsics.model_dump(),
            "rotated": rotated.model_dump(),
            "odometry_fx_drift": record.odometry_intrinsics_drift,
        },
        "preflight": verdict,
        "notes": [ODOMETRY_NOTE],
    }

    file_ops = [
        FileOp(op="write", src=source / "rgb.mp4", dst=destination / "rgb.mp4"),
        FileOp(op="write", src=source / "depth", dst=destination / "depth"),
        FileOp(op="write", src=source / "confidence", dst=destination / "confidence"),
        FileOp(op="write", src=source / "camera_matrix.csv", dst=destination / "camera_matrix.csv"),
        FileOp(
            op="copy",
            src=source / "camera_matrix.csv",
            dst=destination / "camera_matrix_source.csv",
        ),
        FileOp(op="write", src=source / "odometry.csv", dst=destination / "odometry.csv"),
        FileOp(op="copy", src=source / "imu.csv", dst=destination / "imu.csv"),
        FileOp(op="write", src=source, dst=destination / "sidecar.json"),
        FileOp(op="publish", src=source, dst=destination),
    ]
    result = StepResult(
        derived={
            "rotation": rotation.value,
            "k_rewritten": True,
            "n_frames": record.n_frames,
            "dropped_frames": sidecar["dropped_frames"],
            "rgb_output": f"{rgb_size[0]}x{rgb_size[1]}",
            "preflight": verdict["principal_point_inside_image"],
        },
        file_ops=file_ops,
    )
    if not verdict["principal_point_inside_image"]:
        result.warnings.append("rotated principal point falls outside the portrait image")

    if config.dry_run:
        return result

    staging = create_staging_root(destination)
    try:
        rotate_rgb(
            source / "rgb.mp4", staging / "rgb.mp4", rotation, record.n_frames, config.rgb_crf
        )
        rotate_png_stream(source / "depth", staging / "depth", rotation, record.n_frames)
        rotate_png_stream(source / "confidence", staging / "confidence", rotation, record.n_frames)

        write_camera_matrix(staging / "camera_matrix.csv", rotated)
        shutil.copy2(source / "camera_matrix.csv", staging / "camera_matrix_source.csv")
        truncate_csv(source / "odometry.csv", staging / "odometry.csv", record.n_frames)
        for name in PASSTHROUGH:
            shutil.copy2(source / name, staging / name)
        write_json(staging / "sidecar.json", sidecar)

        _validate_exit(staging, record, rgb_size, depth_size)

        if config.overwrite and destination.exists():
            shutil.rmtree(destination)
        publish_staging(staging, destination)
    finally:
        cleanup(staging)

    return result


def _validate_exit(
    staging: Path,
    record: CameraRecord,
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
) -> None:
    """Prove I1 on the staged result before it is published, never after."""

    for name, (width, height) in (("rgb", rgb_size), ("depth", depth_size)):
        if height <= width:
            raise ScanRejected(f"{name} is not portrait after rotation: {width}x{height}")

    for name in ("depth", "confidence"):
        emitted = sorted((staging / name).glob("*.png"))
        if len(emitted) != record.n_frames:
            raise ScanRejected(
                f"{name} frame count after rotation: {len(emitted)} vs {record.n_frames}"
            )

    depth = read_frame(staging / "depth" / "000000.png")
    if depth.dtype != np.uint16:
        raise ScanRejected(f"depth dtype changed during rotation: {depth.dtype}")

    confidence = read_frame(staging / "confidence" / "000000.png")
    outside = sorted(set(np.unique(confidence).tolist()) - CONFIDENCE_VALUES)
    if outside:
        raise ScanRejected(f"confidence values invented during rotation: {outside}")
