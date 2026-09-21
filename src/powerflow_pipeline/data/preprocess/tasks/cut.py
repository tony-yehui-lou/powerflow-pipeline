"""S1 · Cut and cross-camera time alignment -> establishes I2.

The interval owner's operator-declared lift window is the source of truth -- the `side`
camera of a two-camera session, or a single-camera capture itself. `resolve_cut_interval`
converts that window to a real-world epoch interval once per group; `cut_camera` applies the
same interval to each capture in turn, trimming every time-indexed stream by timestamp. Depth and
confidence are always selected together, by the same indices, so neither can drift from the
other. RGB, depth, confidence, and odometry share `odometry.csv:timestamp` as their
authoritative frame clock; IMU remains on its own monotonic clock. `timeline.epoch_ms_series`
bridges both clocks onto epoch milliseconds before selection, never onto a resampled lattice.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import pandas as pd
import yaml
from prefect import task

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.filesystem import (
    cleanup,
    create_staging_root,
    publish_staging,
    write_json,
)
from powerflow_pipeline.data.common.models import FileOp, StepResult
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord, CutInterval, StreamCounts
from powerflow_pipeline.data.preprocess.tasks.ingest import frame_paths
from powerflow_pipeline.data.preprocess.tasks.orient import OUTPUT_TIME_BASE
from powerflow_pipeline.data.preprocess.timeline import (
    creation_time_to_epoch_ms,
    derive_cut_interval,
    epoch_ms_series,
    select_closed_indices,
)

TIME_INDEXED_CSVS = ("odometry.csv", "imu.csv")
FRAME_TIMESTAMP_SOURCE = "odometry.csv:timestamp"


def read_lift_window(path: Path) -> tuple[int, int]:
    """Validate and read the lift window from the raw operator `metadata.yaml`.

    The `..._side_in_ms` field names are identifiers, not assertions about a camera: a
    single-camera capture spells them the same way, and every real August file already does.

    Raises `ScanRejected` with the exact field name at fault: a missing file reads as
    both fields missing, since neither can be trusted.
    """

    lift: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text())
        if isinstance(loaded, dict) and isinstance(loaded.get("lift"), dict):
            lift = loaded["lift"]

    values: dict[str, int] = {}
    for field in ("lift_start_time_side_in_ms", "lift_end_time_side_in_ms"):
        value = lift.get(field)
        if value is None:
            raise ScanRejected(f"missing lift window field: {field}")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ScanRejected(f"non-numeric lift window field: {field}")
        if value < 0:
            raise ScanRejected(f"negative lift window field: {field}")
        values[field] = round(value)

    start, end = values["lift_start_time_side_in_ms"], values["lift_end_time_side_in_ms"]
    if start >= end:
        raise ScanRejected(f"lift window out of order: start={start} end={end}")
    return start, end


def _camera_created_epoch_ms(record: CameraRecord) -> int:
    """`record.creation_time` as epoch milliseconds, or a `ScanRejected` naming the camera."""

    if not record.creation_time:
        raise ScanRejected(f"rgb creation time missing for camera {record.capture_id}")
    try:
        return creation_time_to_epoch_ms(record.creation_time)
    except ValueError as error:
        raise ScanRejected(
            "rgb creation time not convertible to epoch ms for camera "
            f"{record.capture_id}: {record.creation_time!r}"
        ) from error


def _read_timestamps(path: Path, name: str) -> list[float]:
    """The `timestamp` column of a time-indexed CSV, in row order."""

    frame = pd.read_csv(path, skipinitialspace=True)
    if "timestamp" not in frame.columns:
        raise ScanRejected(f"timestamps unavailable for {name}")
    return [float(value) for value in frame["timestamp"]]


def _owner_capture_span_ms(source: Path, created_epoch_ms: int) -> tuple[int, int]:
    """The interval owner's own start/end epoch, from `odometry.csv`'s timestamp column."""

    timestamps = _read_timestamps(source / "odometry.csv", "odometry.csv")
    if not timestamps:
        raise ScanRejected("timestamps unavailable for odometry.csv")
    epochs = epoch_ms_series(created_epoch_ms, timestamps[0], timestamps)
    return epochs[0], epochs[-1]


@task
def resolve_cut_interval(metadata_path: Path, side_record: CameraRecord) -> CutInterval:
    """Read the owner's lift window and derive the epoch interval its whole group shares.

    `metadata_path` is the capture's own governing file, resolved once by discovery, rather
    than rebuilt from a `<date>/<session>` pair the single-camera layout does not have.
    """

    log_task_paths(metadata_path, None)
    lift_start_ms, lift_end_ms = read_lift_window(metadata_path)
    side_created_epoch_ms = _camera_created_epoch_ms(side_record)
    cut_start_epoch_ms, cut_end_epoch_ms = derive_cut_interval(
        side_created_epoch_ms, lift_start_ms, lift_end_ms
    )

    capture_start_ms, capture_end_ms = _owner_capture_span_ms(
        side_record.source, side_created_epoch_ms
    )
    if cut_end_epoch_ms < capture_start_ms or cut_start_epoch_ms > capture_end_ms:
        raise ScanRejected(
            "cut interval outside side capture: "
            f"[{cut_start_epoch_ms},{cut_end_epoch_ms}] vs capture "
            f"[{capture_start_ms},{capture_end_ms}]"
        )

    assert side_record.creation_time is not None  # _camera_created_epoch_ms already proved this
    return CutInterval(
        cut_start_epoch_ms=cut_start_epoch_ms,
        cut_end_epoch_ms=cut_end_epoch_ms,
        side_creation_time=side_record.creation_time,
        side_created_epoch_ms=side_created_epoch_ms,
        lift_start_time_side_in_ms=lift_start_ms,
        lift_end_time_side_in_ms=lift_end_ms,
    )


def trim_rgb(source: Path, destination: Path, frame_epochs_ms: dict[int, int], crf: int) -> None:
    """Re-encode selected frames on a zero-based, odometry-derived timeline.

    `frame_epochs_ms` maps source decode indices to authoritative odometry epochs. Rebasing
    them to the first retained epoch removes the source clip's leading gap while preserving
    irregular capture timing at `OUTPUT_TIME_BASE` precision.
    """

    first_epoch_ms = next(iter(frame_epochs_ms.values()))
    with av.open(str(source)) as inp, av.open(str(destination), "w") as out:
        in_stream = inp.streams.video[0]
        in_stream.thread_type = "AUTO"

        out_stream = out.add_stream(
            "libx264", rate=in_stream.average_rate, options={"crf": str(crf)}
        )
        out_stream.width = in_stream.codec_context.width
        out_stream.height = in_stream.codec_context.height
        out_stream.pix_fmt = "yuv420p"
        out_stream.time_base = OUTPUT_TIME_BASE
        out_stream.codec_context.time_base = OUTPUT_TIME_BASE
        if (creation_time := inp.metadata.get("creation_time")) is not None:
            out.metadata["creation_time"] = creation_time

        for index, frame in enumerate(inp.decode(in_stream)):
            epoch_ms = frame_epochs_ms.get(index)
            if epoch_ms is None:
                continue
            frame.pts = round(Fraction(epoch_ms - first_epoch_ms, 1000) / OUTPUT_TIME_BASE)
            frame.time_base = OUTPUT_TIME_BASE
            for packet in out_stream.encode(frame):
                out.mux(packet)
        for packet in out_stream.encode():
            out.mux(packet)


def trim_png_stream(paths: list[Path], destination: Path, keep_indices: Sequence[int]) -> None:
    """Copy the kept frames byte-for-byte, reindexed to a contiguous `000000..` range."""

    destination.mkdir(parents=True)
    for new_index, old_index in enumerate(keep_indices):
        shutil.copy2(paths[old_index], destination / f"{new_index:06d}.png")


def filter_csv_rows(source: Path, destination: Path, keep_indices: Sequence[int]) -> None:
    """Keep the header plus exactly the data rows at `keep_indices`, in original order."""

    lines = source.read_text().splitlines(keepends=True)
    header, data = lines[0], lines[1:]
    destination.write_text(header + "".join(data[index] for index in keep_indices))


def _validate_exit(staging: Path) -> None:
    """Prove depth and confidence are still a matched pair before publishing."""

    depth_count = len(list((staging / "depth").glob("*.png")))
    confidence_count = len(list((staging / "confidence").glob("*.png")))
    if depth_count != confidence_count:
        raise ScanRejected(
            f"depth/confidence frame count mismatch after cut: {depth_count} vs {confidence_count}"
        )
    if depth_count == 0:
        raise ScanRejected("empty depth stream after cut")


@task(retries=1)
def cut_camera(
    record: CameraRecord, interval: CutInterval, config: PreprocessConfig
) -> tuple[CameraRecord, StepResult]:
    """Trim one camera's time-indexed streams to the shared epoch interval."""

    source = record.source
    destination = config.cut_root / record.relative
    log_task_paths(source, destination)
    created_epoch_ms = _camera_created_epoch_ms(record)

    depth_paths = frame_paths(source / "depth")
    confidence_paths = frame_paths(source / "confidence")
    if len(depth_paths) != len(confidence_paths):
        raise ScanRejected(
            "depth/confidence frame count mismatch before cut: "
            f"{len(depth_paths)} vs {len(confidence_paths)}"
        )

    odometry_ts = _read_timestamps(source / "odometry.csv", "odometry.csv")
    imu_ts = _read_timestamps(source / "imu.csv", "imu.csv")
    if not odometry_ts:
        raise ScanRejected("timestamps unavailable for odometry.csv")

    odom_anchor = odometry_ts[0]
    odom_epochs = epoch_ms_series(created_epoch_ms, odom_anchor, odometry_ts)
    imu_epochs = epoch_ms_series(created_epoch_ms, odom_anchor, imu_ts)

    start_ms, end_ms = interval.cut_start_epoch_ms, interval.cut_end_epoch_ms
    depth_keep = select_closed_indices(odom_epochs, start_ms, end_ms)
    imu_keep = select_closed_indices(imu_epochs, start_ms, end_ms)

    if not depth_keep:
        raise ScanRejected("empty odometry.csv in cut interval")

    paired_count = min(
        record.counts.rgb,
        len(depth_paths),
        len(confidence_paths),
        len(odometry_ts),
    )
    paired_keep = [index for index in depth_keep if index < paired_count]
    if not paired_keep:
        raise ScanRejected(f"no rgb frames in cut interval for camera {record.capture_id}")
    if not imu_keep:
        raise ScanRejected("empty imu.csv in cut interval")

    rgb_keep = paired_keep
    depth_keep = paired_keep
    n_frames = len(paired_keep)

    sidecar = {
        "cut_start_epoch_ms": start_ms,
        "cut_end_epoch_ms": end_ms,
        "side_creation_time": interval.side_creation_time,
        "lift_start_time_side_in_ms": interval.lift_start_time_side_in_ms,
        "lift_end_time_side_in_ms": interval.lift_end_time_side_in_ms,
        "camera_creation_time": record.creation_time,
        "frame_timestamp_source": FRAME_TIMESTAMP_SOURCE,
        "n_frames": n_frames,
        "retained": {
            "rgb": {
                "count": len(rgb_keep),
                "epoch_range": [odom_epochs[rgb_keep[0]], odom_epochs[rgb_keep[-1]]],
            },
            "depth": {
                "count": len(depth_keep),
                "epoch_range": [odom_epochs[depth_keep[0]], odom_epochs[depth_keep[-1]]],
            },
            "confidence": {
                "count": len(depth_keep),
                "epoch_range": [odom_epochs[depth_keep[0]], odom_epochs[depth_keep[-1]]],
            },
            "odometry": {
                "count": len(depth_keep),
                "epoch_range": [odom_epochs[depth_keep[0]], odom_epochs[depth_keep[-1]]],
            },
            "imu": {
                "count": len(imu_keep),
                "epoch_range": [imu_epochs[imu_keep[0]], imu_epochs[imu_keep[-1]]],
            },
        },
    }

    file_ops = [
        FileOp(op="write", src=source / "rgb.mp4", dst=destination / "rgb.mp4"),
        FileOp(op="write", src=source / "depth", dst=destination / "depth"),
        FileOp(op="write", src=source / "confidence", dst=destination / "confidence"),
        FileOp(op="copy", src=source / "camera_matrix.csv", dst=destination / "camera_matrix.csv"),
        FileOp(op="write", src=source / "odometry.csv", dst=destination / "odometry.csv"),
        FileOp(op="write", src=source / "imu.csv", dst=destination / "imu.csv"),
        FileOp(op="write", src=source, dst=destination / "cut_sidecar.json"),
        FileOp(op="publish", src=source, dst=destination),
    ]
    window_ms = interval.lift_end_time_side_in_ms - interval.lift_start_time_side_in_ms
    warnings: list[str] = []
    if window_ms < round(config.cut_min_window_s * 1000):
        # Never a rejection: a snatch really can take under two seconds, and the shortest
        # real windows observed (1.64 s, 1.75 s) are plausible lifts. Worth surfacing, in
        # case the window was mis-measured rather than short.
        warnings.append(
            f"lift window is {window_ms / 1000:.2f} s, below "
            f"cut_min_window_s={config.cut_min_window_s}"
        )

    step = StepResult(
        derived={
            "cut_start_epoch_ms": start_ms,
            "cut_end_epoch_ms": end_ms,
            "n_frames": n_frames,
            "rgb_kept": len(rgb_keep),
            "depth_kept": len(depth_keep),
            "imu_kept": len(imu_keep),
            "frame_timestamp_source": FRAME_TIMESTAMP_SOURCE,
        },
        warnings=warnings,
        file_ops=file_ops,
    )
    cut_record = record.model_copy(
        update={
            "source": destination,
            "counts": StreamCounts(
                rgb=len(rgb_keep),
                depth=len(depth_keep),
                confidence=len(depth_keep),
                odometry=len(depth_keep),
                imu=len(imu_keep),
            ),
            "n_frames": n_frames,
        }
    )

    if config.dry_run:
        return cut_record, step

    staging = create_staging_root(destination)
    try:
        rgb_frame_epochs_ms = {index: odom_epochs[index] for index in rgb_keep}
        trim_rgb(source / "rgb.mp4", staging / "rgb.mp4", rgb_frame_epochs_ms, config.rgb_crf)
        trim_png_stream(depth_paths, staging / "depth", depth_keep)
        trim_png_stream(confidence_paths, staging / "confidence", depth_keep)
        shutil.copy2(source / "camera_matrix.csv", staging / "camera_matrix.csv")
        filter_csv_rows(source / "odometry.csv", staging / "odometry.csv", depth_keep)
        filter_csv_rows(source / "imu.csv", staging / "imu.csv", imu_keep)
        write_json(staging / "cut_sidecar.json", sidecar)

        _validate_exit(staging)

        if config.overwrite and destination.exists():
            shutil.rmtree(destination)
        publish_staging(staging, destination)
    finally:
        cleanup(staging)

    return cut_record, step
