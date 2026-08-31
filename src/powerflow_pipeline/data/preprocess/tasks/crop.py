"""S4 · Common-region crop -> establishes per-camera I5.

Measures residual camera translation from odometry, reads S3's valid-content region, and
crops RGB/depth/confidence to their intersection -- a pure slice, never a reprojection
(see docs/specs/preprocessing/6-cropping.md).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from prefect import task

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.filesystem import (
    cleanup,
    create_staging_root,
    publish_staging,
    write_json,
)
from powerflow_pipeline.data.common.models import CropBounds, FileOp, StepResult
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.crop import (
    crop_fractions,
    crop_intrinsics,
    depth_bounds,
    guard_depth_m,
    intersect_crop,
    max_translation_m,
    motion_bounds_px,
    read_valid_bounds,
    reference_displacements,
    shrink_to_even,
)
from powerflow_pipeline.data.preprocess.models import CameraRecord, Intrinsics
from powerflow_pipeline.data.preprocess.tasks.ingest import (
    CONFIDENCE_VALUES,
    frame_paths,
    read_frame,
)
from powerflow_pipeline.data.preprocess.tasks.orient import (
    OUTPUT_TIME_BASE,
    preflight,
    write_camera_matrix,
)

PASSTHROUGH = ("imu.csv",)


def crop_rgb(
    source: Path, destination: Path, bounds: CropBounds, size: tuple[int, int], crf: int
) -> None:
    """Slice every RGB frame to `bounds`; PTS/time-base carried through unchanged -- a crop
    never touches frame count or timing, only pixel extent (§4)."""

    width, height = size
    with av.open(str(source)) as inp, av.open(str(destination), "w") as out:
        in_stream = inp.streams.video[0]
        in_stream.thread_type = "AUTO"

        out_stream = out.add_stream(
            "libx264", rate=in_stream.average_rate, options={"crf": str(crf)}
        )
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"
        out_stream.time_base = OUTPUT_TIME_BASE
        out_stream.codec_context.time_base = OUTPUT_TIME_BASE
        if (creation_time := inp.metadata.get("creation_time")) is not None:
            out.metadata["creation_time"] = creation_time

        for frame in inp.decode(in_stream):
            array = frame.to_ndarray(format="bgr24")
            cropped = array[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
            out_frame = av.VideoFrame.from_ndarray(cropped.astype(np.uint8), format="bgr24")
            out_frame.pts = frame.pts
            out_frame.time_base = frame.time_base
            for packet in out_stream.encode(out_frame):
                out.mux(packet)
        for packet in out_stream.encode():
            out.mux(packet)


def crop_png_stream(paths: list[Path], destination: Path, bounds: CropBounds) -> None:
    """Slice a PNG frame stream to `bounds`. No interpolation, no value change (§4)."""

    destination.mkdir(parents=True)
    for path in paths:
        frame = read_frame(path)
        cropped = frame[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
        cv2.imwrite(str(destination / path.name), cropped)


def _validate_exit(
    staging: Path,
    record: CameraRecord,
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
    depth_paths: list[Path],
    confidence_paths: list[Path],
    crop_bounds: CropBounds,
    valid_bounds: CropBounds,
    cropped_intrinsics: Intrinsics,
) -> None:
    """Mirror S3 Retilt's `_validate_exit`: prove per-camera I5 on the staged result."""

    with av.open(str(staging / "rgb.mp4")) as container:
        stream = container.streams.video[0]
        out_size = (stream.codec_context.width, stream.codec_context.height)
        n_rgb = sum(1 for _ in container.decode(stream))
    if out_size != rgb_size:
        raise ScanRejected(f"rgb size changed during crop: {out_size} vs expected {rgb_size}")
    if out_size[0] % 2 or out_size[1] % 2:
        raise ScanRejected(f"rgb output dimensions are not both even: {out_size}")
    if n_rgb != record.n_frames:
        raise ScanRejected(f"rgb frame count changed during crop: {n_rgb} vs {record.n_frames}")

    for name, source_paths in (("depth", depth_paths), ("confidence", confidence_paths)):
        emitted = sorted((staging / name).glob("*.png"))
        if [path.name for path in emitted] != [path.name for path in source_paths]:
            raise ScanRejected(f"{name} frame filenames changed during crop")

    depth = read_frame(staging / "depth" / depth_paths[0].name)
    if depth.dtype != np.uint16:
        raise ScanRejected(f"depth dtype changed during crop: {depth.dtype}")
    if (depth.shape[1], depth.shape[0]) != depth_size:
        raise ScanRejected(f"depth size changed during crop: {depth.shape[::-1]} vs {depth_size}")

    confidence = read_frame(staging / "confidence" / confidence_paths[0].name)
    outside = sorted(set(np.unique(confidence).tolist()) - CONFIDENCE_VALUES)
    if outside:
        raise ScanRejected(f"confidence values invented during crop: {outside}")

    if not (
        valid_bounds.x0 <= crop_bounds.x0
        and crop_bounds.x1 <= valid_bounds.x1
        and valid_bounds.y0 <= crop_bounds.y0
        and crop_bounds.y1 <= valid_bounds.y1
    ):
        raise ScanRejected("crop_bounds_px is not inside valid_bounds_px")

    check = preflight(cropped_intrinsics, rgb_size[0], rgb_size[1])
    if not check["principal_point_inside_image"]:
        raise ScanRejected("cropped principal point falls outside the cropped image")


@task(retries=1)
def crop_camera(record: CameraRecord, config: PreprocessConfig) -> tuple[CameraRecord, StepResult]:
    """Measure residual translation, intersect it with S3's valid-content region, and crop
    one camera's streams to the result whole -- establishing per-camera I5.

    `record` is S3's output record; unlike S3 Retilt, no `raw_root` is needed -- every input
    S4 reads lives in S3's own output tree.
    """

    source = record.source
    destination = config.crop_root / record.relative
    log_task_paths(source, destination)

    file_ops = [
        FileOp(op="write", src=source / "rgb.mp4", dst=destination / "rgb.mp4"),
        FileOp(op="write", src=source / "depth", dst=destination / "depth"),
        FileOp(op="write", src=source / "confidence", dst=destination / "confidence"),
        FileOp(op="write", src=source / "camera_matrix.csv", dst=destination / "camera_matrix.csv"),
        FileOp(op="copy", src=source / "odometry.csv", dst=destination / "odometry.csv"),
        FileOp(op="copy", src=source / "imu.csv", dst=destination / "imu.csv"),
        FileOp(op="write", src=source, dst=destination / "crop_sidecar.json"),
        FileOp(op="publish", src=source, dst=destination),
    ]
    crop_record = record.model_copy(update={"source": destination})

    if config.dry_run:
        # Unlike orient_camera, crop's derived values need the S3 sidecar S3 may not have
        # written yet (an upstream stage can also be dry-running) -- plan only the file
        # operations this step would perform, the same shape retilt_camera's own dry-run
        # early-return uses.
        return crop_record, StepResult(file_ops=file_ops)

    sidecar = json.loads((source / "retilt_sidecar.json").read_text())
    valid = read_valid_bounds(sidecar)

    odometry = pd.read_csv(source / "odometry.csv", skipinitialspace=True).iloc[: record.n_frames]
    displacements = reference_displacements(
        odometry[["x", "y", "z"]].to_numpy(dtype=float),
        odometry[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float),
    )
    max_t = max_translation_m(displacements)
    if max_t > config.crop_max_residual_translation_m:
        raise ScanRejected(
            f"camera translated {max_t:.4f} m across the retained frames, exceeding "
            f"crop_max_residual_translation_m={config.crop_max_residual_translation_m}"
        )

    depth_paths = frame_paths(source / "depth")
    confidence_paths = frame_paths(source / "confidence")
    sample = list(range(0, len(depth_paths), config.crop_depth_sample_stride))[
        : config.crop_max_sampled_depth_frames
    ]
    depth_samples: list[np.ndarray] = []
    for index in sample:
        depth_frame = read_frame(depth_paths[index])
        confidence_frame = read_frame(confidence_paths[index])
        mask = (depth_frame > 0) & (confidence_frame > 0)
        depth_samples.append(depth_frame[mask].astype(np.float64) / 1000.0)
    z_guard_m = guard_depth_m(np.concatenate(depth_samples), config.crop_depth_guard_quantile)

    rgb_size = (record.rgb_width, record.rgb_height)
    depth_size = (record.depth_width, record.depth_height)
    motion = motion_bounds_px(
        displacements, record.intrinsics, valid, z_guard_m, config.crop_safety_px, rgb_size
    )
    intersected, bound_source = intersect_crop(valid, motion, rgb_size)
    crop_bounds = shrink_to_even(intersected)
    even_shrink = (
        intersected.width - crop_bounds.width,
        intersected.height - crop_bounds.height,
    )

    frac_w, frac_h = crop_fractions(crop_bounds, rgb_size)
    if frac_w > config.crop_max_crop_fraction or frac_h > config.crop_max_crop_fraction:
        raise ScanRejected(
            f"crop removes {frac_w:.3f}/{frac_h:.3f} of width/height, exceeding "
            f"crop_max_crop_fraction={config.crop_max_crop_fraction}"
        )

    depth_crop = depth_bounds(crop_bounds, rgb_size, depth_size)
    cropped_intrinsics = crop_intrinsics(record.intrinsics, crop_bounds)
    crop_output_size = (crop_bounds.width, crop_bounds.height)
    depth_output_size = (depth_crop.width, depth_crop.height)

    sidecar_out = {
        "reference_frame": 0,
        "max_translation_m": max_t,
        "crop_max_residual_translation_m": config.crop_max_residual_translation_m,
        "z_guard_m": z_guard_m,
        "crop_depth_guard_quantile": config.crop_depth_guard_quantile,
        "crop_safety_px": config.crop_safety_px,
        "valid_bounds_px": [valid.x0, valid.y0, valid.x1, valid.y1],
        "motion_bounds_px": list(motion),
        "crop_bounds_px": [crop_bounds.x0, crop_bounds.y0, crop_bounds.x1, crop_bounds.y1],
        "bound_source": bound_source,
        "even_shrink_px": list(even_shrink),
        "depth_crop_bounds_px": [depth_crop.x0, depth_crop.y0, depth_crop.x1, depth_crop.y1],
        "confidence_crop_bounds_px": [
            depth_crop.x0,
            depth_crop.y0,
            depth_crop.x1,
            depth_crop.y1,
        ],
        "rgb_input_size": list(rgb_size),
        "rgb_output_size": list(crop_output_size),
        "depth_input_size": list(depth_size),
        "depth_output_size": list(depth_output_size),
        "crop_fraction_wh": [frac_w, frac_h],
        "camera_translation_corrected": False,
        "pixels_shifted": False,
        "k_rewritten": True,
    }

    result = StepResult(
        derived={
            "max_translation_m": max_t,
            "z_guard_m": z_guard_m,
            "crop_bounds_px": sidecar_out["crop_bounds_px"],
            "rgb_output_size": list(crop_output_size),
        },
        file_ops=file_ops,
    )

    staging = create_staging_root(destination)
    try:
        crop_rgb(
            source / "rgb.mp4", staging / "rgb.mp4", crop_bounds, crop_output_size, config.rgb_crf
        )
        crop_png_stream(depth_paths, staging / "depth", depth_crop)
        crop_png_stream(confidence_paths, staging / "confidence", depth_crop)

        write_camera_matrix(staging / "camera_matrix.csv", cropped_intrinsics)
        shutil.copy2(source / "odometry.csv", staging / "odometry.csv")
        for name in PASSTHROUGH:
            shutil.copy2(source / name, staging / name)
        write_json(staging / "crop_sidecar.json", sidecar_out)

        _validate_exit(
            staging,
            record,
            crop_output_size,
            depth_output_size,
            depth_paths,
            confidence_paths,
            crop_bounds,
            valid,
            cropped_intrinsics,
        )

        if config.overwrite and destination.exists():
            shutil.rmtree(destination)
        publish_staging(staging, destination)
    finally:
        cleanup(staging)

    crop_record = crop_record.model_copy(
        update={
            "rgb_width": crop_output_size[0],
            "rgb_height": crop_output_size[1],
            "depth_width": depth_output_size[0],
            "depth_height": depth_output_size[1],
            "intrinsics": cropped_intrinsics,
        }
    )
    return crop_record, result
