"""S3 · Floor-plane retilt → establishes I3.

Fits one floor plane per camera from sampled, back-projected depth points (`retilt.py`),
derives the rectifying tilt/roll, and rewarps RGB/depth/confidence with the resulting
homography. Depth is the geometric source of truth; the odometry-derived gravity vector
only checks the fit, warn-only, pending calibration against real sessions (see
`retilt.py`'s module docstring and `docs/specs/preprocessing/4-retilt.md` §7).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import av
import cv2
import numpy as np
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
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.retilt import (
    ConfidenceMode,
    backproject,
    depth_intrinsics,
    depth_scale_map,
    fit_plane,
    gravity_agreement_deg,
    gravity_down_camera,
    homography,
    read_floor_region,
    rectifying_rotation,
    select_floor_pixels,
    tilt_roll_from_normal,
    translation_span_m,
    valid_bounds,
)
from powerflow_pipeline.data.preprocess.tasks.ingest import (
    CONFIDENCE_VALUES,
    frame_paths,
    read_frame,
)
from powerflow_pipeline.data.preprocess.tasks.orient import OUTPUT_TIME_BASE

PASSTHROUGH = ("imu.csv",)


def _read_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ScanRejected(f"missing session metadata: {path}")
    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _sample_indices(n_frames: int, config: PreprocessConfig) -> list[int]:
    indices = list(range(0, n_frames, config.retilt_sample_stride))[
        : config.retilt_max_sampled_frames
    ]
    if not indices:
        raise ScanRejected("no frames available to sample for the floor fit")
    return indices


def _pool_floor_points(
    sample_indices: list[int],
    depth_paths: list[Path],
    confidence_paths: list[Path],
    bounds: tuple[int, int, int, int],
    k_d: object,
    config: PreprocessConfig,
) -> tuple[np.ndarray, list[ConfidenceMode]]:
    """Select and back-project floor pixels from every sampled frame, then pool them."""

    pooled: list[np.ndarray] = []
    modes: list[ConfidenceMode] = []
    for index in sample_indices:
        depth_frame = read_frame(depth_paths[index])
        confidence_frame = read_frame(confidence_paths[index])
        rows, cols, mode = select_floor_pixels(
            depth_frame, confidence_frame, bounds, config.retilt_conf2_area_fraction
        )
        modes.append(mode)
        if rows.size:
            pooled.append(backproject(rows, cols, depth_frame[rows, cols], k_d))  # type: ignore[arg-type]

    points = np.concatenate(pooled, axis=0) if pooled else np.empty((0, 3))
    if points.shape[0] > config.retilt_max_pooled_points:
        stride = max(1, points.shape[0] // config.retilt_max_pooled_points)
        points = points[::stride][: config.retilt_max_pooled_points]
    return points, modes


def _remap_maps(h: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """`cv2.remap` coordinate maps for output-pixel -> source-pixel via `H^-1` (§5)."""

    width, height = size
    h_inv = np.linalg.inv(h)
    cols, rows = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    ones = np.ones_like(cols)
    source = np.stack([cols, rows, ones], axis=-1) @ h_inv.T
    map_x = (source[..., 0] / source[..., 2]).astype(np.float32)
    map_y = (source[..., 1] / source[..., 2]).astype(np.float32)
    return map_x, map_y


def warp_rgb(
    source: Path, destination: Path, h: np.ndarray, size: tuple[int, int], crf: int
) -> None:
    """Warp every RGB frame by `H` (bilinear); PTS/time-base carried through unchanged --
    Retilt never touches frame count or timing, only pixel content (§5)."""

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
            warped = cv2.warpPerspective(array, h, (width, height), flags=cv2.INTER_LINEAR)
            out_frame = av.VideoFrame.from_ndarray(warped.astype(np.uint8), format="bgr24")
            out_frame.pts = frame.pts
            out_frame.time_base = frame.time_base
            for packet in out_stream.encode(out_frame):
                out.mux(packet)
        for packet in out_stream.encode():
            out.mux(packet)


def warp_png_stream(
    paths: list[Path],
    destination: Path,
    map_x: np.ndarray,
    map_y: np.ndarray,
    scale_map: np.ndarray | None,
) -> None:
    """Rectify a PNG frame stream with a precomputed nearest-neighbour remap (§5).

    `scale_map`, when given, rescales the nearest-sampled *value* (depth only) for the
    tilted optical axis; confidence values pass through untouched -- nearest-neighbour
    plus a `0` border never invents a value the sensor did not report.
    """

    destination.mkdir(parents=True)
    for path in paths:
        frame = read_frame(path)
        warped = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if scale_map is not None:
            scaled = warped.astype(np.float64) * scale_map
            warped = np.clip(scaled, 0, np.iinfo(np.uint16).max).round().astype(np.uint16)
        cv2.imwrite(str(destination / path.name), warped)


def _validate_exit(
    staging: Path,
    record: CameraRecord,
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
    depth_paths: list[Path],
    confidence_paths: list[Path],
) -> None:
    """Mirror S1 Orient's `_validate_exit`: prove I3 on the staged result, never after."""

    with av.open(str(staging / "rgb.mp4")) as container:
        stream = container.streams.video[0]
        out_size = (stream.codec_context.width, stream.codec_context.height)
        n_rgb = sum(1 for _ in container.decode(stream))
    if out_size != rgb_size:
        raise ScanRejected(f"rgb size changed during retilt: {out_size} vs {rgb_size}")
    if n_rgb != record.n_frames:
        raise ScanRejected(f"rgb frame count changed during retilt: {n_rgb} vs {record.n_frames}")

    for name, source_paths in (("depth", depth_paths), ("confidence", confidence_paths)):
        emitted = sorted((staging / name).glob("*.png"))
        if [path.name for path in emitted] != [path.name for path in source_paths]:
            raise ScanRejected(f"{name} frame filenames changed during retilt")

    depth = read_frame(staging / "depth" / depth_paths[0].name)
    if depth.dtype != np.uint16:
        raise ScanRejected(f"depth dtype changed during retilt: {depth.dtype}")
    if (depth.shape[1], depth.shape[0]) != depth_size:
        raise ScanRejected(f"depth size changed during retilt: {depth.shape[::-1]} vs {depth_size}")

    confidence = read_frame(staging / "confidence" / confidence_paths[0].name)
    outside = sorted(set(np.unique(confidence).tolist()) - CONFIDENCE_VALUES)
    if outside:
        raise ScanRejected(f"confidence values invented during retilt: {outside}")


@task(retries=1)
def retilt_camera(
    record: CameraRecord, raw_root: Path, config: PreprocessConfig
) -> tuple[CameraRecord, StepResult]:
    """Fit one camera's floor plane, derive its rectifying rotation, and rewarp it whole.

    `record` is S2's *output* record (portrait dimensions, rotated `K`); `raw_root` is
    needed because the floor region lives in the raw session `metadata.yaml` (§1), not
    anywhere in S2's output tree.
    """

    source = record.source
    destination = config.retilt_root / record.relative
    log_task_paths(source, destination)

    metadata = _read_metadata(raw_root / record.date / record.session / "metadata.yaml")
    region = read_floor_region(metadata, record.camera)
    bounds = region.pixel_bounds(record.depth_width, record.depth_height)

    file_ops = [
        FileOp(op="write", src=source / "rgb.mp4", dst=destination / "rgb.mp4"),
        FileOp(op="write", src=source / "depth", dst=destination / "depth"),
        FileOp(op="write", src=source / "confidence", dst=destination / "confidence"),
        FileOp(op="copy", src=source / "camera_matrix.csv", dst=destination / "camera_matrix.csv"),
        FileOp(op="copy", src=source / "odometry.csv", dst=destination / "odometry.csv"),
        FileOp(op="copy", src=source / "imu.csv", dst=destination / "imu.csv"),
        FileOp(op="write", src=source, dst=destination / "retilt_sidecar.json"),
        FileOp(op="publish", src=source, dst=destination),
    ]
    retilt_record = record.model_copy(update={"source": destination})

    if config.dry_run:
        # Unlike orient_camera, retilt's derived values need real pixel reads (the plane
        # fit), not just the CameraRecord's dimensions -- and `source` may not exist yet
        # if an upstream stage also dry-ran. Plan only what needs no pixel data: the
        # floor-region validation above, and the file operations this step would perform.
        return retilt_record, StepResult(derived={"region_px": list(bounds)}, file_ops=file_ops)

    odometry = pd.read_csv(source / "odometry.csv", skipinitialspace=True)
    sample_indices = _sample_indices(record.n_frames, config)
    sampled = odometry.iloc[sample_indices]

    span = translation_span_m(sampled[["x", "y", "z"]].to_numpy(dtype=float))
    if span > config.retilt_max_translation_m:
        raise ScanRejected(
            f"camera translated {span:.4f} m across the sampled frames, exceeding "
            f"retilt_max_translation_m={config.retilt_max_translation_m}"
        )

    rgb_size = (record.rgb_width, record.rgb_height)
    depth_size = (record.depth_width, record.depth_height)
    k_d = depth_intrinsics(record.intrinsics, rgb_size=rgb_size, depth_size=depth_size)

    depth_paths = frame_paths(source / "depth")
    confidence_paths = frame_paths(source / "confidence")
    points, per_frame_modes = _pool_floor_points(
        sample_indices, depth_paths, confidence_paths, bounds, k_d, config
    )
    if points.shape[0] < config.retilt_min_floor_points:
        raise ScanRejected(
            f"only {points.shape[0]} floor points selected, below "
            f"retilt_min_floor_points={config.retilt_min_floor_points}"
        )

    # §1: report "conf2_only" only if every sampled frame individually chose it -- the
    # per-camera field would otherwise vacuously claim the majority's mode.
    confidence_mode: ConfidenceMode = (
        "conf2_only" if all(mode == "conf2_only" for mode in per_frame_modes) else "conf1_and_2"
    )
    plane = fit_plane(points, n_frames_sampled=len(sample_indices), confidence_mode=confidence_mode)
    if plane.rms_residual_m > config.retilt_max_plane_rms_m:
        raise ScanRejected(
            f"plane fit RMS {plane.rms_residual_m:.4f} m exceeds "
            f"retilt_max_plane_rms_m={config.retilt_max_plane_rms_m}"
        )

    tilt_deg, roll_deg = tilt_roll_from_normal(np.array(plane.normal))
    if abs(tilt_deg) > config.retilt_max_tilt_deg:
        raise ScanRejected(
            f"tilt {tilt_deg:.2f} deg exceeds retilt_max_tilt_deg={config.retilt_max_tilt_deg}"
        )
    if abs(roll_deg) > config.retilt_max_roll_deg:
        raise ScanRejected(
            f"roll {roll_deg:.2f} deg exceeds retilt_max_roll_deg={config.retilt_max_roll_deg}"
        )

    rotation = rectifying_rotation(tilt_deg, roll_deg)
    h_rgb = homography(record.intrinsics, rotation)
    h_depth = homography(k_d, rotation)
    bounds_valid = valid_bounds(h_rgb, rgb_size)
    scale_map = depth_scale_map(k_d, rotation, depth_size)

    first = sampled.iloc[0]
    g_camera = gravity_down_camera(
        float(first["qx"]), float(first["qy"]), float(first["qz"]), float(first["qw"])
    )
    agreement = gravity_agreement_deg(np.array(plane.normal), g_camera)

    warnings: list[str] = []
    if agreement > config.retilt_gravity_tolerance_deg:
        # Warn-only (§7): the axis-convention chain behind `g_camera` is unconfirmed
        # against real captures, so this never rejects, only flags the disagreement.
        warnings.append(
            f"gravity disagreement {agreement:.2f} deg exceeds "
            f"retilt_gravity_tolerance_deg={config.retilt_gravity_tolerance_deg}"
        )

    sidecar = {
        "tilt_deg": tilt_deg,
        "roll_deg": roll_deg,
        "floor_normal_cam": plane.normal,
        "plane_rms_residual_m": plane.rms_residual_m,
        "n_floor_points": plane.n_points,
        "n_frames_sampled": plane.n_frames_sampled,
        "confidence_mode": confidence_mode,
        "confidence_mode_per_frame": per_frame_modes,
        "homography_rgb": h_rgb.tolist(),
        "homography_depth": h_depth.tolist(),
        "gravity_agreement_deg": agreement,
        "gravity_check_mode": "warn",
        "valid_bounds_px": [bounds_valid.x0, bounds_valid.y0, bounds_valid.x1, bounds_valid.y1],
        "depth_values_recomputed": True,
        "k_rewritten": False,
        "region_normalized": region.model_dump(),
        "region_px": list(bounds),
        "sample_indices": sample_indices,
        "translation_span_m": span,
    }

    result = StepResult(
        derived={
            "tilt_deg": tilt_deg,
            "roll_deg": roll_deg,
            "gravity_agreement_deg": agreement,
            "plane_rms_residual_m": plane.rms_residual_m,
        },
        warnings=warnings,
        file_ops=file_ops,
    )

    staging = create_staging_root(destination)
    try:
        warp_rgb(source / "rgb.mp4", staging / "rgb.mp4", h_rgb, rgb_size, config.rgb_crf)

        map_x, map_y = _remap_maps(h_depth, depth_size)
        warp_png_stream(depth_paths, staging / "depth", map_x, map_y, scale_map)
        warp_png_stream(confidence_paths, staging / "confidence", map_x, map_y, None)

        shutil.copy2(source / "camera_matrix.csv", staging / "camera_matrix.csv")
        shutil.copy2(source / "odometry.csv", staging / "odometry.csv")
        for name in PASSTHROUGH:
            shutil.copy2(source / name, staging / name)
        write_json(staging / "retilt_sidecar.json", sidecar)

        _validate_exit(staging, record, rgb_size, depth_size, depth_paths, confidence_paths)

        if config.overwrite and destination.exists():
            shutil.rmtree(destination)
        publish_staging(staging, destination)
    finally:
        cleanup(staging)

    return retilt_record, result
