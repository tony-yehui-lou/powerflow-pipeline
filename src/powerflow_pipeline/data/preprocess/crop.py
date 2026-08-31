"""Pure maths for S4 Crop: residual translation, guard band, intersection (6-cropping.md)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import ValidationError

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.models import CropBounds
from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.retilt import quat_to_matrix


def read_valid_bounds(sidecar: dict[str, Any]) -> CropBounds:
    """Parse S3's `valid_bounds_px` flat 4-list, in rectified RGB pixel coordinates (§2)."""

    raw = sidecar.get("valid_bounds_px")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ScanRejected(f"retilt_sidecar.json valid_bounds_px missing or malformed: {raw!r}")
    try:
        x0, y0, x1, y1 = (int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ScanRejected(f"valid_bounds_px entries are not integers: {raw!r}") from exc
    try:
        return CropBounds(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValidationError as exc:
        raise ScanRejected(f"valid_bounds_px degenerate: {raw!r}") from exc


def reference_displacements(positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    """`Δp_i = R_0^T (p_i - p_0)`, expressed in the frame-0 camera frame (§1)."""

    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(quaternions)):
        raise ScanRejected("odometry pose is not finite")
    r0 = quat_to_matrix(*quaternions[0])  # camera-0 -> world
    result: np.ndarray = (positions - positions[0]) @ r0
    return result


def max_translation_m(displacements: np.ndarray) -> float:
    """`max_i ||Δp_i||_2` (§1)."""

    return float(np.linalg.norm(displacements, axis=1).max())


def guard_depth_m(depth_m: np.ndarray, quantile: float) -> float:
    """Lower `quantile` of positive, confidence-nonzero depth samples, in metres (§3)."""

    if depth_m.size == 0:
        raise ScanRejected("no valid depth samples available to derive Z_guard_m")
    return float(np.quantile(depth_m, quantile))


def motion_bounds_px(
    displacements: np.ndarray,
    k: Intrinsics,
    valid: CropBounds,
    z_guard_m: float,
    safety_px: int,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """§3's conservative pixel guard band: `(left, top, right, bottom)`, possibly larger than
    `size` allows -- callers intersect against §2's valid region before using them."""

    r_x = max(k.cx - valid.x0, valid.x1 - 1 - k.cx)
    r_y = max(k.cy - valid.y0, valid.y1 - 1 - k.cy)
    dx, dy, dz = displacements[:, 0], displacements[:, 1], displacements[:, 2]
    m_x = k.fx * np.abs(dx) / z_guard_m + r_x * np.abs(dz) / z_guard_m
    m_y = k.fy * np.abs(dy) / z_guard_m + r_y * np.abs(dz) / z_guard_m
    left = right = math.ceil(float(m_x.max())) + safety_px
    top = bottom = math.ceil(float(m_y.max())) + safety_px
    return left, top, right, bottom


def intersect_crop(
    valid: CropBounds, motion: tuple[int, int, int, int], size: tuple[int, int]
) -> tuple[CropBounds, dict[str, str]]:
    """§3's final rectangle: the intersection of the valid-content region and the motion guard
    band, plus which rectangle determined each edge (a tie resolves to `"valid"`)."""

    left, top, right, bottom = motion
    width, height = size
    x0, x1 = max(valid.x0, left), min(valid.x1, width - right)
    y0, y1 = max(valid.y0, top), min(valid.y1, height - bottom)
    try:
        bounds = CropBounds(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValidationError as exc:
        raise ScanRejected(
            f"motion guard band and valid-content region do not overlap: "
            f"valid={valid}, motion={motion}"
        ) from exc
    source = {
        "left": "valid" if x0 == valid.x0 else "motion",
        "top": "valid" if y0 == valid.y0 else "motion",
        "right": "valid" if x1 == valid.x1 else "motion",
        "bottom": "valid" if y1 == valid.y1 else "motion",
    }
    return bounds, source


def shrink_to_even(bounds: CropBounds) -> CropBounds:
    """Trim at most one row/column off the bottom/right so both extents are even -- libx264 +
    yuv420p refuses an odd width or height."""

    x1 = bounds.x1 - (bounds.width % 2)
    y1 = bounds.y1 - (bounds.height % 2)
    return CropBounds(x0=bounds.x0, y0=bounds.y0, x1=x1, y1=y1)


def depth_bounds(
    crop: CropBounds, rgb_size: tuple[int, int], depth_size: tuple[int, int]
) -> CropBounds:
    """§3's inward-rounded rescale from the RGB crop to depth/confidence resolution."""

    rgb_width, rgb_height = rgb_size
    depth_width, depth_height = depth_size
    x0 = math.ceil(crop.x0 * depth_width / rgb_width)
    x1 = math.floor(crop.x1 * depth_width / rgb_width)
    y0 = math.ceil(crop.y0 * depth_height / rgb_height)
    y1 = math.floor(crop.y1 * depth_height / rgb_height)
    try:
        return CropBounds(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValidationError as exc:
        raise ScanRejected(f"depth crop rectangle degenerate: {[x0, y0, x1, y1]}") from exc


def crop_intrinsics(k: Intrinsics, bounds: CropBounds) -> Intrinsics:
    """§4: a pure origin shift, no resampling -- `fx`/`fy` unchanged."""

    return Intrinsics(fx=k.fx, fy=k.fy, cx=k.cx - bounds.x0, cy=k.cy - bounds.y0, frame=k.frame)


def crop_fractions(bounds: CropBounds, size: tuple[int, int]) -> tuple[float, float]:
    """Fraction of width and of height removed, for §6's `max_crop_fraction` gate."""

    width, height = size
    return 1.0 - bounds.width / width, 1.0 - bounds.height / height
