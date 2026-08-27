"""Pure maths for S3. No I/O, no Prefect, no side effects.

Fits one floor plane per camera from pooled, back-projected depth points, derives the
tilt/roll that levels that plane, and builds the homography (and per-pixel depth-scale
map) that rectifies every stream. One plane, one rotation, one homography per camera --
never per frame (docs/specs/preprocessing/4-retilt.md, Assumptions).

**Correction to §4.** The spec's rectifying-rotation formula reads `R = R_x(tilt) @
R_z(roll)`. Verified numerically against random floor normals (CLAUDE.md: "verify
geometry numerically ... never accept a transform merely because it ran"), that formula
does not level the fitted normal; `R = R_x(tilt) @ R_z(-roll)` does, exactly, to floating
-point precision, for every normal the de-roll step (`n' = R_z(-roll) * n`) actually
de-rolls. `tilt_roll_from_normal` still implements the spec's `roll`/`tilt` formulas
verbatim -- only the final composition's sign is corrected.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.models import CropBounds
from powerflow_pipeline.data.preprocess.models import FloorRegion, Intrinsics, PlaneFit

ConfidenceMode = Literal["conf2_only", "conf1_and_2"]

# 4-connectivity structuring element for the largest-conf-2-component search (§1).
_FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def parse_region_point(raw: str) -> tuple[float, float]:
    """Parse a `"(x, y)"` on-disk region field into a float pair (4-retilt.md §1).

    Real captures write these as parenthesised strings, not YAML sequences -- treating
    the value as already-numeric rejects every real region.
    """

    text = raw.strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"not a parenthesised point: {raw!r}")
    parts = text[1:-1].split(",")
    if len(parts) != 2:
        raise ValueError(f"not a parenthesised point: {raw!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as error:
        raise ValueError(f"not a parenthesised point: {raw!r}") from error


def _region_prefix(camera: str) -> str:
    key = camera.strip().lower()
    if key == "front":
        return "front_"
    if key == "side":
        return "side_"
    raise ScanRejected(f"no floor-region convention for camera {camera!r}")


def read_floor_region(metadata: dict[str, Any], camera: str) -> FloorRegion:
    """Read this camera's operator-annotated floor region from the raw `metadata.yaml`.

    Raises `ScanRejected` naming the exact missing/malformed field, never guesses.
    """

    prefix = _region_prefix(camera)
    video = metadata.get("video")
    if not isinstance(video, dict):
        raise ScanRejected(f"missing floor region for camera {camera}: no video block")

    bl_field = f"{prefix}floor_region_bottom_left_in_pixels"
    tr_field = f"{prefix}floor_region_top_right_in_pixels"

    points: dict[str, tuple[float, float]] = {}
    for field in (bl_field, tr_field):
        raw = video.get(field)
        if raw is None:
            raise ScanRejected(f"missing floor region field: {field}")
        try:
            points[field] = parse_region_point(str(raw))
        except ValueError as error:
            raise ScanRejected(f"unparseable floor region field: {field}") from error

    x0, y0 = points[bl_field]
    x1, y1 = points[tr_field]
    try:
        return FloorRegion(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValueError as error:
        raise ScanRejected(f"invalid floor region for camera {camera}: {error}") from error


def depth_intrinsics(
    k: Intrinsics, rgb_size: tuple[int, int], depth_size: tuple[int, int]
) -> Intrinsics:
    """Scale RGB-resolution `K` to the depth resolution, pixel-centre correct (§2).

    `cx_d = (cx + 0.5) * s - 0.5`, not `cx * s` -- scaling the principal point alone
    biases every back-projected point by up to half a depth pixel.
    """

    rgb_width, rgb_height = rgb_size
    depth_width, depth_height = depth_size
    sx = depth_width / rgb_width
    sy = depth_height / rgb_height
    return Intrinsics(
        fx=k.fx * sx,
        fy=k.fy * sy,
        cx=(k.cx + 0.5) * sx - 0.5,
        cy=(k.cy + 0.5) * sy - 0.5,
        distortion=k.distortion,
        frame=k.frame,
    )


def select_floor_pixels(
    depth: np.ndarray,
    confidence: np.ndarray,
    bounds: tuple[int, int, int, int],
    conf2_area_fraction: float = 1 / 6,
) -> tuple[np.ndarray, np.ndarray, ConfidenceMode]:
    """Select floor pixels within `bounds` per §1; returns `(rows, cols, mode)`.

    `bounds` is `(col_start, row_start, col_end, row_end)`, as `FloorRegion.pixel_bounds`
    returns. The conf-1-and-2 fallback reads from the whole annotated region, not just
    the largest conf-2 component -- that component is conf-2 by construction, so limiting
    the fallback to it would make it vacuous.
    """

    col_start, row_start, col_end, row_end = bounds
    region_confidence = confidence[row_start:row_end, col_start:col_end]
    region_depth = depth[row_start:row_end, col_start:col_end]
    area = region_confidence.shape[0] * region_confidence.shape[1]

    conf2_mask = region_confidence == 2
    labels, n_labels = ndimage.label(conf2_mask, structure=_FOUR_CONNECTED)

    largest_size = 0
    largest_label = 0
    if n_labels:
        sizes = ndimage.sum(conf2_mask, labels, index=range(1, n_labels + 1))
        largest_label = int(np.argmax(sizes)) + 1
        largest_size = int(sizes[largest_label - 1])

    mode: ConfidenceMode
    if largest_size >= conf2_area_fraction * area:
        mask = labels == largest_label
        mode = "conf2_only"
    else:
        mask = np.isin(region_confidence, (1, 2))
        mode = "conf1_and_2"

    mask &= region_depth > 0  # a pixel with no depth return is never selected (§2)
    local_rows, local_cols = np.nonzero(mask)
    return local_rows + row_start, local_cols + col_start, mode


def backproject(
    rows: np.ndarray, cols: np.ndarray, depth_mm: np.ndarray, k_d: Intrinsics
) -> np.ndarray:
    """Back-project selected `(row, col)` pixels with `depth_mm` through `K_d` (§2).

    Camera frame: X right, Y down, Z forward. Returns `(N, 3)` metres.
    """

    z = np.asarray(depth_mm, dtype=np.float64) / 1000.0
    x = (np.asarray(cols, dtype=np.float64) - k_d.cx) / k_d.fx * z
    y = (np.asarray(rows, dtype=np.float64) - k_d.cy) / k_d.fy * z
    return np.stack([x, y, z], axis=-1)


def fit_plane(
    points: np.ndarray,
    *,
    n_frames_sampled: int = 0,
    confidence_mode: ConfidenceMode = "conf2_only",
) -> PlaneFit:
    """Fit `Y = a*X + b*Z + c` with `torch.linalg.lstsq` over the pooled cloud (§3).

    `n_frames_sampled`/`confidence_mode` are not derivable from the points alone; the
    caller (who ran the sampling and selection) passes the true values through.
    """

    x = np.asarray(points[:, 0], dtype=np.float64)
    y = np.asarray(points[:, 1], dtype=np.float64)
    z = np.asarray(points[:, 2], dtype=np.float64)
    n = points.shape[0]

    design = torch.stack(
        [
            torch.as_tensor(x, dtype=torch.float64),
            torch.as_tensor(z, dtype=torch.float64),
            torch.ones(n, dtype=torch.float64),
        ],
        dim=1,
    )
    target = torch.as_tensor(y, dtype=torch.float64).unsqueeze(1)
    solution = torch.linalg.lstsq(design, target).solution.squeeze(1)
    a, b, c = (float(value) for value in solution)

    normal = np.array([a, -1.0, b])
    normal_unit = normal / np.linalg.norm(normal)
    if normal_unit[1] > 0:  # orient toward the camera: "up" is -Y in this convention
        normal_unit = -normal_unit

    predicted_y = a * x + b * z + c
    residuals = (predicted_y - y) / math.sqrt(a**2 + 1 + b**2)
    rms = float(np.sqrt(np.mean(residuals**2)))

    return PlaneFit(
        normal=normal_unit.tolist(),
        rms_residual_m=rms,
        n_points=n,
        n_frames_sampled=n_frames_sampled,
        confidence_mode=confidence_mode,
    )


def _rotation_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tilt_roll_from_normal(n: np.ndarray) -> tuple[float, float]:
    """Derive `(tilt_deg, roll_deg)` from the fitted floor normal, verbatim from §4."""

    n = np.asarray(n, dtype=np.float64)
    roll = math.atan2(n[0], -n[1])
    n_prime = _rotation_z(-roll) @ n
    tilt = math.atan2(n_prime[2], -n_prime[1])
    return math.degrees(tilt), math.degrees(roll)


def rectifying_rotation(tilt_deg: float, roll_deg: float) -> np.ndarray:
    """The rotation that levels the fitted normal to `(0, -1, 0)` (§4, corrected -- see
    the module docstring)."""

    tilt = math.radians(tilt_deg)
    roll = math.radians(roll_deg)
    rotation: np.ndarray = _rotation_x(tilt) @ _rotation_z(-roll)
    return rotation


def homography(k: Intrinsics, r: np.ndarray) -> np.ndarray:
    """`H = K @ R @ K^-1`, the image-plane homography induced by rotating the camera (§5)."""

    k_matrix = np.array(k.to_matrix())
    h: np.ndarray = k_matrix @ r @ np.linalg.inv(k_matrix)
    return h


def depth_scale_map(k_d: Intrinsics, r: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Per-output-pixel `R_row3 . K_d^-1 . x~`, shape `(H, W)` (§5's depth-value correction).

    `size` is `(width, height)`; computed once per camera, applied to every frame's
    nearest-neighbour-sampled depth.
    """

    width, height = size
    k_inv = np.linalg.inv(np.array(k_d.to_matrix()))
    row3_over_k = r[2, :] @ k_inv  # (3,) -- dot with (u, v, 1) per output pixel
    cols, rows = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    scale_map: np.ndarray = row3_over_k[0] * cols + row3_over_k[1] * rows + row3_over_k[2]
    return scale_map


def valid_bounds(h: np.ndarray, size: tuple[int, int]) -> CropBounds:
    """Bounding box of the all-ones mask warped by `H` -- the rectified valid-content region."""

    width, height = size
    mask = np.ones((height, width), dtype=np.uint8)
    warped = cv2.warpPerspective(mask, h, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    rows, cols = np.nonzero(warped)
    if rows.size == 0:
        raise ScanRejected("rectification leaves no valid content in the frame")
    return CropBounds(
        x0=int(cols.min()),
        y0=int(rows.min()),
        x1=int(cols.max()) + 1,
        y1=int(rows.max()) + 1,
    )


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Standard unit-quaternion -> rotation-matrix conversion, `(x, y, z, w)` order."""

    matrix: np.ndarray = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return matrix


_R_Z_90_CW = _rotation_z(math.radians(90.0))


def gravity_down_camera(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Odometry quaternion -> 'down' in S2's rotated portrait camera frame.

    g_world -> ARKit camera axes (R_wc^T) -> CV camera axes (flip Y,Z) -> S2's 90 deg CW
    rotation of the image (not the odometry columns). See 4-retilt.md Assumptions.
    """

    r_wc = _quat_to_matrix(qx, qy, qz, qw)  # world -> ARKit camera-local
    g_world = np.array([0.0, -1.0, 0.0])  # ARKit world is gravity-aligned, Y up
    g_arkit = r_wc.T @ g_world
    g_cv = np.diag([1.0, -1.0, -1.0]) @ g_arkit  # ARKit (Y up, Z back) -> CV (Y down, Z fwd)
    g_portrait: np.ndarray = _R_Z_90_CW @ g_cv  # S2 rotates the image, not odometry
    return g_portrait


def gravity_agreement_deg(normal: np.ndarray, g_camera: np.ndarray) -> float:
    """Angle between the fitted floor normal and the odometry-derived gravity vector (§7)."""

    normal = np.asarray(normal, dtype=np.float64)
    g_camera = np.asarray(g_camera, dtype=np.float64)
    cos_angle = float(
        np.dot(normal, -g_camera) / (np.linalg.norm(normal) * np.linalg.norm(g_camera))
    )
    return math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))


def translation_span_m(xyz: np.ndarray) -> float:
    """Max pairwise distance across sampled-frame `(x, y, z)` odometry rows (§7)."""

    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 2:
        return 0.0
    diffs = xyz[:, None, :] - xyz[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return float(distances.max())
