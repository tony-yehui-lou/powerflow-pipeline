"""Pure maths for S5 Pose's depth-based metric lift. No I/O, no Prefect, no side effects.

Turns one detected joint pixel (in S4 Crop's RGB image space) into a metric 3D point in
`PoseDocument`'s floor-anchored frame, reusing S3 Retilt's own back-projection
(`retilt.backproject`, 4-retilt.md §2) rather than reimplementing it -- the depth stream is
identical in kind, only read at one joint's neighbourhood instead of a pooled floor region.

Checked numerically against known-good geometry (CLAUDE.md: "verify geometry numerically ...
never accept a transform merely because it ran"), like every other pure-maths module here.
"""

from __future__ import annotations

import numpy as np

from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.retilt import backproject

PositionTriple = tuple[float, float, float]


def rgb_pixel_to_depth_pixel(
    row: int, col: int, rgb_size: tuple[int, int], depth_size: tuple[int, int]
) -> tuple[int, int]:
    """Map one RGB-resolution `(row, col)` to its nearest depth-resolution pixel.

    Same proportional scaling `retilt.depth_intrinsics` uses for `K`, rounded for indexing
    into the depth/confidence arrays rather than kept as a continuous intrinsic.
    """

    rgb_width, rgb_height = rgb_size
    depth_width, depth_height = depth_size
    sx = depth_width / rgb_width
    sy = depth_height / rgb_height
    return round(row * sy), round(col * sx)


def lookup_patch_depth_m(
    depth: np.ndarray, confidence: np.ndarray, row: int, col: int, radius: int = 2
) -> float | None:
    """Median metres depth in a `radius`-pixel box around `(row, col)`.

    Prefers confidence-`2` pixels, falls back to confidence `1` and `2` together, and never a
    zero (no-return) depth or a confidence-`0` reading -- the same confidence-based selection
    4-retilt.md §1 uses for the floor region, at the scale of one joint's neighbourhood.

    A confidence-`0` tier was tried and removed: measured on 11 July/50kg_Set3/Front it bought
    4% more joint-frames (5276 -> 5490) while inflating the median upper-arm bone from 0.38 m
    to 1.63 m, because a limb held out against a distant background lets an unvouched-for
    reading land on the floor metres behind the lifter. Coverage is not worth positions that
    are confidently wrong.

    Returns `None` when the box holds no usable depth at all (occlusion, out of LiDAR range,
    e.g. a wrist behind the barbell) -- the caller stores that as a dropped-out frame, never a
    fabricated position.
    """

    height, width = depth.shape
    r0, r1 = max(0, row - radius), min(height, row + radius + 1)
    c0, c1 = max(0, col - radius), min(width, col + radius + 1)
    patch_depth = depth[r0:r1, c0:c1]
    patch_confidence = confidence[r0:r1, c0:c1]

    for mask in (patch_confidence == 2, np.isin(patch_confidence, (1, 2))):
        selected = patch_depth[mask & (patch_depth > 0)]
        if selected.size:
            return float(np.median(selected)) / 1000.0
    return None


def backproject_pixel(row: int, col: int, depth_m: float, k_d: Intrinsics) -> PositionTriple:
    """One pixel's metric 3D point in the (rectified) camera frame.

    `retilt.backproject` for a single point rather than a pooled array; camera frame
    convention is unchanged: X right, Y down, Z forward.
    """

    point = backproject(np.array([row]), np.array([col]), np.array([depth_m * 1000.0]), k_d)
    x, y, z = point[0]
    return float(x), float(y), float(z)


def to_floor_frame(point_cam: PositionTriple, floor_offset_m: float) -> PositionTriple:
    """Camera-frame metres -> `PoseDocument`'s floor-anchored frame.

    S3 Retilt already rotates the camera so the floor normal is exactly `(0, -1, 0)` in the
    rectified frame S5 detects joints in (4-retilt.md §4), so no further rotation is needed
    here -- only a translation along Y from the camera's optical centre down to the floor.
    `floor_offset_m` (`PlaneFit`, fitted pre-rectification) is exactly that translation: a
    distance from the origin is invariant under any rotation about that same origin.

    X and Z pass through unchanged: `PoseDocument`'s floor-frame origin is the optical
    centre's own projection onto the floor, which shares the camera's X and Z.

    Open question, not silently resolved: this makes the floor frame left-handed under a
    strict determinant check once Y is flipped to point up, where `PoseDocument`'s own
    docstring says "right-handed". Nothing in this codebase computes a cross product or
    otherwise depends on handedness today, and keeping X-right/Z-forward matches what a
    viewer expects from the source video far more than an unmotivated axis flip would --
    see `docs/specs/preprocessing/S5-pose-model-recommendation.md`'s Risks section.
    """

    x, y, z = point_cam
    return x, floor_offset_m - y, z


def lift_pixel_to_floor_frame(
    row: int,
    col: int,
    depth: np.ndarray,
    confidence: np.ndarray,
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
    k_d: Intrinsics,
    floor_offset_m: float,
    radius: int = 2,
) -> PositionTriple | None:
    """One joint pixel (RGB image space) -> its floor-frame metric position, or `None`.

    Composes the pixel-space remap, the patch depth lookup, the back-projection, and the
    floor-frame translation into the single call `pose_model.py`'s implementations use per
    joint per frame.
    """

    depth_row, depth_col = rgb_pixel_to_depth_pixel(row, col, rgb_size, depth_size)
    depth_m = lookup_patch_depth_m(depth, confidence, depth_row, depth_col, radius=radius)
    if depth_m is None:
        return None
    point_cam = backproject_pixel(depth_row, depth_col, depth_m, k_d)
    return to_floor_frame(point_cam, floor_offset_m)
