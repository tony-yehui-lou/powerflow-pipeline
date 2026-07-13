"""Pure rotation maths for S1. No I/O, no Prefect, no side effects.

A 90 degree rotation is a transpose plus a flip: every output pixel *is* an input pixel,
moved. Nothing is resampled, so the spec's nearest-neighbour requirement holds by
construction. Never express this as a warp -- `cv2.warpAffine` interpolates, which
fabricates depths across edges and invents a confidence `1` between `0` and `2`.
"""

from __future__ import annotations

import cv2
import numpy as np

from powerflow_pipeline.data.preprocess.config import RotationDirection
from powerflow_pipeline.data.preprocess.models import Intrinsics

_CV2_ROTATION = {
    RotationDirection.CW: cv2.ROTATE_90_CLOCKWISE,
    RotationDirection.CCW: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# ffmpeg's transpose filter, used for the RGB stream.
_FFMPEG_TRANSPOSE = {RotationDirection.CW: "clock", RotationDirection.CCW: "cclock"}


def rotated_size(width: int, height: int) -> tuple[int, int]:
    """Return the `(width, height)` a `width x height` image has after a quarter turn."""

    return height, width


def transpose_filter(rotation: RotationDirection) -> str:
    """Return the ffmpeg `transpose` argument for this rotation."""

    return _FFMPEG_TRANSPOSE[rotation]


def rotate_frame(frame: np.ndarray, rotation: RotationDirection) -> np.ndarray:
    """Rotate one image a quarter turn, moving pixels without resampling them.

    CW maps source `(u, v)` to `(H - 1 - v, u)`; CCW maps it to `(v, W - 1 - u)`.
    """

    rotated: np.ndarray = cv2.rotate(frame, _CV2_ROTATION[rotation])
    return rotated


def rotate_intrinsics(
    intrinsics: Intrinsics, *, width: int, height: int, rotation: RotationDirection
) -> Intrinsics:
    """Rotate `K` to describe the rotated image, given the *source* image size.

    The focal lengths swap axes; the principal point follows the same pixel mapping as
    the image itself. Distortion coefficients are rotation-invariant (and always absent
    in this capture).
    """

    if rotation is RotationDirection.CW:
        cx, cy = height - 1 - intrinsics.cy, intrinsics.cx
    else:
        cx, cy = intrinsics.cy, width - 1 - intrinsics.cx

    return Intrinsics(
        fx=intrinsics.fy,
        fy=intrinsics.fx,
        cx=cx,
        cy=cy,
        distortion=intrinsics.distortion,
        frame="landscape" if intrinsics.frame == "portrait" else "portrait",
    )
