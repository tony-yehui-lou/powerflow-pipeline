"""S8 · The body-plane solve: each joint on a vertical plane beside the bar's mid-plane.

`docs/specs/preprocessing/S8-joint-hva.md` (draft 1) §2.2, now S8's initial guess (draft 3 §3.4).
While the hands hold the bar, the lifter is near-symmetric about the plane through the bar centre
perpendicular to the bar. Joint `j` sits on the parallel plane at signed offset `d_j` along the bar
axis `a`, and its pixel's viewing ray `P = Z r` meets that plane at `Z = (C.a + d_j) / (r.a)`.

Closed form and independent per frame, so one bad frame cannot carry into the next -- which is
what a starting point for a local solver needs. Units are millimetres in S4's levelled camera
frame (X right, Y down, Z forward), as in `bar_geometry`.
"""

from __future__ import annotations

import math
from typing import Final, Literal

from powerflow_pipeline.data.common.models import JointId
from powerflow_pipeline.data.preprocess.bar_geometry import Pixel, Point, normalised
from powerflow_pipeline.data.preprocess.models import Intrinsics

Side = Literal["left", "right"]

# The ray must cross its plane, not run along it: |cos| of the angle between the ray and the bar
# axis below this and a lateral-offset error is amplified beyond use (draft 1 §2.2).
MIN_RAY_AXIS_COS: Final = 0.3


def place_on_plane(
    px: Pixel, k: Intrinsics, centre: Point, axis: Point, offset_mm: float
) -> Point | None:
    """Where `px`'s ray meets the plane `(P - centre).axis = offset_mm`; None if ill-conditioned."""

    x, y = normalised(px, k)
    ray = (x, y, 1.0)
    along = sum(r * a for r, a in zip(ray, axis, strict=True))
    if abs(along) / math.hypot(*ray) < MIN_RAY_AXIS_COS:
        return None
    z = (sum(c * a for c, a in zip(centre, axis, strict=True)) + offset_mm) / along
    if z <= 0:
        return None
    return (x * z, y * z, z)


def place_on_floor(px: Pixel, k: Intrinsics, y_mm: float) -> Point | None:
    """Where `px`'s ray meets the horizontal plane `y_mm` below the camera; None above the
    horizon, where the ray never reaches that plane."""

    x, y = normalised(px, k)
    if y <= 0:
        return None
    z = y_mm / y
    return (x * z, y_mm, z)


def lateral_offsets(
    *, near_side: Side, half_grip_mm: float, shoulder_width_mm: float, hip_width_mm: float
) -> dict[JointId, float]:
    """Each solved joint's signed offset along the bar (cap -> collar) from the bar centre.

    The near side is the one nearer the cap, so its joints sit at negative offsets. Elbows sit
    midway between shoulder and wrist; knees and ankles in line with the hips.
    """

    half = {
        "Wrist": half_grip_mm,
        "Shoulder": shoulder_width_mm / 2.0,
        "Elbow": (half_grip_mm + shoulder_width_mm / 2.0) / 2.0,
        "Hip": hip_width_mm / 2.0,
        "Knee": hip_width_mm / 2.0,
        "Ankle": hip_width_mm / 2.0,
    }
    offsets: dict[JointId, float] = {"head": 0.0}
    for side in ("left", "right"):
        sign = -1.0 if side == near_side else 1.0
        for part, d in half.items():
            offsets[f"{side}{part}"] = sign * d  # type: ignore[index]
    return offsets
