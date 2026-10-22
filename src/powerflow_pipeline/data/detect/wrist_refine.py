"""S6 · Wrist correction against the barbell.

Overhead, the far arm is hidden behind the head, the bar and the plates, and the 2D detector
guesses: on the 10 September captures its far wrist jumps 130-356 px between frames (p95) and
lands on the hip, the far plate, or a bystander. The hands grip the bar, though, and the bar is
known (S5, `bar_geometry.solve_track`). So:

1. **Grip.** The near hand is found on the bar every frame it is seen there confidently; its
   distance from the bar centre is the half grip, constant for the lift (447-464 mm across six
   captures of one lifter).
2. **Mirror.** The far hand sits the same distance on the other side of the centre, along the 3D
   bar, and is projected into the image.
3. **Wrist offset.** The wrist joint is not the bar contact: it sits above the bar at the floor
   and below it overhead. The near wrist is placed at its bar point's depth, and its 3D offset
   from that point is applied at the far hand -- the hands are mirror images, and the offset is
   mostly vertical. That is done wherever the near wrist is itself on the bar; elsewhere the
   offset is interpolated from neighbouring frames, never copied from a wrist that is not on
   this bar. A constant pixel residual learnt at rest absorbs what the copy misses.
4. **Apply while the hands are on the bar.** The far wrist is predicted on every such frame; the
   near wrist only where it left the bar while the hands were still on it (a glitch, or S6 on a
   bystander). After the release the detector's wrists are left alone.

Measured on the prototype: overhead far-wrist step p95 130-356 px -> 14-30 px, as steady as the
near wrist; the prediction sits 3-12 px from confident detections at the floor and 21-37 px
overhead, where confident detections are few (the occlusion this fixes).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from powerflow_pipeline.data.common.bar_geometry import (
    BarPose,
    BarType,
    fill_linear,
    normalised,
    project,
)
from powerflow_pipeline.data.common.capture_models import Intrinsics
from powerflow_pipeline.data.common.models import JointId
from powerflow_pipeline.data.detect.pose_model import JointPixelSeries

# A wrist within this many pixels of its bar point counts as holding the bar.
ON_BAR_PX: Final = 60.0
# Only near-wrist detections at least this confident set the grip and the offset residual.
MIN_CONFIDENCE: Final = 0.7
# The hands are on the bar on a frame when most known frames within this many either side say so,
# so a one- or two-frame near-wrist glitch cannot switch the correction off.
ON_BAR_WINDOW: Final = 5
# Fewer confident near-hand-on-bar frames than this and the grip is not trusted: no correction.
MIN_GRIP_FRAMES: Final = 20
# "At rest" for the offset residual: bar centre within this of its lowest position.
REST_BAND_MM: Final = 100.0
MIN_REST_FRAMES: Final = 10
# A far detection further than this from the uncorrected prediction is not used for the residual.
RESIDUAL_MAX_PX: Final = 120.0

WRISTS: Final[tuple[JointId, JointId]] = ("leftWrist", "rightWrist")


@dataclass(frozen=True)
class WristRefinement:
    left: JointPixelSeries
    right: JointPixelSeries
    near: JointId
    far: JointId
    half_grip_mm: float
    residual_px: tuple[float, float]
    frames_on_bar: int
    far_corrected: tuple[int, ...]
    near_corrected: tuple[int, ...]


@dataclass(frozen=True)
class Skipped:
    reason: str


def _closest_on_bar(bar: BarPose, px: tuple[int, int], k: Intrinsics) -> float:
    """Distance from the cap, along the bar, of the bar point nearest the wrist's viewing ray."""

    u = bar.axis
    x, y = normalised((float(px[0]), float(px[1])), k)
    v = (x, y, 1.0)
    w = bar.cap_mm  # ray origin is the camera centre, so w0 = cap - 0
    a = 1.0
    b = sum(p * q for p, q in zip(u, v, strict=True))
    c = sum(q * q for q in v)
    d = sum(p * q for p, q in zip(u, w, strict=True))
    e = sum(q * q2 for q, q2 in zip(v, w, strict=True))
    return (b * e - c * d) / (a * c - b * b)


def refine_wrists(
    left: JointPixelSeries,
    right: JointPixelSeries,
    bars: Sequence[BarPose | None],
    k: Intrinsics,
    bar: BarType,
) -> WristRefinement | Skipped:
    """Both wrists, corrected against the bar wherever the hands hold it."""

    n = len(bars)
    detected = {"leftWrist": left, "rightWrist": right}

    # Near = the wrist nearer the cap end, i.e. the smaller distance along cap -> collar, judged
    # only where each wrist is on the bar: a far wrist lost to occlusion can land anywhere,
    # including nearer the cap than the near hand.
    reach: dict[JointId, list[float]] = {w: [] for w in WRISTS}
    for i, pose in enumerate(bars):
        if pose is None:
            continue
        for wrist in WRISTS:
            px = detected[wrist].pixel_position[i]
            if px is None:
                continue
            s = _closest_on_bar(pose, px, k)
            proj = project(pose.along(s), k)
            if ((proj[0] - px[0]) ** 2 + (proj[1] - px[1]) ** 2) ** 0.5 <= ON_BAR_PX:
                reach[wrist].append(s)
    if not reach["leftWrist"] and not reach["rightWrist"]:
        return Skipped("neither wrist is ever on a placed bar, so no grip to measure")
    if not reach["leftWrist"] or not reach["rightWrist"]:
        # Only one wrist is ever on the bar: it is the near one, by elimination.
        seen = "leftWrist" if reach["leftWrist"] else "rightWrist"
        reach["rightWrist" if seen == "leftWrist" else "leftWrist"] = [math.inf]
    near, far = sorted(WRISTS, key=lambda w: statistics.median(reach[w]))
    near_s, far_s = detected[near], detected[far]

    # The near wrist on the bar: whether it is, and where it sits relative to the bar's centre.
    # Placed in 3D at the depth of the bar point nearest its viewing ray, it splits into a part
    # along the bar (the grip) and a part perpendicular to it (the wrist's offset from the bar).
    raw_on: list[bool | None] = [None] * n
    along: list[float | None] = [None] * n
    across: list[list[float | None]] = [[None] * n, [None] * n, [None] * n]
    grips: list[float] = []
    for i, pose in enumerate(bars):
        px = near_s.pixel_position[i]
        if pose is None or px is None:
            continue
        point = pose.along(_closest_on_bar(pose, px, k))
        proj = project(point, k)
        raw_on[i] = ((proj[0] - px[0]) ** 2 + (proj[1] - px[1]) ** 2) ** 0.5 <= ON_BAR_PX
        if not raw_on[i]:
            continue
        x, y = normalised((float(px[0]), float(px[1])), k)
        wrist_mm = (x * point[2], y * point[2], point[2])
        rel = [w - c for w, c in zip(wrist_mm, pose.centre_mm, strict=True)]
        axis = pose.axis
        a = sum(r * u for r, u in zip(rel, axis, strict=True))
        along[i] = a
        for j in range(3):
            across[j][i] = rel[j] - a * axis[j]
        if near_s.confidence[i] >= MIN_CONFIDENCE:
            grips.append(-a)
    if len(grips) < MIN_GRIP_FRAMES:
        return Skipped(
            f"only {len(grips)} frames with the near hand confidently on the bar; "
            f"the grip needs {MIN_GRIP_FRAMES}"
        )
    half_grip = statistics.median(grips)  # hands do not slide on the bar during a lift

    on = [False] * n
    for i in range(n):
        votes = [
            v for v in raw_on[max(0, i - ON_BAR_WINDOW) : i + ON_BAR_WINDOW + 1] if v is not None
        ]
        on[i] = bars[i] is not None and bool(votes) and sum(votes) > len(votes) / 2

    # The offset is interpolated across frames where the near wrist is not on the bar, so a wrist
    # that is somewhere else (a glitch, a bystander) never lends its offset to the far hand.
    off = [fill_linear(component) for component in across]

    def predict(pose: BarPose, along_from_centre: float, i: int) -> tuple[float, float]:
        """The wrist at `along_from_centre` along the bar, offset across it like the near wrist:
        the near wrist mirrored through the plane perpendicular to the bar at its centre."""

        ox, oy, oz = off[0][i], off[1][i], off[2][i]
        assert ox is not None and oy is not None and oz is not None
        axis, c = pose.axis, pose.centre_mm
        return project(
            (
                c[0] + along_from_centre * axis[0] + ox,
                c[1] + along_from_centre * axis[1] + oy,
                c[2] + along_from_centre * axis[2] + oz,
            ),
            k,
        )

    far_along, near_along = half_grip, -half_grip
    base: list[tuple[float, float] | None] = [None] * n
    for i, pose in enumerate(bars):
        if on[i] and pose is not None and off[0][i] is not None:
            base[i] = predict(pose, far_along, i)

    # What the copied offset misses, learnt where the far wrist is seen at rest.
    placed = [p.centre_mm[1] for p in bars if p is not None]
    lowest = max(placed)
    residuals = []
    for i, pose in enumerate(bars):
        prediction, px = base[i], far_s.pixel_position[i]
        if (
            pose is None
            or prediction is None
            or px is None
            or far_s.confidence[i] < MIN_CONFIDENCE
            or pose.centre_mm[1] < lowest - REST_BAND_MM
        ):
            continue
        dx, dy = px[0] - prediction[0], px[1] - prediction[1]
        if (dx * dx + dy * dy) ** 0.5 <= RESIDUAL_MAX_PX:
            residuals.append((dx, dy))
    residual = (
        (statistics.median(r[0] for r in residuals), statistics.median(r[1] for r in residuals))
        if len(residuals) >= MIN_REST_FRAMES
        else (0.0, 0.0)
    )

    far_px = list(far_s.pixel_position)
    far_conf = list(far_s.confidence)
    near_px = list(near_s.pixel_position)
    near_conf = list(near_s.confidence)
    far_fixed, near_fixed = [], []
    for i, pose in enumerate(bars):
        prediction = base[i]
        if prediction is None or pose is None:
            continue
        far_px[i] = (round(prediction[0] + residual[0]), round(prediction[1] + residual[1]))
        far_conf[i] = near_s.confidence[i]
        far_fixed.append(i)
        if not raw_on[i]:
            placed_near = predict(pose, near_along, i)
            near_px[i] = (round(placed_near[0]), round(placed_near[1]))
            near_conf[i] = near_s.confidence[i]
            near_fixed.append(i)

    corrected = {
        far: JointPixelSeries(pixel_position=tuple(far_px), confidence=tuple(far_conf)),
        near: JointPixelSeries(pixel_position=tuple(near_px), confidence=tuple(near_conf)),
    }
    return WristRefinement(
        left=corrected["leftWrist"],
        right=corrected["rightWrist"],
        near=near,
        far=far,
        half_grip_mm=half_grip,
        residual_px=residual,
        frames_on_bar=sum(on),
        far_corrected=tuple(far_fixed),
        near_corrected=tuple(near_fixed),
    )
