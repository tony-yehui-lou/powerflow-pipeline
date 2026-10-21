"""Where the barbell is, per frame, from S5's cap and collar pixels alone.

The level-bar solve of `docs/specs/preprocessing/S5-bar-height*.md`, as validated in powerflow-ui's
Bar Height view (`scripts/build_bar_height.py`). S6 uses it to place the hands, which grip the bar.

Frame: S4's camera, which S3 has levelled, so +Y is gravity (X right, Y down, Z forward). A level
bar has both ends at one height Y below the camera, each end's depth is `Z_i = Y / y_i`, and the
known cap-to-opposite-collar distance fixes Y in closed form. Tilt is assumed zero.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final, Literal

from powerflow_pipeline.data.preprocess.models import Intrinsics

Pixel = tuple[float, float]
Point = tuple[float, float, float]
PoseSource = Literal["level", "yaw-interpolated", "collar-only"]


@dataclass(frozen=True)
class BarType:
    cap_to_collar_mm: float  # the visible cap to the collar on the *opposite* end
    cap_to_centre_mm: float  # half the cap-to-cap length


BAR_TYPES: Final[dict[str, BarType]] = {
    "male IWF": BarType(cap_to_collar_mm=1755.0, cap_to_centre_mm=1100.0),
    "female IWF": BarType(cap_to_collar_mm=1660.0, cap_to_centre_mm=1005.0),
}

# A frame's yaw is measured when 2 px of landmark noise moves it by at most this much. Near camera
# height it moves by tens of degrees, and a bar placed with that yaw slid 504 px along its axis in
# one frame (50kg1); such frames take yaw by linear interpolation between measured flanks.
LANDMARK_NOISE_PX: Final = 2.0
YAW_SIGMA_MAX_DEG: Final = 1.0

# A false cap still solves to a well-conditioned bar; only its yaw gives it away (45kg1 f257:
# 159 deg against ~37 on its neighbours). A cap this far from the median of the measured yaws
# within YAW_WINDOW frames either side is rejected, and the frame treated as having no cap.
YAW_OUTLIER_DEG: Final = 20.0
YAW_WINDOW: Final = 15
YAW_MIN_NEIGHBOURS: Final = 3

# A frame with the collar but no cap takes the collar's depth by linear interpolation between
# solved frames either side. Hiding real caps measured median 1-5 mm, p95 11-27 mm, at 30-frame
# gaps; longer gaps, and runs open at either end, are left without a pose.
MAX_COLLAR_GAP: Final = 30


@dataclass(frozen=True)
class BarPose:
    cap_mm: Point
    collar_mm: Point
    centre_mm: Point
    yaw_deg: float  # horizontal plane, from +Z towards +X, along cap -> collar
    source: PoseSource = "level"

    @property
    def axis(self) -> Point:
        """Unit vector along the bar, cap -> collar."""

        d = tuple(b - a for a, b in zip(self.cap_mm, self.collar_mm, strict=True))
        n = math.sqrt(sum(v * v for v in d))
        return (d[0] / n, d[1] / n, d[2] / n)

    def along(self, distance_from_cap_mm: float) -> Point:
        a = self.axis
        return (
            self.cap_mm[0] + distance_from_cap_mm * a[0],
            self.cap_mm[1] + distance_from_cap_mm * a[1],
            self.cap_mm[2] + distance_from_cap_mm * a[2],
        )


def project(point: Point, k: Intrinsics) -> Pixel:
    x, y, z = point
    return (k.fx * x / z + k.cx, k.fy * y / z + k.cy)


def normalised(px: Pixel, k: Intrinsics) -> Pixel:
    return ((px[0] - k.cx) / k.fx, (px[1] - k.cy) / k.fy)


def _pose(cap: Point, collar: Point, bar: BarType, source: PoseSource) -> BarPose:
    t = bar.cap_to_centre_mm / bar.cap_to_collar_mm
    centre = (
        cap[0] + t * (collar[0] - cap[0]),
        cap[1] + t * (collar[1] - cap[1]),
        cap[2] + t * (collar[2] - cap[2]),
    )
    yaw = math.degrees(math.atan2(collar[0] - cap[0], collar[2] - cap[2]))
    return BarPose(cap_mm=cap, collar_mm=collar, centre_mm=centre, yaw_deg=yaw, source=source)


def _axis(yaw_deg: float) -> Point:
    return (math.sin(math.radians(yaw_deg)), 0.0, math.cos(math.radians(yaw_deg)))


def solve_level(cap_px: Pixel, collar_px: Pixel, k: Intrinsics, bar: BarType) -> BarPose | None:
    """The level bar seen as these two pixels; None when an end sits at camera height or the
    ends straddle it (a level bar has both ends on one side of the horizon)."""

    rays = []
    for px in (cap_px, collar_px):
        x, y = normalised(px, k)
        if y == 0:
            return None
        rays.append((x / y, 1.0, 1.0 / y))  # the end's position divided by Y
    if (rays[0][2] > 0) != (rays[1][2] > 0):
        return None
    span = math.dist(rays[0], rays[1])
    if span == 0:
        return None
    y_mm = math.copysign(bar.cap_to_collar_mm / span, rays[0][2])
    cap = (y_mm * rays[0][0], y_mm * rays[0][1], y_mm * rays[0][2])
    collar = (y_mm * rays[1][0], y_mm * rays[1][1], y_mm * rays[1][2])
    return _pose(cap, collar, bar, "level")


def yaw_sigma(
    cap_px: Pixel,
    collar_px: Pixel,
    k: Intrinsics,
    bar: BarType,
    noise_px: float = LANDMARK_NOISE_PX,
) -> float:
    """How far `noise_px` on each landmark coordinate moves the level solve's yaw, in degrees.
    First-order, by finite differences; infinite where any solve has no solution."""

    base = solve_level(cap_px, collar_px, k, bar)
    if base is None:
        return math.inf
    total = 0.0
    for end in range(2):
        for coord in range(2):
            points = [list(cap_px), list(collar_px)]
            points[end][coord] += noise_px
            moved = solve_level((points[0][0], points[0][1]), (points[1][0], points[1][1]), k, bar)
            if moved is None:
                return math.inf
            diff = (moved.yaw_deg - base.yaw_deg + 180.0) % 360.0 - 180.0
            total += diff * diff
    return math.sqrt(total)


def solve_fixed_yaw(
    cap_px: Pixel, collar_px: Pixel, k: Intrinsics, bar: BarType, yaw_deg: float
) -> BarPose | None:
    """A level bar of known yaw. Depth comes from `u` alone, so it stays conditioned at camera
    height where the level solve's does not. None when the axis lies along a viewing ray."""

    (x1, y1), (x2, y2) = normalised(cap_px, k), normalised(collar_px, k)
    if x1 == x2:
        return None
    ax, _, az = _axis(yaw_deg)
    length = bar.cap_to_collar_mm
    z1 = length * (ax - x2 * az) / (x2 - x1)
    z2 = z1 + length * az
    y = (y1 * z1 + y2 * z2) / 2.0
    return _pose((x1 * z1, y, z1), (x2 * z2, y, z2), bar, "yaw-interpolated")


def solve_collar_only(
    collar_px: Pixel, k: Intrinsics, bar: BarType, collar_z_mm: float, yaw_deg: float
) -> BarPose:
    """A level bar placed from its collar alone, given the collar's depth and the bar's yaw."""

    x, y = normalised(collar_px, k)
    collar = (x * collar_z_mm, y * collar_z_mm, collar_z_mm)
    a = _axis(yaw_deg)
    cap = (
        collar[0] - bar.cap_to_collar_mm * a[0],
        collar[1] - bar.cap_to_collar_mm * a[1],
        collar[2] - bar.cap_to_collar_mm * a[2],
    )
    return _pose(cap, collar, bar, "collar-only")


def fill_linear(
    values: Sequence[float | None], *, hold_edges: bool = True, max_gap: int | None = None
) -> list[float | None]:
    """Fill each run of None linearly between its two known flanks (linear in frames). A run at
    either end holds its one flank when `hold_edges`; a run longer than `max_gap` stays None."""

    known = [i for i, v in enumerate(values) if v is not None]
    out = list(values)
    if not known:
        return out

    def fits(run: int) -> bool:
        return max_gap is None or run <= max_gap

    first, last = known[0], known[-1]
    if hold_edges and fits(first):
        out[:first] = [values[first]] * first
    if hold_edges and fits(len(values) - 1 - last):
        out[last + 1 :] = [values[last]] * (len(values) - 1 - last)
    for a, b in pairwise(known):
        va, vb = values[a], values[b]
        assert va is not None and vb is not None
        if b - a - 1 == 0 or not fits(b - a - 1):
            continue
        for i in range(a + 1, b):
            out[i] = va + (vb - va) * (i - a) / (b - a)
    return out


def unwrap_degrees(values: Sequence[float | None]) -> list[float | None]:
    """Remove 360-degree jumps between successive known values."""

    out = list(values)
    previous: float | None = None
    for i, value in enumerate(values):
        if value is None:
            continue
        if previous is not None:
            value = previous + (value - previous + 180.0) % 360.0 - 180.0
        out[i] = value
        previous = value
    return out


def yaw_outliers(yaws: Sequence[float | None]) -> list[int]:
    """Frames whose yaw is over YAW_OUTLIER_DEG from the median of their measured neighbours'.
    Judged against the neighbours as measured, so one outlier cannot shield another."""

    found = []
    for i, value in enumerate(yaws):
        if value is None:
            continue
        lo, hi = max(0, i - YAW_WINDOW), min(len(yaws), i + YAW_WINDOW + 1)
        near = [w for j, w in enumerate(yaws[lo:hi], lo) if j != i and w is not None]
        if len(near) < YAW_MIN_NEIGHBOURS:
            continue
        diffs = [(w - value + 180.0) % 360.0 - 180.0 for w in near]
        if abs(statistics.median(diffs)) > YAW_OUTLIER_DEG:
            found.append(i)
    return found


def solve_track(
    cap_px: Sequence[Pixel | None],
    collar_px: Sequence[Pixel | None],
    k: Intrinsics,
    bar: BarType,
) -> list[BarPose | None]:
    """The bar's pose on every frame it can be placed, None elsewhere.

    1. Level solve wherever both landmarks are seen; its yaw kept only where measurable.
    2. Caps whose yaw disagrees with their neighbours are rejected.
    3. Unmeasurable yaw is linearly interpolated, and the bar placed with it.
    4. Collar-only frames take the collar's depth from solved flanks (both sides, short gaps).
    """

    n = len(collar_px)
    caps = list(cap_px)
    level: list[BarPose | None] = [None] * n
    measured: list[float | None] = [None] * n
    for i in range(n):
        cap, collar = caps[i], collar_px[i]
        if cap is None or collar is None:
            continue
        level[i] = solve_level(cap, collar, k, bar)
        if level[i] is not None and yaw_sigma(cap, collar, k, bar) <= YAW_SIGMA_MAX_DEG:
            measured[i] = level[i].yaw_deg  # type: ignore[union-attr]

    for i in yaw_outliers(measured):
        measured[i] = None
        caps[i] = None
    yaw = fill_linear(unwrap_degrees(measured))

    poses: list[BarPose | None] = [None] * n
    for i in range(n):
        cap, collar, yaw_i = caps[i], collar_px[i], yaw[i]
        if measured[i] is not None:
            poses[i] = level[i]
        elif cap is not None and collar is not None and yaw_i is not None:
            poses[i] = solve_fixed_yaw(cap, collar, k, bar, yaw_i)

    collar_z = fill_linear(
        [p.collar_mm[2] if p else None for p in poses], hold_edges=False, max_gap=MAX_COLLAR_GAP
    )
    for i in range(n):
        collar, z, yaw_i = collar_px[i], collar_z[i], yaw[i]
        if poses[i] is None and collar is not None and z is not None and yaw_i is not None:
            poses[i] = solve_collar_only(collar, k, bar, z, yaw_i)
    return poses


# A 450 mm bumper: the bar centre sits one plate radius above the floor while the bar rests.
PLATE_RADIUS_MM: Final = 225.0
# The opening second of the clip is the bar on the floor: the cut starts before the lift. Rest is
# the median over the frames solved with both landmarks inside it, and needs at least this many;
# without enough, S3's floor plane anchors instead.
REST_WINDOW: Final = 60
MIN_REST_FRAMES: Final = 10

AnchorMethod = Literal["rest-plate", "floor-plane", "none"]


@dataclass(frozen=True)
class FloorAnchor:
    """Where the floor is, as a Y in S4's levelled camera frame (Y down, so positive)."""

    y_floor_mm: float | None
    method: AnchorMethod
    rest_frames: int
    floor_plane_y_mm: float | None  # S3's fit, always reported as the cross-check


def floor_anchor(bars: Sequence[BarPose | None], *, floor_offset_mm: float | None) -> FloorAnchor:
    """The floor the bar's heights and every joint's are measured from (S8 draft 3 §4).

    Preferred, `rest-plate`: the median bar-centre Y over the frames in the first REST_WINDOW
    solved with both landmarks is the bar resting on its plates. A collar-only frame's depth is
    interpolated, so it never sets the zero. Fallback, `floor-plane`: S3's `floor_offset_m`.
    """

    rest = [
        pose.centre_mm[1]
        for pose in bars[:REST_WINDOW]
        if pose is not None and pose.source in ("level", "yaw-interpolated")
    ]
    if len(rest) >= MIN_REST_FRAMES:
        y_floor = statistics.median(rest) + PLATE_RADIUS_MM
        return FloorAnchor(y_floor, "rest-plate", len(rest), floor_offset_mm)
    if floor_offset_mm is not None:
        return FloorAnchor(floor_offset_mm, "floor-plane", 0, floor_offset_mm)
    return FloorAnchor(None, "none", 0, None)
