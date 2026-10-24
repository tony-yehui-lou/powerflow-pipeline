"""S9 · Phase: a snatch's or clean & jerk's events, and the phase intervals between them, from the
bar's height and speed, height bands from the lifter's bones, and S8's knee
(docs/specs/preprocessing/S9-phase-intervals-draft7.md).

Pure: no I/O, no Prefect. The task (`tasks/phase.py`) reads the published inputs and writes the
document; this module builds the bands (§2), bridges short bar gaps (§4.3), finds each event with
a speed rule and a height gate (§1), sizes its uncertainty (§4.4) and assembles the phases.
Nothing is guessed: an event the rules cannot find is `absent`, and every phase it bounds is left
out.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
from scipy.signal import butter, filtfilt

from powerflow_pipeline.data.analyze.config import PhaseThresholds

Lift = Literal["snatch", "clean_and_jerk"]
Status = Literal["found", "fallback", "absent"]

EVENTS: Final[dict[Lift, tuple[str, ...]]] = {
    "snatch": ("liftoff", "barPassesKnee", "fullExtension", "catch", "stand"),
    "clean_and_jerk": (
        "liftoff", "barPassesKnee", "fullExtension", "cleanCatch", "dipStart", "dipBottom",
        "jerkCatch", "stand",
    ),
}  # fmt: skip

# (phase, opening event, closing event); None is the clip's own edge.
PHASES: Final[dict[Lift, tuple[tuple[str, str | None, str | None], ...]]] = {
    "snatch": (
        ("setup", None, "liftoff"),
        ("first_pull", "liftoff", "barPassesKnee"),
        ("second_pull", "barPassesKnee", "fullExtension"),
        ("turnover", "fullExtension", "catch"),
        ("recovery", "catch", "stand"),
    ),
    "clean_and_jerk": (
        ("setup", None, "liftoff"),
        ("first_pull", "liftoff", "barPassesKnee"),
        ("second_pull", "barPassesKnee", "fullExtension"),
        ("turnover", "fullExtension", "cleanCatch"),
        ("clean_recovery", "cleanCatch", "dipStart"),
        ("dip", "dipStart", "dipBottom"),
        ("drive", "dipBottom", "jerkCatch"),
        ("jerk_recovery", "jerkCatch", "stand"),
    ),
}

# Starting plausible durations in seconds (§4.2); the label statistics replace them.
PLAUSIBLE: Final[dict[str, tuple[float, float]]] = {
    "first_pull": (0.30, 1.20),
    "second_pull": (0.08, 0.40),
    "turnover": (0.15, 0.60),
    "clean_recovery": (0.80, 8.00),
    "dip": (0.10, 0.50),
    "drive": (0.20, 0.70),
}

# A peak's uncertainty is the window where velocity stays within this share of it.
PEAK_BAND: Final = 0.05
# The second-order Butterworth filtfilt needs more samples than its padding.
MIN_FILTER_RUN: Final = 10

RESULTS: Final[dict[Lift, frozenset[str]]] = {
    "snatch": frozenset({"made", "missed"}),
    "clean_and_jerk": frozenset({"made", "missed_clean", "missed_jerk"}),
}
# The raw operator metadata's own words for a result.
RESULT_ALIASES: Final[dict[Lift, dict[str, str]]] = {
    "snatch": {"success": "made", "fail": "missed"},
    "clean_and_jerk": {"success": "made"},
}


@dataclass(frozen=True)
class Signals:
    """Per-frame inputs, heights in mm above the floor; None where unknown."""

    t_ms: Sequence[int]
    bar_mm: Sequence[float | None]
    knee_mm: Sequence[float | None]


@dataclass(frozen=True)
class Side:
    """One side's joint height series as S8 publishes it, in mm."""

    height_mm: Sequence[float | None]
    sigma_mm: Sequence[float | None]
    flags: Sequence[Sequence[str]]


@dataclass(frozen=True)
class Body:
    """The lifter's bone lengths as S8 used them, and S6's half grip, in mm (§2.1)."""

    ankle_mm: float
    tibia_mm: float
    femur_mm: float
    torso_mm: float
    upper_arm_mm: float
    forearm_mm: float
    shoulder_width_mm: float
    half_grip_mm: float | None


@dataclass(frozen=True)
class Band:
    """A height range for one position: `contains`, `above` and `below` are against its edges."""

    centre: float
    low: float
    high: float

    @classmethod
    def around(cls, centre: float, tolerance: tuple[float, float]) -> Band:
        below, above = tolerance
        return cls(centre, centre * (1.0 - below), centre * (1.0 + above))

    def contains(self, h: float) -> bool:
        return self.low <= h <= self.high

    def above(self, h: float) -> bool:
        return h > self.high

    def below(self, h: float) -> bool:
        return h < self.low


@dataclass(frozen=True)
class Bands:
    """The body's bands (§2.2). The `floor` band is the bar's own rest height, so `detect`
    builds it from the clip."""

    knee: Band
    hip: Band
    rack: Band
    overhead: Band
    grip_assumed: bool

    def as_dict(self) -> dict[str, Band]:
        return {"knee": self.knee, "hip": self.hip, "rack": self.rack, "overhead": self.overhead}


@dataclass(frozen=True)
class Event:
    frame: int | None
    sigma_ms: float | None
    status: Status
    bar_mm: float | None = None


ABSENT: Final = Event(None, None, "absent")


@dataclass(frozen=True)
class Span:
    """One phase; `end` is exclusive."""

    phase: str
    start: int
    end: int


@dataclass(frozen=True)
class BarTrack:
    """The track `detect` measured every event on, frame by frame (§6 `barTrack`)."""

    raw_mm: Sequence[float | None]  # as S5/S8 gave it, before bridging and the filter
    height_mm: Sequence[float | None]  # bridged and low-passed: what every rule reads
    velocity_mm_per_s: Sequence[float | None]
    bridged_frames: Sequence[int]


WindowEnd = Literal["stand", "missed", "no_lift_end"]


@dataclass(frozen=True)
class Window:
    """The lift window (§6.1): frames `start` to `end` (exclusive) hold the lift."""

    start: int
    end: int
    end_reason: WindowEnd
    margin_ms: float


@dataclass(frozen=True)
class Detection:
    lift: Lift
    events: dict[str, Event]
    auxiliary: dict[str, Event]
    phases: list[Span]
    result: tuple[str, str]  # (value, source)
    flags: list[str]
    derived: dict[str, float | int | None] = field(default_factory=dict)
    bar_track: BarTrack | None = None
    window: Window = Window(0, 0, "no_lift_end", 0.0)


def bands(body: Body, *, thresholds: PhaseThresholds) -> Bands:
    """Every height band of §2.2 for this lifter.

    The overhead reach is the arm's length less its outward lean: the wrists sit `half_grip`
    from the bar centre and the shoulders half a shoulder width, and S6 measures the grip at the
    wrist, so there is no hand term. Without a grip the arms are taken as vertical and the band
    widens below.
    """

    tol = thresholds.bands
    hip = body.ankle_mm + body.tibia_mm + body.femur_mm
    shoulder = hip + body.torso_mm
    arm = body.upper_arm_mm + body.forearm_mm
    grip_assumed = body.half_grip_mm is None
    lean = 0.0 if body.half_grip_mm is None else body.half_grip_mm - body.shoulder_width_mm / 2.0
    reach = math.sqrt(max(arm**2 - lean**2, 0.0))
    overhead_tol = (
        (tol.overhead_below_grip_assumed, tol.overhead[1]) if grip_assumed else tol.overhead
    )
    return Bands(
        knee=Band.around(body.ankle_mm + body.tibia_mm, tol.knee),
        hip=Band.around(hip, tol.hip),
        rack=Band.around(shoulder, tol.rack),
        overhead=Band.around(shoulder + reach, overhead_tol),
        grip_assumed=grip_assumed,
    )


def pair_height(left: Side, right: Side, *, sigma_max_mm: float) -> list[float | None]:
    """Left and right heights per frame, weighted by 1/sigma^2 (§1.1).

    A side flagged `uncertain`, without a sigma, or with sigma above `sigma_max_mm` is left out;
    a frame with neither side left is None.
    """

    paired: list[float | None] = []
    for i in range(len(left.height_mm)):
        total = weight = 0.0
        for side in (left, right):
            h, s = side.height_mm[i], side.sigma_mm[i]
            if h is None or s is None or s > sigma_max_mm or "uncertain" in side.flags[i]:
                continue
            w = 1.0 / max(s, 1e-6) ** 2
            total += w * h
            weight += w
        paired.append(total / weight if weight > 0 else None)
    return paired


class _Track:
    """The bar after gap bridging and filtering, and its velocity, frame by frame."""

    def __init__(self, signals: Signals, thresholds: PhaseThresholds) -> None:
        self.t = np.asarray(signals.t_ms, dtype=float)
        self.n = len(self.t)
        steps = np.diff(self.t)
        self.dt = float(np.median(steps)) if len(steps) else 1000.0 / 60.0
        self.knee = signals.knee_mm
        self.raw = list(signals.bar_mm)  # kept to publish as `barTrack.rawMm`
        bridged, self.gaps = _bridge(self.t, signals.bar_mm, thresholds.max_gap_ms, self.dt)
        self.h = _lowpass(self.t, bridged, thresholds.filter_cutoff_hz)
        self.v = _velocity(self.t, self.h)

    def ms(self, a: int, b: int) -> float:
        return float(self.t[b] - self.t[a])

    def event(self, frame: int, half_width_ms: float, status: Status = "found") -> Event:
        """Sigma at least half a frame, and at least half the bridged gap the frame is in (§4.3)."""

        sigma = max(self.dt / 2.0, half_width_ms, self.gaps.get(frame, 0.0) / 2.0)
        return Event(frame, sigma, status, self.h[frame])

    def window(self, centre: int, keep: Callable[[float], bool]) -> float:
        """Half the duration of the run of frames around `centre` whose velocity passes `keep`."""

        lo = hi = centre
        while lo > 0 and _ok(self.v[lo - 1], keep):
            lo -= 1
        while hi < self.n - 1 and _ok(self.v[hi + 1], keep):
            hi += 1
        return self.ms(lo, hi) / 2.0

    def held(self, i: int, until: int, hold_ms: float, v_max: float, band: Band) -> int | None:
        """The last frame of a stretch from `i`, before `until`, lasting `hold_ms`, with the bar
        slower than `v_max` and inside `band` on every frame of it."""

        for j in range(i, until):
            v, h = self.v[j], self.h[j]
            if v is None or h is None or abs(v) >= v_max or not band.contains(h):
                return None
            if self.ms(i, j) >= hold_ms:
                return j
        return None

    def first_peak(self, after: int, floor_mm: float) -> int | None:
        """The first frame after `after` where the rising bar stops rising, above `floor_mm`."""

        for i in range(after + 1, self.n):
            a, b, h = self.v[i - 1], self.v[i], self.h[i]
            if a is not None and b is not None and h is not None and a > 0 >= b and h > floor_mm:
                return i
        return None

    def bottom(self, after: int, v_rest: float) -> int | None:
        """The first frame after `after` where the falling bar's velocity rises through 0.

        The unbroken falling run that ends there must reach faster than `v_rest`, so near-zero
        noise at a peak, or a rise after a gap in the track, never counts as the bottom.
        """

        fell = False
        for i in range(after + 1, self.n):
            a, b = self.v[i - 1], self.v[i]
            if a is None or b is None or a >= 0:
                fell = False
                continue
            fell = fell or a < -v_rest
            if fell and b >= 0:
                return i
        return None

    def leaves(self, after: int, out: Callable[[float], bool]) -> int:
        """The first frame after `after` where the bar's height passes `out`; else the clip end."""

        for j in range(after + 1, self.n):
            h = self.h[j]
            if h is not None and out(h):
                return j
        return self.n


def _ok(v: float | None, keep: Callable[[float], bool]) -> bool:
    return v is not None and keep(v)


def _bridge(
    t: np.ndarray, bar: Sequence[float | None], max_gap_ms: float, dt: float
) -> tuple[list[float | None], dict[int, float]]:
    """Linearly fill runs of missing bar heights no longer than `max_gap_ms`, flanked on both sides.

    Returns the filled series and, for each filled frame, the gap's duration.
    """

    h = list(bar)
    gaps: dict[int, float] = {}
    i = 0
    while i < len(h):
        if h[i] is not None:
            i += 1
            continue
        j = i
        while j < len(h) and h[j] is None:
            j += 1
        if i > 0 and j < len(h):
            gap_ms = float(t[j] - t[i - 1]) - dt
            if gap_ms <= max_gap_ms + 1e-6:
                a, b = h[i - 1], h[j]
                assert a is not None and b is not None
                for k in range(i, j):
                    w = (t[k] - t[i - 1]) / (t[j] - t[i - 1])
                    h[k] = float(a + w * (b - a))
                    gaps[k] = gap_ms
        i = j
    return h, gaps


def _runs(x: Sequence[float | None]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(x):
        if x[i] is None:
            i += 1
            continue
        j = i
        while j < len(x) and x[j] is not None:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _lowpass(t: np.ndarray, h: Sequence[float | None], cutoff_hz: float) -> list[float | None]:
    """Zero-phase second-order Butterworth low-pass over each contiguous run of heights."""

    out = list(h)
    for a, b in _runs(h):
        if b - a < MIN_FILTER_RUN:
            continue
        fs = 1000.0 / float(np.median(np.diff(t[a:b])))
        if cutoff_hz >= fs / 2.0:
            continue
        num, den = butter(2, cutoff_hz / (fs / 2.0))
        out[a:b] = [float(x) for x in filtfilt(num, den, np.asarray(h[a:b], dtype=float))]
    return out


def _velocity(t: np.ndarray, h: Sequence[float | None]) -> list[float | None]:
    """Central differences over the real `tMs` spacing, one-sided at a run's ends, in mm/s."""

    v: list[float | None] = [None] * len(h)
    for a, b in _runs(h):
        if b - a < 2:
            continue
        for i in range(a, b):
            lo, hi = max(i - 1, a), min(i + 1, b - 1)
            hlo, hhi = h[lo], h[hi]
            assert hlo is not None and hhi is not None
            v[i] = (hhi - hlo) / (t[hi] - t[lo]) * 1000.0
    return v


def _rest_height(track: _Track, frames: int) -> float | None:
    rest = [h for h in track.h[:frames] if h is not None]
    return statistics.median(rest) if rest else None


def _liftoff(track: _Track, rest: float, th: PhaseThresholds) -> Event:
    """E0: off the floor by LIFT_MM and still rising LIFT_HOLD later, stepped back to rest speed."""

    above = rest + th.lift_mm
    for i in range(track.n):
        hi = track.h[i]
        if hi is None or hi <= above:
            continue
        end = None
        for j in range(i, track.n):
            hj = track.h[j]
            if hj is None or hj <= above:
                break
            if track.ms(i, j) >= th.lift_hold_ms:
                end = j
                break
        if end is None or (track.h[end] or 0.0) <= hi:
            continue
        k = i
        while k > 0 and (track.v[k] is None or track.v[k] > th.v_rest_mm_s):  # type: ignore[operator]
            k -= 1
        return track.event(k, track.ms(k, i) / 2.0)
    return ABSENT


def _full_extension(track: _Track, a: int, peak: int) -> Event:
    """E2: peak upward bar velocity between `a` and the bar's first peak."""

    best: int | None = None
    for i in range(a, peak + 1):
        v = track.v[i]
        if v is not None and (best is None or v > track.v[best]):  # type: ignore[operator]
            best = i
    if best is None:
        return ABSENT
    vmax = track.v[best]
    assert vmax is not None
    return track.event(best, track.window(best, lambda v: v >= (1.0 - PEAK_BAND) * vmax))


def _knee_crossing(track: _Track, a: int, peak: int, band: Band | None) -> tuple[Event, str | None]:
    """E1: the bar reaches S8's knee, counted only inside the knee band (§1.2).

    Otherwise the bar reaching the knee band's centre, as a `fallback`, with the flag saying why:
    `knee_unavailable` (no S8 knee) or `knee_out_of_band` (S8's crossing outside the band).
    """

    reason = "knee_unavailable"
    if any(track.knee[i] is not None for i in range(a + 1, peak + 1)):
        for i in range(a + 1, peak + 1):
            h, knee = track.h[i], track.knee[i]
            if h is not None and knee is not None and h >= knee:
                if band is None or band.contains(h):
                    return track.event(i, 0.0), None
                reason = "knee_out_of_band"
                break
    if band is not None:
        for i in range(a + 1, peak + 1):
            h = track.h[i]
            if h is not None and h >= band.centre:
                return track.event(i, 0.0, "fallback"), reason
    return ABSENT, reason


def _bottom_event(
    track: _Track, after: int, th: PhaseThresholds, gate: Callable[[float], bool]
) -> Event:
    """A catch or the dip's bottom: the first rise of velocity through 0 after `after`, if the
    bar's height there passes `gate`. A rejected candidate is `absent`, never the next one."""

    i = track.bottom(after, th.v_rest_mm_s)
    h = None if i is None else track.h[i]
    if i is None or h is None or not gate(h):
        return ABSENT
    return track.event(i, track.window(i, lambda v: abs(v) <= th.v_rest_mm_s))


def _stand(
    track: _Track, after: int, until: int, hold_ms: float, band: Band, th: PhaseThresholds
) -> tuple[Event, int | None]:
    """The first still stretch after `after`, before `until`, inside `band` throughout.

    Returns the event at its first frame and the stretch's last frame.
    """

    for i in range(after + 1, until):
        end = track.held(i, until, hold_ms, th.v_still_mm_s, band)
        if end is not None:
            return track.event(i, 0.0), end
    return ABSENT, None


def _reaches(track: _Track, after: int, until: int, band: Band) -> Event:
    """The stand (E4, E7): the first frame after `after`, before `until`, where the bar is at or
    above `band`'s lower edge. Height alone, no hold: a draft 7 stopgap (Open question 11)."""

    for i in range(after + 1, until):
        h = track.h[i]
        if h is not None and h >= band.low:
            return track.event(i, 0.0)
    return ABSENT


def _pull(
    track: _Track, rest: float, b: Bands | None, th: PhaseThresholds, events: dict[str, Event],
    flags: list[str],
) -> int | None:  # fmt: skip
    """E0-E2, shared by both lifts (§1.2). Returns the bar's first peak, where the catch search
    starts. E0 and E2 outside their bands are flagged, not rejected."""

    e0 = _liftoff(track, rest, th)
    events["liftoff"] = e0
    if e0.frame is None:
        return None
    peak = track.first_peak(e0.frame, rest + th.lift_mm)
    if peak is None:
        return None
    e1, knee_reason = _knee_crossing(track, e0.frame, peak, None if b is None else b.knee)
    if knee_reason is not None:
        flags.append(knee_reason)
    e2 = _full_extension(track, e0.frame if e1.frame is None else e1.frame, peak)
    events["barPassesKnee"] = e1
    events["fullExtension"] = e2
    if b is not None and any(
        e.bar_mm is not None and not band.contains(e.bar_mm)
        for e, band in ((e0, Band(rest, rest - th.lift_mm, rest + th.lift_mm)), (e2, b.hip))
    ):
        flags.append("height_out_of_band")
    return peak


def _snatch(
    track: _Track, peak: int, b: Bands, th: PhaseThresholds, events: dict[str, Event],
    flags: list[str], ends: dict[str, int],
) -> None:  # fmt: skip
    catch = _bottom_event(track, peak, th, b.hip.above)
    events["catch"] = catch
    if catch.frame is None:
        return
    miss = track.leaves(catch.frame, b.hip.below)
    stand = _reaches(track, catch.frame, miss, b.overhead)
    events["stand"] = stand
    if stand.frame is None and miss < track.n:
        flags.append("missed")
        ends["recovery"] = miss


def _clean_and_jerk(
    track: _Track, peak: int, b: Bands, th: PhaseThresholds, events: dict[str, Event],
    auxiliary: dict[str, Event], flags: list[str], ends: dict[str, int],
) -> None:  # fmt: skip
    cj = th.clean_and_jerk
    catch = _bottom_event(track, peak, th, lambda h: b.hip.low <= h <= b.rack.high)
    events["cleanCatch"] = catch
    if catch.frame is None:
        return

    miss = track.leaves(catch.frame, b.hip.below)
    rack, rack_end = _stand(track, catch.frame, miss, cj.rack_hold_ms, b.rack, th)
    auxiliary["rackStand"] = rack
    if rack.frame is None or rack_end is None:
        if miss < track.n:
            flags.append("missed_clean")
            ends["clean_recovery"] = miss
        return

    held = [h for h in track.h[rack.frame : rack_end + 1] if h is not None]
    dipped = statistics.median(held) - cj.dip_mm
    dip = _dip_start(track, rack.frame, dipped, th)
    events["dipStart"] = dip
    if dip.frame is None:
        return
    bottom = _bottom_event(track, dip.frame, th, lambda h: h <= dipped)
    events["dipBottom"] = bottom
    if bottom.frame is None:
        return
    peak2 = track.first_peak(bottom.frame, dipped)
    if peak2 is None:
        return
    jerk = _bottom_event(track, peak2, th, b.rack.above)
    events["jerkCatch"] = jerk
    if jerk.frame is None:
        miss = track.leaves(peak2, b.rack.below)
        if miss < track.n:
            flags.append("missed_jerk")
            ends["drive"] = miss
        return
    miss = track.leaves(jerk.frame, b.rack.below)
    stand = _reaches(track, jerk.frame, miss, b.overhead)
    events["stand"] = stand
    if stand.frame is None and miss < track.n:
        flags.append("missed_jerk")
        ends["jerk_recovery"] = miss


def _dip_start(track: _Track, rack: int, dipped: float, th: PhaseThresholds) -> Event:
    """E4: the bar DIP_MM below the rack and falling, stepped back to the last still frame."""

    for j in range(rack + 1, track.n):
        h, v = track.h[j], track.v[j]
        if h is None or v is None or h > dipped or v >= 0:
            continue
        k = j
        while k > rack and (track.v[k] is None or abs(track.v[k]) > th.v_rest_mm_s):  # type: ignore[arg-type]
            k -= 1
        return track.event(k, track.ms(k, j) / 2.0)
    return ABSENT


def _phases(lift: Lift, events: Mapping[str, Event], ends: Mapping[str, int], n: int) -> list[Span]:
    spans: list[Span] = []
    for phase, opening, closing in PHASES[lift]:
        start = 0 if opening is None else events[opening].frame
        end = n if closing is None else events[closing].frame
        if end is None:
            end = ends.get(phase)
        if start is not None and end is not None and start < end:
            spans.append(Span(phase, start, end))
    return spans


def _window(track: _Track, stand: int | None, ends: Mapping[str, int], margin_ms: float) -> Window:
    """The lift window (§6.1): to `margin_ms` past the stand, or past a miss's drop, clamped to the
    clip end; the whole clip when there is neither."""

    reason: WindowEnd
    if stand is not None:
        anchor, reason = stand, "stand"
    elif ends:
        anchor, reason = max(ends.values()), "missed"
    else:
        return Window(0, track.n, "no_lift_end", margin_ms)
    end = next((i for i in range(anchor, track.n) if track.ms(anchor, i) > margin_ms), track.n)
    return Window(0, end, reason, margin_ms)


def _result(
    lift: Lift, events: Mapping[str, Event], flags: Sequence[str], override: object
) -> tuple[str, str]:
    if isinstance(override, str):
        value = RESULT_ALIASES[lift].get(override, override)
        if value in RESULTS[lift]:
            return value, "metadata"
    for miss in ("missed", "missed_clean", "missed_jerk"):
        if miss in flags:
            return miss, "rules"
    return ("made" if events["stand"].frame is not None else "unknown"), "rules"


def detect(
    signals: Signals,
    lift: Lift,
    thresholds: PhaseThresholds,
    body_bands: Bands | None,
    *,
    result_override: object = None,
    plausible: Mapping[str, tuple[float, float]] = PLAUSIBLE,
) -> Detection:
    """Every event of `lift`, the phases between them, the lift's result and the flags (§1, §4).

    Without bands (no bone lengths), the pull's events still come from speed alone, and every
    gated event after it is `absent`. `result_override` is the raw metadata's `lift.result`; only
    a string naming a result counts, so the operator template's placeholder list is ignored.
    """

    track = _Track(signals, thresholds)
    events: dict[str, Event] = dict.fromkeys(EVENTS[lift], ABSENT)
    auxiliary: dict[str, Event] = {"rackStand": ABSENT} if lift == "clean_and_jerk" else {}
    flags: list[str] = []
    ends: dict[str, int] = {}

    rest = _rest_height(track, thresholds.rest_frames)
    if rest is not None:
        peak = _pull(track, rest, body_bands, thresholds, events, flags)
        if peak is not None and body_bands is not None:
            if lift == "snatch":
                _snatch(track, peak, body_bands, thresholds, events, flags, ends)
            else:
                _clean_and_jerk(track, peak, body_bands, thresholds, events, auxiliary, flags, ends)

    if track.gaps:
        flags.append("bar_gap_bridged")
    if body_bands is not None and body_bands.grip_assumed:
        flags.append("grip_assumed")
    window = _window(track, events["stand"].frame, ends, thresholds.window_margin_ms)
    if window.end_reason == "no_lift_end":
        flags.append("no_lift_end")

    phases = _phases(lift, events, ends, track.n)
    for span in phases:
        if span.phase in plausible:
            lo, hi = plausible[span.phase]
            seconds = track.ms(span.start, min(span.end, track.n - 1)) / 1000.0
            if not lo <= seconds <= hi and "implausible_duration" not in flags:
                flags.append("implausible_duration")

    return Detection(
        lift=lift,
        events=events,
        auxiliary=auxiliary,
        phases=phases,
        result=_result(lift, events, flags, result_override),
        flags=flags,
        derived={"restMm": rest, "bridgedFrames": len(track.gaps)},
        bar_track=BarTrack(
            raw_mm=track.raw,
            height_mm=track.h,
            velocity_mm_per_s=track.v,
            bridged_frames=sorted(track.gaps),
        ),
        window=window,
    )


SCHEMA_VERSION: Final = "1.4.0"
METHOD_SPEC: Final = "docs/specs/preprocessing/S9-phase-intervals-draft10.md"
# The raw operator metadata's `lift.type` for each lift S9 knows.
LIFT_TYPES: Final[dict[str, Lift]] = {"snch": "snatch", "CnJ": "clean_and_jerk"}


def body_from_bones(
    bones: Mapping[str, Mapping[str, object]], ankle_m: object, half_grip_mm: float | None
) -> Body | None:
    """The `Body` behind S8's `bones` and `ankleHeight` blocks (`usedM`, in metres); None if any
    bone S9 needs is missing. Left and right are averaged."""

    def used(name: str) -> float | None:
        value = bones.get(name, {}).get("usedM")
        return float(value) * 1000.0 if isinstance(value, int | float) else None

    def pair(name: str) -> float | None:
        left, right = used(f"left{name}"), used(f"right{name}")
        return None if left is None or right is None else (left + right) / 2.0

    parts = (
        float(ankle_m) * 1000.0 if isinstance(ankle_m, int | float) else None,
        pair("Tibia"), pair("Femur"), pair("Torso"), pair("UpperArm"), pair("Forearm"),
        used("shoulderWidth"),
    )  # fmt: skip
    if any(p is None for p in parts):
        return None
    a, t, f, torso, u, lo, w = (p for p in parts if p is not None)
    return Body(a, t, f, torso, u, lo, w, half_grip_mm)


def _mm(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


def _bands_document(
    detection: Detection, body_bands: Bands | None, thresholds: PhaseThresholds
) -> dict[str, dict[str, float | None]]:
    out: dict[str, Band] = {}
    rest = detection.derived.get("restMm")
    if isinstance(rest, int | float):
        out["floor"] = Band(rest, rest - thresholds.lift_mm, rest + thresholds.lift_mm)
    if body_bands is not None:
        out.update(body_bands.as_dict())
    return {
        name: {"centreMm": _mm(b.centre), "lowMm": _mm(b.low), "highMm": _mm(b.high)}
        for name, b in out.items()
    }


def _bar_track_document(track: BarTrack | None, frames: int) -> dict[str, list[object]]:
    """`barTrack` (§6): one entry per frame, 0.1 rounding; all null with no track, never absent."""

    if track is None:
        empty: list[object] = [None] * frames
        return {"rawMm": empty, "heightMm": list(empty), "velocityMmPerS": list(empty),
                "bridgedFrames": []}  # fmt: skip
    return {
        "rawMm": [_mm(x) for x in track.raw_mm],
        "heightMm": [_mm(x) for x in track.height_mm],
        "velocityMmPerS": [_mm(x) for x in track.velocity_mm_per_s],
        "bridgedFrames": list(track.bridged_frames),
    }


def to_document(
    detection: Detection,
    *,
    capture_id: str,
    t_ms: Sequence[int],
    start_epoch_ms: int,
    body_bands: Bands | None,
    thresholds: PhaseThresholds,
    inputs: Mapping[str, str],
) -> dict[str, object]:
    """`phase.json` (§6) for one capture."""

    def event(e: Event, name: str) -> dict[str, object]:
        known = e.frame is not None
        return {
            "frame": e.frame,
            "tMs": t_ms[e.frame] if e.frame is not None else None,
            "epochMs": start_epoch_ms + t_ms[e.frame] if e.frame is not None else None,
            "sigmaMs": None if e.sigma_ms is None else round(e.sigma_ms, 1),
            "status": e.status,
            "barMm": _mm(e.bar_mm) if known else None,
        }

    detector = {
        name: "rules-v1-fallback" if e.status == "fallback" else "rules-v1"
        for name, e in detection.events.items()
    }
    return {
        "schemaVersion": SCHEMA_VERSION,
        "captureId": capture_id,
        "lift": detection.lift,
        "method": METHOD_SPEC,
        "detector": detector,
        "frames": {"count": len(t_ms), "startEpochMs": start_epoch_ms, "tMs": list(t_ms)},
        "bands": _bands_document(detection, body_bands, thresholds),
        "events": {name: event(e, name) for name, e in detection.events.items()},
        "auxiliary": {name: event(e, name) for name, e in detection.auxiliary.items()},
        "phases": [
            {"phase": s.phase, "startFrame": s.start, "endFrame": s.end} for s in detection.phases
        ],
        "liftWindow": {
            "startFrame": detection.window.start,
            "endFrame": detection.window.end,
            "endReason": detection.window.end_reason,
            "marginMs": detection.window.margin_ms,
        },
        "barTrack": _bar_track_document(detection.bar_track, len(t_ms)),
        "result": {"value": detection.result[0], "source": detection.result[1]},
        "flags": list(detection.flags),
        "provenance": {
            "thresholds": thresholds.model_dump(mode="json"),
            "filterCutoffHz": thresholds.filter_cutoff_hz,
            "inputs": dict(inputs),
            **detection.derived,
        },
    }
