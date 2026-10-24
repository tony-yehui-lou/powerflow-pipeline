"""S9 · Bar HVA: one capture's `bar_hva.json`, from S5's bar track on S8's shared floor.

`docs/specs/preprocessing/S9-hva-tikhonov-draft2.md` §2, §7-§8, §12. Pure: the bar poses, floor
anchor and frame clock arrive already read (`tasks/bar_hva.py` does the I/O), so the whole document
is testable against a synthetic lift. `heightM` is the purely calculated height, exactly as the
level-bar solve places the centre; `heightRegM`, `velocityMps` and `accelMps2` are the regularised
estimate of `tikhonov.py`, solved per segment.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from powerflow_pipeline.data.preprocess import tikhonov as tk
from powerflow_pipeline.data.preprocess.bar_geometry import BAR_TYPES, BarPose, FloorAnchor

SCHEMA_VERSION: Final = "1.0.0"
METHOD_SPEC: Final = "docs/specs/preprocessing/S9-hva-tikhonov-draft2.md"

# P8: a run of empty nodes longer than this (100 ms at 60 fps) splits the series.
MAX_GAP_FRAMES: Final = 6
# P9: below this many observations delta's own uncertainty, 1/sqrt(2n), passes 13%.
MIN_SEGMENT_FRAMES: Final = 30
# P10: snapping further than this fraction of dt means the clock is not the uniform grid.
SNAP_WARN_FRACTION: Final = 0.25
# P11: sigma over its robust estimate above this is heavy-tailed noise, not white.
OUTLIER_RATIO: Final = 1.5
# P17: the data's acceleration over the first and last ENDPOINT_WINDOW_S, above this fraction of
# a_max and above ENDPOINT_SIGMAS standard errors, means the window was cut mid-motion.
ENDPOINT_FRACTION: Final = 0.1
ENDPOINT_WINDOW_S: Final = 0.25
ENDPOINT_SIGMAS: Final = 3.0

CollarOnlyMode = Literal["observe", "mask"]
SegmentStatus = Literal[
    "box_inactive",
    "morozov",
    "data_outside_bounds",
    "noise_dominates",
    "too_short",
    "no_noise_estimate",
    "solver_infeasible",
]


@dataclass(frozen=True)
class Settings:
    """§9.1's decisions: P1, P2, P3, P4 and the calibrated P5."""

    v_max_mps: float
    a_max_mps2: float
    collar_only: CollarOnlyMode
    tau: float
    kappa: float


@dataclass(frozen=True)
class Inputs:
    capture_id: str
    start_epoch_ms: int
    t_ms: Sequence[int]  # offsets from start_epoch_ms, one per frame
    bars: Sequence[BarPose | None]  # `solve_track`'s poses, one per frame
    anchor: FloorAnchor
    bar_type: str
    settings: Settings


@dataclass(frozen=True)
class Built:
    document: dict[str, Any]
    derived: dict[str, Any]
    warnings: list[str]


def _r(value: float | None, digits: int) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


@dataclass
class _Series:
    """Per-node outputs of every solved segment, before they are read back out per frame."""

    h: np.ndarray
    v: np.ndarray
    a: np.ndarray
    flags: list[set[str]]
    solved: np.ndarray


def _segment(
    z: np.ndarray,
    mask: np.ndarray,
    dt: float,
    segment: tk.Segment,
    settings: Settings,
    series: _Series,
) -> tuple[dict[str, Any], list[str]]:
    """Solve one segment into `series` and return its `segments[]` entry and warnings."""

    span = slice(segment.start, segment.stop)
    zs, ms = z[span], mask[span]
    n = segment.stop - segment.start
    entry: dict[str, Any] = {
        "startNode": segment.start,
        "endNode": segment.stop - 1,
        "observed": segment.observed,
        "alpha": None,
        "cutoffHz": None,
        "sigmaM": None,
        "sigmaRobustM": None,
        "kappa": settings.kappa,
        "deltaM": None,
        "tau": settings.tau,
        "residualM": None,
        "rLowM": None,
        "rHighM": None,
        "activeV": 0,
        "activeA": 0,
        "dAlpha": None,
        "bendingEnergy": None,
        "polished": None,
        "solverIterations": 0,
        "alphaEvaluations": 0,
        "atBracket": False,
        "endpointAccelMps2": None,
        "endpointAccelSeMps2": None,
        "heightErrorBoundRmsM": None,
        "accelErrorBoundRmsMps2": None,  # P18: held until the paper's (24)-(26) are checked
    }
    where = f"frames of nodes {segment.start}-{segment.stop - 1}"
    if segment.observed < MIN_SEGMENT_FRAMES:
        entry["status"] = "too_short"
        return entry, [f"bar HVA too_short: {segment.observed} observed {where}"]

    noise = tk.estimate_noise(zs, ms)
    if noise.samples == 0:
        entry["status"] = "no_noise_estimate"
        return entry, [f"bar HVA no_noise_estimate: no four consecutive observations in {where}"]
    delta = settings.kappa * noise.sigma * math.sqrt(segment.observed * dt)
    entry.update(
        sigmaM=_r(noise.sigma, 6), sigmaRobustM=_r(noise.sigma_robust, 6), deltaM=_r(delta, 7)
    )
    warnings: list[str] = []
    outliers = noise.sigma > OUTLIER_RATIO * noise.sigma_robust
    if outliers:
        warnings.append(
            f"bar HVA outliers: sigma {1000 * noise.sigma:.1f} mm against a robust "
            f"{1000 * noise.sigma_robust:.1f} mm in {where}; the noise is heavy-tailed"
        )

    bounds = tk.Bounds(v_max=settings.v_max_mps, a_max=settings.a_max_mps2)
    try:
        fit = tk.discrepancy(zs, ms, dt, bounds, target=settings.tau * delta)
    except tk.SolverError as error:
        entry["status"] = "solver_infeasible"
        return entry, [*warnings, f"bar HVA solver_infeasible in {where}: {error}"]
    entry.update(
        status=fit.status,
        rHighM=_r(fit.r_hi, 7),
        rLowM=_r(fit.r_lo, 7),
        residualM=_r(fit.residual, 7),
        solverIterations=fit.iterations,
        alphaEvaluations=fit.evaluations,
        atBracket=fit.at_bracket,
        polished=fit.polished,
    )
    if fit.h is None:
        return entry, [
            *warnings,
            f"bar HVA data_outside_bounds in {where}: the least-regularised feasible fit leaves "
            f"{1000 * (fit.r_lo or 0):.1f} mm·s^0.5 of residual against a target of "
            f"{1000 * fit.target:.1f}; bounds too tight, noise underestimated, or a track defect",
        ]

    if fit.status == "noise_dominates":
        slope = float(fit.h[1] - fit.h[0]) / dt
        v, a = np.full(n, slope), np.zeros(n)
        warnings.append(f"bar HVA noise_dominates in {where}: published the feasible line")
    else:
        v, a = tk.nodal_derivatives(fit.h, dt)
    entry["bendingEnergy"] = _r(tk.bending_energy(fit.h, dt), 6)

    width = 0
    if fit.alpha is not None:
        entry.update(alpha=fit.alpha, cutoffHz=_r(tk.cutoff_for_alpha(fit.alpha), 3))
        width = math.ceil((fit.alpha * tk.MU_A) ** 0.25 / dt)
        if fit.h_unconstrained is not None:
            gain = tk.energy_norm(fit.h_unconstrained - fit.h, dt, fit.alpha)
            entry["dAlpha"] = _r(gain, 7)
    if fit.at_bracket:
        warnings.append(f"bar HVA alpha at the bracket edge ({tk.CUTOFF_MIN_HZ} Hz) in {where}")
    if not fit.polished:
        warnings.append(f"bar HVA unpolished: the active set did not settle in {where}")

    flags: list[set[str]] = [set() for _ in range(n)]
    if fit.multipliers is not None:
        active = fit.multipliers != 0
        active_v, active_a = active[: n - 1], active[n - 1 :]
        entry.update(activeV=int(active_v.sum()), activeA=int(active_a.sum()))
        for j in np.flatnonzero(active_v):
            flags[j].add("v_active")
            flags[j + 1].add("v_active")
        for j in np.flatnonzero(active_a):
            flags[j + 1].add("a_active")
    for k in range(n):
        if not ms[k]:
            flags[k].add("bridged")
        if k < width or n - 1 - k < width:
            flags[k].add("edge")
        if outliers:
            flags[k].add("outliers")

    window = min(n, round(ENDPOINT_WINDOW_S / dt) + 1)
    ends, errors = tk.end_acceleration(zs, ms, dt, window)
    entry["endpointAccelMps2"] = [_r(float(e), 3) for e in ends]
    entry["endpointAccelSeMps2"] = [_r(float(e), 3) for e in errors]
    for label, value, stderr in zip(("start", "end"), ends, errors, strict=True):
        if abs(value) > max(ENDPOINT_FRACTION * settings.a_max_mps2, ENDPOINT_SIGMAS * stderr):
            warnings.append(
                f"bar HVA endpoint: the data accelerate at {value:.1f} m/s^2 at the {label} of "
                f"{where}; the window looks cut mid-motion, so acceleration near that end "
                "converges at the slower rate (Prop. 4.4)"
            )
    if bool(np.all(ms)):
        entry["heightErrorBoundRmsM"] = _r((1 + settings.tau) * delta / math.sqrt(n * dt), 6)

    series.h[span], series.v[span], series.a[span] = fit.h, v, a
    series.solved[span] = True
    for k in range(n):
        series.flags[segment.start + k] |= flags[k]
    return entry, warnings


def _document(
    inputs: Inputs,
    raw: list[float | None],
    grid: tk.Grid | None,
    series: _Series | None,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(inputs.bars)
    s = inputs.settings
    sources = [None if bar is None else bar.source for bar in inputs.bars]
    if grid is None or series is None:
        reg: list[float | None] = [None] * n
        vel: list[float | None] = [None] * n
        acc: list[float | None] = [None] * n
        flags: list[list[str]] = [[] for _ in range(n)]
    else:
        nodes = grid.nodes
        reg, vel, acc, flags = [], [], [], []
        for i, k in enumerate(nodes):
            solved = bool(series.solved[k])
            bridged = "bridged" in series.flags[k]
            reg.append(_r(series.h[k], 5) if solved and not bridged else None)
            vel.append(_r(series.v[k], 4) if solved else None)
            acc.append(_r(series.a[k], 3) if solved else None)
            frame_flags = set(series.flags[k])
            if sources[i] == "collar-only":
                frame_flags.add("collar_only")
            flags.append(sorted(frame_flags))
        for entry in segments:
            entry["startFrame"] = nodes.index(entry.pop("startNode"))
            entry["endFrame"] = nodes.index(entry.pop("endNode"))
    bar = BAR_TYPES[inputs.bar_type]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "captureId": inputs.capture_id,
        "method": METHOD_SPEC,
        "frames": {"count": n, "startEpochMs": inputs.start_epoch_ms, "tMs": list(inputs.t_ms)},
        "grid": None
        if grid is None
        else {
            "dtS": round(grid.dt_s, 7),
            "nodes": grid.n_nodes,
            "snapErrorMs": round(grid.snap_error_ms, 3),
            "droppedNodes": list(grid.dropped),
        },
        "settings": {
            "vMaxMps": s.v_max_mps,
            "aMaxMps2": s.a_max_mps2,
            "collarOnly": s.collar_only,
            "tau": s.tau,
            "kappa": s.kappa,
            "muA": tk.MU_A,
            "muJ": 0.0,
            "maxGapFrames": MAX_GAP_FRAMES,
            "minSegmentFrames": MIN_SEGMENT_FRAMES,
        },
        "bar": {
            "type": inputs.bar_type,
            "capToCentreM": bar.cap_to_centre_mm / 1000,
            "capToCollarM": bar.cap_to_collar_mm / 1000,
        },
        "floor": {
            "anchor": inputs.anchor.method,
            "yFloorM": _r(
                None if inputs.anchor.y_floor_mm is None else inputs.anchor.y_floor_mm / 1000, 5
            ),
            "restFrames": inputs.anchor.rest_frames,
            "s3FloorPlaneYM": _r(
                None
                if inputs.anchor.floor_plane_y_mm is None
                else inputs.anchor.floor_plane_y_mm / 1000,
                5,
            ),
        },
        "com": {
            "heightM": raw,
            "heightRegM": reg,
            "velocityMps": vel,
            "accelMps2": acc,
            "source": sources,
            "flags": flags,
            "segments": segments,
        },
    }


def build(inputs: Inputs) -> Built:
    """The capture's `bar_hva.json`, its manifest values and its warnings."""

    y_floor = inputs.anchor.y_floor_mm
    raw: list[float | None] = [
        None if bar is None or y_floor is None else _r((y_floor - bar.centre_mm[1]) / 1000, 5)
        for bar in inputs.bars
    ]
    warnings: list[str] = []
    derived: dict[str, Any] = {"bar_hva_floor_anchor": inputs.anchor.method}
    if y_floor is None:
        warnings.append("bar HVA: no floor (no rest-plate anchor and no S3 plane): nothing solved")
        document = _document(inputs, [None] * len(raw), None, None, [])
        return Built(document=document, derived=derived, warnings=warnings)
    try:
        grid = tk.snap_to_grid(inputs.t_ms)
    except tk.GridError as error:
        warnings.append(f"bar HVA: {error}: nothing solved")
        return Built(_document(inputs, raw, None, None, []), derived, warnings)
    if grid.snap_error_ms > SNAP_WARN_FRACTION * grid.dt_s * 1000:
        warnings.append(
            f"bar HVA: frames snap up to {grid.snap_error_ms:.1f} ms off a {1000 * grid.dt_s:.2f} "
            "ms grid; the frame clock is not uniform"
        )

    z = np.full(grid.n_nodes, np.nan)
    mask = np.zeros(grid.n_nodes, dtype=bool)
    observe_collar_only = inputs.settings.collar_only == "observe"
    for i, k in enumerate(grid.nodes):
        bar, height = inputs.bars[i], raw[i]
        if bar is None or height is None:
            continue
        if bar.source == "collar-only" and not observe_collar_only:
            continue
        z[k], mask[k] = height, True

    series = _Series(
        h=np.full(grid.n_nodes, np.nan),
        v=np.full(grid.n_nodes, np.nan),
        a=np.full(grid.n_nodes, np.nan),
        flags=[set() for _ in range(grid.n_nodes)],
        solved=np.zeros(grid.n_nodes, dtype=bool),
    )
    entries = []
    if not mask.any():
        warnings.append("bar HVA: S5's bar was placed on no frame: nothing solved")
    for segment in tk.segments(mask.tolist(), max_gap=MAX_GAP_FRAMES):
        entry, segment_warnings = _segment(z, mask, grid.dt_s, segment, inputs.settings, series)
        entries.append(entry)
        warnings.extend(segment_warnings)
    document = _document(inputs, raw, grid, series, entries)

    velocity = [v for v in document["com"]["velocityMps"] if v is not None]
    peak = max(range(len(velocity)), key=velocity.__getitem__) if velocity else None
    solved_t = [
        t for t, v in zip(inputs.t_ms, document["com"]["velocityMps"], strict=True) if v is not None
    ]
    derived.update(
        bar_hva_status=dict(Counter(entry["status"] for entry in entries)),
        bar_hva_cutoff_hz=[entry["cutoffHz"] for entry in entries],
        bar_hva_sigma_mm=[
            _r(None if e["sigmaM"] is None else 1000 * e["sigmaM"], 2) for e in entries
        ],
        bar_hva_peak_velocity_mps=None if peak is None else velocity[peak],
        bar_hva_peak_velocity_s=None if peak is None else round(solved_t[peak] / 1000, 3),
        bar_hva_active_frames=sum(
            1 for f in document["com"]["flags"] if "v_active" in f or "a_active" in f
        ),
        bar_hva_snap_error_ms=round(grid.snap_error_ms, 3),
    )
    return Built(document=document, derived=derived, warnings=warnings)
