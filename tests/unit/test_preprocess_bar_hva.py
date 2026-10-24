"""S9 · Bar HVA's document, built from a synthetic bar track: `S9-hva-tikhonov-draft2.md` §2, §7-§8,
§12. The maths is `test_preprocess_tikhonov`'s; this checks what reaches `bar_hva.json` and the
manifest, frame by frame, against the trajectory the bar poses were made from.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from powerflow_pipeline.data.preprocess import bar_hva
from powerflow_pipeline.data.preprocess.bar_geometry import BarPose, FloorAnchor, PoseSource
from tests.unit.test_preprocess_tikhonov import bang_bang, profile

Y_FLOOR_MM = 1500.0
ANCHOR = FloorAnchor(
    y_floor_mm=Y_FLOOR_MM, method="rest-plate", rest_frames=60, floor_plane_y_mm=1530.0
)
SETTINGS = bar_hva.Settings(
    v_max_mps=5.0, a_max_mps2=25.0, collar_only="observe", tau=1.1, kappa=1.0
)
FPS = 60.0


def t_ms(n: int) -> list[int]:
    return [round(i * 1000 / FPS) for i in range(n)]


def pose(height_m: float, source: PoseSource = "level") -> BarPose:
    y = Y_FLOOR_MM - 1000.0 * height_m
    return BarPose(
        cap_mm=(-800.0, y, 2500.0),
        collar_mm=(800.0, y, 3500.0),
        centre_mm=(0.0, y, 3000.0),
        yaw_deg=38.0,
        source=source,
    )


def heights(n: int, sigma: float = 0.003, seed: int = 0) -> np.ndarray:
    t = np.arange(n) / FPS
    return profile(t) + np.random.default_rng(seed).normal(0.0, sigma, n)


def inputs(
    bars: Sequence[BarPose | None],
    times: Sequence[int] | None = None,
    anchor: FloorAnchor = ANCHOR,
    settings: bar_hva.Settings = SETTINGS,
) -> bar_hva.Inputs:
    return bar_hva.Inputs(
        capture_id="10 September/50kg3",
        start_epoch_ms=1_700_000_000_000,
        t_ms=t_ms(len(bars)) if times is None else list(times),
        bars=list(bars),
        anchor=anchor,
        bar_type="male IWF",
        settings=settings,
    )


def lift(n: int = 259, **kwargs: object) -> list[BarPose | None]:
    return [pose(float(h)) for h in heights(n, **kwargs)]  # type: ignore[arg-type]


def test_the_raw_height_is_purely_calculated() -> None:
    bars = lift()

    built = bar_hva.build(inputs(bars))

    com = built.document["com"]
    for i, bar in enumerate(bars):
        assert bar is not None
        assert com["heightM"][i] == pytest.approx((Y_FLOOR_MM - bar.centre_mm[1]) / 1000, abs=1e-5)
    assert com["source"] == ["level"] * len(bars)


def test_a_clean_lift_is_box_inactive_and_recovers_the_motion() -> None:
    n = 259
    built = bar_hva.build(inputs(lift(n)))

    document = built.document
    [segment] = document["com"]["segments"]
    assert segment["status"] == "box_inactive"
    assert (segment["startFrame"], segment["endFrame"]) == (0, n - 1)
    assert 2.0 < segment["cutoffHz"] < 20.0
    assert segment["sigmaM"] == pytest.approx(0.003, rel=0.2)
    assert segment["heightErrorBoundRmsM"] == pytest.approx(
        2.1 * segment["deltaM"] / math.sqrt(n / FPS), rel=1e-3
    )
    assert segment["accelErrorBoundRmsMps2"] is None  # P18: held until (24)-(26) are checked

    t = np.arange(n) / FPS
    v = np.array(document["com"]["velocityMps"])
    assert abs(v.max() - profile(t, 1).max()) < 0.05
    assert built.derived["bar_hva_status"] == {"box_inactive": 1}
    assert built.derived["bar_hva_peak_velocity_mps"] == pytest.approx(v.max(), abs=1e-3)
    assert built.derived["bar_hva_floor_anchor"] == "rest-plate"
    assert document["settings"]["vMaxMps"] == 5.0
    assert document["grid"]["nodes"] == n
    assert not built.warnings


def test_edges_are_flagged_by_the_boundary_layer_width() -> None:
    built = bar_hva.build(inputs(lift()))

    flags = built.document["com"]["flags"]
    alpha = built.document["com"]["segments"][0]["alpha"]
    width = math.ceil(alpha**0.25 * FPS)
    assert all("edge" in flags[i] for i in range(width))
    assert "edge" not in flags[width]
    assert "edge" in flags[-1]


def test_a_dropped_frame_is_an_empty_node_and_a_missing_bar_is_bridged() -> None:
    bars = lift()
    times = t_ms(len(bars))
    del bars[100], times[100]  # dropped: no frame at all
    bars[150] = None  # a frame with no bar

    built = bar_hva.build(inputs(bars, times))

    document = built.document
    assert document["grid"]["droppedNodes"] == [100]
    com = document["com"]
    assert com["heightM"][150] is None
    assert com["heightRegM"][150] is None
    assert com["velocityMps"][150] is not None
    assert "bridged" in com["flags"][150]
    assert com["segments"][0]["heightErrorBoundRmsM"] is None  # empty nodes: no certificate


def test_a_long_gap_splits_the_series() -> None:
    bars = lift()
    bars[120:135] = [None] * 15

    built = bar_hva.build(inputs(bars))

    segments = built.document["com"]["segments"]
    assert [(s["startFrame"], s["endFrame"]) for s in segments] == [(0, 119), (135, 258)]
    assert built.document["com"]["velocityMps"][125] is None


def test_a_short_series_is_not_solved() -> None:
    built = bar_hva.build(inputs(lift(20)))

    [segment] = built.document["com"]["segments"]
    assert segment["status"] == "too_short"
    assert built.document["com"]["velocityMps"] == [None] * 20
    assert built.document["com"]["heightM"][0] is not None
    assert any("too_short" in w for w in built.warnings)


def test_without_a_floor_nothing_is_solved() -> None:
    no_floor = FloorAnchor(y_floor_mm=None, method="none", rest_frames=0, floor_plane_y_mm=None)

    built = bar_hva.build(inputs(lift(), anchor=no_floor))

    assert built.document["com"]["heightM"] == [None] * 259
    assert built.document["com"]["segments"] == []
    assert any("no floor" in w for w in built.warnings)


def test_a_grid_collision_is_reported_not_solved() -> None:
    bars = lift(100)
    times = t_ms(100)
    times[50] = times[49] + 2

    built = bar_hva.build(inputs(bars, times))

    assert built.document["com"]["velocityMps"] == [None] * 100
    assert built.document["grid"] is None
    assert any("collision" in w for w in built.warnings)


@pytest.mark.parametrize(("mode", "observed"), [("observe", True), ("mask", False)])
def test_collar_only_frames_follow_p3(mode: str, observed: bool) -> None:
    bars = lift()
    for i in range(82, 87):
        bar = bars[i]
        assert bar is not None
        bars[i] = BarPose(bar.cap_mm, bar.collar_mm, bar.centre_mm, bar.yaw_deg, "collar-only")
    settings = bar_hva.Settings(5.0, 25.0, mode, 1.1, 1.0)  # type: ignore[arg-type]

    built = bar_hva.build(inputs(bars, settings=settings))

    com = built.document["com"]
    assert "collar_only" in com["flags"][85]
    assert (com["heightRegM"][85] is not None) is observed
    assert ("bridged" in com["flags"][85]) is not observed


def test_a_saturating_lift_is_solved_on_the_box_and_flags_the_active_frames() -> None:
    n = 181
    t = np.arange(n) / FPS
    h = bang_bang(t, 4.0) + np.random.default_rng(1).normal(0, 0.003, n)
    settings = bar_hva.Settings(5.0, 4.0, "observe", 1.1, 1.0)

    built = bar_hva.build(inputs([pose(float(x)) for x in h], settings=settings))

    com = built.document["com"]
    [segment] = com["segments"]
    assert segment["status"] == "morozov"
    assert segment["activeA"] > 0
    assert any("a_active" in f for f in com["flags"])
    assert max(abs(a) for a in com["accelMps2"]) <= 4.0 * (1 + 1e-6)
    assert built.derived["bar_hva_active_frames"] > 0


def test_data_outside_the_bounds_publishes_no_derivatives() -> None:
    settings = bar_hva.Settings(0.5, 1.0, "observe", 1.1, 1.0)

    built = bar_hva.build(inputs(lift(), settings=settings))

    assert built.document["com"]["segments"][0]["status"] == "data_outside_bounds"
    assert built.document["com"]["velocityMps"] == [None] * 259
    assert any("data_outside_bounds" in w for w in built.warnings)


def test_a_bar_at_rest_is_noise_dominated() -> None:
    rng = np.random.default_rng(2)
    bars = [pose(0.225 + float(e)) for e in rng.normal(0, 0.003, 120)]

    built = bar_hva.build(inputs(bars))

    com = built.document["com"]
    assert com["segments"][0]["status"] == "noise_dominates"
    assert max(abs(v) for v in com["velocityMps"]) < 0.05
    assert com["accelMps2"] == [0.0] * 120


def test_glitches_raise_the_outlier_flag() -> None:
    bars = lift()
    for i in (60, 130, 200):
        bar = bars[i]
        assert bar is not None
        bars[i] = pose((Y_FLOOR_MM - bar.centre_mm[1]) / 1000 + 0.05)

    built = bar_hva.build(inputs(bars))

    assert "outliers" in built.document["com"]["flags"][0]
    assert any("outliers" in w for w in built.warnings)


def test_a_window_cut_mid_motion_warns_at_the_endpoint() -> None:
    n = 156  # ends at 2.58 s, accelerating at 6 m/s^2 into the recovery
    built = bar_hva.build(inputs(lift(n)))

    assert any("endpoint" in w and "end of" in w for w in built.warnings)
    assert not any("start of" in w for w in built.warnings)  # it starts at rest
    segment = built.document["com"]["segments"][0]
    start, end = segment["endpointAccelMps2"]
    assert abs(start) < 3 * segment["endpointAccelSeMps2"][0]
    window = np.arange(n - 16, n) / FPS  # the 0.25 s the local fit spans
    assert end == pytest.approx(float(np.mean(profile(window, 2))), abs=1.0)  # not the fit's ~0


def test_kappa_inflates_delta() -> None:
    plain = bar_hva.build(inputs(lift()))
    inflated = bar_hva.build(
        inputs(lift(), settings=bar_hva.Settings(5.0, 25.0, "observe", 1.1, 1.5))
    )

    a = plain.document["com"]["segments"][0]
    b = inflated.document["com"]["segments"][0]
    assert b["deltaM"] == pytest.approx(1.5 * a["deltaM"], rel=1e-4)
    assert b["cutoffHz"] < a["cutoffHz"]  # more noise allowed: more smoothing


def test_no_four_consecutive_observations_means_no_noise_estimate() -> None:
    bars: list[BarPose | None] = [pose(0.225 + 0.001 * i) if i % 3 else None for i in range(120)]

    built = bar_hva.build(inputs(bars))

    assert built.document["com"]["segments"][0]["status"] == "no_noise_estimate"


def test_a_frame_clock_far_from_uniform_warns() -> None:
    times = [round(i * 1000 / FPS + (6 if i % 2 else 0)) for i in range(259)]

    built = bar_hva.build(inputs(lift(), times))

    assert any("frame clock" in w for w in built.warnings)


def test_a_bar_placed_on_no_frame_warns() -> None:
    built = bar_hva.build(inputs([None] * 50))

    assert built.document["com"]["segments"] == []
    assert any("placed on no frame" in w for w in built.warnings)
