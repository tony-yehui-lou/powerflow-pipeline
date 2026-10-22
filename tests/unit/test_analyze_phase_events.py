"""S9 Phase rules on synthetic bar curves with known event times (S9 draft 7 §8.2).

Each curve is built from bar-velocity knots: PCHIP keeps every extremum and zero of the velocity
on a knot, so the knot times are the true event times. Heights are the integral of that velocity,
sized to the 10 September lifter's bands (§2.3): knee 503 mm, hip 926, rack 1441, overhead 1959.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from powerflow_pipeline.data.analyze import phase_events as pe
from powerflow_pipeline.data.analyze.config import PhaseThresholds

FRAME_MS = 1000.0 / 60.0
T = PhaseThresholds()
# The 10 September lifter's measured bones (§2.3) and 30kg2's half grip.
BODY = pe.Body(
    ankle_mm=105.0, tibia_mm=398.0, femur_mm=423.0, torso_mm=515.0, upper_arm_mm=290.0,
    forearm_mm=280.0, shoulder_width_mm=440.0, half_grip_mm=457.1,
)  # fmt: skip

# Snatch: rest, first pull (v peak 1250, scoop 1400), E2 at 1600 (1055 mm), peak at 1750, the drop
# under to the catch at 1950 (1221 mm), stand up to overhead (1955 mm) by 2650, then still.
SNATCH_V = [
    (0, 0), (1000, 0), (1250, 1500), (1400, 1200), (1600, 3200), (1750, 0),
    (1850, -800), (1950, 0), (2300, 1900), (2650, 0), (4000, 0),
]  # fmt: skip
# Clean & jerk: the same pull shape, the clean catch at 1950 (990 mm), stand to the rack (1447 mm)
# by 2700, a still rack pause, the dip from 3500 to its bottom at 3800, the drive to a peak at
# 4100, the drop under into the jerk catch at 4330 (1739 mm), the rise to 1897 mm, then still.
CNJ_V = [
    (0, 0), (1000, 0), (1250, 1300), (1400, 1000), (1600, 2200), (1750, 0),
    (1850, -600), (1950, 0), (2300, 1100), (2700, 0), (3500, 0), (3650, -700), (3800, 0),
    (3950, 3000), (4100, 0), (4200, -500), (4330, 0), (4600, 500), (4900, 0), (6500, 0),
]  # fmt: skip


@dataclass(frozen=True)
class Curve:
    t_ms: list[int]
    bar: list[float | None]
    knee: list[float | None]

    def signals(self) -> pe.Signals:
        return pe.Signals(t_ms=self.t_ms, bar_mm=self.bar, knee_mm=self.knee)


def _curve(v_knots: list[tuple[float, float]], *, knee: float | None = 500.0) -> Curve:
    kt, kv = zip(*v_knots, strict=True)
    t = np.arange(0.0, kt[-1], FRAME_MS)
    v = PchipInterpolator(kt, kv)(t)
    h = 225.0 + cumulative_trapezoid(v, t / 1000.0, initial=0.0)
    return Curve(t_ms=[round(x) for x in t], bar=[float(x) for x in h], knee=[knee] * len(t))


def _detect(curve: Curve, lift: pe.Lift, body: pe.Body = BODY, **kwargs: object) -> pe.Detection:
    bands = pe.bands(body, thresholds=T)
    return pe.detect(curve.signals(), lift, T, bands, **kwargs)  # type: ignore[arg-type]


def _ms(curve: Curve, event: pe.Event) -> float:
    assert event.frame is not None, event
    return float(curve.t_ms[event.frame])


def _assert_near(curve: Curve, event: pe.Event, truth_ms: float, frames: float = 1.0) -> None:
    assert event.status in ("found", "fallback"), event
    tolerance = max(frames * FRAME_MS, event.sigma_ms or 0.0) + 1.0
    assert abs(_ms(curve, event) - truth_ms) <= tolerance, (_ms(curve, event), truth_ms, event)


def _first_ms(curve: Curve, at_least: float, after_ms: float = 0.0) -> int:
    return next(
        t
        for t, h in zip(curve.t_ms, curve.bar, strict=True)
        if h is not None and h >= at_least and t > after_ms
    )


OVERHEAD_LOW = pe.bands(BODY, thresholds=T).overhead.low


# --- bands -------------------------------------------------------------------------------------


def test_bands_follow_the_bone_lengths() -> None:
    b = pe.bands(BODY, thresholds=T)

    assert b.knee.centre == pytest.approx(503.0)
    assert b.hip.centre == pytest.approx(926.0)
    assert b.rack.centre == pytest.approx(1441.0)
    reach = ((290.0 + 280.0) ** 2 - (457.1 - 220.0) ** 2) ** 0.5
    assert b.overhead.centre == pytest.approx(1441.0 + reach)
    assert round(b.overhead.centre) == 1959  # §2.3
    assert (b.rack.low, b.rack.high) == pytest.approx((1441.0 * 0.85, 1441.0 * 1.10))
    assert not b.grip_assumed


def test_without_a_grip_the_arms_are_vertical_and_the_band_widens() -> None:
    b = pe.bands(replace(BODY, half_grip_mm=None), thresholds=T)

    assert b.grip_assumed
    assert b.overhead.centre == pytest.approx(1441.0 + 570.0)
    assert b.overhead.low == pytest.approx(2011.0 * 0.75)


def test_band_tests() -> None:
    band = pe.Band(centre=100.0, low=90.0, high=110.0)
    assert band.contains(90.0) and band.contains(110.0) and not band.contains(110.1)
    assert band.above(110.1) and not band.above(110.0)
    assert band.below(89.9) and not band.below(90.0)


# --- snatch ------------------------------------------------------------------------------------


def test_snatch_recovers_every_event() -> None:
    curve = _curve(SNATCH_V)
    found = _detect(curve, "snatch")

    assert list(found.events) == ["liftoff", "barPassesKnee", "fullExtension", "catch", "stand"]
    _assert_near(curve, found.events["liftoff"], 1000)
    _assert_near(curve, found.events["barPassesKnee"], _first_ms(curve, 500.0))
    assert found.events["barPassesKnee"].status == "found"
    _assert_near(curve, found.events["fullExtension"], 1600)
    _assert_near(curve, found.events["catch"], 1950)
    _assert_near(curve, found.events["stand"], _first_ms(curve, OVERHEAD_LOW, after_ms=1950))
    assert found.result == ("made", "rules")
    assert found.flags == []


def test_snatch_phases_cover_every_frame_in_order() -> None:
    curve = _curve(SNATCH_V)
    found = _detect(curve, "snatch")

    names = [p.phase for p in found.phases]
    assert names == ["setup", "first_pull", "second_pull", "turnover", "recovery", "finish"]
    assert found.phases[0].start == 0
    assert found.phases[-1].end == len(curve.t_ms)
    for a, b in zip(found.phases, found.phases[1:], strict=False):
        assert a.end == b.start and a.start < a.end


def test_e1_falls_back_to_the_knee_band_without_s8s_knee() -> None:
    curve = _curve(SNATCH_V, knee=None)
    found = _detect(curve, "snatch")

    e1 = found.events["barPassesKnee"]
    assert e1.status == "fallback"
    _assert_near(curve, e1, _first_ms(curve, 503.0))
    assert "knee_unavailable" in found.flags


def test_an_s8_knee_outside_the_knee_band_falls_back_to_the_band() -> None:
    curve = _curve(SNATCH_V, knee=300.0)  # the bar crosses 300 mm, below the 403-604 mm band
    found = _detect(curve, "snatch")

    e1 = found.events["barPassesKnee"]
    assert e1.status == "fallback"
    _assert_near(curve, e1, _first_ms(curve, 503.0))
    assert "knee_out_of_band" in found.flags
    assert "knee_unavailable" not in found.flags


def test_an_e2_outside_the_hip_band_is_kept_and_flagged() -> None:
    long_femur = replace(BODY, femur_mm=900.0)  # hip band 1122-1684 mm; E2 is at 1055 mm
    found = _detect(_curve(SNATCH_V), "snatch", long_femur)

    assert found.events["fullExtension"].status == "found"
    assert "height_out_of_band" in found.flags


def test_a_still_bar_below_overhead_is_not_a_stand() -> None:
    v = [*SNATCH_V[:8], (2150, 700), (2350, 0), (4000, 0)]  # stands still at about 1300 mm
    found = _detect(_curve(v), "snatch")

    assert found.events["catch"].status == "found"
    assert found.events["stand"].status == "absent"
    assert found.result == ("unknown", "rules")


def test_a_bar_lowered_as_soon_as_it_reaches_overhead_is_a_made_lift() -> None:
    # 30kg1 and 40kg1 (§8.1): overhead, then straight back down with no hold.
    v = [*SNATCH_V[:8], (2300, 1900), (2600, 0), (2700, -600), (2900, -3000), (3300, 0), (4000, 0)]
    curve = _curve(v)
    found = _detect(curve, "snatch")

    assert max(h for h in curve.bar if h is not None) >= OVERHEAD_LOW
    assert found.events["stand"].status == "found"
    assert found.result == ("made", "rules")


def test_lowering_the_bar_after_the_stand_is_a_made_lift() -> None:
    v = [*SNATCH_V[:10], (3200, 0), (3500, -3000), (3900, 0), (4500, 0)]
    curve = _curve(v)
    found = _detect(curve, "snatch")

    assert min(h for h in curve.bar if h is not None) < 700.0
    assert found.events["stand"].status == "found"
    assert "missed" not in found.flags
    assert found.result == ("made", "rules")


def test_missed_snatch_drops_before_standing() -> None:
    v = [*SNATCH_V[:8], (2100, 300), (2200, 0), (2400, -4500), (2650, 0), (4000, 0)]
    curve = _curve(v)
    found = _detect(curve, "snatch")

    assert found.events["catch"].status == "found"
    assert found.events["stand"].status == "absent"
    assert "missed" in found.flags
    assert found.result == ("missed", "rules")
    assert found.phases[-1].phase == "recovery"
    drop = found.phases[-1].end
    assert curve.bar[drop] is not None and curve.bar[drop] < 926.0 * 0.8  # type: ignore[operator]


def test_metadata_result_overrides_the_rules() -> None:
    curve = _curve(SNATCH_V)
    assert _detect(curve, "snatch", result_override="fail").result == ("missed", "metadata")
    # The raw template's placeholder list is not a result.
    placeholder = _detect(curve, "snatch", result_override=["success", "fail"])
    assert placeholder.result == ("made", "rules")


# --- clean & jerk ------------------------------------------------------------------------------


def test_clean_and_jerk_recovers_every_event() -> None:
    curve = _curve(CNJ_V)
    found = _detect(curve, "clean_and_jerk")

    assert list(found.events) == [
        "liftoff", "barPassesKnee", "fullExtension", "cleanCatch", "dipStart", "dipBottom",
        "jerkCatch", "stand",
    ]  # fmt: skip
    _assert_near(curve, found.events["liftoff"], 1000)
    _assert_near(curve, found.events["fullExtension"], 1600)
    _assert_near(curve, found.events["cleanCatch"], 1950)
    _assert_near(curve, found.events["dipStart"], 3500)
    _assert_near(curve, found.events["dipBottom"], 3800)
    _assert_near(curve, found.events["jerkCatch"], 4330)
    _assert_near(curve, found.events["stand"], _first_ms(curve, OVERHEAD_LOW, after_ms=4330))
    assert found.auxiliary["rackStand"].status == "found"
    assert 2550 <= _ms(curve, found.auxiliary["rackStand"]) <= 2750
    assert found.result == ("made", "rules")
    assert found.flags == []
    assert [p.phase for p in found.phases] == [
        "setup", "first_pull", "second_pull", "turnover", "clean_recovery", "dip", "drive",
        "jerk_recovery", "finish",
    ]  # fmt: skip


def test_missed_clean_drops_from_the_rack() -> None:
    v = [*CNJ_V[:8], (2050, 200), (2150, 0), (2300, -3000), (2500, 0), (6500, 0)]
    found = _detect(_curve(v), "clean_and_jerk")

    assert found.events["cleanCatch"].status == "found"
    for name in ("dipStart", "dipBottom", "jerkCatch", "stand"):
        assert found.events[name].status == "absent"
    assert "missed_clean" in found.flags
    assert found.result == ("missed_clean", "rules")
    assert found.phases[-1].phase == "clean_recovery"


def test_missed_jerk_drops_from_overhead() -> None:
    v = [*CNJ_V[:17], (4450, 100), (4550, 0), (4650, -5000), (4900, 0), (6500, 0)]
    found = _detect(_curve(v), "clean_and_jerk")

    assert found.events["jerkCatch"].status == "found"
    assert found.events["stand"].status == "absent"
    assert "missed_jerk" in found.flags
    assert found.result == ("missed_jerk", "rules")
    assert found.phases[-1].phase == "jerk_recovery"


def test_a_jerk_never_caught_is_missed_in_the_drive() -> None:
    v = [*CNJ_V[:15], (4300, -5000), (4500, 0), (6500, 0)]  # straight down from the peak
    found = _detect(_curve(v), "clean_and_jerk")

    assert found.events["dipBottom"].status == "found"
    assert found.events["jerkCatch"].status == "absent"
    assert "missed_jerk" in found.flags
    assert found.phases[-1].phase == "drive"


def test_a_short_rack_pause_still_finds_the_rack_stand_and_dip() -> None:
    v = [*CNJ_V[:10], (3050, 0), (3200, -700), (3350, 0), (3500, 3000), (3650, 0),
         (3750, -500), (3880, 0), (4150, 500), (4450, 0), (6000, 0)]  # fmt: skip
    curve = _curve(v)
    found = _detect(curve, "clean_and_jerk")

    assert found.auxiliary["rackStand"].status == "found"
    _assert_near(curve, found.events["dipStart"], 3050)
    _assert_near(curve, found.events["dipBottom"], 3350)


def test_regrip_bounces_in_the_rack_do_not_start_the_dip() -> None:
    v = [*CNJ_V[:10], (3000, 0), (3050, -100), (3100, 0), (3150, 100), (3200, 0), *CNJ_V[10:]]
    curve = _curve(v)
    _assert_near(curve, _detect(curve, "clean_and_jerk").events["dipStart"], 3500)


def test_a_dip_shallower_than_dip_mm_finds_no_dip() -> None:
    v = [*CNJ_V[:11], (3600, -150), (3700, 0), (6500, 0)]  # about 15 mm down
    found = _detect(_curve(v), "clean_and_jerk")
    assert found.events["dipStart"].status == "absent"
    assert found.events["dipBottom"].status == "absent"


def test_each_catch_gate_rejects_the_wrong_height() -> None:
    # A catch overhead, above the rack band, is not a clean catch.
    tall = [*SNATCH_V[:4], (1600, 5600), *SNATCH_V[5:]]  # catch at 1643 mm
    assert _detect(_curve(tall), "clean_and_jerk").events["cleanCatch"].status == "absent"
    # A catch at the shoulders, under the hip band's top, is not a snatch catch.
    assert _detect(_curve(CNJ_V), "snatch").events["catch"].status == "absent"


# --- robustness --------------------------------------------------------------------------------


def _without(curve: Curve, frames: range) -> Curve:
    return replace(curve, bar=[None if i in frames else h for i, h in enumerate(curve.bar)])


def test_a_short_gap_at_the_catch_is_bridged_and_widens_sigma() -> None:
    curve = _curve(SNATCH_V)
    catch = round(1950 / FRAME_MS)
    found = _detect(_without(curve, range(catch - 2, catch + 3)), "snatch")  # 5 frames, ~83 ms

    assert "bar_gap_bridged" in found.flags
    event = found.events["catch"]
    assert event.status == "found"
    assert event.sigma_ms is not None and event.sigma_ms >= 40.0


def test_a_long_gap_at_the_catch_makes_it_absent() -> None:
    curve = _curve(SNATCH_V)
    catch = round(1950 / FRAME_MS)
    found = _detect(_without(curve, range(catch - 6, catch + 7)), "snatch")

    assert found.events["catch"].status == "absent"
    assert [p.phase for p in found.phases] == ["setup", "first_pull", "second_pull"]


def test_a_dropped_frame_leaves_event_times_unchanged() -> None:
    curve = _curve(SNATCH_V)
    drop = round(1300 / FRAME_MS)
    keep = [i for i in range(len(curve.t_ms)) if i != drop]
    dropped = Curve(
        t_ms=[curve.t_ms[i] for i in keep],
        bar=[curve.bar[i] for i in keep],
        knee=[curve.knee[i] for i in keep],
    )
    a, b = _detect(curve, "snatch"), _detect(dropped, "snatch")
    for name in a.events:
        assert abs(_ms(curve, a.events[name]) - _ms(dropped, b.events[name])) <= FRAME_MS + 1


def test_height_noise_moves_events_by_no_more_than_their_sigma() -> None:
    """2 mm per-frame height noise: the rest-window spread measured on 10 September (1.1-3.4 mm)."""

    curve = _curve(SNATCH_V)
    rng = np.random.default_rng(7)
    noisy = replace(curve, bar=[h + float(rng.normal(0, 2.0)) for h in curve.bar if h is not None])
    clean, found = _detect(curve, "snatch"), _detect(noisy, "snatch")
    for name, event in clean.events.items():
        moved = abs(_ms(curve, event) - _ms(noisy, found.events[name]))
        assert moved <= max(found.events[name].sigma_ms or 0.0, 2 * FRAME_MS) + 1, name


def test_no_bar_makes_every_event_absent() -> None:
    curve = _curve(SNATCH_V)
    found = _detect(replace(curve, bar=[None] * len(curve.t_ms)), "snatch")
    assert all(e.status == "absent" for e in found.events.values())
    assert found.phases == []
    assert found.result == ("unknown", "rules")


def test_no_bands_makes_every_gated_event_absent() -> None:
    found = pe.detect(_curve(SNATCH_V).signals(), "snatch", T, None)
    assert found.events["liftoff"].status == "found"
    assert found.events["fullExtension"].status == "found"
    assert found.events["catch"].status == "absent"


def test_implausible_durations_are_flagged_not_moved() -> None:
    curve = _curve(SNATCH_V)
    found = _detect(curve, "snatch", plausible={"first_pull": (0.01, 0.02)})
    assert "implausible_duration" in found.flags
    assert found.events == _detect(curve, "snatch").events


def test_each_event_reports_the_bar_height_there() -> None:
    curve = _curve(SNATCH_V)
    found = _detect(curve, "snatch")
    stand = found.events["stand"]
    assert stand.frame is not None and stand.bar_mm is not None
    assert stand.bar_mm == pytest.approx(curve.bar[stand.frame], abs=5.0)
    assert found.events["liftoff"].bar_mm == pytest.approx(225.0, abs=5.0)


# --- joint pairing -----------------------------------------------------------------------------


def test_pair_height_weights_by_inverse_variance_and_drops_uncertain() -> None:
    left = pe.Side(height_mm=[100.0, 100.0, None], sigma_mm=[10.0, 10.0, None],
                   flags=[[], ["uncertain"], []])  # fmt: skip
    right = pe.Side(height_mm=[200.0, 200.0, 300.0], sigma_mm=[20.0, 20.0, 80.0],
                    flags=[[], [], []])  # fmt: skip
    paired = pe.pair_height(left, right, sigma_max_mm=50.0)

    assert paired[0] == pytest.approx((100 / 100 + 200 / 400) / (1 / 100 + 1 / 400))
    assert paired[1] == pytest.approx(200.0)
    assert paired[2] is None  # the only side left is above sigma_max
