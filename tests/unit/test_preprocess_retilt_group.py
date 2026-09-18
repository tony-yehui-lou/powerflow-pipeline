"""A capture day's floor fits must agree with each other, or the run says so.

The per-capture gates measure how *tightly* points fit a plane, never whether that plane is
the floor. On real data one candidate rectangle caught a spectator's head and produced a
0.85 cm RMS at -44.9 degrees of tilt -- a beautiful fit to the wrong surface, under the
45-degree gate by a tenth of a degree. For a rig that never moved between captures,
disagreement with the day's median is what catches that.
"""

from __future__ import annotations

import pytest

from powerflow_pipeline.data.preprocess.retilt import assess_plane_consistency

# Three captures from one static tripod, within the spread measured across a real day.
AGREEING = {
    "22 August/Snch/107kgSnch1": (-1.31, -1.24, 1.15),
    "22 August/Snch/110kgSnch1": (-0.63, -0.66, 1.11),
    "22 August/CnJ/48kgCnJ1": (-1.22, -0.89, 1.13),
}
TOLERANCES = {"tilt_tolerance_deg": 3.0, "height_tolerance_m": 0.10}


def test_a_static_rig_day_warns_about_nothing() -> None:
    consistency = assess_plane_consistency(AGREEING, **TOLERANCES)

    assert all(messages == [] for messages in consistency.warnings.values())
    assert consistency.n_captures == 3
    assert consistency.tilt_median_deg == pytest.approx(-1.22)
    assert consistency.height_median_m == pytest.approx(1.13)


def test_the_real_failure_is_flagged_and_the_others_are_not() -> None:
    """The head-occluded fit, verbatim from the measurement that motivated this check."""

    fits = {**AGREEING, "22 August/Snch/55kgSnch1": (-44.89, 37.10, 0.72)}

    consistency = assess_plane_consistency(fits, **TOLERANCES)

    flagged = consistency.warnings["22 August/Snch/55kgSnch1"]
    assert len(flagged) == 3  # tilt, roll and height all disagree
    assert "tilt" in flagged[0] and "median" in flagged[0]
    assert all(consistency.warnings[capture] == [] for capture in AGREEING)


def test_the_median_is_not_dragged_by_one_wrong_fit() -> None:
    """A mean would move toward the outlier and could exonerate it; a median does not."""

    fits = {**AGREEING, "22 August/Snch/50kgSnch1": (15.84, 8.56, 0.10)}

    consistency = assess_plane_consistency(fits, **TOLERANCES)

    assert consistency.tilt_median_deg == pytest.approx(-0.925)  # still among the good fits
    assert consistency.warnings["22 August/Snch/50kgSnch1"] != []


def test_a_height_only_disagreement_is_caught_on_its_own() -> None:
    """A plane parallel to the floor but at the wrong distance has a plausible tilt."""

    fits = {**AGREEING, "22 August/Snch/60kgSnch1": (-1.10, -1.00, 0.55)}

    consistency = assess_plane_consistency(fits, **TOLERANCES)

    (message,) = consistency.warnings["22 August/Snch/60kgSnch1"]
    assert "camera height" in message
    assert "retilt_group_height_tolerance_m" in message


def test_deviations_are_recorded_for_every_capture_not_only_the_flagged_ones() -> None:
    consistency = assess_plane_consistency(AGREEING, **TOLERANCES)

    deviations = consistency.deviations["22 August/Snch/110kgSnch1"]
    assert deviations["tilt_deviation_deg"] == pytest.approx(0.59)
    assert deviations["height_deviation_m"] == pytest.approx(0.02)
