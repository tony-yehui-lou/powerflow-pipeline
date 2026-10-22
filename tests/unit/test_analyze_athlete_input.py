"""S8's reading of the `athlete:` block, and how each bone's length is chosen from it.

The values here are invented for the tests; no lifter's measurements belong in code or spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from powerflow_pipeline.data.analyze import athlete_input as ai
from powerflow_pipeline.data.common.capture_models import AthleteMeta


def _write(tmp_path: Path, athlete: str) -> Path:
    path = tmp_path / "metadata.yaml"
    path.write_text("---\nlift:\n  barbell_type: male IWF\nathlete:\n" + athlete + "...\n")
    return path


FILLED = (
    "  name: string\n"
    "  measureDate: ISO 8601 date\n"
    "  height_in_cm: 170.0\n"
    "  tibia_in_cm: 40.0\n"
    "  femur_in_cm: 41.0\n"
    "  torso_in_cm: 50.0\n"
    "  armspan_in_cm: 172.0\n"
    "  shoulder_width_in_cm: 40.0\n"
    "  upper_arm_in_cm: 31.0\n"
    "  forearm_in_cm: 25.0\n"
    "  hip_width_in_cm: 32.0\n"
    "  ankle_height_in_cm: 9.0\n"
)


def test_athlete_meta_carries_every_field_s8_reads() -> None:
    fields = set(AthleteMeta.model_fields)
    assert set(ai.MEASUREMENT_FIELDS) <= fields
    assert {"shoulder_width_in_cm", "body_fat_percentage"} <= fields


def test_reads_every_numeric_measurement(tmp_path: Path) -> None:
    athlete = ai.read_athlete(_write(tmp_path, FILLED))

    assert athlete.values["height_in_cm"] == 170.0
    assert athlete.values["upper_arm_in_cm"] == 31.0
    assert athlete.values["ankle_height_in_cm"] == 9.0
    assert athlete.warnings == ()


@pytest.mark.parametrize("placeholder", ["float", "string", "ISO 8601 date", "", "true"])
def test_template_placeholders_are_missing_never_numbers(tmp_path: Path, placeholder: str) -> None:
    path = _write(tmp_path, f"  height_in_cm: 170.0\n  tibia_in_cm: {placeholder}\n")

    athlete = ai.read_athlete(path)

    assert "tibia_in_cm" not in athlete.values
    assert athlete.values["height_in_cm"] == 170.0


def test_a_non_positive_value_is_rejected_with_a_warning(tmp_path: Path) -> None:
    athlete = ai.read_athlete(_write(tmp_path, "  height_in_cm: 170.0\n  femur_in_cm: -3\n"))

    assert "femur_in_cm" not in athlete.values
    assert any("femur_in_cm" in w for w in athlete.warnings)


def test_missing_file_or_block_reads_as_nothing(tmp_path: Path) -> None:
    assert ai.read_athlete(tmp_path / "absent.yaml").values == {}
    path = tmp_path / "metadata.yaml"
    path.write_text("lift:\n  barbell_type: male IWF\n")
    assert ai.read_athlete(path).values == {}


def test_measured_bones_use_the_measurement_at_the_measured_sigma(tmp_path: Path) -> None:
    lengths = ai.resolve_lengths(ai.read_athlete(_write(tmp_path, FILLED)))

    tibia = lengths["leftTibia"]
    assert tibia.source == "measured"
    assert tibia.used_mm == pytest.approx(400.0)
    assert tibia.sigma_mm == ai.BONES_BY_ID["leftTibia"].measured_sigma_mm
    assert tibia.population_mm == pytest.approx(0.246 * 1700.0)
    assert lengths["rightTibia"].used_mm == pytest.approx(400.0)
    assert lengths["hipWidth"].used_mm == pytest.approx(320.0)
    assert lengths[ai.ANKLE_HEIGHT].used_mm == pytest.approx(90.0)


def test_missing_measurements_fall_back_to_population_at_the_wider_sigma(tmp_path: Path) -> None:
    lengths = ai.resolve_lengths(ai.read_athlete(_write(tmp_path, "  height_in_cm: 180.0\n")))

    arm = lengths["leftUpperArm"]
    assert arm.source == "population"
    assert arm.measured_mm is None
    assert arm.used_mm == pytest.approx(0.186 * 1800.0)
    assert arm.sigma_mm == ai.BONES_BY_ID["leftUpperArm"].population_sigma_mm
    # The head bone has no field: always population.
    assert lengths["headNeck"].source == "population"


def test_without_height_an_unmeasured_bone_has_no_length(tmp_path: Path) -> None:
    lengths = ai.resolve_lengths(ai.read_athlete(_write(tmp_path, "  tibia_in_cm: 40.0\n")))

    assert lengths["leftTibia"].source == "measured"
    assert lengths["leftFemur"].source == "none"
    assert lengths["leftFemur"].used_mm is None


def test_an_implausible_measurement_is_treated_as_missing(tmp_path: Path) -> None:
    athlete = ai.read_athlete(_write(tmp_path, "  height_in_cm: 180.0\n  tibia_in_cm: 4.0\n"))

    lengths = ai.resolve_lengths(athlete)

    assert lengths["leftTibia"].source == "population"
    assert any("tibia_in_cm" in w for w in lengths["leftTibia"].warnings)


def test_has_any_measurement(tmp_path: Path) -> None:
    assert ai.read_athlete(_write(tmp_path, FILLED)).usable
    assert not ai.read_athlete(_write(tmp_path, "  name: string\n")).usable


def test_armspan_check_flags_an_inconsistent_athlete() -> None:
    used = {"shoulderWidth": 400.0, "leftUpperArm": 310.0, "leftForearm": 250.0}
    consistent = 400.0 + 2 * (310.0 + 250.0 + 0.108 * 1700.0)

    ok = ai.armspan_check(used, height_mm=1700.0, armspan_mm=consistent)
    bad = ai.armspan_check(used, height_mm=1700.0, armspan_mm=consistent - 120.0)

    assert ok is not None and not ok.warn
    assert ok.implied_mm == pytest.approx(consistent)
    assert bad is not None and bad.warn
    assert ai.armspan_check(used, height_mm=None, armspan_mm=consistent) is None
