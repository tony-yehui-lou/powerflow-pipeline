"""S8 · The lifter's body measurements, and the length each bone of the solve is given.

`docs/specs/preprocessing/S8-joint-hva-draft3.md` §2. A bone is a keypoint pair with a target
length and a trust; its length comes from the first source that exists:

1. `measured` -- the `athlete:` field for that bone, at the measured sigma;
2. `population` -- a ratio of `height_in_cm`, at the wider population sigma;
3. `none` -- the bone is left out of the solve.


A tape measurement runs between palpated landmarks, not between the detector's keypoints, so a
measurement is always a prior with a sigma, never a hard constraint.

The pipeline does not read `athlete:` anywhere else yet (S0 writes an empty `AthleteMeta`), so
this reads the capture's raw `metadata.yaml` directly, the way `barbell_input.read_bar_type` does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import yaml

from powerflow_pipeline.data.common.models import JointId

LengthSource = Literal["measured", "population", "none"]
# A bone end is one joint, or the midpoint of two (the head bone ends at the shoulder midpoint).
Endpoint = JointId | tuple[JointId, JointId]

MEASUREMENT_FIELDS: Final = (
    "height_in_cm",
    "tibia_in_cm",
    "femur_in_cm",
    "torso_in_cm",
    "shoulder_width_in_cm",
    "upper_arm_in_cm",
    "forearm_in_cm",
    "hip_width_in_cm",
    "ankle_height_in_cm",
    "armspan_in_cm",
)

# A measurement outside this multiple of its population prior is a typo or a different unit,
# not a body: rejected with a warning and treated as missing.
PLAUSIBLE_RATIO: Final = (0.5, 2.0)

# Hand length as a ratio of stature (Drillis-Contini), for the armspan check only.
HAND_RATIO: Final = 0.108
# |implied - measured armspan| beyond this warns; armspan never enters the solve.
ARMSPAN_TOLERANCE_MM: Final = 50.0

ANKLE_HEIGHT: Final = "ankleHeight"


@dataclass(frozen=True)
class BoneSpec:
    """One length the solve keeps: between `a` and `b`, or (`b is None`) an anchor height."""

    bone_id: str
    a: Endpoint
    b: Endpoint | None
    field: str | None
    ratio: float  # population prior, as a multiple of stature
    offset_mm: float  # added to the population prior
    measured_sigma_mm: float
    population_sigma_mm: float


def _pair(
    side_id: str, a: str, b: str, fld: str, ratio: float, measured: float, population: float
) -> tuple[BoneSpec, BoneSpec]:
    return tuple(  # type: ignore[return-value]
        BoneSpec(
            bone_id=f"{side}{side_id}",
            a=f"{side}{a}",  # type: ignore[arg-type]
            b=f"{side}{b}",  # type: ignore[arg-type]
            field=fld,
            ratio=ratio,
            offset_mm=0.0,
            measured_sigma_mm=measured,
            population_sigma_mm=population,
        )
        for side in ("left", "right")
    )


# Segment ratios from Drillis-Contini and Winter; sigmas from draft 3 §2.2.
BONES: Final[tuple[BoneSpec, ...]] = (
    *_pair("Tibia", "Ankle", "Knee", "tibia_in_cm", 0.246, 20.0, 30.0),
    *_pair("Femur", "Knee", "Hip", "femur_in_cm", 0.245, 20.0, 30.0),
    *_pair("Torso", "Hip", "Shoulder", "torso_in_cm", 0.288, 25.0, 40.0),
    *_pair("UpperArm", "Shoulder", "Elbow", "upper_arm_in_cm", 0.186, 20.0, 40.0),
    *_pair("Forearm", "Elbow", "Wrist", "forearm_in_cm", 0.146, 20.0, 40.0),
    # Biacromial breadth less the keypoint inset; the inset is an assumption.
    BoneSpec("shoulderWidth", "leftShoulder", "rightShoulder", "shoulder_width_in_cm",
             0.259, -40.0, 30.0, 40.0),
    BoneSpec("hipWidth", "leftHip", "rightHip", "hip_width_in_cm", 0.10, 0.0, 25.0, 30.0),
    BoneSpec("headNeck", "head", ("leftShoulder", "rightShoulder"), None, 0.12, 0.0, 50.0, 50.0),
    # Not a bone: a grounded ankle's height above the floor (draft 3 §3.2).
    BoneSpec(ANKLE_HEIGHT, "leftAnkle", None, "ankle_height_in_cm", 0.039, 0.0, 10.0, 15.0),
)  # fmt: skip
BONES_BY_ID: Final[dict[str, BoneSpec]] = {bone.bone_id: bone for bone in BONES}


@dataclass(frozen=True)
class Athlete:
    """The numeric `athlete:` fields that parsed, in centimetres, and why any did not."""

    values: dict[str, float]
    warnings: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return bool(self.values)


@dataclass(frozen=True)
class Length:
    """One bone's length as the solve uses it, with every candidate it was chosen from."""

    bone_id: str
    field: str | None
    measured_mm: float | None
    population_mm: float | None
    used_mm: float | None
    source: LengthSource
    sigma_mm: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArmspanCheck:
    measured_mm: float
    implied_mm: float
    warn: bool


def _number(value: object) -> float | None:
    """A YAML leaf as a number, or None for anything else -- placeholders included."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def read_athlete(metadata_path: Path) -> Athlete:
    """Every numeric measurement in the capture's `athlete:` block, parsed leaf by leaf.

    A template placeholder (`float`, `string`, `ISO 8601 date`) is missing, never a number; a
    non-positive number is rejected with a warning. A missing file or block reads as nothing.
    """

    if not metadata_path.is_file():
        return Athlete(values={})
    loaded = yaml.safe_load(metadata_path.read_text())
    block = loaded.get("athlete") if isinstance(loaded, dict) else None
    if not isinstance(block, dict):
        return Athlete(values={})

    values: dict[str, float] = {}
    warnings: list[str] = []
    for name in MEASUREMENT_FIELDS:
        number = _number(block.get(name))
        if number is None:
            continue
        if number <= 0:
            warnings.append(f"athlete.{name} = {number} is not a length; ignored")
            continue
        values[name] = number
    return Athlete(values=values, warnings=tuple(warnings))


def resolve_lengths(athlete: Athlete) -> dict[str, Length]:
    """Every bone's pass-1 length: `measured` if present and plausible, else `population`."""

    height_mm = athlete.values.get("height_in_cm")
    height_mm = None if height_mm is None else height_mm * 10.0
    lengths: dict[str, Length] = {}
    for bone in BONES:
        population = None if height_mm is None else bone.ratio * height_mm + bone.offset_mm
        measured = None if bone.field is None else athlete.values.get(bone.field)
        measured = None if measured is None else measured * 10.0
        warnings: list[str] = []
        if measured is not None and population is not None:
            lo, hi = (r * population for r in PLAUSIBLE_RATIO)
            if not lo <= measured <= hi:
                warnings.append(
                    f"athlete.{bone.field} = {measured / 10.0} cm is outside "
                    f"{lo / 10.0:.1f}-{hi / 10.0:.1f} cm for this stature; ignored"
                )
                measured = None

        if measured is not None:
            source: LengthSource = "measured"
            used, sigma = measured, bone.measured_sigma_mm
        elif population is not None:
            source, used, sigma = "population", population, bone.population_sigma_mm
        else:
            source, used, sigma = "none", None, bone.population_sigma_mm
        lengths[bone.bone_id] = Length(
            bone_id=bone.bone_id,
            field=bone.field,
            measured_mm=measured,
            population_mm=population,
            used_mm=used,
            source=source,
            sigma_mm=sigma,
            warnings=tuple(warnings),
        )
    return lengths


def armspan_check(
    used_mm: Mapping[str, float | None], *, height_mm: float | None, armspan_mm: float | None
) -> ArmspanCheck | None:
    """Shoulder width + 2 x (upper arm + forearm + hand) against the measured armspan.

    Uses the left arm's lengths as the solve used them. None when any term is unknown.
    """

    terms = [used_mm.get(b) for b in ("shoulderWidth", "leftUpperArm", "leftForearm")]
    if height_mm is None or armspan_mm is None or any(t is None for t in terms):
        return None
    width, upper, fore = (float(t) for t in terms)  # type: ignore[arg-type]
    implied = width + 2.0 * (upper + fore + HAND_RATIO * height_mm)
    return ArmspanCheck(
        measured_mm=armspan_mm,
        implied_mm=implied,
        warn=abs(implied - armspan_mm) > ARMSPAN_TOLERANCE_MM,
    )
