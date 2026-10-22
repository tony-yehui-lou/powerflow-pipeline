"""Run configuration for the analyze pipeline (S8 kinematics, S9 phase)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from powerflow_pipeline.data.common.run_options import RunOptions


class AnalyzeStage(StrEnum):
    """The stages `--stages` can select; each still needs its output root to run."""

    KINEMATICS = "s8"
    PHASE = "s9"


class CleanAndJerkThresholds(BaseModel):
    """S9 rules only the clean & jerk uses (S9-phase-intervals-draft8.md §4.1)."""

    # How long the bar must be still, in the rack band, for the rack stand: E4's search start.
    rack_hold_ms: float = Field(default=300.0, gt=0)
    # Drop below the rack-stand height that counts as the dip; smaller re-grip bounces don't.
    dip_mm: float = Field(default=30.0, gt=0)


class BandTolerances(BaseModel):
    """Each S9 height band's tolerance as (below, above) fractions of its centre (§2.2)."""

    knee: tuple[float, float] = (0.20, 0.20)
    hip: tuple[float, float] = (0.20, 0.20)
    rack: tuple[float, float] = (0.15, 0.10)
    overhead: tuple[float, float] = (0.10, 0.10)
    # Without S6's half grip the arms are assumed vertical, and the overhead band widens below.
    overhead_below_grip_assumed: float = Field(default=0.25, ge=0, lt=1)


class PhaseThresholds(BaseModel):
    """S9 Phase starting thresholds (§4.1), to be tuned against the labels (§8.4).

    Holds are durations: each capture converts them to frames with its own `tMs` cadence.
    """

    lift_mm: float = Field(default=20.0, gt=0)
    lift_hold_ms: float = Field(default=83.0, gt=0)
    v_rest_mm_s: float = Field(default=50.0, gt=0)
    # The rack stand's still speed. The final stand is height alone in draft 7 (§1.2), because
    # 30kg1 and 40kg1 never hold the bar still overhead (§8.1).
    v_still_mm_s: float = Field(default=150.0, gt=0)
    # A knee height with a larger sigma is not used, and E1 falls back to the knee band.
    joint_sigma_max_mm: float = Field(default=50.0, gt=0)
    # Bar gaps up to this are bridged for velocity only; longer ones make an event there absent.
    max_gap_ms: float = Field(default=100.0, ge=0)
    # Zero-phase low-pass on bar height before differencing (Open question 7: 4-10 Hz).
    filter_cutoff_hz: float = Field(default=8.0, gt=0)
    # The bar's rest height is the median over this many opening frames, as S8's floor anchor.
    rest_frames: int = Field(default=60, ge=1)
    bands: BandTolerances = Field(default_factory=BandTolerances)
    clean_and_jerk: CleanAndJerkThresholds = Field(default_factory=CleanAndJerkThresholds)


class AnalyzeConfig(RunOptions):
    """Settings for one analyze run.

    Every camera is read from what preprocess published (`common/camera_records.py`), and
    every input from what detect published. A stage runs when it is in `stages` and its output
    root is set, so `stages=(PHASE,)` re-runs S9 against an S8 tree already in
    `kinematics_root` without solving anything again.
    """

    record_root: Path
    cut_root: Path
    retilt_root: Path
    crop_root: Path
    # S6's published tree: the 2D pose S8 solves, and the grip S8 and S9 read.
    pose_root: Path
    # S5's published tree. Unset, S8 is anchored by the floor alone; S9 needs it.
    barbell_root: Path | None = None
    # S8 Kinematics' tree: written when S8 runs, read by S9 either way.
    kinematics_root: Path | None = None
    # S9 Phase's tree. It reads S8's `joint_hva.json` (bone lengths, floor, knees), S5's bar and
    # S6's grip, so it needs `kinematics_root` and `barbell_root`; setting it without them is
    # rejected.
    phase_root: Path | None = None
    stages: tuple[AnalyzeStage, ...] = (AnalyzeStage.KINEMATICS, AnalyzeStage.PHASE)
    phase: PhaseThresholds = Field(default_factory=PhaseThresholds)
    # S8 ignores S5 detections below this, as S6 does. Real ones score >= 0.95 almost everywhere.
    barbell_min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # Where S6 put each clavicle along the neck -> shoulder line; S8 derives the clavicles the
    # same way, so this must match the detect run that produced `pose_root`.
    rtmpose_clavicle_shoulder_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def runs_kinematics(self) -> bool:
        return AnalyzeStage.KINEMATICS in self.stages and self.kinematics_root is not None

    @property
    def runs_phase(self) -> bool:
        return AnalyzeStage.PHASE in self.stages and self.phase_root is not None
