"""Run configuration for the preprocess pipeline (S0 ingest to S4 crop)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field

from powerflow_pipeline.data.common.run_options import RunOptions


class RotationDirection(StrEnum):
    """The 90 degree rotation that repairs the capture app's landscape output."""

    CW = "cw"
    CCW = "ccw"


class PreprocessConfig(RunOptions):
    """Settings for one preprocess run.

    `rotation` defaults to CW: the direction cannot be derived from any metadata
    (`rgb.mp4` carries no rotation tag), so it was fixed once by looking at a frame.

    `only` narrows the run because a capture day is hours of re-encoding, and correcting a
    floor annotation means re-running one capture at a time. A glob must keep a two-camera
    session whole -- excluding its `side` member leaves the group with no lift window to cut
    to, and the group is rejected exactly as it would be if the camera were missing.
    """

    raw_root: Path
    record_root: Path
    cut_root: Path
    retilt_root: Path
    crop_root: Path
    output_root: Path
    rotation: RotationDirection = RotationDirection.CW
    frame_count_tolerance: int = Field(default=1, ge=0)
    rgb_crf: int = Field(default=16, ge=0, le=51)
    require_stopwatch_attestation: bool = False

    # S3 Retilt tunables (docs/specs/preprocessing/4-retilt.md §3/§7,
    # docs/superpowers/specs/2026-10-21-retilt-gravity-floor-design.md).
    retilt_sample_stride: int = Field(default=10, ge=1)
    retilt_max_sampled_frames: int = Field(default=32, ge=1)
    retilt_max_pooled_points: int = Field(default=200_000, ge=1)
    retilt_min_floor_points: int = Field(default=500, ge=1)
    retilt_max_plane_rms_m: float = Field(default=0.02, gt=0)
    # Rejects since the floor is no longer chosen by eye; 39 real captures over three days
    # measured at most 4.0 deg once the gravity axis convention was corrected.
    retilt_gravity_tolerance_deg: float = Field(default=6.0, gt=0)
    retilt_max_tilt_deg: float = Field(default=45.0, gt=0)
    retilt_max_roll_deg: float = Field(default=45.0, gt=0)
    retilt_max_translation_m: float = Field(default=0.05, gt=0)
    # Floor search: a height histogram along gravity, then a depth-refined plane. Real
    # floors held 41-94% of confident points; one peak per capture on every day measured.
    retilt_floor_bin_m: float = Field(default=0.01, gt=0)
    retilt_floor_min_peak_fraction: float = Field(default=0.15, gt=0, le=1)
    retilt_floor_seed_band_m: float = Field(default=0.03, gt=0)
    retilt_floor_inlier_band_m: float = Field(default=0.02, gt=0)
    retilt_floor_refine_iterations: int = Field(default=3, ge=1)
    retilt_floor_ambiguity_m: float = Field(default=0.30, gt=0)
    # Day-level plane agreement, warn-only. Both defaults sit outside the spread measured
    # across 27 good captures of one static-rig day (1.6 deg, 0.07 m) and well inside the
    # failure that motivated the check (44 deg, 0.4 m). Proposals pending a second day.
    retilt_group_tilt_tolerance_deg: float = Field(default=3.0, gt=0)
    retilt_group_height_tolerance_m: float = Field(default=0.10, gt=0)

    # A lift window shorter than this warns; it never rejects. A snatch really can take
    # under two seconds -- the shortest real windows observed are 1.64 s and 1.75 s.
    cut_min_window_s: float = Field(default=1.0, gt=0)

    # S4 Crop tunables (docs/specs/preprocessing/6-cropping.md §Configuration).
    crop_max_residual_translation_m: float = Field(default=0.025, gt=0)
    crop_depth_guard_quantile: float = Field(default=0.05, gt=0, lt=1)
    crop_safety_px: int = Field(default=8, ge=0)
    crop_max_crop_fraction: float = Field(default=0.25, gt=0, lt=1)
    crop_depth_sample_stride: int = Field(default=10, ge=1)
    crop_max_sampled_depth_frames: int = Field(default=32, ge=1)

    @property
    def stage_roots(self) -> tuple[Path, ...]:
        """Every tree this run publishes into, in stage order.

        `metadata.yaml` is written to each: a stage's output says what the pixels are, never
        which lift they came from, and a consumer should not have to reach back into an
        earlier stage's tree to find out. A new stage adds its root here and inherits the copy.

        The detect and analyze trees deliberately do not: their stages don't run for every
        camera in every run, so they would leave orphan `metadata.yaml` files with no output
        beside them whenever a stage is skipped.
        """

        return (self.record_root, self.cut_root, self.retilt_root, self.crop_root, self.output_root)
