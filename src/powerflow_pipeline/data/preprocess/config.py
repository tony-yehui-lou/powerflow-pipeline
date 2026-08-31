"""Run configuration for the preprocess pipeline (S0 ingest, S1 cut, S2 orient, S3 retilt)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class RotationDirection(StrEnum):
    """The 90 degree rotation that repairs the capture app's landscape output."""

    CW = "cw"
    CCW = "ccw"


class PreprocessConfig(BaseModel):
    """Settings for one preprocess run.

    `rotation` defaults to CW: the direction cannot be derived from any metadata
    (`rgb.mp4` carries no rotation tag), so it was fixed once by looking at a frame.
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
    overwrite: bool = False
    dry_run: bool = False

    # S3 Retilt tunables (docs/specs/preprocessing/4-retilt.md §3/§7).
    retilt_sample_stride: int = Field(default=10, ge=1)
    retilt_max_sampled_frames: int = Field(default=32, ge=1)
    retilt_max_pooled_points: int = Field(default=200_000, ge=1)
    retilt_min_floor_points: int = Field(default=500, ge=1)
    retilt_max_plane_rms_m: float = Field(default=0.02, gt=0)
    retilt_gravity_tolerance_deg: float = Field(default=5.0, gt=0)
    retilt_max_tilt_deg: float = Field(default=45.0, gt=0)
    retilt_max_roll_deg: float = Field(default=45.0, gt=0)
    retilt_max_translation_m: float = Field(default=0.05, gt=0)
    retilt_conf2_area_fraction: float = Field(default=1 / 6, gt=0, le=1)

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
        """

        return (self.record_root, self.cut_root, self.retilt_root, self.crop_root, self.output_root)
