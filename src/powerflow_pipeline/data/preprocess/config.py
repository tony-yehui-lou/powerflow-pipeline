"""Run configuration for the preprocess pipeline (S0 ingest, S1 cut, S2 orient, S3 retilt)."""

from __future__ import annotations

from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel, Field


class PoseDetector(StrEnum):
    """Which 2D keypoint model S5 Pose runs (S5-rtmpose-migration.md §7)."""

    MEDIAPIPE = "mediapipe"
    RTMPOSE = "rtmpose"


class RTMPoseVariant(StrEnum):
    """RTMPose speed/accuracy tier; each pins its own detector and pose checkpoints."""

    LIGHTWEIGHT = "lightweight"
    BALANCED = "balanced"
    PERFORMANCE = "performance"


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
    # S5 Pose is optional -- unset until a `PoseModel` (issue #118) is wired in at the flow
    # call site. `preprocess()` skips the stage entirely when `pose_model` isn't given, so
    # this stays unset for every existing caller.
    pose_root: Path | None = None
    # S6 Lift is optional in the same way, and independently: it reads S5's published document
    # rather than S5's in-memory result, so a run can detect without lifting (while the depth
    # sampler is being revised) or lift an earlier run's detections without re-detecting.
    # Unset skips the stage; setting it without `pose_root` is rejected, since there would be
    # nothing to lift.
    lift_root: Path | None = None
    # Reuse whatever S5 already published in `pose_root` instead of re-running detection.
    # `pose_root` names the S5 tree either way -- written when S5 runs, read when it doesn't --
    # so lifting an earlier run's detections is `skip_detect` plus a `lift_root`.
    skip_detect: bool = False
    # S5 Pose's 2D detector. MediaPipe stays the default until RTMPose beats it on the
    # measurements in docs/specs/preprocessing/S5-rtmpose-migration.md §6; both stay installed
    # and selectable so the comparison can be run, and so neither becomes load-bearing.
    pose_detector: PoseDetector = PoseDetector.MEDIAPIPE
    # Half-width of the depth patch S6 medians around each joint pixel, in depth pixels. 2
    # gives the 5x5 box S5 used before the split. One depth pixel spans ~7.5 RGB pixels, so
    # this is already wider than a forearm -- see S5-bone-constrained-reconstruction.md.
    lift_patch_radius: int = Field(default=2, ge=0)
    rtmpose_variant: RTMPoseVariant = RTMPoseVariant.BALANCED
    # Run the person detector every Nth frame and track in between (§3). 1 re-detects every
    # frame -- the safe starting point; raising it is faster but track drift over a fast lift
    # is unmeasured.
    rtmpose_det_frequency: int = Field(default=1, ge=1)
    # Where a clavicle sits along the neck -> shoulder line (§2); 0.5 is the collarbone midpoint.
    rtmpose_clavicle_shoulder_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    # Whether S5 may read depth to pick *which person* is the athlete (§4 rule 1). This never
    # produces a coordinate -- it returns an index -- but it is the one place S5 opens a depth
    # frame at all. False makes S5 purely pixel-space, falling back to largest-bounding-box
    # selection, which is worse only on captures that actually contain bystanders.
    rtmpose_subject_depth: bool = True
    # Restrict the run to capture ids matching this glob. A capture day is hours of
    # re-encoding, and correcting a floor annotation means re-running one capture at a
    # time; unset, every discovered capture is processed. A glob must keep a two-camera
    # session whole -- excluding its `side` member leaves the group with no lift window to
    # cut to, and the group is rejected exactly as it would be if the camera were missing.
    only: str | None = None
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

    def selects(self, capture_id: str) -> bool:
        """Whether `--only` admits this capture. Unset admits everything."""

        return self.only is None or fnmatch(capture_id, self.only)

    @property
    def stage_roots(self) -> tuple[Path, ...]:
        """Every tree this run publishes into, in stage order.

        `metadata.yaml` is written to each: a stage's output says what the pixels are, never
        which lift they came from, and a consumer should not have to reach back into an
        earlier stage's tree to find out. A new stage adds its root here and inherits the copy.

        S5 Pose deliberately does not: unlike S0-S4, it doesn't run for every camera in every
        run (see `pose_root`), so it would leave orphan `metadata.yaml` files with no pose
        output beside them whenever it's unset or a camera is skipped.
        """

        return (self.record_root, self.cut_root, self.retilt_root, self.crop_root, self.output_root)
