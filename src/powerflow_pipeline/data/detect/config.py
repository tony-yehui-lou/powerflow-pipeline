"""Run configuration for the detect pipeline (S5 barbell, S6 pose, S7 lift)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field

from powerflow_pipeline.data.common.run_options import RunOptions


class PoseDetector(StrEnum):
    """Which 2D keypoint model S6 Pose runs (S6-rtmpose-migration.md §7)."""

    MEDIAPIPE = "mediapipe"
    RTMPOSE = "rtmpose"


class RTMPoseVariant(StrEnum):
    """RTMPose speed/accuracy tier; each pins its own detector and pose checkpoints."""

    LIGHTWEIGHT = "lightweight"
    BALANCED = "balanced"
    PERFORMANCE = "performance"


class BarbellDevice(StrEnum):
    """Where S5 runs its detector; `auto` picks mps, then cuda, then cpu."""

    AUTO = "auto"
    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


class DetectConfig(RunOptions):
    """Settings for one detect run.

    Every camera is read from what preprocess published: S0's record, S1's timing, S3's floor
    and S4's cropped streams (`common/camera_records.py`). Each stage runs only when asked:
    S5 with a checkpoint, S6 with a detector, S7 with a `lift_root`.
    """

    record_root: Path
    cut_root: Path
    retilt_root: Path
    crop_root: Path
    # S6 Pose's tree: written when S6 runs (a detector is given), read by S7 either way.
    pose_root: Path | None = None
    # S7 Lift is off by default (phase-split design): it back-projects S6's pixels through S4's
    # LiDAR depth, and nothing downstream reads it -- S8 solves its joints from bone lengths.
    # It reads S6's published document, so it can lift an earlier run's detections without
    # re-detecting. Setting it without `pose_root` is rejected: there would be nothing to lift.
    lift_root: Path | None = None
    # S5 Barbell's tree: `barbell.json` per capture under a root mirroring the raw layout --
    # written when S5 runs, read by S6 either way. When set, S6 corrects both wrists against the
    # bar the hands hold (`wrist_refine.py`): overhead the far wrist is occluded and the
    # detector's guess jumps hundreds of pixels. Unset leaves S6's wrists exactly as detected.
    barbell_root: Path | None = None
    # A promoted S5 checkpoint (`barbell_model.read_card`). Set, S5 runs and publishes into
    # `barbell_root`; unset, S5 doesn't run and `barbell_root` is read as already published.
    barbell_checkpoint: Path | None = None
    barbell_device: BarbellDevice = BarbellDevice.AUTO
    # S5 records candidates down to this score, so a consumer never loses evidence it can't
    # recover; false caps on a bystander scored 0.31-0.33 and are recorded too.
    barbell_record_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    # Consumers (S6, S8) ignore detections below this. Real ones score >= 0.95 almost everywhere.
    barbell_min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    barbell_batch_size: int = Field(default=4, ge=1)
    # Also publish `overlay.mp4`, every frame with its boxes, for review. Re-encodes the video.
    barbell_render_overlay: bool = False
    # S5 warns, never rejects, when fewer frames than this carry a cap / a collar. 45kg1 has a
    # cap on about a quarter of its frames and is still valid S5 output.
    barbell_warn_cap_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    barbell_warn_collar_coverage: float = Field(default=0.9, ge=0.0, le=1.0)
    # Reuse whatever S6 already published in `pose_root` instead of re-running detection.
    # `pose_root` names the S6 tree either way -- written when S6 runs, read when it doesn't --
    # so lifting an earlier run's detections is `skip_detect` plus a `lift_root`.
    skip_detect: bool = False
    # S6 Pose's 2D detector. RTMPose is the default (decided 2026-10-22, ahead of the
    # measurements in docs/specs/preprocessing/S6-rtmpose-migration.md §6, which are still to
    # be run). MediaPipe stays installed and selectable so that comparison can still be made.
    pose_detector: PoseDetector = PoseDetector.RTMPOSE
    # Half-width of the depth patch S7 medians around each joint pixel, in depth pixels. 2
    # gives the 5x5 box S6 used before the split. One depth pixel spans ~7.5 RGB pixels, so
    # this is already wider than a forearm -- see S6-bone-constrained-reconstruction.md.
    lift_patch_radius: int = Field(default=2, ge=0)
    rtmpose_variant: RTMPoseVariant = RTMPoseVariant.BALANCED
    # Run the person detector every Nth frame and track in between (§3). 1 re-detects every
    # frame -- the safe starting point; raising it is faster but track drift over a fast lift
    # is unmeasured.
    rtmpose_det_frequency: int = Field(default=1, ge=1)
    # Where a clavicle sits along the neck -> shoulder line (§2); 0.5 is the collarbone midpoint.
    rtmpose_clavicle_shoulder_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    # Whether S6 may read depth to pick *which person* is the athlete (§4 rule 1). This never
    # produces a coordinate -- it returns an index -- but it is the one place S6 opens a depth
    # frame at all. False makes S6 purely pixel-space, falling back to largest-bounding-box
    # selection, which is worse only on captures that actually contain bystanders.
    rtmpose_subject_depth: bool = True
