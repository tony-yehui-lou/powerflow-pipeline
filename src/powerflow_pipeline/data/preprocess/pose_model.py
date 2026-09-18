"""The joint-detection model boundary for S5 Pose (GitHub issue #118, "Find a Suitable Model").

`PoseModel.predict` is what `tasks/pose.py` calls through; `Detector2D.detect` is the smaller
seam a concrete `PoseModel` (`depth_pose_model.DepthBackedPoseModel`) composes with a 2D
keypoint detector (`mediapipe_pose_detector.MediaPipePoseDetector`). Splitting them keeps the
depth back-projection + floor-frame lift (`pose_geometry.py`) independent of which 2D detector
produced the pixels, so either half can be swapped or unit-tested without the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from powerflow_pipeline.data.common.models import JointId, JointSeries
from powerflow_pipeline.data.preprocess.models import Intrinsics

PixelPair = tuple[int, int]


@dataclass(frozen=True)
class JointPixelSeries:
    """One joint's 2D trajectory only -- a detector's raw output, before any metric lift.

    Mirrors `JointSeries`'s `pixel_position`/`confidence` fields exactly (same drop-out
    convention: `None` pixel iff zero confidence) but omits `position`, which no 2D-only
    detector can supply.
    """

    pixel_position: tuple[PixelPair | None, ...]
    confidence: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.pixel_position)


@runtime_checkable
class Detector2D(Protocol):
    """Something that locates the 15 skeleton joints in pixel space, per frame of one video."""

    def detect(self, rgb_path: Path, n_frames: int) -> dict[JointId, JointPixelSeries]:
        """One `JointPixelSeries` per skeleton joint, each `n_frames` long, for `rgb_path`."""
        ...


@runtime_checkable
class PoseModel(Protocol):
    """Something that turns one camera's cropped RGB into a per-joint pose time series.

    The model owns pixel-space detection and the lift to `JointSeries.position` (floor-frame
    metres) alike -- S5 only assembles whatever it returns into a `PoseDocument` and publishes
    it, unchanged. `depth_dir`/`confidence_dir`/`intrinsics`/`rgb_size`/`depth_size`/
    `floor_offset_m` are S4 Crop's and S3 Retilt's own outputs, already computed and sitting
    on disk next to `rgb_path` -- `tasks/pose.py` reads them once and passes them through so a
    depth-backed implementation doesn't need to re-derive or re-read anything the pipeline
    already has. A `PoseModel` that ignores them (a pure monocular estimator) is free to do so.
    """

    def predict(
        self,
        rgb_path: Path,
        n_frames: int,
        *,
        depth_dir: Path,
        confidence_dir: Path,
        intrinsics: Intrinsics,
        rgb_size: tuple[int, int],
        depth_size: tuple[int, int],
        floor_offset_m: float,
    ) -> dict[JointId, JointSeries]:
        """One `JointSeries` per skeleton joint, each `n_frames` long, for `rgb_path`."""
        ...
