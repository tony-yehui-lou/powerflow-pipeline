"""Composes a `Detector2D` with S3/S4's own depth and intrinsics into a full `PoseModel`.

The 2D detector (e.g. `mediapipe_pose_detector.MediaPipePoseDetector`) is responsible only for
pixel-space joints; this module does the metric lift (`pose_geometry.py`) using the LiDAR depth
S3 Retilt and S4 Crop already computed, never a learned monocular scale estimate -- see
docs/specs/preprocessing/S5-pose-model-recommendation.md for why.
"""

from __future__ import annotations

from pathlib import Path

from powerflow_pipeline.data.common.models import JointId, JointSeries
from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.pose_geometry import lift_pixel_to_floor_frame
from powerflow_pipeline.data.preprocess.pose_model import Detector2D
from powerflow_pipeline.data.preprocess.retilt import depth_intrinsics
from powerflow_pipeline.data.preprocess.tasks.ingest import frame_paths, read_frame


class DepthBackedPoseModel:
    """A `PoseModel` (structurally, via `pose_model.PoseModel`) using real depth for scale."""

    def __init__(self, detector: Detector2D, *, patch_radius: int = 2) -> None:
        self._detector = detector
        self._patch_radius = patch_radius

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
        pixel_series = self._detector.detect(rgb_path, n_frames)
        k_d = depth_intrinsics(intrinsics, rgb_size, depth_size)

        depth_paths = frame_paths(depth_dir)
        confidence_paths = frame_paths(confidence_dir)

        positions: dict[JointId, list[tuple[float, float, float] | None]] = {
            joint: [None] * n_frames for joint in pixel_series
        }
        confidences: dict[JointId, list[float]] = {
            joint: list(series.confidence) for joint, series in pixel_series.items()
        }
        pixels: dict[JointId, list[tuple[int, int] | None]] = {
            joint: list(series.pixel_position) for joint, series in pixel_series.items()
        }

        for frame_index in range(n_frames):
            depth_frame = read_frame(depth_paths[frame_index])
            confidence_frame = read_frame(confidence_paths[frame_index])

            for joint, pixel_row in pixels.items():
                pixel = pixel_row[frame_index]
                if pixel is None:
                    continue
                col, row = pixel  # PixelPair is (x, y): col, row
                point = lift_pixel_to_floor_frame(
                    row,
                    col,
                    depth_frame,
                    confidence_frame,
                    rgb_size,
                    depth_size,
                    k_d,
                    floor_offset_m,
                    radius=self._patch_radius,
                )
                if point is None:
                    # `JointSeries` requires position and pixelPosition to drop out
                    # together, and a zero confidence wherever position is None -- a
                    # detected-but-depth-unusable joint (occlusion, out of LiDAR range) is
                    # indistinguishable, in this schema, from a joint the detector never
                    # placed at all.
                    pixels[joint][frame_index] = None
                    confidences[joint][frame_index] = 0.0
                else:
                    positions[joint][frame_index] = point

        return {
            joint: JointSeries(
                position=tuple(positions[joint]),
                pixel_position=tuple(pixels[joint]),
                confidence=tuple(confidences[joint]),
            )
            for joint in pixel_series
        }
