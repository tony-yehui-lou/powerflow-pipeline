"""S6 Lift's core: give S5's pixel-space joints floor-frame metric positions from LiDAR depth.

This is the second half of what `depth_pose_model.DepthBackedPoseModel` used to do in one pass
with detection. Separating them makes the expensive half (detection) and the half still under
active revision (depth sampling) independently re-runnable -- see
docs/specs/preprocessing/S5-bone-constrained-reconstruction.md, which changes only this side.

The per-pixel geometry lives in `pose_geometry.py`; this module is the per-joint, per-frame loop
over it plus the depth/confidence frame reads, and nothing else. It uses real LiDAR depth for
scale, never a learned monocular estimate -- see S5-pose-model-recommendation.md for why.
"""

from __future__ import annotations

from pathlib import Path

from powerflow_pipeline.data.common.models import JointId, JointSeries
from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.pose_geometry import lift_pixel_to_floor_frame
from powerflow_pipeline.data.preprocess.retilt import depth_intrinsics
from powerflow_pipeline.data.preprocess.tasks.ingest import frame_paths, read_frame

DEFAULT_PATCH_RADIUS = 2


def lift_joint_series(
    joints: dict[JointId, JointSeries],
    *,
    depth_dir: Path,
    confidence_dir: Path,
    intrinsics: Intrinsics,
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
    floor_offset_m: float,
    patch_radius: int = DEFAULT_PATCH_RADIUS,
) -> dict[JointId, JointSeries]:
    """S5's 2D series, plus a floor-frame position wherever depth supports one.

    `joints` is expected to be pixel-only (`position is None`); a series that already carries
    positions is rejected rather than silently relifted, since the second lift would be reading
    its own stage's output.
    """

    already_lifted = sorted(
        joint for joint, series in joints.items() if series.position is not None
    )
    if already_lifted:
        raise ValueError(f"joints already carry positions, refusing to relift: {already_lifted}")

    n_frames = len(next(iter(joints.values()))) if joints else 0
    k_d = depth_intrinsics(intrinsics, rgb_size, depth_size)

    depth_paths = frame_paths(depth_dir)
    confidence_paths = frame_paths(confidence_dir)

    positions: dict[JointId, list[tuple[float, float, float] | None]] = {
        joint: [None] * n_frames for joint in joints
    }
    confidences: dict[JointId, list[float]] = {
        joint: list(series.confidence) for joint, series in joints.items()
    }
    pixels: dict[JointId, list[tuple[int, int] | None]] = {
        joint: list(series.pixel_position) for joint, series in joints.items()
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
                radius=patch_radius,
            )
            if point is None:
                # `JointSeries` requires position and pixelPosition to drop out together, and a
                # zero confidence wherever position is None -- so a joint S5 detected but whose
                # depth is unusable (occlusion, out of LiDAR range) loses its pixel here too.
                # S5's own document keeps that pixel; this is the loss the bone-constrained
                # reconstruction spec exists to recover, and the reason the stages are split.
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
        for joint in joints
    }
