"""`lift_joint_series`: S6's metric lift, against real depth PNGs on disk.

No detector here -- the lift takes S5's published `JointSeries`, so this isolates the depth
back-projection and JointSeries reassembly it actually owns (CLAUDE.md: TDD, synthetic
fixtures). Previously the same coverage sat on `DepthBackedPoseModel`, which combined this with
detection before the stages were split.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from powerflow_pipeline.data.common.models import HUMAN_SKELETON, JointId, JointSeries
from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.pose_lift import lift_joint_series

INTRINSICS = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, frame="portrait")


def _write_depth_confidence(
    tmp_path: Path, n_frames: int, size: tuple[int, int], depth_mm: int, confidence: int
) -> tuple[Path, Path]:
    """`size` is `(width, height)`. Every pixel of every frame gets the same depth/confidence."""

    width, height = size
    depth_dir = tmp_path / "depth"
    confidence_dir = tmp_path / "confidence"
    depth_dir.mkdir()
    confidence_dir.mkdir()

    depth_frame = np.full((height, width), depth_mm, dtype=np.uint16)
    confidence_frame = np.full((height, width), confidence, dtype=np.uint8)
    for index in range(n_frames):
        cv2.imwrite(str(depth_dir / f"{index:06d}.png"), depth_frame)
        cv2.imwrite(str(confidence_dir / f"{index:06d}.png"), confidence_frame)
    return depth_dir, confidence_dir


def _detected(
    pixels: tuple[tuple[int, int] | None, ...], confidence: tuple[float, ...]
) -> dict[JointId, JointSeries]:
    """One 2D-only `JointSeries` per skeleton joint, as S5 publishes them."""

    return {
        joint: JointSeries(position=None, pixel_position=pixels, confidence=confidence)
        for joint in HUMAN_SKELETON.joints
    }


def _lift(
    joints: dict[JointId, JointSeries],
    depth_dir: Path,
    confidence_dir: Path,
    floor_offset_m: float,
    size: tuple[int, int] = (100, 100),
) -> dict[JointId, JointSeries]:
    return lift_joint_series(
        joints,
        depth_dir=depth_dir,
        confidence_dir=confidence_dir,
        intrinsics=INTRINSICS,
        rgb_size=size,
        depth_size=size,
        floor_offset_m=floor_offset_m,
    )


def test_lifts_every_detected_pixel_to_a_metric_floor_frame_position(tmp_path: Path) -> None:
    n_frames = 3
    depth_dir, confidence_dir = _write_depth_confidence(
        tmp_path, n_frames, size=(100, 100), depth_mm=2000, confidence=2
    )
    joints = _detected(((50, 50),) * n_frames, (0.9,) * n_frames)

    result = _lift(joints, depth_dir, confidence_dir, floor_offset_m=2.0)

    assert set(result) == set(HUMAN_SKELETON.joints)
    series = result["head"]
    assert len(series) == n_frames
    assert series.position is not None
    for position in series.position:
        assert position == pytest.approx((0.0, 2.0, 2.0))
    assert series.confidence == (0.9, 0.9, 0.9)
    assert series.pixel_position == ((50, 50),) * n_frames


def test_a_frame_with_no_detected_pixel_stays_dropped_out(tmp_path: Path) -> None:
    depth_dir, confidence_dir = _write_depth_confidence(
        tmp_path, 2, size=(100, 100), depth_mm=1500, confidence=2
    )
    joints = _detected(((50, 50), None), (0.8, 0.0))

    series = _lift(joints, depth_dir, confidence_dir, floor_offset_m=1.5)["leftWrist"]

    assert series.position is not None
    assert series.position[1] is None
    assert series.pixel_position[1] is None
    assert series.confidence[1] == 0.0
    assert series.position[0] is not None


def test_unusable_depth_forces_the_frame_to_drop_out_even_with_a_confident_pixel(
    tmp_path: Path,
) -> None:
    # Zero depth everywhere: S5 found a pixel, but there's nothing to back-project.
    depth_dir, confidence_dir = _write_depth_confidence(
        tmp_path, 1, size=(100, 100), depth_mm=0, confidence=2
    )
    joints = _detected(((50, 50),), (0.95,))

    series = _lift(joints, depth_dir, confidence_dir, floor_offset_m=1.8)["rightKnee"]

    # JointSeries's own invariant: position/pixelPosition drop out together, confidence 0. This
    # is the loss S5's own document does *not* suffer, which is why the stages are separable.
    assert series.position is not None
    assert series.position[0] is None
    assert series.pixel_position[0] is None
    assert series.confidence[0] == 0.0


def test_refuses_to_relift_joints_that_already_carry_positions(tmp_path: Path) -> None:
    depth_dir, confidence_dir = _write_depth_confidence(
        tmp_path, 1, size=(10, 10), depth_mm=1000, confidence=2
    )
    already = {
        joint: JointSeries(position=((0.0, 1.0, 2.0),), pixel_position=((5, 5),), confidence=(0.9,))
        for joint in HUMAN_SKELETON.joints
    }

    with pytest.raises(ValueError, match="already carry positions"):
        _lift(already, depth_dir, confidence_dir, floor_offset_m=1.0, size=(10, 10))
