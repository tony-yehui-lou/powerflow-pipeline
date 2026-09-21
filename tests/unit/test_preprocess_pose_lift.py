"""`DepthBackedPoseModel`: composes a fake 2D detector with real depth PNGs on disk.

No mediapipe import here -- `Detector2D` is faked, isolating the depth back-projection +
JointSeries-assembly logic this module actually owns (CLAUDE.md: TDD, synthetic fixtures).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from powerflow_pipeline.data.common.models import HUMAN_SKELETON, JointId
from powerflow_pipeline.data.preprocess.depth_pose_model import DepthBackedPoseModel
from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.pose_model import JointPixelSeries


class _FakeDetector2D:
    """Returns a fixed, caller-supplied `dict[JointId, JointPixelSeries]`."""

    def __init__(self, result: dict[JointId, JointPixelSeries]) -> None:
        self._result = result
        self.calls: list[tuple[Path, int]] = []

    def detect(self, rgb_path: Path, n_frames: int) -> dict[JointId, JointPixelSeries]:
        self.calls.append((rgb_path, n_frames))
        return self._result


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


def test_lifts_every_detected_pixel_to_a_metric_floor_frame_position(tmp_path: Path) -> None:
    n_frames = 3
    depth_dir, confidence_dir = _write_depth_confidence(
        tmp_path, n_frames, size=(100, 100), depth_mm=2000, confidence=2
    )
    pixel_series = {
        joint: JointPixelSeries(pixel_position=((50, 50),) * n_frames, confidence=(0.9,) * n_frames)
        for joint in HUMAN_SKELETON.joints
    }
    intrinsics = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, frame="portrait")
    model = DepthBackedPoseModel(_FakeDetector2D(pixel_series))

    result = model.predict(
        rgb_path=tmp_path / "rgb.mp4",
        n_frames=n_frames,
        depth_dir=depth_dir,
        confidence_dir=confidence_dir,
        intrinsics=intrinsics,
        rgb_size=(100, 100),
        depth_size=(100, 100),
        floor_offset_m=2.0,
    )

    assert set(result) == set(HUMAN_SKELETON.joints)
    series = result["head"]
    assert len(series) == n_frames
    for position in series.position:
        assert position == pytest.approx((0.0, 2.0, 2.0))
    assert series.confidence == (0.9, 0.9, 0.9)
    assert series.pixel_position == ((50, 50),) * n_frames


def test_a_frame_with_no_detected_pixel_stays_dropped_out(tmp_path: Path) -> None:
    n_frames = 2
    depth_dir, confidence_dir = _write_depth_confidence(
        tmp_path, n_frames, size=(100, 100), depth_mm=1500, confidence=2
    )
    pixel_series = {
        joint: JointPixelSeries(pixel_position=((50, 50), None), confidence=(0.8, 0.0))
        for joint in HUMAN_SKELETON.joints
    }
    intrinsics = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, frame="portrait")
    model = DepthBackedPoseModel(_FakeDetector2D(pixel_series))

    result = model.predict(
        rgb_path=tmp_path / "rgb.mp4",
        n_frames=n_frames,
        depth_dir=depth_dir,
        confidence_dir=confidence_dir,
        intrinsics=intrinsics,
        rgb_size=(100, 100),
        depth_size=(100, 100),
        floor_offset_m=1.5,
    )

    series = result["leftWrist"]
    assert series.position[1] is None
    assert series.pixel_position[1] is None
    assert series.confidence[1] == 0.0
    assert series.position[0] is not None


def test_unusable_depth_forces_the_frame_to_drop_out_even_with_a_confident_pixel(
    tmp_path: Path,
) -> None:
    n_frames = 1
    # Zero depth everywhere: the detector found a pixel, but there's nothing to back-project.
    depth_dir, confidence_dir = _write_depth_confidence(
        tmp_path, n_frames, size=(100, 100), depth_mm=0, confidence=2
    )
    pixel_series = {
        joint: JointPixelSeries(pixel_position=((50, 50),), confidence=(0.95,))
        for joint in HUMAN_SKELETON.joints
    }
    intrinsics = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, frame="portrait")
    model = DepthBackedPoseModel(_FakeDetector2D(pixel_series))

    result = model.predict(
        rgb_path=tmp_path / "rgb.mp4",
        n_frames=n_frames,
        depth_dir=depth_dir,
        confidence_dir=confidence_dir,
        intrinsics=intrinsics,
        rgb_size=(100, 100),
        depth_size=(100, 100),
        floor_offset_m=1.8,
    )

    series = result["rightKnee"]
    # JointSeries's own invariant: position/pixelPosition drop out together, confidence 0.
    assert series.position[0] is None
    assert series.pixel_position[0] is None
    assert series.confidence[0] == 0.0


def test_passes_rgb_path_and_n_frames_through_to_the_detector(tmp_path: Path) -> None:
    depth_dir, confidence_dir = _write_depth_confidence(
        tmp_path, 1, size=(10, 10), depth_mm=1000, confidence=2
    )
    empty_series = JointPixelSeries(pixel_position=(None,), confidence=(0.0,))
    detector = _FakeDetector2D({joint: empty_series for joint in HUMAN_SKELETON.joints})
    model = DepthBackedPoseModel(detector)
    rgb_path = tmp_path / "rgb.mp4"

    model.predict(
        rgb_path=rgb_path,
        n_frames=1,
        depth_dir=depth_dir,
        confidence_dir=confidence_dir,
        intrinsics=Intrinsics(fx=10.0, fy=10.0, cx=5.0, cy=5.0, frame="portrait"),
        rgb_size=(10, 10),
        depth_size=(10, 10),
        floor_offset_m=1.0,
    )

    assert detector.calls == [(rgb_path, 1)]
