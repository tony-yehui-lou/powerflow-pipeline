"""`MediaPipePoseDetector`: a real inference smoke test.

A synthetic frame has no actual person in it, so this cannot assert detection accuracy --
only that the real MediaPipe Tasks-API call path (frame decode -> `landmarker.detect` -> the
no-pose-found branch) runs to completion, on the pinned `mediapipe==0.10.35`, without the
macOS 1.x Metal crash this module's own docstring documents, and produces correctly-shaped
output. Detection quality against real lift footage is a separate, tracked follow-up
(docs/specs/preprocessing/S5-pose-model-recommendation.md, Risks: "Accuracy validation").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import av
import numpy as np
import pytest

from powerflow_pipeline.data.common.models import HUMAN_SKELETON
from powerflow_pipeline.data.preprocess.mediapipe_pose_detector import (
    MediaPipePoseDetector,
    pad_to_square,
)


def test_pad_to_square_squares_a_portrait_frame_without_moving_a_pixel() -> None:
    array = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3)  # h=4, w=3

    padded = pad_to_square(array)

    assert padded.shape == (4, 4, 3)
    # Every original pixel keeps its own coordinate -- that's what lets a landmark
    # normalized against the square map straight back with no offset to undo.
    assert np.array_equal(padded[:4, :3], array)
    assert np.array_equal(padded[:, 3], np.zeros((4, 3), dtype=np.uint8))


def test_pad_to_square_squares_a_landscape_frame() -> None:
    array = np.full((2, 5, 3), 7, dtype=np.uint8)

    padded = pad_to_square(array)

    assert padded.shape == (5, 5, 3)
    assert np.array_equal(padded[:2, :5], array)
    assert np.array_equal(padded[2:], np.zeros((3, 5, 3), dtype=np.uint8))


def test_pad_to_square_leaves_an_already_square_frame_alone() -> None:
    array = np.full((6, 6, 3), 3, dtype=np.uint8)

    assert pad_to_square(array) is array


def _write_blank_rgb(path: Path, n_frames: int, size: tuple[int, int] = (64, 64)) -> None:
    width, height = size
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_detect_runs_end_to_end_and_returns_every_skeleton_joint(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.mp4"
    n_frames = 2
    _write_blank_rgb(rgb_path, n_frames)

    detector = MediaPipePoseDetector()
    result = detector.detect(rgb_path, n_frames)

    assert set(result) == set(HUMAN_SKELETON.joints)
    for series in result.values():
        assert len(series) == n_frames
        # A blank frame has no person: every joint stays at its dropped-out default.
        assert series.pixel_position == (None,) * n_frames
        assert series.confidence == (0.0,) * n_frames


class _StubOptions:
    """Captures the kwargs `MediaPipePoseDetector` builds `PoseLandmarkerOptions` with."""

    last_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        _StubOptions.last_kwargs = kwargs


class _StubLandmarker:
    @classmethod
    def create_from_options(cls, options: object) -> _StubLandmarker:
        return cls()


def _patch_mediapipe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = "powerflow_pipeline.data.preprocess.mediapipe_pose_detector"
    monkeypatch.setattr(f"{module}.vision.PoseLandmarkerOptions", _StubOptions)
    monkeypatch.setattr(f"{module}.vision.PoseLandmarker", _StubLandmarker)
    (tmp_path / "model.task").touch()


def test_keeps_mediapipes_own_default_confidence_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deliberately NOT loosened: a lower threshold doesn't recover a missed lifter, it
    # invents phantom skeletons (see `pad_to_square` for the defect that actually caused
    # the zero-detection session this once tried to paper over).
    _patch_mediapipe(monkeypatch, tmp_path)

    MediaPipePoseDetector(model_path=tmp_path / "model.task")

    assert _StubOptions.last_kwargs["min_pose_detection_confidence"] == 0.5
    assert _StubOptions.last_kwargs["min_pose_presence_confidence"] == 0.5


def test_accepts_a_custom_confidence_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_mediapipe(monkeypatch, tmp_path)

    MediaPipePoseDetector(
        model_path=tmp_path / "model.task",
        min_pose_detection_confidence=0.2,
        min_pose_presence_confidence=0.1,
    )

    assert _StubOptions.last_kwargs["min_pose_detection_confidence"] == 0.2
    assert _StubOptions.last_kwargs["min_pose_presence_confidence"] == 0.1
