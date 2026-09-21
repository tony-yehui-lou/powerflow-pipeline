"""Subject selection, candidate depth, and frame assembly for the RTMPose detector.

Subject selection (S5-rtmpose-migration.md §4) is this class's own responsibility rather than
something the model hides, so it is tested directly on synthetic arrays. `detect` is exercised
with `rtmlib`'s tracker stubbed out: the decode/select/map/provenance wiring is ours and worth
testing, while the inference behind it is not, and needs a real checkpoint download to run.
Whether the model finds the right person on real footage is a §6.3 visual check against real
captures, which no unit test can stand in for.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from powerflow_pipeline.data.common.models import HUMAN_SKELETON
from powerflow_pipeline.data.preprocess.rtmpose_detector import (
    RTMPoseDetector,
    candidate_depth_m,
    select_subject,
)


def _person(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """26 keypoints spread over the given box."""

    xs = np.linspace(x0, x1, 26)
    ys = np.linspace(y0, y1, 26)
    return np.stack([xs, ys], axis=-1)


def _scores(value: float) -> np.ndarray:
    return np.full(26, value, dtype=float)


# --- candidate_depth_m --------------------------------------------------------------------


def test_candidate_depth_is_the_median_over_its_keypoints() -> None:
    depth = np.full((50, 50), 3000, dtype=np.uint16)
    confidence = np.full((50, 50), 2, dtype=np.uint8)

    result = candidate_depth_m(
        _person(10, 10, 40, 40), depth, confidence, rgb_size=(50, 50), depth_size=(50, 50)
    )

    assert result == pytest.approx(3.0)


def test_candidate_depth_is_none_when_no_keypoint_has_usable_depth() -> None:
    depth = np.zeros((50, 50), dtype=np.uint16)
    confidence = np.full((50, 50), 2, dtype=np.uint8)

    result = candidate_depth_m(
        _person(10, 10, 40, 40), depth, confidence, rgb_size=(50, 50), depth_size=(50, 50)
    )

    assert result is None


def test_candidate_depth_ignores_keypoints_outside_the_depth_frame() -> None:
    depth = np.full((50, 50), 2500, dtype=np.uint16)
    confidence = np.full((50, 50), 2, dtype=np.uint8)

    # Keypoints running well past the frame must not raise or wrap around.
    result = candidate_depth_m(
        _person(-200, -200, 400, 400), depth, confidence, rgb_size=(50, 50), depth_size=(50, 50)
    )

    assert result == pytest.approx(2.5)


# --- select_subject -----------------------------------------------------------------------


def test_prefers_the_nearest_person_by_depth() -> None:
    # The athlete is on the platform nearest the camera; bystanders stand behind.
    near, far = _person(10, 10, 20, 20), _person(30, 30, 45, 45)
    depth = np.full((50, 50), 8000, dtype=np.uint16)
    depth[:25, :25] = 2000  # the near candidate's region
    confidence = np.full((50, 50), 2, dtype=np.uint8)

    index, rule = select_subject(
        np.stack([far, near]),
        np.stack([_scores(0.9), _scores(0.9)]),
        depth,
        confidence,
        rgb_size=(50, 50),
        depth_size=(50, 50),
    )

    assert index == 1  # `near`, despite being the smaller box
    assert rule == "nearest_depth"


def test_falls_back_to_largest_bbox_when_depth_is_unusable() -> None:
    small, large = _person(10, 10, 15, 15), _person(0, 0, 45, 45)
    depth = np.zeros((50, 50), dtype=np.uint16)
    confidence = np.zeros((50, 50), dtype=np.uint8)

    index, rule = select_subject(
        np.stack([small, large]),
        np.stack([_scores(0.9), _scores(0.9)]),
        depth,
        confidence,
        rgb_size=(50, 50),
        depth_size=(50, 50),
    )

    assert index == 1
    assert rule == "largest_bbox"


def test_falls_back_to_largest_bbox_when_no_depth_is_available_at_all() -> None:
    small, large = _person(10, 10, 15, 15), _person(0, 0, 45, 45)

    index, rule = select_subject(
        np.stack([small, large]),
        np.stack([_scores(0.9), _scores(0.9)]),
        None,
        None,
        rgb_size=(50, 50),
        depth_size=None,
    )

    assert index == 1
    assert rule == "largest_bbox"


def test_ignores_candidates_below_the_score_floor() -> None:
    # A large but barely-scored blob must not outrank a confident smaller person.
    junk, athlete = _person(0, 0, 49, 49), _person(10, 10, 25, 25)

    index, rule = select_subject(
        np.stack([junk, athlete]),
        np.stack([_scores(0.05), _scores(0.9)]),
        None,
        None,
        rgb_size=(50, 50),
        depth_size=None,
    )

    assert index == 1
    assert rule == "largest_bbox"


def test_reports_no_subject_when_every_candidate_is_below_the_score_floor() -> None:
    index, rule = select_subject(
        np.stack([_person(0, 0, 10, 10)]),
        np.stack([_scores(0.01)]),
        None,
        None,
        rgb_size=(50, 50),
        depth_size=None,
    )

    assert index is None
    assert rule == "none"


# --- detect / _record, with the rtmlib tracker stubbed out ---------------------------------


class _StubTracker:
    """Stands in for `rtmlib.PoseTracker`: returns canned (keypoints, scores) per call."""

    def __init__(self, frames: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self._frames = frames
        self.calls = 0

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = self._frames[min(self.calls, len(self._frames) - 1)]
        self.calls += 1
        return result


def _detector_without_rtmlib(
    tracker: _StubTracker, *, use_depth_for_subject: bool = True
) -> RTMPoseDetector:
    """Build the detector past its rtmlib-loading constructor, which needs real checkpoints."""

    detector = object.__new__(RTMPoseDetector)
    detector._tracker = tracker  # type: ignore[attr-defined]
    detector._variant = "balanced"  # type: ignore[attr-defined]
    detector._det_frequency = 1  # type: ignore[attr-defined]
    detector._clavicle_shoulder_weight = 0.5  # type: ignore[attr-defined]
    detector._use_depth_for_subject = use_depth_for_subject  # type: ignore[attr-defined]
    detector.last_provenance = {}
    return detector


def _write_rgb(path: Path, n_frames: int, size: tuple[int, int] = (64, 64)) -> None:
    width, height = size
    image = np.full((height, width, 3), 120, dtype=np.uint8)
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


def test_detect_maps_a_tracked_person_into_every_skeleton_joint(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.mp4"
    _write_rgb(rgb, 2)
    person = _person(10, 10, 50, 50)[None, :, :]
    tracker = _StubTracker([(person, _scores(0.8)[None, :])])

    result = _detector_without_rtmlib(tracker).detect(rgb, 2)

    assert set(result) == set(HUMAN_SKELETON.joints)
    for series in result.values():
        assert len(series) == 2
        assert all(p is not None for p in series.pixel_position)
        assert all(c == pytest.approx(0.8) for c in series.confidence)


def test_detect_counts_a_frame_with_no_person_as_a_track_loss(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.mp4"
    _write_rgb(rgb, 2)
    empty = (np.zeros((0, 26, 2)), np.zeros((0, 26)))
    tracker = _StubTracker([empty])

    detector = _detector_without_rtmlib(tracker)
    result = detector.detect(rgb, 2)

    assert detector.last_provenance["track_losses"] == 2
    for series in result.values():
        assert series.pixel_position == (None, None)
        assert series.confidence == (0.0, 0.0)


def test_detect_records_provenance(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.mp4"
    _write_rgb(rgb, 1)
    person = _person(10, 10, 50, 50)[None, :, :]
    tracker = _StubTracker([(person, _scores(0.7)[None, :])])

    detector = _detector_without_rtmlib(tracker)
    detector.detect(rgb, 1)

    provenance = detector.last_provenance
    assert provenance["detector"] == "rtmpose"
    assert provenance["keypoint_format"] == "halpe26"
    assert provenance["variant"] == "balanced"
    assert provenance["subject_selection_rules"] == ["largest_bbox"]  # tmp_path has no depth/
    assert provenance["rgb_size"] == [64, 64]
    assert "latest" not in provenance["pose_checkpoint_url"]


def test_detect_reads_no_depth_when_subject_depth_is_disabled(tmp_path: Path) -> None:
    """S5's one remaining depth read is subject selection; this turns it off completely.

    The depth directory here is unreadable garbage -- if the detector opened it, it would
    raise. With the switch off it must not look, leaving S5 a pure pixel-space stage.
    """

    rgb = tmp_path / "rgb.mp4"
    _write_rgb(rgb, 1)
    (tmp_path / "depth").mkdir()
    (tmp_path / "confidence").mkdir()
    (tmp_path / "depth" / "000000.png").write_bytes(b"not a png")
    (tmp_path / "confidence" / "000000.png").write_bytes(b"not a png")
    person = _person(10, 10, 50, 50)[None, :, :]
    tracker = _StubTracker([(person, _scores(0.8)[None, :])])

    detector = _detector_without_rtmlib(tracker, use_depth_for_subject=False)
    result = detector.detect(rgb, 1)

    assert set(result) == set(HUMAN_SKELETON.joints)
    assert detector.last_provenance["subject_depth"] is False
    assert detector.last_provenance["subject_selection_rules"] == ["largest_bbox"]


def test_provenance_records_that_subject_depth_was_available(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.mp4"
    _write_rgb(rgb, 1)
    person = _person(10, 10, 50, 50)[None, :, :]
    tracker = _StubTracker([(person, _scores(0.8)[None, :])])

    detector = _detector_without_rtmlib(tracker)
    detector.detect(rgb, 1)

    assert detector.last_provenance["subject_depth"] is True


def test_detect_stops_at_n_frames_even_if_the_video_is_longer(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.mp4"
    _write_rgb(rgb, 5)
    person = _person(10, 10, 50, 50)[None, :, :]
    tracker = _StubTracker([(person, _scores(0.8)[None, :])])

    result = _detector_without_rtmlib(tracker).detect(rgb, 3)

    assert all(len(series) == 3 for series in result.values())
