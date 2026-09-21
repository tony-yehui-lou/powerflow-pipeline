"""`to_pixel_only_series`: where a detector's 2D output becomes the pipeline's own type.

S5 publishes pixels and nothing else, so the interesting assertions are about what is *absent*
(no positions) and about the `Detector2D` contract being enforced at this boundary rather than
surfacing later as a `PoseDocument` frame-count error.
"""

from __future__ import annotations

import pytest

from powerflow_pipeline.data.common.models import HUMAN_SKELETON, JointId
from powerflow_pipeline.data.preprocess.pose_model import JointPixelSeries, to_pixel_only_series


def _pixel_series(n_frames: int) -> dict[JointId, JointPixelSeries]:
    return {
        joint: JointPixelSeries(
            pixel_position=tuple((index, 2 * index) for index in range(n_frames)),
            confidence=(0.9,) * n_frames,
        )
        for joint in HUMAN_SKELETON.joints
    }


def test_passes_the_detector_pixels_through_and_carries_no_positions() -> None:
    result = to_pixel_only_series(_pixel_series(3), 3)

    assert set(result) == set(HUMAN_SKELETON.joints)
    series = result["head"]
    assert series.position is None
    assert series.pixel_position == ((0, 0), (1, 2), (2, 4))
    assert series.confidence == (0.9, 0.9, 0.9)
    assert len(series) == 3


def test_keeps_an_undetected_frame_dropped_out() -> None:
    pixel_series = {
        joint: JointPixelSeries(pixel_position=((5, 5), None), confidence=(0.8, 0.0))
        for joint in HUMAN_SKELETON.joints
    }

    series = to_pixel_only_series(pixel_series, 2)["leftWrist"]

    assert series.pixel_position == ((5, 5), None)
    assert series.confidence == (0.8, 0.0)
    assert series.position is None


def test_rejects_a_detector_that_omits_a_skeleton_joint() -> None:
    partial = _pixel_series(1)
    del partial["head"]

    with pytest.raises(ValueError, match="head"):
        to_pixel_only_series(partial, 1)


def test_rejects_a_detector_series_that_disagrees_with_the_frame_count() -> None:
    with pytest.raises(ValueError, match="expected 3"):
        to_pixel_only_series(_pixel_series(2), 3)
