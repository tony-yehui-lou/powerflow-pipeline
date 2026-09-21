"""The joint-detection model boundary for S5 Pose (GitHub issue #118, "Find a Suitable Model").

`Detector2D.detect` is the whole seam: S5 Pose locates joints in pixel space and stops there.
The metric lift to floor-frame metres is S6 Lift's job (`pose_lift.py`, `tasks/lift.py`), which
reads S5's published document rather than being composed into the detector. Keeping them apart
means a detector can be swapped, or re-run, without touching depth -- and the depth lift can be
re-run without paying for detection again, which on RTMPose is most of the wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from powerflow_pipeline.data.common.models import HUMAN_SKELETON, JointId, JointSeries

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


def to_pixel_only_series(
    pixel_series: dict[JointId, JointPixelSeries], n_frames: int
) -> dict[JointId, JointSeries]:
    """Wrap a detector's 2D output as the `JointSeries` S5 publishes, with no positions.

    `position` is `None` for the whole series rather than a tuple of `None`s: S5 did not fail
    to place these joints, it never computes positions at all. See `JointSeries`.

    The `Detector2D` contract (all 15 joints, each series `n_frames` long) is enforced here,
    at the boundary where a detector's output first enters the pipeline's own types, so a
    misbehaving detector is named as such instead of surfacing later as a frame-count error
    against a `PoseDocument`.
    """

    missing = set(HUMAN_SKELETON.joints) - set(pixel_series)
    if missing:
        raise ValueError(f"detector omitted skeleton joints: {sorted(missing)}")

    for joint, series in pixel_series.items():
        if len(series) != n_frames:
            raise ValueError(
                f"detector returned {len(series)} frames for {joint!r}, expected {n_frames}"
            )

    return {
        joint: JointSeries(
            position=None,
            pixel_position=series.pixel_position,
            confidence=series.confidence,
        )
        for joint, series in pixel_series.items()
    }
