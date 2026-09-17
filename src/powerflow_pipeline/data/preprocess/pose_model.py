"""The joint-detection model boundary for S5 Pose.

Which model to run and how to run it is GitHub issue #118 ("Find a Suitable Model") --
out of scope here. This module only defines the seam `tasks/pose.py` calls through, so the
S5 stage's plumbing (config, file layout, flow wiring, the manifest) can be built and tested
against a stand-in before a real model is chosen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from powerflow_pipeline.data.common.models import JointId, JointSeries


@runtime_checkable
class PoseModel(Protocol):
    """Something that turns one camera's cropped RGB into a per-joint pose time series.

    The model owns pixel-space detection and the lift to `JointSeries.position` (floor-frame
    metres) alike -- S5 only assembles whatever it returns into a `PoseDocument` and publishes
    it, unchanged.
    """

    def predict(self, rgb_path: Path, n_frames: int) -> dict[JointId, JointSeries]:
        """One `JointSeries` per skeleton joint, each `n_frames` long, for `rgb_path`."""
        ...
