"""Reading a stream's PNG frames, shared by every stage that opens depth or confidence."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from powerflow_pipeline.data.common.errors import ScanRejected


def frame_paths(directory: Path) -> list[Path]:
    """Return the PNG frames of a stream in index order."""

    return sorted(directory.glob("*.png"))


def read_frame(path: Path) -> np.ndarray:
    """Read one stream frame exactly as stored: no depth conversion, no colour mapping."""

    frame: np.ndarray | None = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ScanRejected(f"unreadable frame: {path.name}")
    return frame
