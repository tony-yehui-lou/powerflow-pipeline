"""S5's barbell detections and the declared bar, as S6's wrist correction and S8 read them.

S5 publishes one `barbell.json` per capture under `barbell_root`, mirroring the raw layout
(`tasks/barbell.py`). This module is the one place a consumer reads it: box centres per frame,
gated at the consumer's own `min_score`, never at the lower score S5 recorded down to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from powerflow_pipeline.data.common.bar_geometry import Pixel
from powerflow_pipeline.data.common.barbell_storage import (
    BARBELL_FILENAME,
    barbell_path,
    read_barbell,
)
from powerflow_pipeline.data.common.models import BarbellClass

DETECTIONS_FILENAME: Final = BARBELL_FILENAME

# Real detections score >= 0.95 almost everywhere; false caps on a bystander scored 0.31-0.33.
MIN_SCORE: Final = 0.5


@dataclass(frozen=True)
class Detections:
    """Cap and collar box centres per frame, and which checkpoint produced them."""

    caps: list[Pixel | None]
    collars: list[Pixel | None]
    checkpoint_sha256: str


def detections_path(barbell_root: Path, relative: Path) -> Path:
    return barbell_path(barbell_root, relative)


def read_detections(
    path: Path, *, n_frames: int, frame_size: tuple[int, int], min_score: float = MIN_SCORE
) -> Detections:
    """Cap and collar centres per frame, `n_frames` long; None where missing or below
    `min_score`. Refuses detections drawn in an image of a different size from `frame_size`
    (width, height), or over a different number of frames: either means they came from some
    other video, and a pixel means nothing outside the image it was drawn in. A malformed
    document is a `ValueError` too (pydantic's `ValidationError` is one)."""

    document = read_barbell(path)
    space = (document.pixel_space.width, document.pixel_space.height)
    if space != frame_size:
        raise ValueError(
            f"{path}: detections are {space[0]}x{space[1]}, the capture is "
            f"{frame_size[0]}x{frame_size[1]}"
        )
    if document.frames.count != n_frames:
        raise ValueError(
            f"{path}: detections span {document.frames.count} frames, the capture has {n_frames}"
        )

    def centres(name: BarbellClass) -> list[Pixel | None]:
        return [
            hit.centre if hit is not None and hit.score >= min_score else None
            for hit in document.landmarks[name]
        ]

    return Detections(
        caps=centres("cap"),
        collars=centres("collar"),
        checkpoint_sha256=document.detector.checkpoint_sha256,
    )


def read_bar_type(metadata_path: Path) -> str | None:
    """`lift.barbell_type` from the capture's raw operator metadata, or None if undeclared."""

    if not metadata_path.is_file():
        return None
    loaded = yaml.safe_load(metadata_path.read_text())
    lift = loaded.get("lift") if isinstance(loaded, dict) else None
    value = lift.get("barbell_type") if isinstance(lift, dict) else None
    return value if isinstance(value, str) else None
