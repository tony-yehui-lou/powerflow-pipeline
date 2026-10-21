"""S5's barbell detections and the declared bar, read the way S6's wrist correction needs them.

Today S5's detections are `predictions.json` from `scripts/predict_landmarks.py`, published per
capture under a root that mirrors the raw layout. `S5-barbell-detection-stage.md` replaces that
with `barbell.json` under `s5_barbell_output/`; this module is the one place that changes then.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import yaml

from powerflow_pipeline.data.preprocess.bar_geometry import Pixel

DETECTIONS_FILENAME: Final = "predictions.json"

# Real detections score >= 0.95 almost everywhere; false caps on a bystander scored 0.31-0.33.
MIN_SCORE: Final = 0.5


def detections_path(barbell_root: Path, relative: Path) -> Path:
    return barbell_root / relative / DETECTIONS_FILENAME


def read_detections(
    path: Path, *, n_frames: int, frame_size: tuple[int, int], min_score: float = MIN_SCORE
) -> tuple[list[Pixel | None], list[Pixel | None]]:
    """Cap and collar box centres per frame, `n_frames` long; None where missing or below
    `min_score`. Refuses detections drawn in an image of a different size from `frame_size`
    (width, height): a pixel means nothing outside the image space it was drawn in."""

    document = json.loads(path.read_text(encoding="utf-8"))
    height, width = document["frame_size"]
    if (width, height) != frame_size:
        raise ValueError(
            f"{path}: detections are {width}x{height}, the capture is "
            f"{frame_size[0]}x{frame_size[1]}"
        )

    caps: list[Pixel | None] = [None] * n_frames
    collars: list[Pixel | None] = [None] * n_frames
    for key, frame in document["predictions"].items():
        i = int(key)
        if i >= n_frames:
            continue
        for name, out in (("cap", caps), ("collar", collars)):
            hit = frame.get(name)
            if hit and hit["score"] >= min_score:
                x0, y0, x1, y1 = hit["box"]
                out[i] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return caps, collars


def read_bar_type(metadata_path: Path) -> str | None:
    """`lift.barbell_type` from the capture's raw operator metadata, or None if undeclared."""

    if not metadata_path.is_file():
        return None
    loaded = yaml.safe_load(metadata_path.read_text())
    lift = loaded.get("lift") if isinstance(loaded, dict) else None
    value = lift.get("barbell_type") if isinstance(lift, dict) else None
    return value if isinstance(value, str) else None
