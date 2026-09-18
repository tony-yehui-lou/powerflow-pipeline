"""Fetches MediaPipe's pretrained Pose Landmarker model bundle on demand.

The `mediapipe` pip package ships no model weights -- the Tasks API
(`mediapipe.tasks.python.vision.PoseLandmarker`) requires a separate `.task` bundle, published
by Google at a fixed, versioned URL. Downloaded once into `models_cache/` (git-ignored) and
reused on every later run; never committed to the repo.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Literal

ModelVariant = Literal["lite", "full", "heavy"]

# Google's official MediaPipe model storage (developers.google.com/mediapipe/solutions/
# vision/pose_landmarker). Pinned to a specific model per variant, not "latest", so a run
# is reproducible across machines and time.
_MODEL_URLS: dict[ModelVariant, str] = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
}

# Real bundles are several MB; anything smaller is a failed download (an HTML error page,
# a truncated transfer), never a valid model.
_MIN_VALID_BYTES = 1_000_000

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[4] / "models_cache"


def default_model_path(variant: ModelVariant = "lite", cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / f"pose_landmarker_{variant}.task"


def ensure_pose_landmarker_model(
    variant: ModelVariant = "lite", cache_dir: Path = DEFAULT_CACHE_DIR
) -> Path:
    """Return a local path to `variant`'s model bundle, downloading it if not already cached."""

    path = default_model_path(variant, cache_dir)
    if path.exists() and path.stat().st_size >= _MIN_VALID_BYTES:
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".task.part")
    urllib.request.urlretrieve(_MODEL_URLS[variant], tmp_path)

    if tmp_path.stat().st_size < _MIN_VALID_BYTES:
        tmp_path.unlink()
        raise RuntimeError(f"downloaded {variant} pose landmarker model looks truncated/invalid")

    tmp_path.replace(path)
    return path
