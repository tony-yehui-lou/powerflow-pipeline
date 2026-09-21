"""Fetches RTMPose's ONNX checkpoints on demand (S5-rtmpose-migration.md §7).

Mirrors `mediapipe_assets.py`: pinned URLs per variant (never `latest`), a minimum-size check,
and an atomic rename so a truncated download is never mistaken for a valid model. Downloads
into the git-ignored `models_cache/` rather than rtmlib's own `~/.cache/rtmlib`, so a run's
models sit beside the pipeline that uses them and can be cleared with the rest of the cache.

`rtmlib`'s tools use a path directly when it exists on disk and only download when it doesn't,
so handing them a local path from here bypasses their cache entirely.

The URLs are the ones `rtmlib.BodyWithFeet.MODE` pins, copied deliberately rather than read out
of rtmlib at runtime: a pinned checkpoint is provenance, and it should change when someone
edits this file, not when a dependency bumps.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Literal

RTMPoseVariant = Literal["lightweight", "balanced", "performance"]

# Halpe-26 (BodyWithFeet) detector + pose checkpoints, per variant. YOLOX is the person
# detector; it is Apache-2.0 (S5-rtmpose-migration.md §8).
_MODEL_URLS: dict[RTMPoseVariant, dict[str, str]] = {
    "lightweight": {
        "det": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
        ),
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip"
        ),
    },
    "balanced": {
        "det": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
        ),
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip"
        ),
    },
    "performance": {
        "det": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "yolox_x_8xb8-300e_humanart-a39d44ed.zip"
        ),
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip"
        ),
    },
}

# The input sizes each checkpoint was exported at. Passing the wrong one silently degrades
# accuracy rather than failing, so they travel with the URLs.
INPUT_SIZES: dict[RTMPoseVariant, dict[str, tuple[int, int]]] = {
    "lightweight": {"det": (416, 416), "pose": (192, 256)},
    "balanced": {"det": (640, 640), "pose": (192, 256)},
    "performance": {"det": (640, 640), "pose": (288, 384)},
}

# Real checkpoints are several MB; anything smaller is a failed download (an HTML error page,
# a truncated transfer), never a valid model.
_MIN_VALID_BYTES = 1_000_000

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[4] / "models_cache"


def model_path(role: str, variant: RTMPoseVariant, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Where `role` ("det" or "pose") for `variant` is cached locally."""

    return cache_dir / f"rtmpose_{variant}_{role}.onnx"


def _extract_single_onnx(archive: Path, destination: Path) -> None:
    """Pull the one `.onnx` out of a checkpoint zip and put it at `destination`, atomically."""

    with tempfile.TemporaryDirectory() as raw_temp:
        temp = Path(raw_temp)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(temp)
        found = sorted(temp.rglob("*.onnx"))
        if len(found) != 1:
            raise RuntimeError(f"expected exactly one .onnx in {archive.name}, found {len(found)}")
        if found[0].stat().st_size < _MIN_VALID_BYTES:
            raise RuntimeError(f"{archive.name} holds a truncated/invalid model")
        staged = destination.with_suffix(".onnx.part")
        shutil.move(str(found[0]), staged)
        staged.replace(destination)


def ensure_rtmpose_model(
    role: str, variant: RTMPoseVariant = "balanced", cache_dir: Path = DEFAULT_CACHE_DIR
) -> Path:
    """Return a local path to `variant`'s `role` checkpoint, downloading it if not cached."""

    if role not in ("det", "pose"):
        raise ValueError(f"unknown checkpoint role: {role!r}")

    path = model_path(role, variant, cache_dir)
    if path.exists() and path.stat().st_size >= _MIN_VALID_BYTES:
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as raw_temp:
        archive = Path(raw_temp) / "checkpoint.zip"
        urllib.request.urlretrieve(_MODEL_URLS[variant][role], archive)
        _extract_single_onnx(archive, path)
    return path


def checkpoint_url(role: str, variant: RTMPoseVariant) -> str:
    """The pinned URL a cached checkpoint came from -- provenance for the sidecar (§5)."""

    return _MODEL_URLS[variant][role]
