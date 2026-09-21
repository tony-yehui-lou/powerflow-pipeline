#!/usr/bin/env python
"""Regenerate the published human body model documents from the normative constants.

Nothing here is hand-maintained: `skeleton.human-v2.json` is rendered from `HUMAN_SKELETON`, so
the code and the published document cannot drift apart. Run after changing the body model:

    uv run python scripts/generate_body_model_documents.py

Normative source: "Human Body Model Data Storage v1" (docs/spec, GitHub issue #117).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from powerflow_pipeline.data.common.models import (
    HUMAN_SKELETON,
    Frames,
    JointId,
    JointSeries,
    PoseDocument,
    skeleton_document,
)
from powerflow_pipeline.data.common.pose_storage import dumps

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PIPELINE_ROOT.parent / "powerflow-ui" / "docs" / "schema-normalized" / "examples"

# A three-frame slice of the ascent of 11 July/30kg_Set1, Side camera. Metres in the floor frame
# (+y up), pixels in s4_crop image space. leftWrist is occluded on the middle frame to show the
# null convention.
_ASCENT: dict[str, tuple[tuple[float, float, float], ...]] = {
    "leftAnkle": ((-0.11, 0.08, 1.92), (-0.11, 0.08, 1.92), (-0.11, 0.08, 1.92)),
    "rightAnkle": ((0.11, 0.08, 1.92), (0.11, 0.08, 1.92), (0.11, 0.08, 1.92)),
    "leftKnee": ((-0.13, 0.44, 1.88), (-0.13, 0.47, 1.87), (-0.13, 0.51, 1.86)),
    "rightKnee": ((0.13, 0.44, 1.88), (0.13, 0.47, 1.87), (0.13, 0.51, 1.86)),
    "leftHip": ((-0.10, 0.78, 1.90), (-0.10, 0.83, 1.89), (-0.10, 0.89, 1.88)),
    "rightHip": ((0.10, 0.78, 1.90), (0.10, 0.83, 1.89), (0.10, 0.89, 1.88)),
    "leftShoulder": ((-0.18, 1.24, 1.91), (-0.18, 1.29, 1.90), (-0.18, 1.35, 1.89)),
    "rightShoulder": ((0.18, 1.24, 1.91), (0.18, 1.29, 1.90), (0.18, 1.35, 1.89)),
    "leftElbow": ((-0.33, 1.20, 1.93), (-0.33, 1.25, 1.92), (-0.33, 1.31, 1.91)),
    "rightElbow": ((0.33, 1.20, 1.93), (0.33, 1.25, 1.92), (0.33, 1.31, 1.91)),
    "leftWrist": ((-0.41, 1.42, 1.94), (-0.41, 1.47, 1.93), (-0.41, 1.53, 1.92)),
    "rightWrist": ((0.41, 1.42, 1.94), (0.41, 1.47, 1.93), (0.41, 1.53, 1.92)),
    "leftClavicle": ((-0.06, 1.38, 1.91), (-0.06, 1.43, 1.90), (-0.06, 1.49, 1.89)),
    "rightClavicle": ((0.06, 1.38, 1.91), (0.06, 1.43, 1.90), (0.06, 1.49, 1.89)),
    "head": ((0.00, 1.63, 1.90), (0.00, 1.68, 1.89), (0.00, 1.74, 1.88)),
}

OCCLUDED = ("leftWrist", 1)  # joint, frame index


def _pixel(point: tuple[float, float, float]) -> tuple[int, int]:
    """A stand-in projection: metres to s4_crop pixels, origin top-left, y growing downward."""

    return (round(702 + point[0] * 700), round(1580 - point[1] * 700))


def _example_pose() -> PoseDocument:
    joints: dict[JointId, JointSeries] = {}
    for joint in HUMAN_SKELETON.joints:
        track = _ASCENT[joint]
        positions: list[tuple[float, float, float] | None] = list(track)
        pixels: list[tuple[int, int] | None] = [_pixel(point) for point in track]
        confidence = [0.97, 0.96, 0.97]
        if joint == OCCLUDED[0]:
            index = OCCLUDED[1]
            positions[index] = None
            pixels[index] = None
            confidence[index] = 0.0
        joints[joint] = JointSeries(
            position=tuple(positions),
            pixel_position=tuple(pixels),
            confidence=tuple(confidence),
        )
    return PoseDocument(
        capture_id="11 July/30kg_Set1/Side",
        role="side",
        stage="s4_crop",
        frames=Frames(count=3, fps=60.0, start_epoch_ms=1783749250270, t_ms=(0, 17, 33)),
        joints=joints,
    )


def _replace(path: Path, payload: object) -> None:
    """Regeneration overwrites; the write-once rule governs stage outputs, not published docs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(payload) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    if not EXAMPLES.is_dir():
        print(f"powerflow-ui examples directory not found: {EXAMPLES}", file=sys.stderr)
        return 1
    _replace(EXAMPLES / "skeleton.human-v2.json", skeleton_document())
    _replace(EXAMPLES / "pose.11-july-30kg-set1.side.json", _example_pose().to_document())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
