#!/usr/bin/env python
"""Quick look: run a promoted S5 model over one cropped capture and watch what it finds.

    uv run python scripts/predict_landmarks.py "10 September/50kg3" \
        --checkpoint ../data/models/s5_barbell/frcnn-v2-2026-10-19.pt --frames 120

Writes only `overlay.mp4`, under `../data/interim/s5_barbell_preview/<capture>/` by default:
every frame with its boxes and scores, through the same detector and the same drawing as the
S5 stage. It publishes nothing. Detections for S6 and S8 come from the stage
(`powerflow preprocess --barbell ... --barbell-checkpoint ...`), never from here, and nothing
writes into the labelled corpus under `annotations/` any more.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import av

from powerflow_pipeline.data.preprocess.barbell_model import TorchvisionBarbellDetector
from powerflow_pipeline.data.preprocess.config import BarbellDevice
from powerflow_pipeline.data.preprocess.tasks.barbell import render_overlay

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help='e.g. "10 September/50kg3"')
    parser.add_argument("--checkpoint", type=Path, required=True, help="a promoted model")
    parser.add_argument("--frames", type=int, default=None, help="detect only the first N")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", type=BarbellDevice, default=BarbellDevice.AUTO)
    parser.add_argument("--crop-root", type=Path, default=DATA_ROOT / "s4_crop_output")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    video = args.crop_root / args.capture / "rgb.mp4"
    if not video.is_file():
        raise SystemExit(f"no such capture: {video}")
    out = args.out or DATA_ROOT / "interim" / "s5_barbell_preview" / args.capture
    out.mkdir(parents=True, exist_ok=True)

    detector = TorchvisionBarbellDetector(
        args.checkpoint, device=args.device, record_threshold=args.score_threshold, batch_size=4
    )
    n_frames = args.frames or _frame_count(video)
    started = time.time()
    found = detector.detect(video, n_frames=n_frames)
    elapsed = time.time() - started
    render_overlay(video, found, out / "overlay.mp4")

    done = len(found["cap"])
    both = sum(1 for c, k in zip(found["cap"], found["collar"], strict=True) if c and k)
    print(
        f"{done} frames in {elapsed:.0f}s ({elapsed / max(1, done):.2f}s/frame) on "
        f"{detector.provenance.device}; both landmarks on {both}/{done}"
    )
    print(f"wrote {out / 'overlay.mp4'}")
    return 0


def _frame_count(video: Path) -> int:
    with av.open(str(video)) as container:
        return sum(1 for _ in container.demux(container.streams.video[0]) if _.size)


if __name__ == "__main__":
    sys.exit(main())
