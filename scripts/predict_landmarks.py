#!/usr/bin/env python
"""Run the trained detector over a capture and render what it found.

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --with torchvision --with pillow \
        python scripts/predict_landmarks.py "10 September/50kg3"

This is the end-to-end check: real video in, cap and collar boxes out, drawn on every frame so
the result can be watched rather than inferred from a metric. No pairing, no scale, no curves --
just whether the detector holds the two landmarks through a whole lift.

**One cap and one collar per frame, by construction.** The scene contains exactly one of each, so
the highest-scoring detection per class is taken and everything else dropped. That is why
precision barely matters for this task: spurious boxes cannot survive a top-1 rule.

Output, beside the capture's annotations:
    predictions.mp4     every frame with its boxes, confidences and a frame counter
    predictions.json    per-frame boxes and scores, for whatever consumes them next
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PIPELINE_ROOT.parent / "data"
STAGE_ROOT = DATA_ROOT / "s4_crop_output"
V3_ROOT = DATA_ROOT / "annotations" / "barbell" / "v3"
RUNS_ROOT = V3_ROOT / "runs"

CLASS_NAMES = {1: "cap", 2: "collar"}
CLASS_COLOURS = {1: (60, 220, 255), 2: (120, 255, 120)}


def latest_checkpoint() -> Path:
    runs = sorted(RUNS_ROOT.glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"no checkpoint under {RUNS_ROOT}; train one first")
    return runs[-1]


def draw(frame: np.ndarray, found: dict[str, Any], index: int, total: int) -> np.ndarray:
    canvas = frame.copy()
    for name, hit in found.items():
        if hit is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in hit["box"])
        colour = CLASS_COLOURS[1 if name == "cap" else 2]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 3)
        cv2.circle(canvas, ((x0 + x1) // 2, (y0 + y1) // 2), 3, colour, -1)
        label = f"{name} {hit['score']:.2f}"
        cv2.putText(canvas, label, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    missing = [n for n, h in found.items() if h is None]
    banner = f"frame {index}/{total - 1}"
    if missing:
        banner += f"   MISSING: {', '.join(missing)}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        banner,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if missing else (255, 255, 255),
        2,
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help='e.g. "10 September/50kg3"')
    parser.add_argument("--checkpoint", default=None, help="default: the most recent best.pt")
    parser.add_argument("--every", type=int, default=1, help="run on every Nth frame")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    video_path = STAGE_ROOT / args.capture / "rgb.mp4"
    if not video_path.exists():
        raise SystemExit(f"no such capture: {video_path}")

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_landmark_detector import build_model

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint()
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    name = blob.get("model", "fasterrcnn_resnet50_fpn_v2")
    device = args.device or (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"checkpoint {checkpoint_path.name} ({name}, epoch {blob.get('epoch')}) on {device}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    height, width = frames[0].shape[:2]
    print(f"{args.capture}: {len(frames)} frames at {width}x{height}")

    # Weights were trained at native resolution; inference must match or the boxes shift.
    model = build_model(
        name, min_size=min(width, height), max_size=max(width, height), weights=None
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()

    out_dir = Path(args.out) if args.out else V3_ROOT / args.capture
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_dir / "predictions.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit("could not open predictions.mp4 for writing")

    records: dict[int, dict[str, Any]] = {}
    started = time.time()
    missing_counts = {"cap": 0, "collar": 0}
    with torch.inference_mode():
        for index, frame in enumerate(frames):
            if index % args.every:
                writer.write(frame)
                continue
            tensor = (
                torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
                / 255
            )
            output = model([tensor.to(device)])[0]
            boxes = output["boxes"].cpu().tolist()
            scores = output["scores"].cpu().tolist()
            labels = output["labels"].cpu().tolist()
            found: dict[str, Any] = {"cap": None, "collar": None}
            for box, score, label in zip(boxes, scores, labels, strict=True):
                cls = CLASS_NAMES.get(label)
                if cls is None or score < args.score_threshold:
                    continue
                if found[cls] is None or score > found[cls]["score"]:
                    found[cls] = {"box": [round(v, 1) for v in box], "score": round(score, 4)}
            for cls, hit in found.items():
                if hit is None:
                    missing_counts[cls] += 1
            records[index] = found
            writer.write(draw(frame, found, index, len(frames)))
            if index % 50 == 0:
                print(f"  frame {index}/{len(frames)}", flush=True)
    writer.release()

    done = len(records)
    both = sum(1 for f in records.values() if f["cap"] and f["collar"])
    (out_dir / "predictions.json").write_text(
        json.dumps(
            {
                "capture": args.capture,
                "checkpoint": str(checkpoint_path),
                "model": name,
                "frame_size": [height, width],
                "fps": fps,
                "frames_run": done,
                "frames_with_both": both,
                "missing": missing_counts,
                "score_threshold": args.score_threshold,
                "rule": "highest-scoring detection per class, one cap and one collar per frame",
                "predictions": {str(k): v for k, v in sorted(records.items())},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    elapsed = time.time() - started
    print(
        f"\n{done} frames in {elapsed:.0f}s ({elapsed / max(1, done):.2f}s/frame)\n"
        f"both landmarks on {both}/{done} frames; "
        f"missing cap on {missing_counts['cap']}, collar on {missing_counts['collar']}"
    )
    print(f"wrote {out_dir / 'predictions.mp4'} and predictions.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
