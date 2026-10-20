#!/usr/bin/env python
"""Pass B's automatable half: clean Pass A's masks, derive attributes, flag what a human must see.

Reads one capture's SAM 2 masks, fits an ellipse per plate per frame, derives every attribute the
annotation guide asks for except `stack_index`, runs the guide's §5 quality checks, and writes a
per-capture `plates.json`. What it cannot decide it flags, so the human pass is a review of a
handful of frames instead of all of them.

    uv run python scripts/plate_labels_qc.py "22 August/Snch/107kgSnch1"
    uv run python scripts/plate_labels_qc.py "22 August/Snch/107kgSnch1" --stack-index 1:0,2:0

`sleeve` comes from the median x of each track, `occluded` from the area-to-median ratio,
`truncated` from a bounding box on the frame edge, `blurred` from Laplacian variance inside the
mask, `edge_on` from the axis ratio. `stack_index` is the one a human must supply -- the loading
does not change mid-lift, so it is one decision per capture, not per frame.

See docs/specs/preprocessing/S5-barbell-annotation-guide.md §3 (Pass B) and §5.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PIPELINE_ROOT.parent / "data"
STAGE_ROOT = DATA_ROOT / "s4_crop_output"
OUT_ROOT = DATA_ROOT / "annotations" / "barbell" / "v1"

# Area relative to a track's own median. Below PARTIAL the plate is being cut into; below HEAVY
# an ellipse fitted to what is left has a centre no one should trust.
OCCLUSION_PARTIAL = 0.85
OCCLUSION_HEAVY = 0.50
# Laplacian variance relative to the capture's median, inside the mask.
BLUR_RATIO = 0.5
# Cleanup that discards less than this is a speck, not a lost piece of plate.
SPECK_PX = 20
SPECK_SHARE = 0.02
# An ellipse this squashed is a rim seen edge-on, not a plate face.
EDGE_ON_AXIS_RATIO = 0.35
# Flags for the §5 smoothness checks.
AXIS_OUTLIER = 0.35
CENTRE_JUMP_PX = 12.0
SEPARATION_OUTLIER = 0.05


def load_frames(video_path: Path, scale: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(bgr)
    cap.release()
    return frames


def measure(mask: np.ndarray, gray: np.ndarray) -> dict[str, Any] | None:
    """One plate on one frame: largest contour, fitted ellipse, and the numbers attributes need."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Keep the largest contour and drop the rest: SAM 2 leaves the odd few-pixel speck, and a
    # speck is never a plate.
    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 5:
        return None
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [biggest], -1, 1, thickness=cv2.FILLED)
    discarded = int(mask.sum()) - int(cleaned.sum())
    (cx, cy), (minor, major), angle = cv2.fitEllipse(biggest)
    x, y, w, h = cv2.boundingRect(biggest)
    roi = gray[y : y + h, x : x + w]
    blur = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0
    return {
        "ellipse": {
            "cx": round(float(cx), 2),
            "cy": round(float(cy), 2),
            "major_px": round(float(major), 2),
            "minor_px": round(float(minor), 2),
            "angle_deg": round(float(angle), 2),
        },
        "bbox": [int(x), int(y), int(w), int(h)],
        "area_px": int(cleaned.sum()),
        "n_contours_before_cleanup": len(contours),
        "discarded_px": max(discarded, 0),
        "laplacian_var": round(blur, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help='e.g. "22 August/Snch/107kgSnch1"')
    parser.add_argument(
        "--stack-index",
        default=None,
        help='human input, e.g. "1:0,2:0" -- 0 is the plate against the bar shoulder',
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="also write review/ images for every flagged frame",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing plates.json (it is derived from the masks, so this is cheap)",
    )
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.capture
    run_path = out_dir / "sam2_run.json"
    if not run_path.exists():
        source = STAGE_ROOT / args.capture / "rgb.mp4"
        if not source.exists():
            raise SystemExit(f"no such capture: {source}")
        raise SystemExit(
            f"no Pass A output for this capture yet ({run_path} is missing).\n"
            "Run Pass A first:\n"
            "  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --with transformers --with torchvision \\\n"
            f'      python scripts/sam2_plate_label.py "{args.capture}"'
        )
    plates_path = out_dir / "plates.json"
    if plates_path.exists() and not args.force:
        # Unlike the masks, this file is derived: same masks plus the same --stack-index give the
        # same document every time, and supplying stack_index on a second run is ordinary use.
        # So it is replaceable -- but only when asked, so nobody loses one by accident.
        raise SystemExit(
            f"{plates_path} already exists -- re-run with --force to replace it "
            "(it is derived from the masks, so nothing captured is lost)"
        )

    run = json.loads(run_path.read_text())
    scale = float(run["decode_scale"])
    object_ids = sorted(run.get("tracked_object_ids") or [int(k) for k in run["prompts"]])
    # The subject's pair is recorded, never inferred: which bar the capture is about is not a
    # visual property, so nothing here tries to work it out from the pixels.
    subject_ids = [int(o) for o in run.get("subject_object_ids", [1, 2]) if int(o) in object_ids]
    stack_index: dict[int, int] = {}
    if args.stack_index:
        for pair in args.stack_index.split(","):
            obj, idx = pair.split(":")
            stack_index[int(obj)] = int(idx)

    frames = load_frames(STAGE_ROOT / args.capture / "rgb.mp4", scale)
    if not frames:
        raise SystemExit(f"could not decode {STAGE_ROOT / args.capture / 'rgb.mp4'}")
    height, width = frames[0].shape[:2]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # Pass 1: measure every plate on every frame.
    per_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for mask_path in sorted((out_dir / "masks").glob("*.png")):
        idx = int(mask_path.stem)
        if idx >= len(frames):
            continue
        indexed = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        per_frame[idx] = {}
        for obj_id in object_ids:
            m = (indexed == obj_id).astype(np.uint8)
            got = measure(m, grays[idx])
            if got:
                per_frame[idx][obj_id] = got

    # Pass 2: per-track statistics, so attributes are relative to this capture, not a constant.
    stats: dict[int, dict[str, float]] = {}
    for obj_id in object_ids:
        areas = [f[obj_id]["area_px"] for f in per_frame.values() if obj_id in f]
        majors = [f[obj_id]["ellipse"]["major_px"] for f in per_frame.values() if obj_id in f]
        xs = [f[obj_id]["ellipse"]["cx"] for f in per_frame.values() if obj_id in f]
        blurs = [f[obj_id]["laplacian_var"] for f in per_frame.values() if obj_id in f]
        stats[obj_id] = {
            "median_area": float(np.median(areas)) if areas else 0.0,
            "median_major": float(np.median(majors)) if majors else 0.0,
            "median_x": float(np.median(xs)) if xs else 0.0,
            "median_blur": float(np.median(blurs)) if blurs else 0.0,
        }
    # `sleeve` is decided once per track from its median x, not per frame: the two plates never
    # swap sides, and a per-frame decision would flip them on any frame where one is missing. It
    # is meaningful only for the subject's bar; every other plate is `none`.
    sleeve = dict.fromkeys(object_ids, "none")
    if len(subject_ids) == 2:
        by_x = sorted(subject_ids, key=lambda o: stats[o]["median_x"])
        sleeve[by_x[0]] = "left"
        sleeve[by_x[-1]] = "right"

    # Pass 3: attributes, per-frame flags, and the §5 checks.
    records: list[dict[str, Any]] = []
    flagged: dict[int, list[str]] = {}
    previous: dict[int, tuple[float, float]] = {}
    separations: list[float] = []
    for idx in sorted(per_frame):
        centres: dict[int, tuple[float, float]] = {}
        for obj_id in object_ids:
            got = per_frame[idx].get(obj_id)
            reasons: list[str] = []
            if got is None:
                flagged.setdefault(idx, []).append(f"obj{obj_id}: no mask")
                records.append(
                    {
                        "frame": idx,
                        "object_id": obj_id,
                        "present": False,
                        "subject": obj_id in subject_ids,
                        "sleeve": sleeve[obj_id],
                    }
                )
                continue
            e = got["ellipse"]
            centre = (e["cx"], e["cy"])
            centres[obj_id] = centre
            area_ratio = got["area_px"] / (stats[obj_id]["median_area"] or 1)
            if area_ratio < OCCLUSION_HEAVY:
                occluded = "heavy"
            elif area_ratio < OCCLUSION_PARTIAL:
                occluded = "partial"
            else:
                occluded = "none"
            x, y, w, h = got["bbox"]
            truncated = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
            blurred = got["laplacian_var"] < BLUR_RATIO * (stats[obj_id]["median_blur"] or 1)
            axis_ratio = e["minor_px"] / (e["major_px"] or 1)

            if occluded != "none":
                reasons.append(f"obj{obj_id}: {occluded} occlusion (area {area_ratio:.0%})")
            if abs(e["major_px"] - stats[obj_id]["median_major"]) > AXIS_OUTLIER * (
                stats[obj_id]["median_major"] or 1
            ):
                reasons.append(f"obj{obj_id}: major axis {e['major_px']:.0f}px off median")
            if obj_id in previous:
                jump = float(np.hypot(*(np.array(centre) - np.array(previous[obj_id]))))
                if jump > CENTRE_JUMP_PX:
                    reasons.append(f"obj{obj_id}: centre jumped {jump:.0f}px")
            # Only worth a human's time if cleanup discarded something that could plausibly be
            # plate. SAM 2 routinely leaves a speck of a few pixels; flagging that trains the
            # reviewer to ignore flags.
            if (
                got["discarded_px"] > SPECK_PX
                and got["discarded_px"] > SPECK_SHARE * got["area_px"]
            ):
                reasons.append(f"obj{obj_id}: cleanup discarded {got['discarded_px']}px")
            previous[obj_id] = centre
            if reasons:
                flagged.setdefault(idx, []).extend(reasons)

            records.append(
                {
                    "frame": idx,
                    "object_id": obj_id,
                    "present": True,
                    **got,
                    "area_ratio": round(area_ratio, 3),
                    "subject": obj_id in subject_ids,
                    "sleeve": sleeve[obj_id],
                    "stack_index": stack_index.get(obj_id) if obj_id in subject_ids else None,
                    "occluded": occluded,
                    "truncated": bool(truncated),
                    "blurred": bool(blurred),
                    "edge_on": bool(axis_ratio < EDGE_ON_AXIS_RATIO),
                }
            )
        if len(subject_ids) == 2 and all(o in centres for o in subject_ids):
            a, b = (np.array(centres[o]) for o in subject_ids)
            separations.append(float(np.hypot(*(a - b))))

    sep = np.array(separations) if separations else np.array([np.nan])
    sep_cv = float(np.nanstd(sep) / (np.nanmean(sep) or 1))
    frames_with_masks = len(per_frame)
    document = {
        "capture": args.capture,
        "schema": "plate-labels/1",
        "source_masks": str((out_dir / "masks").relative_to(DATA_ROOT)),
        "decode_scale": scale,
        "frame_size": [height, width],
        "n_frames": len(frames),
        "frames_with_masks": frames_with_masks,
        "subject_object_ids": subject_ids,
        "objects": {
            str(o): {
                "subject": o in subject_ids,
                "sleeve": sleeve[o],
                "stack_index": stack_index.get(o) if o in subject_ids else None,
                "median_area_px": round(stats[o]["median_area"], 1),
                "median_major_px": round(stats[o]["median_major"], 2),
            }
            for o in object_ids
        },
        "quality": {
            "plate_separation_px_mean": round(float(np.nanmean(sep)), 2),
            "plate_separation_cv": round(sep_cv, 4),
            "plate_separation_pass": bool(sep_cv <= SEPARATION_OUTLIER),
            "flagged_frames": len(flagged),
            "flagged_share": round(len(flagged) / max(frames_with_masks, 1), 3),
        },
        "stack_index_supplied": bool(stack_index),
        "written_at": datetime.now(UTC).isoformat(),
        "guide": "docs/specs/preprocessing/S5-barbell-annotation-guide.md",
        "annotations": records,
    }
    replacing = plates_path.exists()
    plates_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    if replacing:
        print(f"replaced the existing {plates_path.name}")

    print(f"capture           {args.capture}")
    print(f"frames            {frames_with_masks}/{len(frames)} carry masks")
    for o in object_ids:
        role = "SUBJECT" if o in subject_ids else "other  "
        print(
            f"  object {o} {role} sleeve={sleeve[o]} stack_index={stack_index.get(o)} "
            f"median area={stats[o]['median_area']:.0f}px major={stats[o]['median_major']:.1f}px"
        )
    verdict = "PASS" if sep_cv <= SEPARATION_OUTLIER else "FAIL"
    print(
        f"bar-length check  subject plates {np.nanmean(sep):.1f}px apart, "
        f"coefficient of variation {sep_cv:.1%} -> {verdict}"
    )
    if len(subject_ids) == 2:
        mm_per_px = 450.0 / (stats[subject_ids[0]]["median_major"] or 1)
        cys = [
            r["ellipse"]["cy"]
            for r in records
            if r.get("present") and r["object_id"] == subject_ids[0]
        ]
        if cys:
            excursion_m = (max(cys) - min(cys)) * mm_per_px / 1000.0
            note = "" if 0.8 <= excursion_m <= 2.0 else "   <-- outside the 1.06-1.72 m seen so far"
            print(f"lift excursion    {excursion_m:.2f} m (plate ruler, 450 mm){note}")
    print(f"flagged frames    {len(flagged)}/{frames_with_masks} need a human eye")
    for idx in sorted(flagged)[:12]:
        print(f"  frame {idx:>4}: {'; '.join(flagged[idx])}")
    if len(flagged) > 12:
        print(f"  ... and {len(flagged) - 12} more (all listed in plates.json)")
    if not stack_index:
        print("\nstack_index not supplied -- re-run with --stack-index once you have looked at")
        print("the overlay and know which plate sits against each bar shoulder.")

    if args.contact_sheet and flagged:
        review = out_dir / "review"
        review.mkdir(exist_ok=True)
        for idx in sorted(flagged):
            frame = frames[idx].copy()
            for obj_id in object_ids:
                got = per_frame[idx].get(obj_id)
                if not got:
                    continue
                e = got["ellipse"]
                cv2.ellipse(
                    frame,
                    ((e["cx"], e["cy"]), (e["minor_px"], e["major_px"]), e["angle_deg"]),
                    (60, 200, 255) if obj_id == object_ids[0] else (255, 120, 60),
                    2,
                )
            cv2.putText(
                frame,
                "; ".join(flagged[idx])[:90],
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            cv2.imwrite(str(review / f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"\nwrote {len(flagged)} review images to {review}")
    print(f"\nwrote {plates_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
