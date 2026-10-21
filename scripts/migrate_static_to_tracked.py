#!/usr/bin/env python
"""Upgrade captures labelled under the old "static distractors" method to tracked plates.

The earlier method clicked the non-subject plates on a single frame and held that annotation
constant for the whole clip. That assumption is false -- measured in `38kgSnch1` and `60kgSnch2`,
background plates move, get carried off, and new ones appear -- so those plates have to be
propagated like the subject's.

Nobody re-clicks anything: `static/static.json` recorded every click, so this replays them through
SAM 2 with propagation and merges the result into the capture's existing `masks/`.

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --with transformers --with torchvision \
        python scripts/migrate_static_to_tracked.py --apply

Subject plates keep ids 1-2 untouched; the former static plates become ids 3..N. The merge is
reversible by dropping ids >= 3, so the subject annotation can never be damaged by a bad run.

See docs/specs/preprocessing/S5-barbell-annotation-guide.md §0.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PIPELINE_ROOT.parent / "data"
STAGE_ROOT = DATA_ROOT / "s4_crop_output"
OUT_ROOT = DATA_ROOT / "annotations" / "barbell" / "v1"
SUBJECT_IDS = (1, 2)


def plan(static_json: Path) -> dict[str, object]:
    capture_dir = static_json.parent.parent
    capture = str(capture_dir.relative_to(OUT_ROOT))
    meta = json.loads(static_json.read_text())
    run_path = capture_dir / "sam2_run.json"
    if not run_path.exists():
        return {"capture": capture, "status": "no Pass A output; skipped"}
    run = json.loads(run_path.read_text())
    if run.get("migrated_at"):
        return {"capture": capture, "status": "already migrated"}
    prompts = meta.get("prompts", {})
    if not prompts:
        return {"capture": capture, "status": "no clicks recorded; skipped"}
    return {
        "capture": capture,
        "status": "would migrate",
        "n_other_plates": len(prompts),
        "prompt_frame": meta.get("prompt_frame", 0),
        "new_ids": list(range(3, 3 + len(prompts))),
    }


def migrate(static_json: Path, model: object, processor: object, device: str) -> dict[str, object]:
    from sam2_plate_label import load_frames, masks_to_indexed, segment_frame

    capture_dir = static_json.parent.parent
    capture = str(capture_dir.relative_to(OUT_ROOT))
    meta = json.loads(static_json.read_text())
    run_path = capture_dir / "sam2_run.json"
    run = json.loads(run_path.read_text())
    prompts = meta["prompts"]
    prompt_frame = int(meta.get("prompt_frame", 0))
    scale = float(meta["decode_scale"])

    frames, _ = load_frames(STAGE_ROOT / capture / "rgb.mp4", scale)
    height, width = frames[0].shape[:2]
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    # Replay the recorded clicks, renumbered to sit after the subject's pair.
    remap = {int(old): 3 + i for i, old in enumerate(sorted(prompts, key=int))}
    points = {remap[int(k)]: [tuple(p) for p in v["points"]] for k, v in prompts.items()}
    labels = {remap[int(k)]: list(v["labels"]) for k, v in prompts.items()}

    _, session = segment_frame(
        model, processor, rgb, prompt_frame, points, labels, device, height, width
    )
    merged = 0
    for reverse in (False, True):
        if reverse and prompt_frame == 0:
            continue
        for output in model.propagate_in_video_iterator(session, reverse=reverse):  # type: ignore[attr-defined]
            masks = processor.post_process_masks(  # type: ignore[attr-defined]
                [output.pred_masks], original_sizes=[[height, width]], binarize=True
            )[0]
            per_object = {
                obj_id: masks[slot, 0].cpu().numpy().astype(bool)
                for slot, obj_id in enumerate(session.obj_ids)
            }
            mask_path = capture_dir / "masks" / f"{output.frame_idx:06d}.png"
            if not mask_path.exists():
                continue
            existing = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            # Subject plates win every overlap: their annotation is the one nothing may damage.
            addition = masks_to_indexed(per_object, (height, width))
            keep = (existing == 0) & (addition > 0)
            existing[keep] = addition[keep]
            cv2.imwrite(str(mask_path), existing)
            merged += 1
            # \r only works on a terminal; through a pipe it writes one line per frame and
            # buries the result under hundreds of them.
            if sys.stdout.isatty():
                print(f"  {capture}: frame {output.frame_idx}   ", end="\r", flush=True)
            elif merged % 50 == 0:
                print(f"  {capture}: {merged} frames merged", flush=True)
    if sys.stdout.isatty():
        print()

    run["tracked_object_ids"] = sorted(set(SUBJECT_IDS) | set(remap.values()))
    run["subject_object_ids"] = list(SUBJECT_IDS)
    run["prompts"] = {
        **run.get("prompts", {}),
        **{str(remap[int(k)]): v for k, v in prompts.items()},
    }
    run["migrated_at"] = datetime.now(UTC).isoformat()
    run["migrated_from"] = "static one-frame annotation, replayed with propagation"
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return {
        "capture": capture,
        "status": "migrated",
        "frames_merged": merged,
        "new_ids": sorted(remap.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", help="one capture; default is every one found")
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    found = (
        [OUT_ROOT / args.capture / "static" / "static.json"]
        if args.capture
        else sorted(OUT_ROOT.glob("**/static/static.json"))
    )
    found = [f for f in found if f.exists()]
    if not found:
        raise SystemExit("no captures with a static annotation to migrate")

    plans = [plan(f) for f in found]
    for p in plans:
        extra = f"  {p.get('n_other_plates', '')} plates -> ids {p.get('new_ids', '')}"
        print(f"{p['capture']}: {p['status']}{extra if 'n_other_plates' in p else ''}")
    todo = [f for f, p in zip(found, plans, strict=True) if p["status"] == "would migrate"]
    if not args.apply:
        print(f"\nDRY RUN -- {len(todo)} capture(s) would be migrated; pass --apply to write")
        return 0
    if not todo:
        print("\nnothing to do")
        return 0

    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nloading {args.model} on {device} ...")
    model = Sam2VideoModel.from_pretrained(args.model).to(device).eval()
    processor = Sam2VideoProcessor.from_pretrained(args.model)
    for f in todo:
        r = migrate(f, model, processor, device)
        print(f"{r['capture']}: {r['status']}, {r['frames_merged']} frames, ids {r['new_ids']}")
    print(f"\nmigrated {len(todo)} capture(s); re-run plate_labels_qc.py --force on each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
