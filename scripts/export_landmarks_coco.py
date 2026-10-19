#!/usr/bin/env python
"""Turn the hand-drawn landmark boxes into a COCO dataset, split by capture.

    uv run python scripts/export_landmarks_coco.py
    uv run python scripts/export_landmarks_coco.py --val 45kg1 --test 50kg3

Two categories, `cap` and `collar`. Every box becomes an annotation; `blurred` rides along as a
per-annotation attribute so a trainer can down-weight or exclude it and an evaluator can report
sharp and blurred frames separately.

**Splits are whole captures, never frames.** Two frames of the same lift from a tripod that never
moved are almost the same picture, so a frame-level split would report a score that says nothing
about a lift the model has not seen.

The declared bar type travels with each image, because metric scale depends on it: a cap and the
opposite collar are 1755 mm apart on a men's bar and 1660 mm on a women's.

Output under ../data/annotations/barbell/v3/coco/:
    train.json, val.json, test.json     COCO instances, absolute image paths
    summary.json                        what went where, and the class counts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = PIPELINE_ROOT.parent / "data" / "annotations" / "barbell" / "v3"
CATEGORIES = [{"id": 1, "name": "cap"}, {"id": 2, "name": "collar"}]
CATEGORY_IDS = {c["name"]: c["id"] for c in CATEGORIES}


def load_captures() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(V3_ROOT.glob("**/landmarks.json")):
        doc = json.loads(path.read_text())
        doc["_dir"] = path.parent
        out[doc["capture"]] = doc
    return out


def build(captures: list[str], docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = ann_id = 1
    for capture in captures:
        doc = docs[capture]
        height, width = doc["frame_size"]
        for frame_str, entries in sorted(doc["boxes"].items(), key=lambda kv: int(kv[0])):
            frame = int(frame_str)
            jpg = doc["_dir"] / "frames" / f"{frame:06d}.jpg"
            if not jpg.exists():
                continue
            images.append(
                {
                    "id": image_id,
                    "file_name": str(jpg),
                    "width": width,
                    "height": height,
                    "capture": capture,
                    "frame": frame,
                    "barbell_type": doc.get("barbell_type"),
                    "cap_to_opposite_collar_mm": doc.get("cap_to_opposite_collar_mm"),
                }
            )
            for entry in entries:
                x0, y0, x1, y1 = entry["box"]
                w, h = x1 - x0, y1 - y0
                if w <= 0 or h <= 0:
                    continue
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": CATEGORY_IDS[entry["class"]],
                        "bbox": [x0, y0, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "blurred": bool(entry.get("blurred")),
                    }
                )
                ann_id += 1
            image_id += 1
    return {
        "info": {
            "description": "PowerFlow barbell landmarks (cap, collar)",
            "created": datetime.now(UTC).isoformat(),
            "guide": "docs/specs/preprocessing/S7-barbell-annotation-guide.md",
        },
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }


def describe(split: dict[str, Any]) -> dict[str, Any]:
    per_class = Counter(
        next(c["name"] for c in CATEGORIES if c["id"] == a["category_id"])
        for a in split["annotations"]
    )
    return {
        "captures": sorted({i["capture"] for i in split["images"]}),
        "images": len(split["images"]),
        "annotations": len(split["annotations"]),
        "per_class": dict(per_class),
        "blurred": sum(1 for a in split["annotations"] if a["blurred"]),
        "images_without_annotations": sum(
            1
            for i in split["images"]
            if not any(a["image_id"] == i["id"] for a in split["annotations"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val", nargs="*", default=["45kg1"], help="captures held out for validation"
    )
    parser.add_argument("--test", nargs="*", default=["50kg3"], help="captures held out for test")
    parser.add_argument("--out", default=str(V3_ROOT / "coco"))
    args = parser.parse_args()

    docs = load_captures()
    if not docs:
        raise SystemExit(f"no landmark annotations under {V3_ROOT}")

    def match(names: list[str]) -> list[str]:
        chosen = []
        for name in names:
            hits = [c for c in docs if c.endswith(name) or c == name]
            if not hits:
                raise SystemExit(f"no capture matching {name!r}; have: {sorted(docs)}")
            chosen.extend(hits)
        return chosen

    val = match(args.val)
    test = match(args.test)
    train = [c for c in docs if c not in set(val) | set(test)]
    if not train:
        raise SystemExit("every capture is held out; nothing left to train on")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, captures in (("train", train), ("val", val), ("test", test)):
        split = build(captures, docs)
        (out_dir / f"{name}.json").write_text(json.dumps(split) + "\n", encoding="utf-8")
        summary[name] = describe(split)
        d = summary[name]
        print(
            f"{name:<5} {d['images']:>4} images  {d['annotations']:>4} boxes  "
            f"{d['per_class']}  blurred={d['blurred']}  captures={d['captures']}"
        )

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "created": datetime.now(UTC).isoformat(),
                "split_rule": "whole captures, never frames",
                "splits": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir}/train.json, val.json, test.json, summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
