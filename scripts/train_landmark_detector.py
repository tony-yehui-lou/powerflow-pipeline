#!/usr/bin/env python
"""Fine-tune a torchvision detector on the cap/collar landmark boxes.

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --with torchvision --with pillow \
        python scripts/train_landmark_detector.py --model fasterrcnn_resnet50_fpn_v2

    ... --list-models                      # what you can swap to
    ... --model retinanet_resnet50_fpn_v2 --epochs 30 --lr 5e-4

Every architecture in `MODELS` is swapped with one flag. They all start from COCO-pretrained
weights, because 200 training images is far too few to learn detection from scratch, and only the
classification head is replaced to predict background / cap / collar.

**Resolution matters more than architecture here.** A cap is ~30x40 px at native resolution, and
torchvision's default transform would resize the 1416 px images down to a 800 px short side,
shrinking the cap to ~17 px. So `--min-size` defaults to native. Lower it only if you run out of
memory, and expect small-object accuracy to fall when you do.

Evaluation is deliberately not COCO mAP: it is precision/recall at IoU 0.5 plus **median centre
error in pixels**, reported separately for sharp and blurred boxes. Centre error is what the bar
path actually consumes, and the sharp/blurred split is the only honest way to quote a number when
the fast frames are the ones that matter.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
COCO_ROOT = PIPELINE_ROOT.parent / "data" / "annotations" / "barbell" / "v3" / "coco"
RUNS_ROOT = PIPELINE_ROOT.parent / "data" / "annotations" / "barbell" / "v3" / "runs"

# One builder for training and inference: S5 loads the weights this script saves through the
# same registry, so the two can never construct different models.
from powerflow_pipeline.data.preprocess.barbell_model import (  # noqa: E402
    CLASS_NAMES,
    MODELS,
    build_model,
)


class LandmarkDataset:
    """COCO instances, as torchvision detection wants them."""

    def __init__(self, coco_path: Path, hflip: float = 0.0, jitter: float = 0.0) -> None:
        doc = json.loads(coco_path.read_text())
        self.images = doc["images"]
        self.by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in doc["annotations"]:
            self.by_image[ann["image_id"]].append(ann)
        self.hflip = hflip
        self.jitter = jitter

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import torch
        from PIL import Image
        from torchvision.transforms import functional as F

        meta = self.images[index]
        image = Image.open(meta["file_name"]).convert("RGB")
        anns = self.by_image.get(meta["id"], [])
        boxes = [
            [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
            for a in anns
        ]
        labels = [a["category_id"] for a in anns]
        blurred = [1 if a["blurred"] else 0 for a in anns]

        tensor = F.to_tensor(image)
        if self.jitter and random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.jitter, self.jitter)
            tensor = (tensor * factor).clamp(0, 1)
        if self.hflip and random.random() < self.hflip:
            # A cap stays a cap when mirrored. Flipping is on by default precisely to stop the
            # model learning this rig's "cap is image-left" habit, which is not a property of
            # barbells.
            tensor = tensor.flip(-1)
            width = tensor.shape[-1]
            boxes = [[width - b[2], b[1], width - b[0], b[3]] for b in boxes]

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "blurred": torch.as_tensor(blurred, dtype=torch.int64),
            "image_id": torch.as_tensor([meta["id"]]),
        }
        return tensor, target


def collate(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[list[Any], list[dict[str, Any]]]:
    return [b[0] for b in batch], [b[1] for b in batch]


def iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def evaluate(
    model: Any, dataset: LandmarkDataset, device: str, score_threshold: float
) -> dict[str, Any]:
    """Precision/recall at IoU 0.5 and centre error, split by class and by blur."""

    import torch

    model.eval()
    matched: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            output = model([image.to(device)])[0]
            keep = output["scores"] >= score_threshold
            pred_boxes = output["boxes"][keep].cpu().tolist()
            pred_labels = output["labels"][keep].cpu().tolist()
            counts["predictions"] += len(pred_boxes)

            gt_boxes = target["boxes"].tolist()
            gt_labels = target["labels"].tolist()
            gt_blur = target["blurred"].tolist()
            used: set[int] = set()
            for gi, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels, strict=True)):
                name = CLASS_NAMES.get(glabel, str(glabel))
                counts[f"gt_{name}"] += 1
                best, best_iou = -1, 0.5
                for pi, (pbox, plabel) in enumerate(zip(pred_boxes, pred_labels, strict=True)):
                    if pi in used or plabel != glabel:
                        continue
                    value = iou(gbox, pbox)
                    if value >= best_iou:
                        best, best_iou = pi, value
                if best >= 0:
                    used.add(best)
                    counts[f"hit_{name}"] += 1
                    pbox = pred_boxes[best]
                    error = (
                        ((pbox[0] + pbox[2]) / 2 - (gbox[0] + gbox[2]) / 2) ** 2
                        + ((pbox[1] + pbox[3]) / 2 - (gbox[1] + gbox[3]) / 2) ** 2
                    ) ** 0.5
                    matched[name].append(error)
                    matched["blurred" if gt_blur[gi] else "sharp"].append(error)

    def median(values: list[float]) -> float | None:
        if not values:
            return None
        values = sorted(values)
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    result: dict[str, Any] = {"predictions": counts["predictions"]}
    total_hits = 0
    total_gt = 0
    for name in CLASS_NAMES.values():
        gt = counts[f"gt_{name}"]
        hit = counts[f"hit_{name}"]
        total_hits += hit
        total_gt += gt
        result[name] = {
            "gt": gt,
            "recall": round(hit / gt, 4) if gt else None,
            "median_centre_px": None
            if median(matched[name]) is None
            else round(median(matched[name]), 2),
        }
    result["recall"] = round(total_hits / total_gt, 4) if total_gt else None
    result["precision"] = (
        round(total_hits / counts["predictions"], 4) if counts["predictions"] else None
    )
    for kind in ("sharp", "blurred"):
        value = median(matched[kind])
        result[f"median_centre_px_{kind}"] = None if value is None else round(value, 2)
        result[f"n_{kind}"] = len(matched[kind])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn_v2")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="train from random init; only for ablation -- 200 images cannot learn detection cold",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--min-size", type=int, default=1416, help="native short side; lower only for memory"
    )
    parser.add_argument("--max-size", type=int, default=1792)
    parser.add_argument("--hflip", type=float, default=0.5)
    parser.add_argument("--jitter", type=float, default=0.2, help="brightness jitter, 0 to disable")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--coco", default=str(COCO_ROOT))
    parser.add_argument("--out", default=None, help="run directory (default: runs/<model>-<time>)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if args.list_models:
        print("models (swap with --model):")
        for name, spec in MODELS.items():
            print(f"  {name:<40} family={spec.family}")
        return 0

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    coco = Path(args.coco)
    train_set = LandmarkDataset(coco / "train.json", hflip=args.hflip, jitter=args.jitter)
    val_set = LandmarkDataset(coco / "val.json")
    print(f"model      {args.model} on {device}")
    print(f"data       {len(train_set)} train / {len(val_set)} val images")
    print(f"resolution min_size={args.min_size} max_size={args.max_size}")

    weights = None if args.no_pretrained else "DEFAULT"
    model = build_model(args.model, args.min_size, args.max_size, weights).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.workers,
    )

    out_dir = Path(args.out or RUNS_ROOT / f"{args.model}-{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        total = 0.0
        for images, targets in loader:
            images = [i.to(device) for i in images]
            targets = [
                {k: v.to(device) for k, v in t.items() if k in ("boxes", "labels")} for t in targets
            ]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss)
        scheduler.step()
        metrics = evaluate(model, val_set, device, args.score_threshold)
        record = {
            "epoch": epoch,
            "train_loss": round(total / max(1, len(loader)), 4),
            "seconds": round(time.time() - started, 1),
            **metrics,
        }
        history.append(record)
        print(
            f"epoch {epoch:>3}  loss {record['train_loss']:.3f}  "
            f"recall {metrics['recall']}  precision {metrics['precision']}  "
            f"centre px sharp={metrics['median_centre_px_sharp']} "
            f"blurred={metrics['median_centre_px_blurred']}  ({record['seconds']}s)"
        )
        score = metrics["recall"] or 0.0
        if score > best:
            best = score
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                },
                out_dir / "best.pt",
            )
        (out_dir / "history.json").write_text(
            json.dumps({"args": vars(args), "history": history}, indent=2) + "\n", encoding="utf-8"
        )

    print(f"\nbest val recall {best}; checkpoint and history in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
