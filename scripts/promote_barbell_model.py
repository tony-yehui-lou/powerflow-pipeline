#!/usr/bin/env python
"""Promote a training run's checkpoint to a named S5 model, once.

    uv run python scripts/promote_barbell_model.py \
        ../data/annotations/barbell/v3/runs/fasterrcnn_resnet50_fpn_v2-20261019-003709/best.pt \
        ../data/models/s5_barbell/frcnn-v2-2026-10-19.pt

Copies the weights and writes the card (`<model>.json`: architecture, classes, training run,
split, val metrics, sha256) beside them. S5 loads a checkpoint only through its card, and a
promoted model is never rewritten: promote a new name instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from powerflow_pipeline.data.preprocess.barbell_model import card_path, promote_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="a training run's best.pt")
    parser.add_argument("destination", type=Path, help="../data/models/s5_barbell/<name>.pt")
    args = parser.parse_args()

    card = promote_checkpoint(args.source, args.destination)
    print(f"promoted {args.source} -> {args.destination}")
    print(f"card {card_path(args.destination)}: {card.architecture}, epoch {card.training_epoch}")
    print(f"sha256 {card.sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
