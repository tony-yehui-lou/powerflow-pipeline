#!/usr/bin/env python
"""Rewrite the 22 August `video:` floor-region blocks to the form S3 Retilt can read.

Every one of the 29 files ships the same block, and it is wrong three ways
(docs/superpowers/specs/2026-09-18-single-camera-oblique-capture-design.md §4):

    video:
      front_floor_region_bottom_left_in_pixels: (0.65204, 0.66118)   # wrong prefix
      floor_region_top_right_in_pixels: (0.79080, 0.86117)           # y inverted

and, once both are corrected, the rectangle still lands on a PA speaker cabinet in the near
foreground rather than the floor -- fitted RMS 7-10 cm against S3's 2 cm ceiling.

This writes the measured replacement: the black mat between the camera and the platform,
which fitted cleanly on 27 of 29 captures (RMS median 0.79 cm, camera height 1.13 m).
`Snch/55kgSnch1` gets a rectangle clear of the spectator's head that occludes the default
one there. `Snch/50kgSnch1` has no rectangle that clears the gate at all (best of ten trial
rectangles: 2.96 cm) and is expected to be rejected; it is left on the default so the
rejection is visible rather than papered over.

Prints what it would change and exits. Pass `--apply` to write, which first copies every
file it touches to `<file>.bkp` (git-ignored).

    uv run python scripts/fix_august_floor_regions.py
    uv run python scripts/fix_august_floor_regions.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

RAW_AUGUST = Path(__file__).resolve().parents[2] / "data" / "raw" / "22 August"

# Measured, not guessed -- see the spec's §4.2 and §4.3 for the numbers behind each.
DEFAULT_REGION = ("(0.20, 0.78)", "(0.55, 0.68)")
PER_CAPTURE_REGION = {"Snch/55kgSnch1": ("(0.38, 0.78)", "(0.62, 0.68)")}

OLD_BOTTOM_LEFT = "front_floor_region_bottom_left_in_pixels"
OLD_TOP_RIGHT = "floor_region_top_right_in_pixels"
NEW_BOTTOM_LEFT = "floor_region_bottom_left_in_pixels"
NEW_TOP_RIGHT = "floor_region_top_right_in_pixels"


def rewrite(body: str, bottom_left: str, top_right: str) -> str:
    """Replace the two region lines in place, leaving every other line untouched."""

    out: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{OLD_BOTTOM_LEFT}:") or stripped.startswith(f"{NEW_BOTTOM_LEFT}:"):
            out.append(f"  {NEW_BOTTOM_LEFT}: {bottom_left}")
        elif stripped.startswith(f"{OLD_TOP_RIGHT}:"):
            out.append(f"  {NEW_TOP_RIGHT}: {top_right}")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the files (default: preview)")
    args = parser.parse_args()

    if not RAW_AUGUST.is_dir():
        print(f"not found: {RAW_AUGUST}", file=sys.stderr)
        return 1

    touched = 0
    for path in sorted(RAW_AUGUST.glob("*/*/metadata.yaml")):
        capture = str(path.parent.relative_to(RAW_AUGUST))
        bottom_left, top_right = PER_CAPTURE_REGION.get(capture, DEFAULT_REGION)
        before = path.read_text()
        after = rewrite(before, bottom_left, top_right)
        if after == before:
            continue

        touched += 1
        print(f"{capture}: {NEW_BOTTOM_LEFT}: {bottom_left} / {NEW_TOP_RIGHT}: {top_right}")
        if args.apply:
            shutil.copy2(path, path.with_suffix(".yaml.bkp"))
            path.write_text(after)

    verb = "rewrote" if args.apply else "would rewrite"
    print(f"\n{verb} {touched} file(s)")
    if not args.apply:
        print("re-run with --apply to write (originals are copied to <file>.yaml.bkp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
