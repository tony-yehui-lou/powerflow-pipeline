#!/usr/bin/env python
"""Check a capture's bar-landmark boxes before anyone trains on them.

    uv run python scripts/bar_landmark_qc.py "10 September/50kg1"
    uv run python scripts/bar_landmark_qc.py            # every labelled capture

Nothing here needs a model. Each check catches a different way a hand-drawn dataset goes wrong,
and each one names the frames to look at rather than just a verdict:

* **Coverage** -- sampled frames still undecided, and frames deliberately left empty.
* **Composition** -- how many caps and collars per frame. Two collars in one frame is what the
  1310 mm collar-to-collar ruler needs; without it there is no bar-type-independent scale.
* **Box size** -- a box far off its class median is usually a slip, except on blurred frames
  where the smear is genuinely longer.
* **Side consistency** -- if the cap is image-left on most frames and image-right on a few,
  those few are likely the wrong end.
* **Trajectory** -- a landmark that leaps between neighbouring samples is either a mislabel or a
  genuinely fast frame; the two are told apart by whether the blur flag is set.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PIPELINE_ROOT.parent / "data" / "annotations" / "barbell" / "v3"

SIZE_TOLERANCE = 2.2  # times the class median area, before a box is worth a second look
JUMP_PX_PER_FRAME = 25.0  # a landmark moving faster than this between samples


def centre(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def area(box: list[int]) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def check(path: Path) -> tuple[str, list[str], list[str]]:
    """Return (capture, problems, notes). Problems are worth fixing; notes are for information."""

    doc = json.loads(path.read_text())
    capture = doc["capture"]
    boxes = {int(k): v for k, v in doc.get("boxes", {}).items()}
    sampled = doc.get("sampled_frames", sorted(boxes))
    height, width = doc.get("frame_size", [0, 0])
    problems: list[str] = []
    notes: list[str] = []

    undecided = [f for f in sampled if f not in boxes]
    if undecided:
        problems.append(f"{len(undecided)} sampled frame(s) never decided: {undecided[:10]}")
    empty = sorted(f for f, v in boxes.items() if not v)
    if empty:
        notes.append(f"{len(empty)} frame(s) deliberately empty: {empty[:10]}")

    composition = Counter(
        (
            sum(1 for e in v if e["class"] == "cap"),
            sum(1 for e in v if e["class"] == "collar"),
        )
        for v in boxes.values()
    )
    notes.append(
        "per-frame (caps, collars): "
        + ", ".join(f"{k}x{n}" for k, n in sorted(composition.items(), key=lambda kv: -kv[1]))
    )
    # Scale comes from a pair of landmarks a known distance apart. The far collar is usually
    # hidden behind its own plates, so the pair that is actually available is the visible cap
    # and the opposite collar: 1755 mm on a men's bar, 1660 mm on a women's. That difference is
    # why the bar type has to be declared -- in the corpus by assumption, in the app by asking
    # the lifter. Collar-to-collar (1310 mm) is better when both are visible, because it is the
    # same on both bar types, but it rarely is.
    two_collars = sum(1 for v in boxes.values() if sum(1 for e in v if e["class"] == "collar") >= 2)
    pairs = sum(
        1
        for v in boxes.values()
        if any(e["class"] == "cap" for e in v) and any(e["class"] == "collar" for e in v)
    )
    if two_collars:
        notes.append(
            f"{two_collars} frame(s) carry both collars: 1310 mm apart, bar-type independent"
        )
    if pairs:
        notes.append(
            f"{pairs} frame(s) pair a cap with a collar: 1755 mm on a men's bar, 1660 mm on a "
            "women's, so scale depends on the declared bar type"
        )
    if not pairs and not two_collars:
        problems.append(
            "no frame carries a landmark pair of known separation (two collars, or a cap and a "
            "collar), so this capture cannot supply metric scale"
        )

    blurred = sum(1 for v in boxes.values() for e in v if e.get("blurred"))
    notes.append(f"{blurred} box(es) flagged blurred")

    for cls in ("cap", "collar"):
        entries = [(f, e) for f, v in boxes.items() for e in v if e["class"] == cls]
        if not entries:
            problems.append(f"no {cls} boxes at all")
            continue
        areas = [area(e["box"]) for _, e in entries]
        median = statistics.median(areas)
        widths = [e["box"][2] - e["box"][0] for _, e in entries]
        heights = [e["box"][3] - e["box"][1] for _, e in entries]
        notes.append(
            f"{cls}: {len(entries)} box(es), median {statistics.median(widths):.0f}x"
            f"{statistics.median(heights):.0f} px"
        )
        odd = [
            f
            for f, e in entries
            if not e.get("blurred")
            and (
                area(e["box"]) > SIZE_TOLERANCE * median or area(e["box"]) * SIZE_TOLERANCE < median
            )
        ]
        if odd:
            problems.append(
                f"{cls}: {len(odd)} box(es) far off the median size: {sorted(odd)[:10]}"
            )
        bad = [
            f
            for f, e in entries
            if e["box"][0] < 0
            or e["box"][1] < 0
            or (width and e["box"][2] > width)
            or (height and e["box"][3] > height)
            or area(e["box"]) <= 0
        ]
        if bad:
            problems.append(
                f"{cls}: {len(bad)} degenerate or out-of-frame box(es): {sorted(bad)[:10]}"
            )

        singles = sorted(
            (f, centre(e["box"]))
            for f, e in entries
            if sum(1 for x in boxes[f] if x["class"] == cls) == 1
        )
        jumps = []
        for (f0, p0), (f1, p1) in itertools.pairwise(singles):
            gap = f1 - f0
            if 0 < gap <= 12:
                speed = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5 / gap
                if speed > JUMP_PX_PER_FRAME:
                    blur_here = any(e.get("blurred") for e in boxes[f1] if e["class"] == cls)
                    jumps.append((f0, f1, speed, blur_here))
        for f0, f1, speed, blur_here in jumps:
            tag = "blur flagged, so probably genuine" if blur_here else "NOT flagged blurred"
            notes.append(f"{cls}: {speed:.0f} px/frame between f{f0} and f{f1} ({tag})")

    # Which side is each class usually on? A flip is usually the wrong end of the bar.
    sides = {"cap": [], "collar": []}
    for f, v in boxes.items():
        cap = [e for e in v if e["class"] == "cap"]
        col = [e for e in v if e["class"] == "collar"]
        if len(cap) == 1 and len(col) == 1:
            sides["cap"].append((f, centre(cap[0]["box"])[0] < centre(col[0]["box"])[0]))
    if sides["cap"]:
        left = sum(1 for _, is_left in sides["cap"] if is_left)
        total = len(sides["cap"])
        minority = [f for f, is_left in sides["cap"] if is_left != (left * 2 > total)]
        notes.append(f"cap is image-left of the collar on {left}/{total} paired frames")
        if minority:
            problems.append(f"side flips on {len(minority)} frame(s): {sorted(minority)[:10]}")
    return capture, problems, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", help="one capture; default is every labelled one")
    args = parser.parse_args()

    found = (
        [OUT_ROOT / args.capture / "landmarks.json"]
        if args.capture
        else sorted(OUT_ROOT.glob("**/landmarks.json"))
    )
    found = [f for f in found if f.exists()]
    if not found:
        raise SystemExit(f"no landmark annotations under {OUT_ROOT}")

    worst = 0
    for path in found:
        capture, problems, notes = check(path)
        print(f"\n=== {capture}")
        for note in notes:
            print(f"    {note}")
        for problem in problems:
            print(f"  ! {problem}")
        if not problems:
            print("  ok -- nothing to fix")
        worst = max(worst, len(problems))
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())
