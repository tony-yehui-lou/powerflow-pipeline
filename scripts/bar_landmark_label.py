#!/usr/bin/env python
"""Draw boxes round the bar's metal landmarks, frame by frame. No model, no propagation.

Two classes, because between them something is always visible:

**cap** -- the end face of the sleeve. The two caps point in opposite directions along the axis,
so at most one can face the camera at a time, and when the bar is fronto-parallel neither does.

**collar** -- the bar's own stepped inner shoulder, where the shaft meets the plates. Visible on
both ends, including the end whose cap is hidden, and sitting against a coloured plate rather
than a grey floor, so it reads well. It is part of the bar, not a lifter-added clamp (those go
outside the plates), so its position is fixed: **the two collars are ~1310 mm apart on both
men's and women's bars**, which makes them a ruler that needs no knowledge of the bar type or
the loading.

Propagation is the wrong tool here -- SAM 2 drifts on a ~30 px specular target and has memory to
corrupt, while a detector sees one frame at a time and cannot drift. So these are drawn by hand
on sampled frames, which is how most detection datasets are built anyway.

    uv run python scripts/bar_landmark_label.py "10 September/50kg1"

Boxes are stored in **full-resolution** pixel coordinates whatever scale you label at, because
the detector should train at native resolution where a 28 px cap is still 28 px.

Three rules decide whether this dataset is worth anything:

1. **Box a cap only when its face reads as a ring or an ellipse.** The side of the chrome
   cylinder is not a cap.
2. **Box a collar when the stepped metal between shaft and plate is distinguishable**, even
   partly occluded by a hand -- say over half of it.
3. **Motion blur is labelled, not skipped, and the box covers the whole smear.** The fast frames
   are where peak velocity is measured, so a model that has never been shown a blurred cap fails
   exactly where it matters. For linear blur the smear's centre is the position at mid-exposure,
   which is the position the bar path wants, so one box serves both. Press `b` to mark boxes as
   blurred -- it stays on until pressed again, because blur arrives in runs. Leave the frame
   empty only when the landmark is genuinely indistinguishable from the background.
4. **A frame with nothing visible is a real label, not a skipped frame.** Press `n` on it.

Controls:
    1 / 2           choose the class: 1 = cap, 2 = collar
    b               toggle the blurred flag on boxes you draw
    left drag       draw a box of the active class
    u               undo the last box on this frame
    n / space       next frame (an empty frame is recorded as "no cap visible")
    p               previous frame
    g               toggle the magnifier
    s               save now (it also saves after every frame and on quit)
    q               save and quit

Output, under ../data/annotations/barbell/v3/<capture>/:
    landmarks.json  boxes per sampled frame, each with a class, full-res coordinates
    frames/         the sampled frames as JPEGs, ready for a COCO export

Each box carries its class, so one pass produces both.

Re-running resumes: frames already decided are skipped unless --redo.

See docs/specs/preprocessing/S7-barbell-annotation-guide.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PIPELINE_ROOT.parent / "data"
STAGE_ROOT = DATA_ROOT / "s4_crop_output"
RAW_ROOT = DATA_ROOT / "raw"
OUT_ROOT = DATA_ROOT / "annotations" / "barbell" / "v3"

MAGNIFIER = (
    3  # a 30 px landmark is 15 px at the default display scale; the inset makes it clickable
)
MAGNIFIER_SRC = 70

CAP, COLLAR = 1, 2
CLASSES = {CAP: "cap", COLLAR: "collar"}
CLASS_COLOURS = {CAP: (60, 220, 255), COLLAR: (120, 255, 120)}


def read_barbell_type(capture: str) -> str | None:
    """The operator's declared bar type, from the raw capture's metadata.

    It is read from `raw/` rather than the stage output because the preprocess metadata task
    carries a fixed field set and drops `barbell_type` (along with `weight_in_kg` and `type`)
    on the way to s4. Scale depends on this value -- cap to opposite collar is 1755 mm on a
    men's bar and 1660 mm on a women's -- so it is recorded with the annotations rather than
    assumed by whoever reads them later.
    """

    path = RAW_ROOT / capture / "metadata.yaml"
    if not path.exists():
        return None
    try:
        import yaml

        doc = yaml.safe_load(path.read_text()) or {}
    except Exception:  # a hand-edited metadata file should not stop the labelling
        return None
    lift = doc.get("lift") or {}
    value = lift.get("barbell_type")
    return str(value) if value is not None else None


def sample_indices(n_frames: int, count: int) -> list[int]:
    """Evenly spaced frames across the clip, endpoints included.

    Even spacing rather than clustering: the clip is already cut to the lift, so every phase is
    represented, and a detector gains nothing from near-duplicate neighbours.
    """

    if count >= n_frames:
        return list(range(n_frames))
    return sorted({round(i * (n_frames - 1) / (count - 1)) for i in range(count)})


def load_frames(
    video_path: Path, wanted: set[int] | None = None
) -> tuple[dict[int, np.ndarray], int, float]:
    """Decode, keeping only the frames asked for. Full resolution -- scaling is display-only."""

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames: dict[int, np.ndarray] = {}
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if wanted is None or idx in wanted:
            frames[idx] = bgr
        idx += 1
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {video_path}")
    return frames, idx, fps if fps > 0 else 30.0


def draw_magnifier(canvas: np.ndarray, source: np.ndarray, cursor: tuple[int, int]) -> None:
    """A zoomed inset round the cursor, drawn into the canvas corner.

    Without it a cap is a handful of pixels on screen and the box corners are guesswork.
    """

    h, w = source.shape[:2]
    cx, cy = cursor
    half = MAGNIFIER_SRC // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half), min(h, cy + half)
    patch = source[y0:y1, x0:x1]
    if patch.size == 0:
        return
    zoom = cv2.resize(
        patch,
        (patch.shape[1] * MAGNIFIER, patch.shape[0] * MAGNIFIER),
        interpolation=cv2.INTER_NEAREST,
    )
    zh, zw = zoom.shape[:2]
    # Keep the inset away from the cursor so it never hides what is being boxed.
    px = 8 if cx > w // 2 else w - zw - 8
    py = h - zh - 8
    canvas[py : py + zh, px : px + zw] = zoom
    cv2.rectangle(canvas, (px, py), (px + zw, py + zh), (0, 255, 255), 1)
    cv2.drawMarker(canvas, (px + zw // 2, py + zh // 2), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help='e.g. "10 September/50kg1"')
    parser.add_argument("--count", type=int, default=40, help="frames to sample (default 40)")
    parser.add_argument(
        "--scale", type=float, default=0.5, help="display scale; boxes are stored at full res"
    )
    parser.add_argument("--redo", action="store_true", help="revisit frames already decided")
    args = parser.parse_args()

    video_path = STAGE_ROOT / args.capture / "rgb.mp4"
    if not video_path.exists():
        raise SystemExit(f"no such capture: {video_path}")
    out_dir = OUT_ROOT / args.capture
    out_path = out_dir / "landmarks.json"

    probe = cv2.VideoCapture(str(video_path))
    n_total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    wanted = sample_indices(n_total, args.count)

    done: dict[int, list[dict[str, object]]] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        done = {int(k): v for k, v in existing.get("boxes", {}).items()}
        print(f"resuming: {len(done)} of {len(wanted)} sampled frames already decided")

    todo = wanted if args.redo else [i for i in wanted if i not in done]
    if not todo:
        print(f"every sampled frame is already decided for {args.capture}")
        return 0

    print(f"decoding {len(wanted)} of {n_total} frames ...")
    frames, _, fps = load_frames(video_path, set(wanted))
    full_h, full_w = frames[wanted[0]].shape[:2]
    barbell_type = read_barbell_type(args.capture)
    print(f"{args.capture}: {n_total} frames at {full_w}x{full_h}; labelling {len(todo)}")
    if barbell_type:
        print(f"bar type (from raw metadata): {barbell_type}")
    else:
        print("WARNING: no barbell_type in the raw metadata -- metric scale will be ambiguous")

    scale = args.scale
    position = 0
    drag_from: tuple[int, int] | None = None
    cursor = (0, 0)
    magnify = True
    active = COLLAR  # the workhorse: visible on both ends, on every frame
    blurred = False  # sticky, because blur arrives in runs of consecutive frames

    window = f"cap boxes - {args.capture}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event: int, x: int, y: int, flags: int, _p: object) -> None:
        nonlocal drag_from, cursor
        cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_from = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_from is not None:
            x0, y0 = drag_from
            drag_from = None
            if abs(x - x0) < 3 or abs(y - y0) < 3:
                return  # a click, not a box
            box = [
                round(min(x0, x) / scale),
                round(min(y0, y) / scale),
                round(max(x0, x) / scale),
                round(max(y0, y) / scale),
            ]
            done.setdefault(todo[position], []).append(
                {"class": CLASSES[active], "box": box, "blurred": blurred}
            )

    cv2.setMouseCallback(window, on_mouse)

    def save() -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frames").mkdir(exist_ok=True)
        for idx in sorted(done):
            if idx in frames:
                target = out_dir / "frames" / f"{idx:06d}.jpg"
                if not target.exists():
                    cv2.imwrite(str(target), frames[idx], [cv2.IMWRITE_JPEG_QUALITY, 95])
        caps = sum(1 for v in done.values() for e in v if e["class"] == "cap")
        collars = sum(1 for v in done.values() for e in v if e["class"] == "collar")
        empty = sum(1 for v in done.values() if not v)
        out_path.write_text(
            json.dumps(
                {
                    "capture": args.capture,
                    "schema": "bar-landmarks/1",
                    "classes": list(CLASSES.values()),
                    "barbell_type": barbell_type,
                    "collar_separation_mm": 1310,
                    "cap_to_opposite_collar_mm": (
                        None
                        if not barbell_type
                        else 1755
                        if "male" in barbell_type.lower() and "female" not in barbell_type.lower()
                        else 1660
                    ),
                    "coordinates": "full-resolution xyxy",
                    "frame_size": [full_h, full_w],
                    "n_frames_total": n_total,
                    "sampled_frames": wanted,
                    "frames_decided": len(done),
                    "cap_boxes": caps,
                    "collar_boxes": collars,
                    "blurred_boxes": sum(1 for v in done.values() for e in v if e.get("blurred")),
                    "frames_with_nothing_visible": empty,
                    "fps": fps,
                    "boxes": {str(k): v for k, v in sorted(done.items())},
                    "written_at": datetime.now(UTC).isoformat(),
                    "guide": "docs/specs/preprocessing/S7-barbell-annotation-guide.md",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    while True:
        idx = todo[position]
        frame = frames[idx]
        shown = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        canvas = shown.copy()
        for entry in done.get(idx, []):
            box = entry["box"]
            colour = CLASS_COLOURS[CAP if entry["class"] == "cap" else COLLAR]
            p0 = (int(box[0] * scale), int(box[1] * scale))
            p1 = (int(box[2] * scale), int(box[3] * scale))
            cv2.rectangle(canvas, p0, p1, colour, 2)
            cv2.putText(
                canvas,
                str(entry["class"]),
                (p0[0], max(12, p0[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                colour,
                1,
            )
        if drag_from is not None:
            cv2.rectangle(canvas, drag_from, cursor, CLASS_COLOURS[active], 1)
        if magnify:
            draw_magnifier(canvas, shown, cursor)
        here = done.get(idx, [])
        caps = sum(1 for v in done.values() for e in v if e["class"] == "cap")
        collars = sum(1 for v in done.values() for e in v if e["class"] == "collar")
        banner = (
            f"{position + 1}/{len(todo)}  frame {idx}  |  drawing {CLASSES[active].upper()}"
            f"{' + BLUR' if blurred else ''}  |  {len(here)} box(es) here  |  "
            f"{len(done)} decided: {caps} caps, {collars} collars"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(canvas, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(
            canvas,
            "1=cap 2=collar  b=blur  drag=box  u=undo  n/space=next (empty is a label)  "
            "p=prev  g=magnifier  q=save+quit",
            (8, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            done.setdefault(idx, [])
            save()
            break
        if key == ord("b"):
            blurred = not blurred
        elif key == ord("1"):
            active = CAP
        elif key == ord("2"):
            active = COLLAR
        elif key == ord("u") and done.get(idx):
            done[idx].pop()
        elif key in (ord("n"), ord(" ")):
            done.setdefault(idx, [])  # an empty frame is a decision, not a gap
            save()
            if position + 1 >= len(todo):
                print("every sampled frame decided")
                break
            position += 1
        elif key == ord("p"):
            position = max(0, position - 1)
        elif key == ord("g"):
            magnify = not magnify
        elif key == ord("s"):
            save()
            print(f"saved {len(done)} frames")

    cv2.destroyAllWindows()
    caps = sum(1 for v in done.values() for e in v if e["class"] == "cap")
    collars = sum(1 for v in done.values() for e in v if e["class"] == "collar")
    empty = sum(1 for v in done.values() if not v)
    print(f"{len(done)} frames decided: {caps} cap boxes, {collars} collar boxes, {empty} empty")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
