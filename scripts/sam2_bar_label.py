#!/usr/bin/env python
"""Label the barbell itself -- the whole bar as one object -- with refinement until it holds.

The subject of S5 is the bar, not the plates: a bar looks the same whatever is loaded on it and
whatever gym it is in, where a bumper changes colour between plates and a gym floor is covered in
other bumpers. Bar ends come from the mask's extremes along its own axis, so no separate sleeve
detection is needed.

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --with transformers --with torchvision \
        python scripts/sam2_bar_label.py "10 September/50kg1"

Measured 2026-10-18 on `10 September/50kg1`: five clicks (shaft, both plates, both sleeve tips)
produce a 21k px mask of the whole bar, and propagation holds it for the entire lift -- zero lost
frames, area steady at 19.1-20.2k through the overhead catch -- drifting only in the last 25
frames, after the bar is dropped. One click is not enough: on this footage a single click selects
a *part* (the shaft alone, or one plate), not the bar.

There is one object, `barbell`. Its bar ends are the mask's extremes along its own axis, which is
why the sleeve tips deserve a click each.

Prompt phase:
    left / right click   positive / negative point on the active object
    [ / ]                step the prompt frame back / forward by 5
    space                preview this frame
    enter                propagate through the clip
    q                    quit without writing

Review phase (after the first propagation):
    , / .                step one frame back / forward
    [ / ]                step ten frames
    f                    jump to the next frame this tool thinks drifted
    left / right click   add a refinement point for the active object, here
    enter                re-propagate with the refinements
    w                    write the result
    q                    quit without writing

See docs/specs/preprocessing/S5-barbell-annotation-guide.md.
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
OUT_ROOT = DATA_ROOT / "annotations" / "barbell" / "v2"

# One object: the barbell, whole. Measured on 10 September footage, a single click selects a
# *part* -- the shaft alone, or one plate -- so the bar is defined by several clicks on one
# object rather than by clicking it once. Bar ends come from the mask's extremes along its own
# axis, which is why the sleeve tips are worth a click each.
BARBELL = 1
ROLES = {BARBELL: "barbell"}
COLOURS = {BARBELL: (60, 200, 255)}

# A track that suddenly doubles in area or leaps across the frame has almost certainly swapped
# objects -- the observed failure was a 3.4x area jump and an 84 px centre move in one frame.
AREA_JUMP = 1.8
CENTRE_JUMP_PX = 25.0


def load_frames(video_path: Path, scale: float) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    # Read fps BEFORE release: a released capture answers -1.0, which is truthy, so an
    # `or 30.0` fallback never fires and VideoWriter then fails to open without raising.
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(bgr)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {video_path}")
    return frames, fps if fps > 0 else 30.0


def overlay_masks(frame: np.ndarray, indexed: np.ndarray | None) -> np.ndarray:
    out = frame.copy()
    if indexed is None:
        return out
    for obj_id in (BARBELL,):
        mask = indexed == obj_id
        if not mask.any():
            continue
        tint = np.zeros_like(out)
        tint[:] = COLOURS[obj_id]
        out[mask] = (0.55 * out[mask] + 0.45 * tint[mask]).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, COLOURS[obj_id], 2)
    return out


def masks_to_indexed(per_object: dict[int, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    indexed = np.zeros(shape, dtype=np.uint8)
    for obj_id in (BARBELL,):
        mask = per_object.get(obj_id)
        if mask is not None:
            indexed[mask] = obj_id
    return indexed


def suspect_frames(masks: dict[int, np.ndarray]) -> list[int]:
    """Frames where a track probably swapped objects, ordered by frame index.

    This is the whole point of the review phase: a 15 px specular sleeve drifts, and finding the
    drift by scrubbing 400 frames by hand is worse than the labelling itself.
    """

    flagged: set[int] = set()
    for obj_id in (BARBELL,):
        previous: tuple[float, float, float] | None = None
        for idx in sorted(masks):
            mask = masks[idx] == obj_id
            area = float(mask.sum())
            if area == 0:
                if previous is not None:
                    flagged.add(idx)
                previous = None
                continue
            ys, xs = np.nonzero(mask)
            centre = (float(xs.mean()), float(ys.mean()))
            if previous is not None:
                prev_area, px, py = previous
                grew = prev_area > 0 and (
                    area / prev_area > AREA_JUMP or prev_area / area > AREA_JUMP
                )
                moved = float(np.hypot(centre[0] - px, centre[1] - py)) > CENTRE_JUMP_PX
                if grew or moved:
                    flagged.add(idx)
            previous = (area, centre[0], centre[1])
    return sorted(flagged)


class Prompts:
    """Every click made, per object, per frame -- the record this tool exists to produce."""

    def __init__(self) -> None:
        self.points: dict[int, dict[int, list[tuple[int, int]]]] = {BARBELL: {}}
        self.labels: dict[int, dict[int, list[int]]] = {BARBELL: {}}
        self.active = BARBELL

    def add(self, frame_idx: int, x: int, y: int, positive: bool) -> None:
        self.points[self.active].setdefault(frame_idx, []).append((x, y))
        self.labels[self.active].setdefault(frame_idx, []).append(1 if positive else 0)

    def at(self, frame_idx: int) -> dict[int, tuple[list[tuple[int, int]], list[int]]]:
        out = {}
        for obj_id in (BARBELL,):
            pts = self.points[obj_id].get(frame_idx)
            if pts:
                out[obj_id] = (pts, self.labels[obj_id][frame_idx])
        return out

    def clear_frame(self, frame_idx: int) -> None:
        for obj_id in (BARBELL,):
            self.points[obj_id].pop(frame_idx, None)
            self.labels[obj_id].pop(frame_idx, None)

    def total(self) -> int:
        return sum(len(v) for obj in self.points.values() for v in obj.values())

    def as_json(self) -> dict[str, Any]:
        return {
            ROLES[obj_id]: {
                str(frame): {"points": pts, "labels": self.labels[obj_id][frame]}
                for frame, pts in sorted(self.points[obj_id].items())
            }
            for obj_id in (BARBELL,)
        }


def draw(
    frame: np.ndarray,
    indexed: np.ndarray | None,
    prompts: Prompts,
    frame_idx: int,
    n_frames: int,
    mode: str,
    flagged: list[int],
) -> np.ndarray:
    canvas = overlay_masks(frame, indexed)
    for obj_id in (BARBELL,):
        for (x, y), label in zip(
            prompts.points[obj_id].get(frame_idx, []),
            prompts.labels[obj_id].get(frame_idx, []),
            strict=True,
        ):
            marker = cv2.MARKER_CROSS if label == 1 else cv2.MARKER_TILTED_CROSS
            cv2.drawMarker(canvas, (x, y), COLOURS[obj_id], marker, 18, 2)
    flag_note = f"  |  {len(flagged)} suspect frame(s)" if flagged else ""
    first = (
        f"frame {frame_idx}/{n_frames - 1}  |  {ROLES[prompts.active]}  |  "
        f"{mode}{'  <-- SUSPECT' if frame_idx in flagged else ''}{flag_note}"
    )
    keys = (
        "space=preview  enter=propagate  [ ]=frame  q=quit"
        if mode == "PROMPT"
        else ", . [ ]=scrub   f=next suspect   click=refine   enter=re-propagate   w=write   q=quit"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(canvas, first, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(canvas, keys, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    return canvas


def open_overlay_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"could not open {path} for writing at {fps} fps, {width}x{height}")
    return writer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help='e.g. "22 August/Snch/107kgSnch1"')
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    video_path = STAGE_ROOT / args.capture / "rgb.mp4"
    if not video_path.exists():
        raise SystemExit(f"no such capture: {video_path}")
    out_dir = OUT_ROOT / args.capture
    if (out_dir / "bar_run.json").exists() or (out_dir / "masks").exists():
        raise SystemExit(f"already labelled: {out_dir} -- bump to v3 rather than relabelling")

    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"decoding {video_path} at scale {args.scale} ...")
    frames, fps = load_frames(video_path, args.scale)
    height, width = frames[0].shape[:2]
    print(f"{len(frames)} frames at {width}x{height}, device={device}")
    print(f"loading {args.model} ...")
    model = Sam2VideoModel.from_pretrained(args.model).to(device).eval()
    processor = Sam2VideoProcessor.from_pretrained(args.model)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    session = processor.init_video_session(video=rgb, inference_device=device)
    prompts = Prompts()
    masks: dict[int, np.ndarray] = {}
    flagged: list[int] = []
    frame_idx = 0
    mode = "PROMPT"

    window = f"SAM2 bar labelling - {args.capture}"
    # AUTOSIZE: in a resizable window OpenCV reports clicks in the window's scaled space, so
    # every click would land off target by the zoom factor.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    if height > 1000:
        print(f"note: window is {width}x{height}; re-run with a smaller --scale if it overflows")

    def on_mouse(event: int, x: int, y: int, flags: int, _p: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            prompts.add(frame_idx, x, y, positive=True)
        elif event == cv2.EVENT_RBUTTONDOWN:
            prompts.add(frame_idx, x, y, positive=False)

    cv2.setMouseCallback(window, on_mouse)

    def send_prompts(at_frame: int) -> None:
        """Hand this frame's clicks to the session. Batched: one call, every object."""

        here = prompts.at(at_frame)
        if not here:
            return
        obj_ids = sorted(here)
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=at_frame,
            obj_ids=obj_ids,
            input_points=[[[list(p) for p in here[o][0]] for o in obj_ids]],
            input_labels=[[list(here[o][1]) for o in obj_ids]],
        )

    def to_indexed(output: Any) -> np.ndarray:
        got = processor.post_process_masks(
            [output.pred_masks], original_sizes=[[height, width]], binarize=True
        )[0]
        per_object = {
            obj_id: got[slot, 0].cpu().numpy().astype(bool)
            for slot, obj_id in enumerate(session.obj_ids)
        }
        return masks_to_indexed(per_object, (height, width))

    def propagate(start: int) -> None:
        """Track outward from `start`, which must be a frame whose prompts were just sent.

        `start_frame_idx` is passed explicitly rather than inferred: without it, and without a
        forward pass on the prompted frame first, the iterator raises "Cannot determine the
        starting frame index" -- after the clicking is done, which is the worst moment.
        """

        nonlocal masks, flagged
        anchors = sorted({f for obj in prompts.points.values() for f in obj})
        print(f"propagating from {start} (anchors: {anchors}) ...")
        masks[start] = to_indexed(model(inference_session=session, frame_idx=start))
        for reverse in (False, True):
            if reverse and start == 0:
                continue
            for output in model.propagate_in_video_iterator(
                session, start_frame_idx=start, reverse=reverse
            ):
                masks[output.frame_idx] = to_indexed(output)
                if sys.stdout.isatty():
                    print(f"  {'back' if reverse else 'fwd '} {output.frame_idx}", end="\r")
        print()
        flagged = suspect_frames(masks)
        print(f"{len(masks)} frames tracked; {len(flagged)} look like drift: {flagged[:12]}")

    while True:
        cv2.imshow(
            window,
            draw(
                frames[frame_idx],
                masks.get(frame_idx),
                prompts,
                frame_idx,
                len(frames),
                mode,
                flagged,
            ),
        )
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            print("quit without writing")
            return 1
        if key == ord("[") or key == ord("]"):
            step = -1 if key == ord("[") else 1
            step *= 5 if mode == "PROMPT" else 10
            if mode == "PROMPT":
                prompts.clear_frame(frame_idx)
            frame_idx = max(0, min(len(frames) - 1, frame_idx + step))
        elif key == ord(",") or key == ord("."):
            frame_idx = max(0, min(len(frames) - 1, frame_idx + (1 if key == ord(".") else -1)))
        elif key == ord("f") and flagged:
            later = [f for f in flagged if f > frame_idx]
            frame_idx = later[0] if later else flagged[0]
            print(f"suspect frame {frame_idx}")
        elif key == ord(" ") and mode == "PROMPT":
            if not prompts.at(frame_idx):
                print("click the sleeve and the shaft first")
                continue
            send_prompts(frame_idx)
            masks[frame_idx] = to_indexed(model(inference_session=session, frame_idx=frame_idx))
        elif key in (13, 10):
            here = prompts.at(frame_idx)
            if mode == "PROMPT" and not here:
                print("click the bar first -- shaft, then each plate, then each sleeve tip")
                continue
            if not here and mode == "REVIEW":
                print("add a refinement click here first, or press w to write")
                continue
            send_prompts(frame_idx)
            propagate(frame_idx)
            mode = "REVIEW"
        elif key == ord("w") and mode == "REVIEW":
            break

    cv2.destroyAllWindows()
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True)
    for idx, indexed in sorted(masks.items()):
        cv2.imwrite(str(masks_dir / f"{idx:06d}.png"), indexed)
    writer = open_overlay_writer(out_dir / "overlay.mp4", fps, width, height)
    for i, frame in enumerate(frames):
        writer.write(overlay_masks(frame, masks.get(i)))
    writer.release()
    (out_dir / "bar_run.json").write_text(
        json.dumps(
            {
                "capture": args.capture,
                "source": str(video_path),
                "model": args.model,
                "device": device,
                "decode_scale": args.scale,
                "mask_size": [height, width],
                "n_frames": len(frames),
                "frames_with_masks": len(masks),
                "object_roles": {str(k): v for k, v in ROLES.items()},
                "subject_object_ids": [BARBELL],
                "n_prompts": prompts.total(),
                "prompts": prompts.as_json(),
                "suspect_frames_at_write": suspect_frames(masks),
                "written_at": datetime.now(UTC).isoformat(),
                "guide": "docs/specs/preprocessing/S5-barbell-annotation-guide.md",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(masks)} frames to {masks_dir} from {prompts.total()} click(s)")
    print(f"review {out_dir / 'overlay.mp4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
