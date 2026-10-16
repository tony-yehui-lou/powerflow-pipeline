#!/usr/bin/env python
"""Pass A of the barbell annotation guide: click every plate once, track them all.

Opens one S4-cropped capture, lets you click every plate face in a frame of your choosing -- the
subject's two bar plates first, then every other plate in the room -- and runs SAM 2 through the
whole clip, writing one indexed mask per frame plus a review overlay.

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --with transformers --with torchvision \
        python scripts/sam2_plate_label.py "22 August/Snch/42kgSnch1"

`transformers` and `torchvision` are layered in by `uv run --with` rather than added to the
project: SAM 2 is a labelling tool, and the pipeline's runtime dependency list deliberately does
not grow for it. `torchvision` is not optional -- SAM 2's image processor imports it, and the
project pins `torch` without it.

Nothing is treated as static scenery. Other athletes lift mid-capture and plates are carried in
and out of frame, so every plate is propagated frame by frame like the subject's.

Objects 1 and 2 are the subject's bar plates, by convention: `subject_object_ids` in the run
record keys off it, and every downstream consumer reads that rather than guessing. The detector
never sees the distinction -- see docs/specs/preprocessing/S7-barbell-model-recommendation.md C3.

Controls (in the click window):
    left click      positive point on the plate under the cursor
    right click     negative point (use it when the mask bleeds onto the bar or a leg)
    tab             cycle the active plate
    n               start a new plate
    [ / ]           step the prompt frame back / forward by 5
    space           preview the masks for the current prompts
    enter           accept and propagate through the clip
    r               clear all points
    q               quit without writing

Output (write-once, refuses to clobber):
    ../data/annotations/barbell/v1/<capture>/masks/<frame>.png   indexed: 0 bg, 1..N plates
    ../data/annotations/barbell/v1/<capture>/sam2_run.json       clicks, model, subject ids
    ../data/annotations/barbell/v1/<capture>/overlay.mp4         for eyeballing before Pass B

See docs/specs/preprocessing/S7-barbell-annotation-guide.md §3.
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

# The subject's bar plates are clicked first, always. Everything downstream reads this rather
# than re-deriving "which bar matters" from the pixels, which is not a visual question.
SUBJECT_OBJECT_IDS = (1, 2)
SUBJECT_COLOURS = {1: (60, 200, 255), 2: (255, 120, 60)}


def object_colour(obj_id: int) -> tuple[int, int, int]:
    """Stable, distinguishable colours however many plates a frame turns out to hold."""

    if obj_id in SUBJECT_COLOURS:
        return SUBJECT_COLOURS[obj_id]
    hue = int((obj_id * 47) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def load_frames(video_path: Path, scale: float) -> tuple[list[np.ndarray], float]:
    """Decode the whole clip once, optionally downscaled."""

    cap = cv2.VideoCapture(str(video_path))
    # Read fps BEFORE release: a released capture answers -1.0, which is truthy, so an
    # `or 30.0` fallback never fires and the -1.0 reaches VideoWriter, which then fails to
    # open without raising and silently writes no overlay at all.
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


class ClickState:
    """The points clicked so far, per plate, on the current prompt frame."""

    def __init__(self) -> None:
        self.object_ids = list(SUBJECT_OBJECT_IDS)
        self.points: dict[int, list[tuple[int, int]]] = {i: [] for i in self.object_ids}
        self.labels: dict[int, list[int]] = {i: [] for i in self.object_ids}
        self.active = self.object_ids[0]
        self.preview: np.ndarray | None = None

    def add(self, x: int, y: int, positive: bool) -> None:
        self.points[self.active].append((x, y))
        self.labels[self.active].append(1 if positive else 0)
        self.preview = None

    def add_object(self) -> int:
        new_id = max(self.object_ids) + 1
        self.object_ids.append(new_id)
        self.points[new_id] = []
        self.labels[new_id] = []
        self.active = new_id
        self.preview = None
        return new_id

    def next_object(self) -> None:
        i = self.object_ids.index(self.active)
        self.active = self.object_ids[(i + 1) % len(self.object_ids)]

    def clear(self) -> None:
        for i in self.object_ids:
            self.points[i].clear()
            self.labels[i].clear()
        self.preview = None

    def prompted(self) -> list[int]:
        return [i for i in self.object_ids if self.points[i]]

    def subject_ready(self) -> bool:
        """The subject's pair is mandatory; further plates are as many as the frame holds."""

        return all(self.points[i] for i in SUBJECT_OBJECT_IDS)


def overlay_masks(frame: np.ndarray, indexed: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for obj_id in range(1, int(indexed.max()) + 1):
        mask = indexed == obj_id
        if not mask.any():
            continue
        colour = object_colour(obj_id)
        tint = np.zeros_like(out)
        tint[:] = colour
        out[mask] = (0.55 * out[mask] + 0.45 * tint[mask]).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, colour, 2)
    return out


def masks_to_indexed(per_object: dict[int, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Collapse per-object boolean masks into one indexed image. Later objects win overlaps."""

    indexed = np.zeros(shape, dtype=np.uint8)
    for obj_id in sorted(per_object):
        mask = per_object[obj_id]
        if mask is not None:
            indexed[mask] = obj_id
    return indexed


def draw_ui(frame: np.ndarray, state: ClickState, frame_idx: int, n_frames: int) -> np.ndarray:
    canvas = frame.copy()
    if state.preview is not None:
        canvas = overlay_masks(canvas, state.preview)
    for obj_id in state.object_ids:
        colour = object_colour(obj_id)
        for (x, y), label in zip(state.points[obj_id], state.labels[obj_id], strict=True):
            marker = cv2.MARKER_CROSS if label == 1 else cv2.MARKER_TILTED_CROSS
            cv2.drawMarker(canvas, (x, y), colour, marker, 18, 2)
    role = "SUBJECT bar" if state.active in SUBJECT_OBJECT_IDS else "other plate"
    banner = (
        f"frame {frame_idx}/{n_frames - 1}  |  plate {state.active} ({role})  |  "
        f"space=preview  enter=track  tab=switch  n=new plate  [ ]=frame  r=reset  q=quit"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, banner, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return canvas


def capture_status(out_dir: Path) -> str:
    """What already exists for this capture."""

    pass_a = (out_dir / "sam2_run.json").exists()
    qc = (out_dir / "plates.json").exists()
    mark = {True: "done", False: "not yet"}
    return f"{out_dir.name}: Pass A (all plates tracked) {mark[pass_a]} | Pass B QC {mark[qc]}"


def segment_frame(
    model: Any,
    processor: Any,
    rgb_frames: list[np.ndarray],
    frame_idx: int,
    points: dict[int, list[tuple[int, int]]],
    labels: dict[int, list[int]],
    device: str,
    height: int,
    width: int,
) -> tuple[dict[int, np.ndarray], Any]:
    """Segment every prompted plate on one frame, and hand back the session for propagation.

    One batched call for every object, never one call per object: adding objects separately makes
    SAM 2 treat the second call as a non-initial conditioning frame and it fails with
    "maskmem_features in conditioning outputs cannot be empty". Ragged point counts are fine --
    the processor pads them.
    """

    session = processor.init_video_session(video=rgb_frames, inference_device=device)
    prompted = [obj_id for obj_id in sorted(points) if points[obj_id]]
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=frame_idx,
        obj_ids=prompted,
        input_points=[[[list(p) for p in points[obj_id]] for obj_id in prompted]],
        input_labels=[[list(labels[obj_id]) for obj_id in prompted]],
    )
    outputs = model(inference_session=session, frame_idx=frame_idx)
    masks = processor.post_process_masks(
        [outputs.pred_masks], original_sizes=[[height, width]], binarize=True
    )[0]
    per_object = {
        obj_id: masks[slot, 0].cpu().numpy().astype(bool)
        for slot, obj_id in enumerate(session.obj_ids)
    }
    return per_object, session


def open_overlay_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """A writer that is verified open. `cv2.VideoWriter` reports failure only via isOpened()."""

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"could not open {path} for writing at {fps} fps, {width}x{height}")
    return writer


def render_overlay(capture: str, scale: float) -> int:
    """Rebuild overlay.mp4 from masks already on disk."""

    out_dir = OUT_ROOT / capture
    masks_dir = out_dir / "masks"
    if not masks_dir.is_dir():
        raise SystemExit(f"no masks to render: {masks_dir}")
    target = out_dir / "overlay.mp4"
    if target.exists():
        raise SystemExit(f"refusing to overwrite: {target} already exists")
    frames, fps = load_frames(STAGE_ROOT / capture / "rgb.mp4", scale)
    height, width = frames[0].shape[:2]
    writer = open_overlay_writer(target, fps, width, height)
    drawn = 0
    for i, frame in enumerate(frames):
        mask_path = masks_dir / f"{i:06d}.png"
        if mask_path.exists():
            indexed = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            writer.write(overlay_masks(frame, indexed))
            drawn += 1
        else:
            writer.write(frame)
    writer.release()
    print(f"wrote {target} ({drawn}/{len(frames)} frames carried a mask)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help='e.g. "22 August/Snch/42kgSnch1"')
    parser.add_argument(
        "--model", default="facebook/sam2.1-hiera-small", help="HuggingFace SAM 2.1 checkpoint"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="decode scale; SAM 2 resizes to 1024 internally, so 0.5 costs no real accuracy",
    )
    parser.add_argument("--device", default=None, help="mps, cpu or cuda (default: best available)")
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="rebuild overlay.mp4 from masks already written, without re-running SAM 2",
    )
    args = parser.parse_args()

    if args.overlay_only:
        return render_overlay(args.capture, args.scale)

    # Resolve and guard every output path BEFORE importing torch or loading a model: a refusal
    # should cost a second, not a model load and a session of clicking.
    video_path = STAGE_ROOT / args.capture / "rgb.mp4"
    if not video_path.exists():
        raise SystemExit(f"no such capture: {video_path}")
    out_dir = OUT_ROOT / args.capture
    print(capture_status(out_dir))
    if (out_dir / "sam2_run.json").exists() or (out_dir / "masks").exists():
        raise SystemExit(
            f"Pass A output already exists for {args.capture} "
            f"({out_dir / 'masks'}) -- bump to v2 rather than relabelling in place"
        )

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
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    state = ClickState()
    frame_idx = 0
    window = f"SAM2 plate prompts - {args.capture}"
    # AUTOSIZE, deliberately: in a resizable window OpenCV reports clicks in the window's own
    # scaled space, so every click would land off the plate by the zoom factor. At 1:1 the
    # callback's coordinates are the image's. If the window overflows the screen, lower --scale.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    if height > 1000:
        print(f"note: window is {width}x{height}; re-run with a smaller --scale if it overflows")

    def on_mouse(event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.add(x, y, positive=True)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.add(x, y, positive=False)

    cv2.setMouseCallback(window, on_mouse)

    def run_prompt_frame() -> tuple[dict[int, np.ndarray], Any]:
        prompted = {obj_id: state.points[obj_id] for obj_id in state.prompted()}
        labels = {obj_id: state.labels[obj_id] for obj_id in state.prompted()}
        return segment_frame(
            model, processor, rgb_frames, frame_idx, prompted, labels, device, height, width
        )

    session = None
    while True:
        cv2.imshow(window, draw_ui(frames[frame_idx], state, frame_idx, len(frames)))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            print("quit without writing")
            return 1
        if key == ord("\t"):
            state.next_object()
        elif key == ord("n"):
            print(f"new plate {state.add_object()}")
        elif key == ord("["):
            frame_idx = max(0, frame_idx - 5)
            state.clear()
        elif key == ord("]"):
            frame_idx = min(len(frames) - 1, frame_idx + 5)
            state.clear()
        elif key == ord("r"):
            state.clear()
        elif key == ord(" "):
            if not state.prompted():
                print("click a plate first")
                continue
            print("segmenting prompt frame ...")
            per_object, _ = run_prompt_frame()
            state.preview = masks_to_indexed(per_object, (height, width))
        elif key in (13, 10):  # enter
            if not state.subject_ready():
                print("plates 1 and 2 are the subject's bar and are required (tab switches plate)")
                continue
            per_object, session = run_prompt_frame()
            break

    cv2.destroyAllWindows()

    tracked = sorted(state.prompted())
    print(f"propagating {len(tracked)} plate(s) from frame {frame_idx} ...")
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True)
    writer = open_overlay_writer(out_dir / "overlay.mp4", fps, width, height)
    covered: dict[int, np.ndarray] = {}
    # Forward from the prompt frame, then backward: a prompt frame chosen mid-lift (because that
    # is where the plates are unobstructed) would otherwise leave every earlier frame unlabelled.
    for reverse in (False, True):
        if reverse and frame_idx == 0:
            continue
        for output in model.propagate_in_video_iterator(session, reverse=reverse):
            masks = processor.post_process_masks(
                [output.pred_masks], original_sizes=[[height, width]], binarize=True
            )[0]
            per_object = {
                obj_id: masks[slot, 0].cpu().numpy().astype(bool)
                for slot, obj_id in enumerate(session.obj_ids)
            }
            indexed = masks_to_indexed(per_object, (height, width))
            covered[output.frame_idx] = indexed
            cv2.imwrite(str(masks_dir / f"{output.frame_idx:06d}.png"), indexed)
            direction = "back" if reverse else "fwd "
            print(f"  {direction} frame {output.frame_idx}", end="\r", flush=True)
    print()

    for i, frame in enumerate(frames):
        indexed = covered.get(i)
        writer.write(overlay_masks(frame, indexed) if indexed is not None else frame)
    writer.release()

    (out_dir / "sam2_run.json").write_text(
        json.dumps(
            {
                "capture": args.capture,
                "source": str(video_path),
                "model": args.model,
                "device": device,
                "decode_scale": args.scale,
                "mask_size": [height, width],
                "n_frames": len(frames),
                "frames_with_masks": len(covered),
                "prompt_frame": frame_idx,
                "tracked_object_ids": tracked,
                "subject_object_ids": list(SUBJECT_OBJECT_IDS),
                "prompts": {
                    str(obj_id): {
                        "points": state.points[obj_id],
                        "labels": state.labels[obj_id],
                    }
                    for obj_id in tracked
                },
                "written_at": datetime.now(UTC).isoformat(),
                "guide": "docs/specs/preprocessing/S7-barbell-annotation-guide.md",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(covered)} masks for {len(tracked)} plate(s) to {masks_dir}")
    print(f"review {out_dir / 'overlay.mp4'} before moving to Pass B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
