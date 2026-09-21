"""Concrete `Detector2D` backed by RTMPose (Halpe-26) through `rtmlib`.

Implements docs/specs/preprocessing/S5-rtmpose-migration.md. Unlike MediaPipe/BlazePose, this
is **top-down**: a YOLOX person detector proposes boxes for every person in frame, then pose is
estimated per box. That inverts the three BlazePose properties the spec blames for the measured
per-capture detection rates (99%, 98%, 42%, 12%) -- it is multi-person, its detector is not
head/torso-anchored, and `rtmlib.PoseTracker` carries a subject between frames.

Being multi-person means choosing the athlete is now this class's job rather than something the
model hides (§4). A competition frame from `11 July/50kg_Set3/Front` holds seven people.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import numpy as np

from powerflow_pipeline.data.common.models import HUMAN_SKELETON, JointId
from powerflow_pipeline.data.preprocess.halpe26_mapping import (
    DEFAULT_CLAVICLE_SHOULDER_WEIGHT,
    N_KEYPOINTS,
    map_keypoints_to_joints,
)
from powerflow_pipeline.data.preprocess.pose_geometry import (
    lookup_patch_depth_m,
    rgb_pixel_to_depth_pixel,
)
from powerflow_pipeline.data.preprocess.pose_model import JointPixelSeries
from powerflow_pipeline.data.preprocess.rtmpose_assets import (
    INPUT_SIZES,
    RTMPoseVariant,
    checkpoint_url,
    ensure_rtmpose_model,
)
from powerflow_pipeline.data.preprocess.tasks.ingest import frame_paths, read_frame

PixelPair = tuple[int, int]

# How far a candidate's per-joint scores must reach before it is considered a person at all.
# Below this the box is kept out of subject selection rather than competing with the athlete.
_MIN_CANDIDATE_SCORE = 0.3


def _bbox(keypoints: np.ndarray) -> tuple[float, float, float, float]:
    """`(x0, y0, x1, y1)` around one candidate's keypoints."""

    xs, ys = keypoints[:, 0], keypoints[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _bbox_area(keypoints: np.ndarray) -> float:
    x0, y0, x1, y1 = _bbox(keypoints)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def candidate_depth_m(
    keypoints: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
) -> float | None:
    """Median confident depth across a candidate's own keypoints, or `None`.

    §4's rule 1 reads "the median depth inside the candidate's bounding box". Sampling at the
    keypoints rather than over the whole box is the same idea applied where the person actually
    is: a bounding box around a bent-over lifter is mostly background, and averaging that in is
    what the whole depth-sampling problem is about.
    """

    readings: list[float] = []
    for x, y in keypoints:
        depth_row, depth_col = rgb_pixel_to_depth_pixel(round(y), round(x), rgb_size, depth_size)
        if not (0 <= depth_row < depth.shape[0] and 0 <= depth_col < depth.shape[1]):
            continue
        metres = lookup_patch_depth_m(depth, confidence, depth_row, depth_col)
        if metres is not None:
            readings.append(metres)
    return float(np.median(readings)) if readings else None


def select_subject(
    keypoints: np.ndarray,
    scores: np.ndarray,
    depth: np.ndarray | None,
    confidence: np.ndarray | None,
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int] | None,
) -> tuple[int | None, str]:
    """Pick the athlete among detected people; returns `(index, rule_that_fired)` (§4).

    Rule 1 is nearest by depth -- the lifter is on the platform nearest the camera and
    bystanders are behind them, and this pipeline has real LiDAR depth rather than having to
    guess from pixels. Rule 2, largest bounding box, is the fallback when depth is unusable for
    every candidate, which is itself diagnostic: on `11 July/50kg_Set3/Front` the subject sits
    9-11 m out, beyond reliable LiDAR range, so rule 1 cannot fire there at all.
    """

    viable = [
        index
        for index in range(len(keypoints))
        if float(np.max(scores[index])) >= _MIN_CANDIDATE_SCORE
    ]
    if not viable:
        return None, "none"

    if depth is not None and confidence is not None and depth_size is not None:
        depths = {
            index: candidate_depth_m(keypoints[index], depth, confidence, rgb_size, depth_size)
            for index in viable
        }
        measured = {index: value for index, value in depths.items() if value is not None}
        if measured:
            return min(measured, key=lambda index: measured[index]), "nearest_depth"

    return max(viable, key=lambda index: _bbox_area(keypoints[index])), "largest_bbox"


class RTMPoseDetector:
    """A `Detector2D` (structurally, via `pose_model.Detector2D`) backed by RTMPose Halpe-26."""

    def __init__(
        self,
        variant: RTMPoseVariant = "balanced",
        *,
        det_frequency: int = 1,
        clavicle_shoulder_weight: float = DEFAULT_CLAVICLE_SHOULDER_WEIGHT,
        use_depth_for_subject: bool = True,
        backend: str = "onnxruntime",
        device: str = "cpu",
    ) -> None:
        from functools import partial

        from rtmlib import BodyWithFeet, PoseTracker

        self._variant = variant
        self._det_frequency = det_frequency
        self._clavicle_shoulder_weight = clavicle_shoulder_weight
        self._use_depth_for_subject = use_depth_for_subject

        # `PoseTracker` constructs the solution itself, forwarding only backend/device/mode, so
        # the checkpoint paths are bound here instead. Passing `det`/`pose` explicitly makes
        # `BodyWithFeet` skip its own `MODE` table and its cache, which is what keeps the
        # pinned `models_cache/` copies (§7) the ones actually loaded.
        sizes = INPUT_SIZES[variant]
        solution = partial(
            BodyWithFeet,
            det=str(ensure_rtmpose_model("det", variant)),
            det_input_size=sizes["det"],
            pose=str(ensure_rtmpose_model("pose", variant)),
            pose_input_size=sizes["pose"],
        )
        self._tracker = PoseTracker(
            solution,
            det_frequency=det_frequency,
            tracking=True,
            backend=backend,
            device=device,
        )
        self.last_provenance: dict[str, Any] = {}

    def detect(self, rgb_path: Path, n_frames: int) -> dict[JointId, JointPixelSeries]:
        """Run RTMPose over one capture's frames, tracking a single chosen subject.

        This is the **only** place S5 touches depth, and it never produces a coordinate from
        it: depth is read solely to decide *which person in frame is the athlete* (§4's rule 1,
        nearest by depth), and the answer is an index into the detected people. Every keypoint
        this method returns comes from RGB alone. The metric lift is S6 (`pose_lift.py`).

        `use_depth_for_subject=False` removes even that read, leaving S5 a pure pixel-space
        stage, at the cost of falling back to §4's rule 2 (largest bounding box) when a capture
        holds bystanders. Depth is read from `rgb_path`'s own directory, since S4 Crop
        publishes `depth/` and `confidence/` beside `rgb.mp4`; their absence degrades selection
        the same way rather than failing.
        """

        pixel_positions: dict[JointId, list[PixelPair | None]] = {
            joint: [None] * n_frames for joint in HUMAN_SKELETON.joints
        }
        confidences: dict[JointId, list[float]] = {
            joint: [0.0] * n_frames for joint in HUMAN_SKELETON.joints
        }

        depth_dir, confidence_dir = rgb_path.parent / "depth", rgb_path.parent / "confidence"
        has_depth = self._use_depth_for_subject and depth_dir.is_dir() and confidence_dir.is_dir()
        depth_paths = frame_paths(depth_dir) if has_depth else []
        confidence_paths = frame_paths(confidence_dir) if has_depth else []
        depth_size: tuple[int, int] | None = None
        if depth_paths:
            sample = read_frame(depth_paths[0])
            depth_size = (sample.shape[1], sample.shape[0])

        rules: list[str] = []
        track_losses = 0
        rgb_size = (0, 0)

        with av.open(str(rgb_path)) as container:
            stream = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index >= n_frames:
                    break
                # rtmlib follows OpenCV's BGR convention, as its own examples do by reading
                # frames through cv2.imread.
                array = frame.to_ndarray(format="rgb24")[:, :, ::-1]
                rgb_size = (array.shape[1], array.shape[0])
                keypoints, scores = self._tracker(array)
                if len(keypoints) == 0:
                    track_losses += 1
                    continue

                depth = confidence = None
                if has_depth and frame_index < len(depth_paths):
                    depth = read_frame(depth_paths[frame_index])
                    confidence = read_frame(confidence_paths[frame_index])

                index, rule = select_subject(
                    keypoints, scores, depth, confidence, rgb_size, depth_size
                )
                rules.append(rule)
                if index is None:
                    track_losses += 1
                    continue

                self._record(
                    keypoints[index], scores[index], frame_index, pixel_positions, confidences
                )

        self.last_provenance = {
            "detector": "rtmpose",
            "variant": self._variant,
            "keypoint_format": "halpe26",
            "det_frequency": self._det_frequency,
            "det_checkpoint_url": checkpoint_url("det", self._variant),
            "pose_checkpoint_url": checkpoint_url("pose", self._variant),
            "det_input_size": list(INPUT_SIZES[self._variant]["det"]),
            "pose_input_size": list(INPUT_SIZES[self._variant]["pose"]),
            "subject_depth": self._use_depth_for_subject,
            "subject_selection_rules": sorted(set(rules)),
            "track_losses": track_losses,
            "rgb_size": list(rgb_size),
        }

        return {
            joint: JointPixelSeries(
                pixel_position=tuple(pixel_positions[joint]),
                confidence=tuple(confidences[joint]),
            )
            for joint in HUMAN_SKELETON.joints
        }

    def _record(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        frame_index: int,
        pixel_positions: dict[JointId, list[PixelPair | None]],
        confidences: dict[JointId, list[float]],
    ) -> None:
        raw = {
            index: (float(keypoints[index][0]), float(keypoints[index][1]), float(scores[index]))
            for index in range(min(N_KEYPOINTS, len(keypoints)))
        }
        joints = map_keypoints_to_joints(raw, self._clavicle_shoulder_weight)
        for joint, (x, y, score) in joints.items():
            clamped = max(0.0, min(1.0, score))
            if clamped <= 0.0:
                continue  # treat as not detected: keep the pixel/confidence default together
            pixel_positions[joint][frame_index] = (round(x), round(y))
            confidences[joint][frame_index] = clamped
