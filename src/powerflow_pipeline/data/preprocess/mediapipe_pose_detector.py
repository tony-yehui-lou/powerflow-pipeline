"""Concrete `Detector2D` wrapping MediaPipe's Pose Landmarker (Tasks API).

Pinned to `mediapipe==0.10.35` in `pyproject.toml`: newer 1.x releases hard-crash
`PoseLandmarker` on macOS via an unconditional Metal graph-service initialization inside the
compiled graph -- an uncatchable native `abort()`, not a Python exception, reproduced while
building this module and tracked upstream at
https://github.com/google-ai-edge/mediapipe/issues/6356. See
docs/specs/preprocessing/S5-pose-model-recommendation.md for the full model-choice writeup.
"""

from __future__ import annotations

from pathlib import Path

import av
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from powerflow_pipeline.data.common.models import HUMAN_SKELETON, JointId
from powerflow_pipeline.data.preprocess.mediapipe_assets import ensure_pose_landmarker_model
from powerflow_pipeline.data.preprocess.pose_landmark_mapping import map_landmarks_to_joints
from powerflow_pipeline.data.preprocess.pose_model import JointPixelSeries

PixelPair = tuple[int, int]


def pad_to_square(array: np.ndarray) -> np.ndarray:
    """Pad an RGB frame to a square with black, anchored top-left.

    MediaPipe mis-projects its ROI on non-square input -- it warns `Using NORM_RECT without
    IMAGE_DIMENSIONS is only supported for the square ROI` on every such frame, and on a real
    1420x1668 capture (11 July/50kg_Set3/Front) that put the entire detected skeleton up in
    the ceiling rafters instead of on the lifter filling the lower two-thirds of the frame.
    Padding bottom/right only leaves every original pixel at its own coordinate, so a landmark
    normalized against the square maps back by multiplying by the square's side -- there is no
    offset to undo, unlike a centred letterbox.
    """

    height, width = array.shape[:2]
    side = max(height, width)
    if height == width:
        return array
    padded = np.zeros((side, side, array.shape[2]), dtype=array.dtype)
    padded[:height, :width] = array
    return padded


class MediaPipePoseDetector:
    """A `Detector2D` (structurally, via `pose_model.Detector2D`) backed by BlazePose.

    Runs each frame independently in `IMAGE` mode -- simpler than `VIDEO` mode's monotonic-
    timestamp bookkeeping, at the cost of no temporal smoothing between frames. Good enough
    for a first S5 implementation; a future pass can move to `VIDEO` mode without changing
    this class's `Detector2D` contract.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
    ) -> None:
        """`min_pose_detection_confidence`/`min_pose_presence_confidence` gate whether a pose
        is found at all (not the per-joint `visibility` this class stores as `confidence`).

        Both keep MediaPipe's own 0.5 default. An earlier revision lowered them to 0.05 after
        a real session (11 July/50kg_Set3/Front) detected nothing at 0.5 -- that was treating
        the symptom: the real defect was non-square input (see `pad_to_square`), and the low
        threshold did not recover the lifter, it manufactured phantom skeletons in the ceiling
        rafters which then back-projected to ~10 m of nonsense. With the padding fixed, 0.5
        finds the lifter with per-joint visibility of 0.67-1.00 on that same footage, so there
        is no reason to accept detections the model itself doubts.
        """

        path = model_path or ensure_pose_landmarker_model("full")
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, rgb_path: Path, n_frames: int) -> dict[JointId, JointPixelSeries]:
        pixel_positions: dict[JointId, list[PixelPair | None]] = {
            joint: [None] * n_frames for joint in HUMAN_SKELETON.joints
        }
        confidences: dict[JointId, list[float]] = {
            joint: [0.0] * n_frames for joint in HUMAN_SKELETON.joints
        }

        with av.open(str(rgb_path)) as container:
            stream = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index >= n_frames:
                    break
                self._detect_one_frame(frame, frame_index, pixel_positions, confidences)

        return {
            joint: JointPixelSeries(
                pixel_position=tuple(pixel_positions[joint]),
                confidence=tuple(confidences[joint]),
            )
            for joint in HUMAN_SKELETON.joints
        }

    def _detect_one_frame(
        self,
        frame: av.VideoFrame,
        frame_index: int,
        pixel_positions: dict[JointId, list[PixelPair | None]],
        confidences: dict[JointId, list[float]],
    ) -> None:
        padded = pad_to_square(frame.to_ndarray(format="rgb24"))
        side = padded.shape[0]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return  # no pose found this frame -- every joint stays at its None/0.0 default

        # Both axes scale by `side`, not by the original width/height: the landmarks are
        # normalized against the padded square (see `pad_to_square`).
        raw = {
            index: (landmark.x * side, landmark.y * side, landmark.visibility or 0.0)
            for index, landmark in enumerate(result.pose_landmarks[0])
        }
        for joint, (x, y, confidence) in map_landmarks_to_joints(raw).items():
            clamped = max(0.0, min(1.0, confidence))
            if clamped <= 0.0:
                continue  # treat as not detected: keep the pixel/confidence default together
            pixel_positions[joint][frame_index] = (round(x), round(y))
            confidences[joint][frame_index] = clamped
