"""Pure mapping from a 33-point BlazePose-style landmark set to PowerFlow's 15-joint skeleton.

No I/O, no `mediapipe` import -- takes plain per-landmark `(x, y, confidence)` triples (already
extracted from whatever detector produced them), so this mapping is unit-testable without
running real inference (CLAUDE.md: TDD, synthetic fixtures). `x`/`y` are passed through
whatever unit the caller used (normalized `[0, 1]` or pixel) -- values from the same source
frame are always in the same unit, so this function's arithmetic is unit-agnostic.
"""

from __future__ import annotations

from powerflow_pipeline.data.common.models import JointId

LandmarkTriple = tuple[float, float, float]  # (x, y, confidence)

# BlazePose's fixed 33-landmark topology (developers.google.com/mediapipe/solutions/vision/
# pose_landmarker), stable since the original BlazePose paper. Only the landmarks this
# mapping actually uses are named.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# JointId -> the single landmark index it maps to directly.
_DIRECT_MAP: dict[JointId, int] = {
    "leftAnkle": LEFT_ANKLE,
    "rightAnkle": RIGHT_ANKLE,
    "leftKnee": LEFT_KNEE,
    "rightKnee": RIGHT_KNEE,
    "leftHip": LEFT_HIP,
    "rightHip": RIGHT_HIP,
    "leftShoulder": LEFT_SHOULDER,
    "rightShoulder": RIGHT_SHOULDER,
    "leftElbow": LEFT_ELBOW,
    "rightElbow": RIGHT_ELBOW,
    "leftWrist": LEFT_WRIST,
    "rightWrist": RIGHT_WRIST,
    "head": NOSE,
}

# BlazePose has no clavicle landmark. Approximated as a point mostly at the shoulder, pulled
# slightly toward the nose -- the collarbone's anatomical position, between the shoulder joint
# and the base of the neck (docs/specs/preprocessing/S5-pose-model-recommendation.md, Gaps:
# "Joint-to-landmark mapping"). Weighted toward the shoulder so a bad interpolation doesn't
# collapse two visually distinct skeleton joints onto the same point.
_CLAVICLE_SHOULDER_WEIGHT = 0.7


def _interpolate(a: LandmarkTriple, b: LandmarkTriple, weight_a: float) -> LandmarkTriple:
    ax, ay, av = a
    bx, by, bv = b
    return (
        weight_a * ax + (1 - weight_a) * bx,
        weight_a * ay + (1 - weight_a) * by,
        min(av, bv),  # a derived point's confidence never exceeds either source's
    )


def map_landmarks_to_joints(landmarks: dict[int, LandmarkTriple]) -> dict[JointId, LandmarkTriple]:
    """One frame's raw `{landmark_index: (x, y, confidence)}` -> `{JointId: (x, y, confidence)}`.

    A landmark missing from `landmarks` (the detector found no pose at all this frame, or ran
    with a model variant that drops some points) propagates as a missing joint, never a
    fabricated `(0, 0, 0)` -- the caller is responsible for turning "missing" into
    `JointSeries`'s `None`/zero-confidence convention.
    """

    result: dict[JointId, LandmarkTriple] = {}
    for joint_id, index in _DIRECT_MAP.items():
        if index in landmarks:
            result[joint_id] = landmarks[index]

    if LEFT_SHOULDER in landmarks and NOSE in landmarks:
        result["leftClavicle"] = _interpolate(
            landmarks[LEFT_SHOULDER], landmarks[NOSE], _CLAVICLE_SHOULDER_WEIGHT
        )
    if RIGHT_SHOULDER in landmarks and NOSE in landmarks:
        result["rightClavicle"] = _interpolate(
            landmarks[RIGHT_SHOULDER], landmarks[NOSE], _CLAVICLE_SHOULDER_WEIGHT
        )
    return result
