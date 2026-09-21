"""Pure mapping from RTMPose's Halpe-26 keypoints to PowerFlow's 15-joint skeleton.

No I/O, no `rtmlib` import -- takes plain per-keypoint `(x, y, score)` triples, so this mapping
is unit-testable without running inference (CLAUDE.md: TDD, synthetic fixtures). Mirrors
`pose_landmark_mapping.py`, which does the same job for MediaPipe's 33 BlazePose landmarks.

Halpe-26 is chosen over COCO-17 because it carries an explicit `head` and `neck`
(docs/specs/preprocessing/S5-rtmpose-migration.md §2), and both matter here:

- `head` is a real head point rather than the nose, which is what makes BlazePose's head height
  read oddly whenever the athlete is looking down -- most of a lift.
- the clavicles interpolate along **neck -> shoulder**, anatomically where a collarbone lies,
  rather than MediaPipe's shoulder -> nose line, which has no anatomical basis and was chosen
  only because BlazePose offers nothing better.

Index order verified against rtmlib's own `BodyWithFeet` output on
`11 July/50kg_Set3/Front` frame 185 (§2.1 requires this; a transposed index is exactly the
defect class that once put a whole skeleton in the ceiling rafters).
"""

from __future__ import annotations

from powerflow_pipeline.data.common.models import JointId

KeypointTriple = tuple[float, float, float]  # (x, y, score)

# Halpe-26 keypoint indices. 0-16 are COCO-17; 17-19 and the six foot points are Halpe's
# additions. Only the ones this mapping uses are named.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
HEAD = 17
NECK = 18
HIP_CENTRE = 19

N_KEYPOINTS = 26

# JointId -> the single Halpe-26 index it maps to directly.
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
    "head": HEAD,
}

# How far from the neck toward the shoulder a clavicle sits. A parameter rather than a constant
# baked into the mapping (§2.1); 0.5 places it at the midpoint of the collarbone.
DEFAULT_CLAVICLE_SHOULDER_WEIGHT = 0.5


def _interpolate(
    neck: KeypointTriple, shoulder: KeypointTriple, shoulder_weight: float
) -> KeypointTriple:
    nx, ny, ns = neck
    sx, sy, ss = shoulder
    return (
        shoulder_weight * sx + (1 - shoulder_weight) * nx,
        shoulder_weight * sy + (1 - shoulder_weight) * ny,
        min(ns, ss),  # a derived point's score never exceeds either source's
    )


def map_keypoints_to_joints(
    keypoints: dict[int, KeypointTriple],
    clavicle_shoulder_weight: float = DEFAULT_CLAVICLE_SHOULDER_WEIGHT,
) -> dict[JointId, KeypointTriple]:
    """One person's `{halpe26_index: (x, y, score)}` -> `{JointId: (x, y, score)}`.

    A keypoint missing from `keypoints` propagates as a missing joint, never a fabricated
    `(0, 0, 0)` -- the caller turns "missing" into `JointSeries`'s `None`/zero-confidence
    convention.
    """

    result: dict[JointId, KeypointTriple] = {}
    for joint_id, index in _DIRECT_MAP.items():
        if index in keypoints:
            result[joint_id] = keypoints[index]

    if NECK in keypoints:
        if LEFT_SHOULDER in keypoints:
            result["leftClavicle"] = _interpolate(
                keypoints[NECK], keypoints[LEFT_SHOULDER], clavicle_shoulder_weight
            )
        if RIGHT_SHOULDER in keypoints:
            result["rightClavicle"] = _interpolate(
                keypoints[NECK], keypoints[RIGHT_SHOULDER], clavicle_shoulder_weight
            )
    return result
