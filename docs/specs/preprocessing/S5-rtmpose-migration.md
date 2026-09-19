# S5 Pose — migrating the 2D detector to RTMPose

Status: **draft** | Derived from: `S5-pose-model-recommendation.md` (issue [#118](https://github.com/tony-yehui-lou/PowerFlow/issues/118)) | Updated: 2026-09-19

## Purpose

Replace S5 Pose's 2D keypoint detector — today MediaPipe/BlazePose — with RTMPose, run through
[`rtmlib`](https://github.com/Tau-J/rtmlib), while leaving the rest of the stage untouched:

```text
S4 Crop -> [ 2D detector ] -> depth back-projection -> floor-frame transform -> PoseDocument
                ^^^^^^^^^^
            this, and only this
```

The motivation is measured, not theoretical. MediaPipe's per-capture 2D detection rate (frames
in which it found the athlete at all) splits sharply:

| capture | frames | 2D detection rate |
|---|---|---|
| `11 July/50kg_Set3/Front` | 371 | 99% |
| `22 August/Snch/85kgSnch1` | 296 | 98% |
| `22 August/CnJ/75kgCnJ1` | 763 | **42%** |
| `22 August/Snch/38kgSnch1` | 430 | **12%** |

Same venue and rig across the August three, so this is per-capture (athlete distance, occlusion,
bystanders), not a systemic misconfiguration. Three properties of BlazePose explain it, none of
them tunable:

- **Single-subject by design.** It detects one ROI and tracks it; `num_poses=1` means that in
  competition footage with coaches, judges and waiting athletes in frame, it can lock onto the
  wrong person and never say so.
- **Head/torso-anchored detection.** BlazePose's detector stage descends from BlazeFace. An
  athlete bent double, facing away, head tucked between the arms — the entire first half of a
  snatch or clean — is a weak anchor.
- **Per-frame independence.** S5 runs it in `IMAGE` mode, so every frame is a fresh guess with
  no continuity across the clip.

RTMPose is top-down: a generic person detector proposes boxes for *every* person, then pose is
estimated per box. That inverts all three properties.

**In:** unchanged — S4 Crop's `rgb.mp4` for one capture, plus `n_frames`.
**Out:** unchanged — `dict[JointId, JointPixelSeries]`, the existing `Detector2D` contract.

## Scope, and what this explicitly does not fix

This changes **one implementation behind an existing seam**. No change to `pose_geometry.py`,
`depth_pose_model.py`, `tasks/pose.py`, `flow.py`, `pose_storage.py`, or `common/models.py`.

**It does not improve joint coverage in the stored document.** Measured separately: on
`22 August/CnJ/75kgCnJ1` and `38kgSnch1`, **42% of successful 2D joint detections are discarded
by the depth lookup**, because `JointSeries` requires `position` and `pixelPosition` to drop out
together and there is no depth to pair the pixel with. A perfect 2D detector still loses those.
Partial-skeleton frames are a depth problem and are out of scope here; only all-null frames are
in scope. Do not measure this migration by counting complete skeletons in `pose.json` — measure
it at the `Detector2D` boundary (§6).

## Assumptions

- The athlete is the person of interest in every capture, and is distinguishable from bystanders
  by position and distance (§4). No capture contains two people lifting simultaneously.
- The camera is static for the clip (already assumed pipeline-wide), so a subject chosen on one
  frame remains the subject unless they leave frame.
- Inference is offline batch on a Prefect worker. There is no real-time budget; accuracy and
  coverage outrank throughput.
- The project is research, not commercial (§8 — this is load-bearing for the weights).

## 1. Dependencies

`rtmlib` runs RTMPose from ONNX with **no mmcv, mmpose or mmdet**. Its requirements are numpy,
opencv-python, opencv-contrib-python and onnxruntime — of which the pipeline already has numpy
and opencv-contrib-python.

New dependencies, both requiring approval per `CLAUDE.md` ("Ask before adding dependencies"):

- `rtmlib`
- `onnxruntime` (CPU; `onnxruntime-gpu`, openvino or tensorrt are drop-in accelerations later)

`mediapipe` stays a dependency: MediaPipe remains selectable and remains the default until
RTMPose beats it on §6's criteria.

## 2. Model and skeleton choice

Use **Halpe-26**, not COCO-17.

COCO-17 carries nose, eyes, ears, shoulders, elbows, wrists, hips, knees and ankles. Halpe-26
adds an explicit **head**, **neck** and hip-centre point, and both of the extras matter here:

- `head` maps to a real head point instead of the nose. BlazePose's nose is why head height
  currently reads oddly whenever the athlete is looking down, which in weightlifting is most of
  the lift.
- `leftClavicle`/`rightClavicle` interpolate along **neck → shoulder**, which is anatomically
  where a collarbone lies. The current implementation interpolates 70/30 between shoulder and
  *nose*, a line with no anatomical basis; it was chosen only because BlazePose offers nothing
  better.

Neither skeleton has true clavicles, so those two joints remain derived. This is inherent to
every 2D keypoint model and is not resolvable within this architecture (see
`S5-pose-model-recommendation.md` for the body-model alternatives that do have them).

### 2.1 Joint mapping

| `JointId` | Halpe-26 source |
|---|---|
| `leftAnkle` / `rightAnkle` | 15 / 16 |
| `leftKnee` / `rightKnee` | 13 / 14 |
| `leftHip` / `rightHip` | 11 / 12 |
| `leftShoulder` / `rightShoulder` | 5 / 6 |
| `leftElbow` / `rightElbow` | 7 / 8 |
| `leftWrist` / `rightWrist` | 9 / 10 |
| `head` | 17 (`head`) |
| `leftClavicle` | interpolate(18 `neck`, 5 `leftShoulder`) |
| `rightClavicle` | interpolate(18 `neck`, 6 `rightShoulder`) |

**These indices are stated from the published Halpe-26 layout and MUST be verified against
rtmlib's actual output ordering before being trusted.** A silently transposed index is exactly
the class of defect that put a whole skeleton in the ceiling rafters last time; the check is
cheap (§6.3) and skipping it is not acceptable.

The clavicle interpolation weight is a parameter, not a constant baked into the mapping.

## 3. Tracking and continuity

Use rtmlib's **`PoseTracker`**, not bare per-frame inference. It tracks the subject between
frames and runs the person detector only every `det_frequency` frames.

This is the mechanism that addresses the all-null frames: a tracked subject carries through
frames where from-scratch detection fails, which is precisely the bent-over/facing-away portion
of each lift.

- `det_frequency` is a config parameter. Lower values re-detect more often (more robust to track
  drift, slower); higher values lean on tracking.
- **Track loss is recorded, not hidden.** A frame where the tracker holds no subject emits the
  same all-`None`/zero-confidence series the current detector does — S5 must never interpolate a
  position it did not observe.
- rtmlib documents no temporal smoothing. Smoothing (OneEuro, Savitzky-Golay) is **out of scope**
  for this migration: it changes stored values rather than recovering missing ones, and belongs
  in a consumer or a later, separately-validated stage.

## 4. Subject selection

Top-down detection returns every person in frame. Choosing among them is a new responsibility
that MediaPipe's single-subject design hid rather than solved — badly, by taking whoever ranked
highest.

On the first frame where any person is detected, select the subject by, in order:

1. **Nearest by depth.** The median depth inside the candidate's bounding box, using the same
   confidence-gated lookup as `pose_geometry.lookup_patch_depth_m`. The lifter is on the platform
   nearest the camera; bystanders are behind them. This pipeline has real LiDAR depth available
   and should use it rather than guess from pixels.
2. **Largest bounding-box area**, if depth is unusable for every candidate (which is itself
   diagnostic — see the 11 July Front capture, whose subject sat 9–11 m out, beyond reliable
   LiDAR range).

Thereafter the `PoseTracker`'s identity governs; re-selection happens only after a track loss.
The chosen subject's bounding box and the rule that selected it are recorded per capture as
provenance (§5).

## 5. Output contract

The `Detector2D` return value is unchanged: `dict[JointId, JointPixelSeries]`, all 15 joints
present, each series exactly `n_frames` long, `pixel_position` `None` exactly where `confidence`
is `0.0`.

Two changes in what fills it:

- **Confidence** is RTMPose's per-keypoint score, passed through unchanged. This is a cleaner
  quantity than BlazePose's `visibility`, which conflates "occluded" with "outside the frame".
- **Pixel coordinates** are in S4 Crop's image space, as before. RTMPose is given the frame
  as-is: the square-padding workaround in `mediapipe_pose_detector.pad_to_square` is a MediaPipe
  ROI-projection defect and **must not** be carried over without first confirming RTMPose needs
  it (it should not — top-down models crop to the person box themselves).

A sidecar of per-capture detector provenance is written alongside the pose document: detector
name and model variant, checkpoint URL and file hash, `det_frequency`, keypoint format, the
subject-selection rule that fired, and the number of track losses. Without this, two `pose.json`
files produced by different detectors are indistinguishable on disk.

## 6. Validation

A migration is accepted only on measurements, against the four reference captures in §Purpose.

### 6.1 Coverage, at the `Detector2D` boundary

Measure frames in which all 15 joints were detected in 2D, **before** the depth lookup discards
any. RTMPose must be no worse than MediaPipe on every reference capture, and materially better
on `75kgCnJ1` (42%) and `38kgSnch1` (12%).

### 6.2 Anatomical plausibility

Bone length is pose-invariant and is the sharpest available proxy for correctness. Current
medians on `11 July/50kg_Set3/Front`, with MediaPipe and the depth path fixed: femur 0.35 m,
tibia 0.36 m, upper arm 0.38 m. Plausible adult ranges: femur 0.38–0.50, tibia 0.36–0.45, torso
0.45–0.60, upper arm 0.26–0.36.

RTMPose must not regress these medians, and should narrow their spread. Note that bone length is
measured *after* the depth lift, so a regression here may indict depth rather than the detector —
report both the 2D pixel-space limb lengths and the 3D metric ones so the two are separable.

### 6.3 Subject and index verification (mandatory)

Render the detected skeleton onto source frames for each reference capture and inspect them. The
skeleton must land on the athlete, and left/right and head/foot must not be transposed. This is
the check that would have caught both the ceiling-rafter phantoms and would catch a wrong
Halpe-26 index; it costs one image per capture.

### 6.4 Runtime

Record wall-clock seconds per 1,000 frames at the capture's native resolution, on CPU. There is
no pass/fail threshold — this is an offline batch stage — but a 10x regression is worth knowing
before it is deployed across the corpus.

## 7. Configuration and rollout

- `PreprocessConfig` gains a `pose_detector` field: `"mediapipe"` (default) or `"rtmpose"`, plus
  RTMPose tunables (`rtmpose_model`, `rtmpose_det_frequency`, `rtmpose_keypoint_format`).
- `cli.py` gains `--pose-detector`, replacing the hardcoded
  `DepthBackedPoseModel(MediaPipePoseDetector())` construction.
- Both detectors remain installed and selectable. MediaPipe stays the default until §6 is
  satisfied; the default flips in a separate, small change that cites the measurements.
- Model assets download on first use into the git-ignored `models_cache/`, mirroring
  `mediapipe_assets.py`: pinned URLs per variant (never `latest`), a minimum-size check, and an
  atomic rename so a truncated download is never mistaken for a valid model.

## 8. Licensing

The RTMPose *code* is Apache-2.0. The *checkpoints* are not clearly covered: OpenMMLab has an
open, unanswered request to state the licence of published model-zoo weights
([mmpose#3273](https://github.com/open-mmlab/mmpose/issues/3273)), and the weights are trained
partly on AI Challenger, whose terms are likewise unclear.

For the current research use this is acceptable. **It is a blocker if PowerFlow is ever
commercialised**, and the swap would then have to be revisited — so the detector must remain
selectable rather than becoming load-bearing. `S5-pose-model-recommendation.md` previously
described these weights as "Apache-2.0 licensed end to end (code and released weights)"; that
claim is wrong and is corrected here.

The person detector shipped with rtmlib (RTMDet or YOLOX) carries its own licence and must be
checked separately — YOLOX is Apache-2.0, but confirm what rtmlib actually downloads.

## Open questions

- **Which person detector, and at what size.** rtmlib offers RTMDet and YOLOX in several sizes.
  The tradeoff is detector recall on a bent-over athlete versus runtime; undecided pending §6.1.
- **Whether `det_frequency` can be raised safely.** Cheaper, but track drift over a fast lift is
  unmeasured. Start at every frame and raise it only against §6.1.
- **Halpe-26 index order in rtmlib's actual output** (§2.1). Stated from the published layout;
  unverified in code.
- **Whether the subject-selection depth rule survives out-of-range captures.** On 11 July Front
  the athlete sits 9–11 m out with confidence-`0` depth across their whole body, so rule 1 will
  fall through to rule 2 on exactly the capture where bystander confusion is least likely. Rule 1
  is untested on a capture that has both bystanders *and* usable depth.
- **Whether RTMPose needs any input-shape accommodation** at 1420x1668 and 1416x1886 (§5). It
  should not, but MediaPipe should not have either.
