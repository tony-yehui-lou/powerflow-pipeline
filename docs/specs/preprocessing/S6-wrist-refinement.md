# S6 Pose — Wrist Refinement Against the Barbell

Status: **implemented** | Scope: S6 Pose; reads S5 Barbell's detections | Updated: 2026-10-20

## Problem

The pose detector's **far wrist** is unreliable whenever the arm is hidden. The arm is hidden
overhead, behind the head, the bar and the plates. On the `10 September` captures:
- the far wrist jumps **130–356 px between frames** (95th percentile of steps while overhead);
- its confidence is 0.43–0.69;
- it lands on the hip, on the far plate, or on a bystander.

The near wrist is steady (18–40 px p95). The far elbow has the same problem; it is not addressed
here.

## Idea

The hands grip the bar, the grip doesn't change during a lift, and S5 tells us where the bar is
on every frame. The near hand is seen; the far hand is its mirror image across the bar's centre.

## Method

The code is `preprocess/bar_geometry.py` (where the bar is) and `preprocess/wrist_refine.py`
(where the wrists go), called from `tasks/pose.py` after detection and before `pose.json` is
written.

### 1. Place the bar

`bar_geometry.solve_track` gives the bar's 3D pose per frame from S5's cap and collar pixels, in
S4's levelled camera frame. It is the level-bar solve used by powerflow-ui's Bar Height view:
- detections scoring below 0.5 are dropped;
- the yaw is interpolated where 2 px of noise moves it by more than 1°;
- a cap is rejected when its yaw is more than 20° off its neighbours';
- a frame with the collar but no cap takes the collar's depth from solved frames on both sides,
  for gaps of up to 30 frames.

The bar type comes from the capture's raw `metadata.yaml` (`lift.barbell_type`) and is never
guessed.

### 2. Find the near wrist on the bar

For each frame:
1. Find the bar point nearest the near wrist's viewing ray. The wrist is **on the bar** when that
   point projects within 60 px of it.
2. On those frames, place the wrist in 3D at that point's depth and split its position relative
   to the bar centre into:
   - a part **along** the bar: the grip;
   - a part **across** it: the wrist's offset from the bar. The wrist sits above the bar at the
     floor and below it overhead.

**Near vs. far** is decided by which wrist sits nearer the cap end, counting only frames where
that wrist is on the bar. Occluded far-wrist guesses can land nearer the cap than the near hand.

### 3. Grip

The half grip is the median along-bar distance over frames where the near wrist is on the bar
with confidence ≥ 0.7. It needs at least 20 such frames; otherwise nothing is corrected. Hands
don't slide on the bar, so this is one number per capture.

### 4. Mirror

The far wrist is placed at **centre + half grip** along the bar, plus the near wrist's across-bar
offset: the near wrist reflected through the plane perpendicular to the bar at its centre. Where
the near wrist itself is off the bar (a glitch, or the detector on a bystander), its offset is
interpolated from neighbouring frames instead of copied.

A constant pixel residual is learned at rest and added. It absorbs what the mirror misses, about
(+10, −7) px. It is learned from frames where:
- the bar is within 100 mm of its lowest position;
- the far wrist is confident (≥ 0.7);
- the far wrist is within 120 px of the uncorrected prediction.

It needs at least 10 such frames; otherwise the residual is zero.

### 5. Apply only while the hands hold the bar

A frame counts as hands-on-bar when the majority of known near-wrist states within ±5 frames say
so, so a one- or two-frame glitch can't switch the correction off. On those frames:
- **the far wrist** is always replaced by the mirrored prediction;
- **the near wrist** is replaced only where it is itself off the bar.

After the release the detector's wrists are left alone. A corrected point takes the **near
wrist's confidence**, because the prediction is built from the near hand.

## Output

- **`pose.json`**: unchanged schema, with the corrected wrists. S7 Lift lifts them like any other
  joint.
- **`wrist_refinement.json`** beside it: which wrist is near and far, the half grip, the residual,
  the number of frames with a placed bar and with hands on the bar, and the exact frame lists
  `farCorrected` and `nearCorrected`. It is written once, and replaced with `--overwrite`.
- **Manifest `derived`**: `bar_frames_placed`, `wrist_far`, `wrist_half_grip_mm`,
  `wrist_frames_on_bar`, `wrist_far_corrected`, `wrist_near_corrected`.
- **When the correction cannot run** (no detections, an undeclared bar, detections in another image
  size, or too few grip frames): a manifest warning, and the wrists exactly as detected. Never a
  rejection.

## Configuration

| | |
|---|---|
| `--barbell PATH` / `barbell_root` | S5's detections: `<PATH>/<capture path>/predictions.json`. Unset means S6 is unchanged. |
| `barbell_min_score` | default 0.5 |

Thresholds are module constants, each documented where it is defined.

Today the detections are `predictions.json` from `scripts/predict_landmarks.py`
(`../data/annotations/barbell/v3`). When the S5 stage of `S5-barbell-detection-stage.md` lands,
only `barbell_input.py` changes.

## Results

These come from the pipeline code, run read-only on the existing S6 detections. Overhead means
the bar centre is at least 1.4 m above its lowest position.

| Capture | Far-wrist step p95 (before → after) | Largest step | Half grip | Frames corrected |
|---|---|---|---|---|
| 30kg1 | 300 → 15 px | 366 → 21 px | 468 mm | 260 / 261 |
| 30kg2 | 320 → 16 px | 349 → 174 px | 461 mm | 347 / 356 |
| 40kg1 | 172 → 15 px | 182 → 17 px | 466 mm | 395 / 395 |
| 45kg1 | 208 → 58 px | 431 → 70 px | **544 mm** | 76 / 367 |
| 50kg1 | 356 → 13 px | 632 → 28 px | 472 mm | 375 / 398 |
| 50kg2 | 130 → 14 px | 236 → 25 px | 465 mm | 449 / 476 |
| 50kg3 | 166 → 15 px | 410 → 31 px | 472 mm | 385 / 433 |

After correction the far wrist is as steady as the near one (18–40 px p95). In the prototype the
prediction sat 3–12 px from confident far-wrist detections at the floor.

Unit tests: a synthetic lift (rest, rise, hold, release) with a far wrist that jumps, a near-wrist
glitch, and a wrist offset that flips overhead. The corrected far wrist stays within 5 px of truth
throughout, and the release is untouched.

## Known limits

- **45kg1:** the cap is never seen at rest, so the bar is placed on only 139 of 367 frames. Its
  grip of 544 mm is out of line with the lifter's 461–472 mm on every other capture, so treat its
  corrections with suspicion. A grip carried from the same lifter's other captures would fix it.
- **The mirror assumes a symmetric posture,** with both wrists the same distance off the bar. That
  is true of a snatch held overhead. A press-out, or an uneven catch, breaks it.
- **The near wrist is placed at its bar point's depth.** A wrist 40 mm off the bar is not quite
  there, which costs up to ~5 px on the synthetic lift.
- **Tilt is assumed zero,** as everywhere in the bar solve.
- **No ground truth.** The overhead accuracy is judged by smoothness and by the few confident
  far-wrist detections, not by a measured hand position.
- **The far elbow is not addressed.** A bone-length constraint from the corrected wrist is the
  natural next step (`S6-bone-constrained-reconstruction.md`).
