# S8 Kinematics — Joint Angles from the Solved 3D Joints

Status: **draft 1** | Scope: GitHub issue [#70](https://github.com/tony-yehui-lou/PowerFlow/issues/70) (Joint Angle Calculation) under epic [#65](https://github.com/tony-yehui-lou/PowerFlow/issues/65) (Joint) | Builds on: [`S8-joint-hva-draft3.md`](S8-joint-hva-draft3.md) | Updated: 2026-10-21

**Decided before this draft (user, 2026-10-21):**

1. Angles are **included angles**: 180° means the limb is in a straight line.
2. **3D only.** No angle is measured on the 2D image, and there is no 2D fallback.
3. **No true ankle angle yet.** The skeleton has no foot points. Keeping RTMPose's toe and heel
   keypoints is a separate issue (§8). Until then, the shin's angle to vertical stands in.

**In:** S8's own per-frame solve (`skeleton_solve.FrameFit`), S5's bar track, and S6's near side,
all of which `joint_hva.build` already holds.

**Out:** additions to `joint_hva.json`. There is no new file, stage or CLI flag. The additions are:
- per joint per frame: `angleDeg`, its source and its flags;
- a per-capture `segments` block: shin, thigh and trunk angles to vertical.

---

## 1. Why 3D, and why S8

A joint angle is a property of three points in space. Measuring it on the image gives the angle
of their **projection**, which equals the true angle only when the limb lies in a plane facing
the camera square-on. Two facts rule that out here:

- **Most captures are oblique.** The 22 August captures are one ~45° three-quarter view
  (`2026-09-18-single-camera-oblique-capture-design.md`). In that view, a knee bent forward
  towards the camera shows a projected angle far from its real one.
- **S8 already places every joint in 3D.** It does so per frame, with the bone lengths held to
  the lifter's measurements (`S8-joint-hva-draft3.md` §3).

So the angle is computed from S8's solved points, inside S8. It follows the rule set for heights
(draft 3, feedback 2026-10-20, later): **purely calculated from that frame's solution**. There is
no smoothing, no temporal filter and no adjustment.

---

## 2. Definitions

### 2.1 Frame and axes

S8's points are in S4's levelled camera frame, in millimetres: X right, Y down, Z forward. S3
levels the floor, so:

- **up** is exactly `Û = (0, −1, 0)`;
- the frame has **no "forward" axis tied to the athlete**, because S3 never corrects yaw.

A signed angle needs to know which way the athlete faces. That comes from the
**left-to-right axis** `m̂`, per frame, taken from the first source that exists:

1. **`bar`:** S5's bar axis on that frame (`BarPose.axis`, cap → collar). Its sign is fixed so
   it points from the lifter's left to right:
   - by `m · (rightHip − leftHip) > 0`, when both hips are solved;
   - otherwise by S6's near side, assuming the detected cap is on the near end (open
     question 3).
2. **`hips`:** `rightHip − leftHip`, when there is no bar on the frame.
3. **`null`:** the angles are published unsigned and flagged `unsigned` (§2.5).

Before use, `m` is projected onto the horizontal plane and normalised:
`m̂ = normalise(m − (m·Û)Û)`. The bar is level, so this changes it only by noise. A hip vector
can tilt.

The lifter's **forward** axis is then `F̂ = Û × m̂`. This holds because the camera frame is
right-handed, as is the body's (forward × up = right). A synthetic test pins it down (§6.1).

### 2.2 The included angle

At joint B, with neighbours A and C, let `u = A − B` and `v = C − B`. Then:

```
θ = atan2(‖u × v‖, u · v)          ∈ [0°, 180°]
```

`atan2` is used rather than `acos(û·v̂)`, because `acos` loses precision near 0° and 180°. That is
exactly where a locked-out knee or elbow sits.

### 2.3 The sign: which side of straight

Taken alone, `θ` cannot tell a hip flexed 10° from a hip extended 10° past straight. Both read
170°. A full extension at the top of the pull and an elbow locked overhead both live there.

So for each joint the angle is measured **on the side it flexes towards**, over the full circle:

```
σ = s_j · ((u × v) · m̂)
angleDeg = θ          if σ ≥ 0      (the flexion side: ≤ 180°)
         = 360° − θ   if σ < 0      (past straight: > 180°)
```

`s_j` is a fixed constant per joint, chosen so that anatomical flexion gives `σ ≥ 0`. It is
worked out in the athlete's own axes, with `m̂ = R` (right), `Û = U`, `F̂ = F`, and
`U × F = −R`:

| Joint | `u` | `v` | Flexion example | `(u×v)·m̂` at flexion | `s_j` |
|---|---|---|---|---|---|
| Knee | hip − knee | ankle − knee | thigh up, shank back: `u=U, v=−F` | `+1` | `+1` |
| Hip | shoulder − hip | knee − hip | trunk up, thigh forward: `u=U, v=F` | `−1` | `−1` |
| Elbow | shoulder − elbow | wrist − elbow | upper arm up, forearm forward: `u=U, v=F` | `−1` | `−1` |

The same `m̂` serves both sides of the body, because the left and right legs flex in the same
sense about the left-to-right axis.

At straight (180°), the cross product vanishes and both branches give 180°, so the signed angle
is **continuous through straight**. A noisy lockout reads 178°, 181°, 179°, rather than folding
back at 180°.

### 2.4 The shoulder

**What it measures:** the angle between the trunk and the upper arm, at the shoulder, with
`u = hip − shoulder` and `v = elbow − shoulder` (same side). It is the same included angle as
§2.2:

- about **0°**: the arm hangs along the trunk (the pull);
- about **90°**: the arm is horizontal in front (a high front rack has the elbows up near here);
- about **180°**: the arm is overhead, in line with the trunk (snatch receive, jerk lockout);
- **past 180°**: the arm is behind the line of the trunk overhead. In the snatch, "arms behind the
  ears" overhead is a coached position, and this is where it shows.

**Range:** the shoulder spends much of a lift near 0° and near 180°, so a single wrap point at
either one would make the number jump. Its signed angle is therefore reported over
**(−90°, 270°]**. Forward of the trunk is positive. The sign rule is that of §2.3, with
`s_shoulder = +1`, since at flexion (`u = −U, v = F`) `(u×v)·m̂ = +1`:

```
angleDeg = θ          if (u × v)·m̂ ≥ 0
         = −θ         otherwise, then + 360° if below −90°
```

A hanging arm slightly behind the trunk reads −5°. An arm overhead slightly behind the ears reads
190°.

**Caveat:** the shoulder is the weakest keypoint in this lift. The bar crosses it, and S6
interpolates the clavicles from it. Expect its flags to fire more often than the knee's. Whether
the UI shows it at all is open question 1.

### 2.5 Joints that get an angle

| Joint | Angle | Notes |
|---|---|---|
| `leftKnee`, `rightKnee` | §2.3 | |
| `leftHip`, `rightHip` | §2.3 | The trunk segment is hip → same-side shoulder, so spinal flexion is folded into the hip angle |
| `leftElbow`, `rightElbow` | §2.3 | |
| `leftShoulder`, `rightShoulder` | §2.4 | |
| `leftAnkle`, `rightAnkle` | none | No foot point: see `segments.*Shin` (§3) and §8 |
| `leftWrist`, `rightWrist` | none | No hand point |
| clavicles, `head` | none | Not anatomical joints of interest |

For a joint with no angle, `angleDeg` is `null` as a whole, not an array of nulls. That matches
the UI spec's existing convention.

### 2.6 When an angle is null

On a frame, `angleDeg[i]` is `null` when any of these holds:

- the frame has no `constrained` solve (`frameStatus.method[i] != "constrained"`):
  - this includes `bar_plane` frames. Their points are placed on one plane at fixed lateral
    offsets, so their angle is a projected angle, which is the thing decision 2 rules out;
- any of the three joints is null or `unseen`;
- either segment is shorter than `MIN_SEGMENT_MM = 50`. The shortest segment used is the
  forearm, at a few hundred mm, so this catches only a broken solve.

When `m̂` is null, the angle is still published, as `θ` alone (≤ 180°), and flagged `unsigned`.

### 2.7 Provenance and flags, per joint per frame

- **`angleSource[i]`:**
  - `"observed"`: all three joints are `observed`;
  - `"weak"`: otherwise.

  `unseen` never reaches here, because §2.6 already nulls the angle.
- **`angleFlags[i]`:**
  - inherits `uncertain` and `ground_conflict` from any of the three joints;
  - adds `branch_ambiguous` when the frame is;
  - adds `unsigned` when the frame has no `m̂`;
  - adds `off_sagittal` when `θ < 170°` and the angle's plane normal `n̂ = (u × v)/‖u × v‖`
    meets `m̂` at more than 45° (`|n̂ · m̂| < cos 45°`). The limb is then bending mostly sideways
    (knees caving, an elbow flared). The magnitude is still the true 3D angle, but the
    flexion/extension sign means little.

No angle σ in v1: S8 exports only each height's σ, not each joint's 3D covariance (open
question 2).

---

## 3. Segment angles to vertical

These are yaw-independent, because they need only the vertical, which S3 makes exact. They are
also the measures coaches read most directly: shin angle, back angle.

For a segment with **proximal** end P and **distal** end D, let `d = P − D`:

```
inclinationDeg = atan2(d · F̂, d · Û)       ∈ (−180°, 180°]
```

- 0° is vertical;
- positive means the proximal end is ahead of the distal end;
- with no `m̂`, it is published as the unsigned `atan2(‖d − (d·Û)Û‖, d · Û)` and flagged
  `unsigned`.

| Segment | P | D | Reads as |
|---|---|---|---|
| `leftShin`, `rightShin` | knee | ankle | Shin angle; knees ahead of the ankles is positive |
| `leftThigh`, `rightThigh` | hip | knee | Hips behind the knees (a deep squat) is negative |
| `trunk` | mid-shoulder | mid-hip | Back angle; a forward lean is positive |

`mid-shoulder` and `mid-hip` are the midpoints of the two solved shoulders and the two solved
hips. If either side is missing, that midpoint is null. Nulls, sources and flags follow
§2.6–§2.7, with two endpoints in place of three joints. `off_sagittal` does not apply.

---

## 4. Output

`joint_hva.json` goes to **`schemaVersion` 1.1.0**. Fields are only added, and a 1.0.0 reader
ignores them.

Each joint in `joints` gains:

```jsonc
"angleDeg":    "[<float> | null, ...] | null",   // null as a whole for joints in §2.5 with no angle
"angleSource": "[\"observed\" | \"weak\" | null, ...] | null",
"angleFlags":  "[[<flag>, ...], ...] | null"
```

`frameStatus` gains `angleAxis[]`: `"bar" | "hips" | null` per frame (§2.1).

The document gains, at the top level:

```jsonc
"angleConvention": {
  "joint": "included; 180 = straight; > 180 = past straight (knee, hip, elbow)",
  "shoulder": "included trunk-to-upper-arm; (-90, 270]; 0 = arm along trunk, positive = forward",
  "segment": "inclination from vertical; positive = proximal end ahead of distal end"
},
"segments": {
  "<segmentId>": {
    "inclinationDeg": "[<float> | null, ...]",
    "source":         "[\"observed\" | \"weak\" | null, ...]",
    "flags":          "[[<flag>, ...], ...]"
  }
}
```

Angles are rounded to 0.1°.

Manifest `derived` gains:
- `angles_frames_by_joint`: frames with a non-null angle, per joint;
- `angles_axis`: frames per `angleAxis` value;
- `angles_off_sagittal`: the `off_sagittal` count.

---

## 5. Code layout

- **`preprocess/joint_angles.py` (new), pure maths:**
  - takes one frame's points in mm, the joint sources and flags, and `m̂`;
  - returns each joint's angle, source and flags, and each segment's inclination;
  - imports nothing from tasks and does no I/O.
- **`preprocess/joint_hva.py`:**
  - picks `m̂` per frame (§2.1) from the `bars` and `near_side` it already receives;
  - calls `joint_angles` per frame on `constrained` frames;
  - writes the §4 fields;
  - bumps `SCHEMA_VERSION`.
- **`tasks/kinematics.py`: unchanged.** It already passes everything `build` needs.
- **No CLI change.** Angles come with every `--kinematics` run.

---

## 6. Validation

### 6.1 Synthetic (unit tests, written first)

- **The angle maths, per joint:**
  - poses built from set angles give back those angles to 1e-6°, at 0°, 90°, 179.9° and
    180° (straight);
  - past straight: hip 190°, elbow 185°;
  - shoulder at −5°, 90°, 180° and 190°.
- **Handedness:** a hand-built athlete facing the camera, side-on and at 45°, with a flexed hip
  and knee, reads < 180° at both. This pins `F̂ = Û × m̂` and every `s_j`.
- **Invariances:**
  - rotating the whole pose about the vertical changes no angle;
  - translating it changes no angle;
  - mirroring left and right swaps the sides and changes no value.
- **Continuity:** a joint swept from 170° through 180° to 190° gives a monotone series with no
  jump.
- **Nulls and flags:**
  - a missing joint, a short segment, a `bar_plane` frame and no `m̂` each give the §2.6
    result;
  - a knee bending sideways gets `off_sagittal`.
- **The axis:**
  - the bar axis given reversed (collar → cap) is re-signed by the hips;
  - with no bar, the hips are used.

### 6.2 Through the solve

On the synthetic lift already used by `test_preprocess_skeleton_solve.py`:
- set the joint angles;
- project to pixels;
- run `joint_hva.build`;
- check the angles come back within **1°** without noise.

With 1.5 px noise, report the error rather than gating on it. It sets the tolerance for §6.3.

**Illustrative sensitivity:** a joint off by `δ` perpendicular to a bone of length `L` turns
that segment by about `δ/L`. With 20 mm on a 400 mm bone, that is about 2.9°. Two segments with
independent errors give about 4°. Depth errors along the camera ray matter most for a segment
that points towards the camera.

### 6.3 Real captures

These wait on S8 running on real captures, which needs the `athlete:` blocks filled in.

- **Ranges:**
  - knee about 170–185° at lockout;
  - knee about 30–60° in a full squat catch (included, so deep is small);
  - elbow about 175–190° overhead.
- **Left/right:** the sides agree on frames where the lift is symmetric. Report the median
  `|left − right|` per joint.
- **The July two-camera captures:** the side camera's 2D angle agrees with the 3D angle, where
  the limb faces it square-on. This is a check only. Nothing 2D is published (decision 2).

---

## 7. Risks

- **The bar's cap/collar orientation** (§2.1, open question 3) decides the sign on frames where
  the hips cannot settle it. A wrong assumption flips every past-straight reading on those
  frames. The hip check makes this rare, and the handedness test catches a wrong `s_j`.
- **Hip angle includes the spine.** With no spine keypoints, a rounded back reads as hip
  flexion. The `trunk` segment does not separate them either.
- **Shoulder keypoint quality** (§2.4).
- **No σ:** a consumer cannot yet tell a 2° angle from a 15° one except through the flags.

---

## 8. Follow-ups outside this spec

- **True ankle angle:** keep Halpe-26's big toe, small toe and heel (RTMPose already outputs
  them; `halpe26_mapping.py` drops them). That needs a new skeleton version (`human-v3`) and an
  S6 schema bump. The ankle angle is then shank-to-foot, under §2.3's rules.
- **powerflow-ui:** `datatypespec.md` says `angleDeg` is a "flexion angle". It becomes an
  included angle under this spec's convention. The UI's joint list has no shoulders (open
  question 1).

---

## Open questions

1. **Should the UI show the shoulder angle?** S8 computes it either way (§2.4). Showing it means
   adding `leftShoulder` and `rightShoulder` to the UI's `jointName` enum and `skeleton.joints`.
2. **Angle σ.** Should S8 export each joint's 3×3 posterior covariance, which it already has as
   `(JᵀJ)⁻¹`? Then each angle could carry a propagated σ, and an `uncertain` flag at a stated
   threshold (say 5°).
3. **Which end is the cap?** §2.1 assumes S5's detected cap is the bar's near end. That needs
   checking against S5's detections, or replacing with the hips alone.
4. **Segments:** are shin, thigh and trunk the right set? Upper arm to vertical ("arms long and
   vertical in the pull") is a one-line addition.
5. **`bar_plane` frames** get no angle in v1 (§2.6). Is that right for a capture with no athlete
   measurements, where those are the only solved frames?

---

## Feedback (2026-10-21)

- **User, shoulder (open question 1):** the UI does not need the shoulder for now, and it is
  ignored there. S8 still computes it in `joint_hva.json` (§2.4). The UI follow-up in §8 is
  dropped.
- **User, bar ends (open question 3):** S5 detects the **near cap** and the **far collar**. So
  `BarPose.axis` (cap → collar) always points along the bar from the near side to the far side.
  §2.1's assumption holds, and the sign can be fixed from S6's near side alone. The hip check
  becomes a consistency check that warns when it disagrees.
- **User, athlete measurements (§6.3, open question 5):** asked whether every September lift has
  them. **Checked in `data/raw/`:**

  | Date | Captures | `athlete:` block |
  |---|---|---|
  | 10 September | 7 | **Complete:** all 11 numeric fields, including `forearm`, `upper_arm` and `ankle_height` |
  | 22 August | 29 | **Partial:** 8 fields; no `forearm_in_cm`, `upper_arm_in_cm`, `ankle_height_in_cm` |
  | 11 July | 2 | **Empty** |

  - All 7 September captures have S6 output (`data/s6_pose_output/10 September/`), so they are
    the real-data target for §6.3. Their frames solve as `constrained`, so open question 5 does
    not arise for them.
  - **None has S5 output yet** (no `s5_barbell_output`). Without it, S8 anchors by the floor
    alone and `m̂` falls back to the hips (§2.1). Running S5 on September comes first.
  - The UI spec's "every `athlete:` block is unfilled" is out of date.
