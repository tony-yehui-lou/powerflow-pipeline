# S8 Kinematics — Joint HVA: 3D Joints from the Lifter's Body Measurements

Status: **draft 3** | Supersedes: [`S8-joint-hva-draft2.md`](S8-joint-hva-draft2.md) (feedback at its bottom) | Scope: GitHub issue [#66](https://github.com/tony-yehui-lou/PowerFlow/issues/66) (Joint HVA) under epic [#65](https://github.com/tony-yehui-lou/PowerFlow/issues/65) | Updated: 2026-10-20

**Changes from draft 2:**

1. **No lifter's values appear in the spec.** It defines which `athlete:` fields are read, which
   keypoint pair each one constrains, how much it is trusted, and what happens when it is
   missing. The values themselves live only in each capture's `metadata.yaml`.
2. **Four new `athlete:` fields:** `upper_arm_in_cm`, `forearm_in_cm`, `hip_width_in_cm` and
   `ankle_height_in_cm`. When present, each turns a fitted bone or anchor into a measured one
   (§2.2).
3. **The armspan check is a formula** with a stated tolerance, not a worked example (§2.4).

Carried over from draft 2:
- the per-frame least-squares solve (pixels, bone lengths, hands on the bar, feet on the floor);
- draft 1's body-plane solve as the initial guess;
- inferred joints marked as inferred;
- the two-pass bone calibration;
- hold-one-anchor-out validation;
- velocity and acceleration left to the user's Tikhonov regulariser.

The stage number S8 is still a proposal.

**In:**
- S6's `pose.json` (2D, wrists refined) and `wrist_refinement.json`;
- S5's bar track (`bar_geometry.solve_track`);
- S4's `camera_matrix.csv`;
- S3's `retilt_sidecar.json`, as a cross-check only;
- the capture's raw `metadata.yaml`: `lift.barbell_type` and the **`athlete:` block**.

**Out:** `joint_hva.json` per capture: per joint per frame, the 3D position, the height, its
uncertainty and its provenance. Velocity and acceleration are reserved for §5.

---

## 1. Why the measurements make this solvable

A pixel fixes a joint's **direction** from the camera. One number is missing per joint: how far
along that ray it sits. S6 gives the pixels. Everything else in this spec exists to supply the
missing depths without LiDAR.

The body supplies them. A bone of known length `L` from a joint whose 3D position is known puts the
next joint on a sphere. Where that sphere meets the next joint's ray (a quadratic) gives its depth.
Chained from an anchor, every joint gets placed. This lift has **two anchors, one at each end of
the body**:

- **Hands → bar.** S5 places the bar in 3D, and the wrists hold it at the measured half grip.
- **Feet → floor.** A grounded ankle sits at a known height above the floor. Its ray meets that
  horizontal plane at a single point. The ankle is well below camera height, so this is
  well-conditioned.

The chains run ankle → knee → hip → shoulder ← elbow ← wrist. They meet at the shoulder from both
ends, so the problem is **over-determined**. The leftover disagreement, the *closure error*, is a
per-frame quality measure that costs nothing (§7.2).

A single ray–sphere step is fragile on its own:
- When the bone lies across the line of sight, the two roots merge. A small length error then
  moves the depth a lot: `δZ ≈ √(2·L·δL)`. For a 0.4 m bone, a 20 mm error gives ~130 mm at
  tangency.
- When the bone points at the camera, the roots sit `±L` apart, and one of them is wrong.

That is why this spec does not chain greedily. It solves **all joints in a frame at once**, with
every constraint pulling together (§3). It initialises from draft 1's body-plane solve, which lands
on the right root (§3.4).

---

## 2. The body measurements

### 2.1 How a measurement is used

Every length the solve uses is a **bone**: a keypoint pair with a target length `L` and a trust
`σ`. Each bone's `L` comes from the first of these sources that exists:

1. **`estimated`:** the capture's own calibrated length (§3.5). Pass 2 only, and only when it
   passes its gate.
2. **`measured`:** the `athlete:` field for that bone, at the measured σ.
3. **`population`:** a ratio of `height_in_cm`, at the wider population σ.
4. **`none`:** the bone is left out of the solve, and its joints rely on the other terms.

**The tape and the keypoint measure different things.** A tape measurement runs between palpated
landmarks. The solve needs the distance between the detector's keypoints, which is offset by an
amount that depends on the model (`S6-bone-constrained-reconstruction.md` §1). So:
- **a measurement is a prior with a σ, never a hard constraint**;
- §3.5 re-estimates every length from the capture, and **reports the tape-vs-estimate gap per
  bone**, never averaging the two.

### 2.2 Field table

Existing fields are marked *existing*; the four marked **new** are added by this spec. All are
optional per capture.

| `athlete:` field | | constrains (keypoint pair) | measured σ | population prior (ratio of H) | population σ |
|---|---|---|---|---|---|
| `tibia_in_cm` | existing | ankle – knee, both sides | 20 mm | 0.246·H | 30 mm |
| `femur_in_cm` | existing | knee – hip, both sides | 20 mm | 0.245·H | 30 mm |
| `torso_in_cm` | existing | hip – shoulder (same side), both sides | 25 mm | 0.288·H | 40 mm |
| `shoulder_width_in_cm` | existing | leftShoulder – rightShoulder | 30 mm | 0.259·H − 0.04 m† | 40 mm |
| **`upper_arm_in_cm`** | **new** | shoulder – elbow, both sides | 20 mm | 0.186·H | 40 mm |
| **`forearm_in_cm`** | **new** | elbow – wrist, both sides | 20 mm | 0.146·H | 40 mm |
| **`hip_width_in_cm`** | **new** | leftHip – rightHip | 25 mm | 0.10·H | 30 mm |
| **`ankle_height_in_cm`** | **new** | grounded ankle's height above the floor (§3.2), not a bone | 10 mm | 0.039·H | 15 mm |
| — | — | head – shoulder midpoint | — | 0.12·H | 50 mm |
| `height_in_cm` | existing | scales every population prior | — | — | — |
| `armspan_in_cm` | existing | **consistency check only** (§2.4) | — | — | — |
| `weight_in_kg`, `body_fat_percentage` | existing | not used here (COM, #68) | — | — | — |

The ratios are segment-length ratios from Drillis–Contini and Winter. †Biacromial breadth less
the keypoint inset. That inset is an assumption until §3.5 measures it.

If `height_in_cm` is also missing, a bone with no measurement has source `none`.

### 2.3 What each new field means, and how to measure it

These are the definitions this spec assumes. They are written to sit near the keypoints the
detector places, and the operator's protocol must match them (open question 1).

| field | from | to | posture |
|---|---|---|---|
| `upper_arm_in_cm` | acromion | lateral epicondyle of the humerus | arm hanging, relaxed |
| `forearm_in_cm` | lateral epicondyle | radial styloid (wrist crease, thumb side) | elbow at 90° |
| `hip_width_in_cm` | greater trochanter | opposite greater trochanter (bitrochanteric breadth, as taped; §3.5 measures the inset to the hip joint centres) | standing, feet hip-width |
| `ankle_height_in_cm` | floor | centre of the lateral malleolus | standing, barefoot or in lifting shoes as filmed |

`ankle_height_in_cm` must be measured **in the shoes worn for the capture**. Lifting shoes raise
the heel by roughly 2 cm, and that height is exactly what the floor anchor relies on.

The existing fields also need their definitions confirmed (open question 1), especially:
- `torso`: hip-to-shoulder, trochanter-to-acromion, or seated height?
- `shoulder_width`: biacromial or bideltoid?

### 2.4 Armspan consistency check

From the arm measurements, the implied armspan is:

```text
armspan_implied = shoulder width + 2·(upper arm + forearm + hand),   hand ≈ 0.108·H
```

Here each term is the `usedM` length of §6, whatever its source. If `|armspan_implied − armspan|`
exceeds 50 mm, the stage warns, naming which terms were measured and which were estimated.
Armspan never enters the solve. It mixes shoulder breadth and hand length into one number, and the
August captures have already shown it to be inconsistent with the other fields.

### 2.5 Reading the block

The pipeline does not read `athlete:` today: S0 writes an empty `AthleteMeta`. S8 reads it
straight from the capture's raw `metadata.yaml`, the way `barbell_input.read_bar_type` reads
`lift.barbell_type`.
- Each leaf is parsed **on its own**. Template placeholders (`string`, `ISO 8601 date`, `float`)
  are treated as missing, never as numbers.
- Units are taken from the field name (`_in_cm`). The value must be positive and within a
  plausibility range (0.5–2× its population prior); otherwise it is rejected with a warning and
  treated as missing.
- If the block has no usable field at all, including `height_in_cm`, the stage falls back to draft
  1's body-plane solve for the whole capture (`method="bar_plane"`) and warns.

Follow-ups outside this stage (§6):
- the four new fields go into `AthleteMeta` and into the README's `metadata.yaml` template;
- plumbing the block through S0 (the bone spec's §1.1) is still the right fix later.

Neither blocks this stage.

---

## 3. The per-frame solve

### 3.1 Unknowns

Frame `t` has **13 joints**:
- ankles, knees, hips, shoulders, elbows and wrists, both sides;
- `head`.

Each is a free 3D point `P_j` in the S4 levelled camera frame (X right, Y down, Z forward), so a
frame has 39 unknowns. The clavicles are derived afterwards, at S6's own interpolation weight
between each shoulder and the shoulder midpoint. They are never solved, since they carry no
information of their own.

These are full 3D points, not depths along fixed rays. That lets a noisy or misplaced keypoint
be outvoted by the bone lengths, instead of forcing the skeleton through it.

### 3.2 Residuals

Minimise `Σ ρ(r_k / σ_k)²` over the terms below, with `scipy.optimize.least_squares` (scipy is a
core dependency) and a robust loss `ρ` (`soft_l1`) on the reprojection terms:

| term | per frame | residual | σ |
|---|---|---|---|
| **reprojection** | each joint | `project(P_j) − pixel_j` (2D) | `σ_px / confidence_j`, with `σ_px` = 3 px. A joint below `min_conf` (0.3) or with no pixel gets no term. |
| **bone length** | each bone in §2.2 | `‖P_a − P_b‖ − L_ab` | the bone's σ for its source (§2.1) |
| **wrist on bar** | each wrist, hands-on-bar frames | `(P_w − C)·a ∓ g` (the measured half grip), plus distance from the bar axis beyond 60 mm | 20 mm |
| **ankle on floor** | each **grounded** ankle (§3.3) | `(Y_floor − P_y) − h_ankle` | the ankle-height σ for its source (§2.2) |
| **symmetry** (weak) | hip midpoint and shoulder midpoint, hands-on-bar | distance from the bar mid-plane, `(M − C)·a` | 50 mm |
| **head on mid-plane** (weak) | head, hands-on-bar | `(P_head − C)·a` | 80 mm |

`C`, `a` and `g` come from the bar track and `wrist_refinement.json`. `Y_floor` comes from the
bar's rest-plate anchor, shared with the bar (§4). The symmetry terms are weak on purpose: they
hold an occluded far-side joint in a sensible place, but they give way to the bones and pixels
whenever those disagree. That is the split-jerk case draft 1 had to reject.

### 3.3 When an ankle counts as grounded

Grounded is decided in **2D**, before the solve, so the solve cannot argue itself onto the floor.
An ankle is grounded when:
- its pixel moves less than `ground_px` (default 4 px) per frame over ±3 frames;
- its confidence is at least `min_conf`.

This correctly drops the heel rise of the second pull, the jump, and a split. After the solve, a
grounded ankle whose floor residual exceeds 3σ is reported as `ground_conflict`, so a
misclassification shows up rather than hiding.

### 3.4 Initial guess, and choosing the right root

Least squares finds the minimum nearest where it starts. Every frame starts from **draft 1's
body-plane solve**:
- each joint is placed on a plane at its lateral offset from the bar mid-plane;
- the offsets are half the shoulder width, the half grip, and half the hip width, each from its
  §2.1 source;
- the solve is closed form and independent per frame, so one bad frame cannot carry into the next.

When the bar is absent, the start is the previous frame's solution, and failing that, the nearest
solved frame.

The frame is also solved a second time, starting from the previous frame's solution. If the two
runs reach different minima, the lower-cost one is kept and the frame is flagged
`branch_ambiguous`. That makes the two-root problem of §1 measurable instead of silent.

### 3.5 Two passes: calibrate the bones, then solve

1. **Pass 1.** Solve every frame with each bone at its `measured` or `population` length.
2. **Calibrate.** For each bone, take the median of `‖P_a − P_b‖` over well-conditioned frames:
   - both endpoints `observed` (confidence ≥ 0.7);
   - posterior σ below the median.

   Ankle height is calibrated the same way: the median height of grounded ankles in frames that
   are anchored by the bar.

   The estimate is accepted if at least 30 frames survive and it lies within `3σ` of the pass-1
   length. It then becomes source `estimated`, at the σ of its own spread. Otherwise the pass-1
   length stays and the **disagreement is reported**. The two are never averaged.
3. **Pass 2.** Solve again with the lengths from step 2.

A bundle adjustment would solve all frames and the bone lengths together. It is the principled
upgrade, and it has a sparse Jacobian of ~16k unknowns per capture. It is deferred until the two
passes show they are not enough.

### 3.6 Which frames get a solve

| anchors present | what fixes the scale | result |
|---|---|---|
| bar **and** a grounded ankle | both, over-determined | `method="constrained"`; closure error reported |
| bar only (the jump, the catch landing) | wrists on the bar | `constrained` |
| a grounded ankle only (before the grip, after the release, bar gaps) | feet on the floor | `constrained`; no symmetry terms |
| neither | bone lengths alone: scale is determined, but root ambiguity is high | null, `reason="no_anchor"` |
| fewer than 6 joints `observed` | — | null, `reason="too_few_joints"` |
| solver fails to converge, or cost > `max_cost` | — | null, `reason="no_fit"` |

### 3.7 Provenance and uncertainty, per joint per frame

- `source`:
  - `"observed"`: the joint had its own reprojection term with confidence ≥ 0.7;
  - `"weak"`: confidence between `min_conf` and 0.7;
  - `"inferred"`: no pixel term, so it was placed by bones and symmetry alone. The occluded far
    elbow is the typical case.

  A consumer must be able to tell a measured joint from an inferred one; that was the bone spec's
  §5 lesson.
- `heightSigmaM`: the height's posterior σ, from `(JᵀJ)⁻¹` at the solution, scaled by the
  residual variance. Heights with `σ > max_height_sigma` (default 50 mm) are kept but flagged
  `uncertain`.

### 3.8 Sensitivity, worked (illustrative)

- **Ankle on the floor.** A 15 mm error in ankle height (the population σ) moves the ankle's depth
  by `δh / y_n`, about 50 mm at `y_n ≈ 0.3`. Chained to a knee at `y_n ≈ 0.2`, that is ~10 mm of
  knee height. A measured `ankle_height_in_cm` (σ 10 mm) cuts both by a third.
- **Bone length.** A 20 mm tibia error, in a frame where the knee is also held by the femur, the
  hip and its own pixel, spreads across all of those terms instead of landing on the knee. The
  posterior σ reports how much reached each joint.
- **Arms.** The elbow sits between the bar-anchored wrist and the leg-anchored shoulder. With the
  arm lengths at the population σ of 40 mm, it is the least constrained interior joint. Measuring
  `upper_arm_in_cm` and `forearm_in_cm` is what most improves the elbow, and with it the
  early-arm-bend issue.
- **Closure.** The shoulder, reached from the ankles and from the wrists, should agree to roughly
  the root-sum-square of the bone σs along both chains. That is ~40–50 mm with measured legs and
  population arms, and less with all four new fields. A frame whose closure error is far beyond
  this has a bad keypoint or a wrong anchor.

---

## 4. Floor anchor

Unchanged from draft 1 §2.4. `Y_floor` comes from the bar's rest-plate anchor (rest bar centre
plus the plate radius), falling back to S3's floor plane. Both are reported. It moves into the
pipeline next to `bar_geometry.py`, so the bar and the joints share it.

## 5. Velocity and acceleration (placeholder)

To be specified with the user's **Tikhonov regulariser**. This draft fixes only the interface it
receives, per joint:
- the timestamps `startEpochMs + tMs[i]`, which have VFR jitter;
- `heightM[i]` or null;
- `heightSigmaM[i]`, available as per-sample weights;
- `source[i]`, so inferred samples can be down-weighted.

It returns `velocityMps[]` and `accelMps2[]` (positive upward; kinematic, so gravity is not added).

## 6. Code layout and output

- `preprocess/athlete_input.py` (new): reads the `athlete:` block leaf by leaf, including the four
  new fields (§2.5), and resolves each bone's source (§2.1).
- `preprocess/body_plane.py` (new): draft 1's closed-form solve, used as the initial guess (§3.4).
- `preprocess/skeleton_solve.py` (new), pure maths:
  - the residuals (§3.2);
  - grounded detection (§3.3);
  - the per-frame solve;
  - bone and ankle-height calibration (§3.5);
  - posterior σ;
  - the armspan check (§2.4).
- `preprocess/tasks/kinematics.py` (new `@task`). `cli.py` gains `--kinematics <root>`, which
  requires `--pose`. `--barbell` is optional (§3.6).
- **Outside the stage:**
  - `common/models.py` `AthleteMeta` gains the four new fields (plus `shoulder_width_in_cm` and
    `body_fat_percentage`, already missing per the bone spec §1.1);
  - the README's `metadata.yaml` example lists them, with §2.3's definitions.

`joint_hva.json` has draft 1's shape (§5 there), with these changes:
- `method` takes the values `constrained | bar_plane | null`;
- each joint adds `heightSigmaM[]`, `source[]` and `closureErrorM[]` (per frame, the same for all
  joints);
- `velocityMps[]` and `accelMps2[]` stay null until §5 is specified;
- a per-capture `bones` table, one entry per bone in §2.2, replaces `lateralOffsetsM`. Every length
  is in metres and is `null` when it does not exist:

```jsonc
"bones": {
  "<boneId>": {
    "field":       "<athlete field name> | null",   // e.g. "tibia_in_cm"; null for head
    "measuredM":   "<float> | null",                // from metadata.yaml
    "populationM": "<float> | null",                // ratio × height_in_cm
    "estimatedM":  "<float> | null",                // §3.5 median, or null if rejected
    "usedM":       "<float>",                       // what pass 2 used
    "source":      "estimated | measured | population | none",
    "sigmaM":      "<float>",
    "frames":      "<int>",                         // §3.5 survivors
    "iqrM":        "<float> | null",
    "tapeGapM":    "<float> | null"                 // estimatedM − measuredM
  }
},
"ankleHeight": { /* same fields as a bone */ },
"armspanCheck": { "measuredM": "<float> | null", "impliedM": "<float>", "warn": "<bool>" }
```

Manifest `derived`:
- the source of each bone, and each `tapeGapM`;
- frames per anchor class (§3.6);
- median and p95 closure error;
- the `branch_ambiguous` count;
- the `ground_conflict` count;
- the armspan check.

## 7. Validation

Captures: the seven `10 September` captures, with 50kg3 held out.

### 7.1 Synthetic (unit tests)

A known skeleton, camera and bar, rendered to pixels with noise at the measured σ:
- The solve recovers the joints to tolerance with every anchor present.
- It also recovers them with each anchor removed in turn.
- With the far side occluded, the far joints come back as `inferred` with honest σ.
- In a tangent bone, a pointing bone and a split stance, the wrong root is never chosen silently.
- Each bone's source resolves correctly with:
  - every field present;
  - each field missing;
  - template placeholders;
  - out-of-range values;
  - `height_in_cm` missing.
- The armspan check fires on an inconsistent synthetic athlete and stays quiet on a consistent
  one.

### 7.2 Hold-one-anchor-out: the primary real-data test

Solve each frame with **only the feet** anchoring it, then predict where the wrists should be.
Compare that with the bar-derived wrist position. Then do the reverse, predicting the grounded
ankles from the bar alone. The two anchors are independent: S5 against the floor plane and S6. So
this measures the whole chain end to end without ground truth.

Acceptance: the median wrist and ankle prediction error is under 50 mm on every capture.

### 7.3 Measurements vs the capture

The `tapeGapM` of every measured bone, per capture. This shows whether the metadata helps or
biases. Captures of the same lifter should agree, so a gap that changes between captures points at
the detector, not the tape.

Once captures with the four new fields exist, rerun §7.2 with each new field in turn replaced by
its population prior. The change in closure error is what each measurement is worth.

### 7.4 Consistency

- **Near-wrist vs bar vertical motion:** independent detectors.
- **Left-minus-right height in the pull:** median ≈ 0.
- **Height σ in the rest window:** lifter static.
- **LiDAR cross-check:** reported only.
- **Draft 1's body-plane heights vs this solve:** where they disagree, the symmetry assumption is
  the suspect.

### 7.5 Ground truth (proposed capture)

The same as draft 1 §6.5: a still standing pose at the start, with landmark heights tape-measured.
Take all the `athlete:` fields in the same session by §2.3's definitions, and record the protocol.

## 8. Risks

- **Measurement definitions.** If a field is measured between different landmarks than §2.3
  assumes, its prior is wrong by centimetres in every frame. §3.5 catches this as a large
  `tapeGapM`, but only after the fact.
- **The new fields will be missing on existing captures.** Every capture recorded so far lacks
  the four new fields, so those bones run at population σ until new captures are taken. §7.3
  measures what that costs.
- **Shoes.** An `ankle_height_in_cm` measured barefoot, on a lift filmed in lifting shoes, puts
  every grounded ankle about 2 cm too low.
- **Inferred joints look like measured ones.** This is mitigated only by `source[]` reaching the
  UI. The graphs should draw inferred stretches differently, as the bar view dashes its
  cap-missing frames.
- **Grounded misclassification** pins a lifted foot to the floor. That is bounded by the soft σ
  and surfaced by `ground_conflict`.
- **Local minima.** Mitigated by the closed-form start and the two-start check of §3.4, and
  counted by `branch_ambiguous`.
- **Few bodies.** The corpus so far is one lifter. The priors, σs and thresholds are unproven for
  other proportions.

## Open questions

1. **How are the `athlete:` fields measured?** The operator protocol must match §2.3, and the
   existing `torso` and `shoulder_width` definitions must be confirmed. Blocking for the mapping
   in §2.2.
2. **Hip width:** bitrochanteric breadth (§2.3) is easy to tape, but it sits wider than the hip
   joint centres the keypoints track. Should §3.5's estimate be trusted over it by default?
3. **Should frames with no anchor** use bone lengths alone, with an honest σ, or stay null? Null
   in v1.
4. **Two passes vs a bundle adjustment** (§3.5). This depends on how much the bone estimates
   scatter between frames.
5. **The Tikhonov interface** (§5): does it take per-sample weights from `heightSigmaM` and
   `source`?

---

## Feedback (2026-10-20)

- **User:** implement everything except velocity and acceleration; the user will add the athlete
  measurements to the metadata themselves.
- **Implemented** (uncommitted), heights only:
  - `preprocess/athlete_input.py` reads the athlete block and resolves each bone's length;
  - `preprocess/body_plane.py` holds the initial guess;
  - `preprocess/skeleton_solve.py` holds the solve and the calibration;
  - `preprocess/joint_hva.py` builds the document;
  - `preprocess/tasks/kinematics.py` is the task;
  - `bar_geometry.floor_anchor` is the shared floor;
  - `AthleteMeta` has the new fields;
  - the CLI has `--kinematics` and `--kinematics-calibrate`;
  - the README has the athlete-measurement table and Recipe C.

  Velocity and acceleration are published as nulls. **Not yet run on real captures.**
- **Deviations, each measured on the synthetic lift in `tests/unit/test_preprocess_skeleton_solve.py`:**
  1. **Cauchy loss, not soft-L1 (§3.2).**
     - **Measured:** with soft-L1, a knee pixel 210 px off still dragged the ankle 85 mm.
     - **Change:** Cauchy's influence falls away with distance.
     - **Still unsolved:** the misplaced knee itself is not pulled back. It can swing about the
       hip–ankle line without changing either bone. Fixing that needs a knee-hinge constraint,
       which this spec does not have.
  2. **Calibration holds each bone out (§3.5).**
     - **Measured:** a pass-1 fit cannot measure a bone it was told to keep. A tibia taped 30 mm
       long stayed 30 mm long, with an IQR of 0.25 mm across frames.
     - **Change:** each bone is now measured from re-solves without its own length term. Ankle
       height comes from bar-only re-solves, run after the bones are calibrated.
     - **Not used:** the posterior-σ median gate. It is replaced by sampling at most 60
       eligible frames per bone.
  3. **Calibration is reported, not applied, by default.** `--kinematics-calibrate` applies it.
     - **Measured:** on 70 frames, median height error with pass 1 alone vs after calibration:

       | scenario | pass 1 alone | after calibration |
       |---|---|---|
       | correct tape | 0 mm | 0 mm |
       | tibia taped 30 mm long | 17.2 mm | 19.8 mm |
       | correct tape, 1.5 px noise | 5.3 mm | 5.9 mm |
       | tibia +30 mm, 1.5 px noise | 18.7 mm | 17.1 mm |
     - **Where the error goes:** the wrong tibia is found 8.5 mm long, and the rest leaks into
       its neighbours: femur +8 mm, ankle height +15 mm.
     - **Repeating calibration does not converge.** Five rounds wandered (tibia 428 → 432 → 422 →
       429 → 415 mm).
     - **Conclusion:** one camera only weakly observes a bone's length. The `tapeGapM` report is
       a check that the tape is right, not a way to correct it.
  4. **A floor-only frame has exact mirror solutions.** From a cold start it landed 470 mm off,
     with zero residual.
     - **Change:** the floor-only initial guess places every joint on its ray at the ankles'
       depth. Each frame also starts from the previous frame's solution.
     - **Ties:** fits within 0.2 RMS of each other are decided by nearness to the previous frame.
       Those frames are flagged `branch_ambiguous`.
     - **Closure solves** also start from the frame's own solution, not the cold guess, since
       cold-started leg chains can flip.
  5. **A wrong ankle height does not show at the ankle.**
     - **Measured:** a 120 mm error spreads across the bones and the bar instead.
     - **So:** `ground_conflict` fires only on gross errors. The closure error is the detector:
       below 5 mm when the ankle height is right, above 50 mm with the 120 mm error.
  6. **Joints with no pixel get a wide pull** (σ 150 mm) towards their starting position. This
     keeps an occluded elbow's problem well-posed, since the bones alone leave it free on a
     circle. Its height σ is reported, not assumed.
  7. **Posterior σ** is scaled by `max(1, residual variance)`, so a near-perfect fit cannot
     report near-zero uncertainty.
  8. **`method` and `reason` are per frame,** in `frameStatus`, since the solve is per frame.
     `source`, the height σ and `flags` stay per joint.
  9. **`positionM` is in S4's levelled camera frame,** in metres. Height is `y_floor − Y`.
  10. **Hands-on-bar frames** are read from `wrist_refinement.json`'s `farCorrected`. No
      explicit `handsOnBar` key has been added to S6.
- **Speed:** the Jacobian is analytic, checked against finite differences. A finite-difference
  Jacobian needs 40 residual evaluations, and a 35-frame build was 6.8 s, mostly in calibration
  (about 600 re-solves).

---

## Feedback (2026-10-20, later)

- **User:** no velocity or acceleration yet. Remove all related code: **height only**, and only
  the purely calculated height, with no regularised or adjusted value.
- **Removed from the implementation:**
  - the null `velocityMps` / `accelMps2` arrays;
  - the `--kinematics-calibrate` option, which re-solved with the capture's own lengths;
  - the weak symmetry and head-on-mid-plane priors (§3.2);
  - the σ 150 mm pull towards the start for joints with no pixel (implementation note 6 above).
- **Unseen joints:** a joint S6 did not see (confidence below 0.3, or no pixel) is now
  `unseen`. It is not solved, and its position and height are null. Bones ending at an unseen
  joint are left out of the solve. §3.7's `inferred` no longer exists.
- **The capture-measured length report (§3.5) is removed as well.**
  - **Measured:** without the priors, it found only 7 mm of a 30 mm tibia tape error. It
    flagged nothing at all for a 120 mm error, because the solve absorbs the wrong length.
  - **So:** a check that misses a 120 mm error should not be published. `bones` now records only
    the length and source each bone was solved with.
- **What a height is now:** each frame's own least-squares solution of exactly four kinds of
  term: S6's pixels, the bone lengths, the wrists on the bar, and the grounded ankles on the
  floor. Two things still shape which solution is kept, without changing any value:
  - the Cauchy loss down-weights a keypoint far from the rest;
  - where two solutions fit equally well (a floor-only frame's mirror pair), the one nearer the
    previous frame's solution is kept, and the frame is flagged `branch_ambiguous`.
- **Kept:** the closure error is a per-frame consistency check. It does not change any height.

---

## Feedback (2026-10-22): the stage's home

- **User:** preprocess ends at S4. S5 and S6 are the detection phase. S8 and S9 are the
  quantitative-analysis phase.
- **Decided** in `docs/superpowers/specs/2026-10-22-pipeline-phase-split-design.md`:
  - S8 Kinematics moves out of `preprocess` into a new **`analyze`** pipeline, alongside S9
    Phase. `detect` holds S5 Barbell, S6 Pose and an optional S7 Lift (off by default).
  - S-numbers stay global. **S8** is no longer a proposal.
  - The pipelines talk only through published output. S8 reads S0's record, S3's floor, S4's
    sidecar, S5's `barbell.json` and S6's `pose.json` / `wrist_refinement.json` from disk,
    never from S4's in-memory record.
  - Code moves: `athlete_input`, `body_plane`, `skeleton_solve`, `joint_hva` and
    `tasks/kinematics` to `analyze/`; `bar_geometry` (shared with S6 wrist refinement and S9)
    to `common/`.
  - The CLI flag `--kinematics` moves to `powerflow analyze`.
- **For draft 4:** update the stage header, the inputs and §6's code layout. Also fold in both
  2026-10-20 feedback rounds above (no velocity or acceleration, no priors, `unseen` joints,
  no calibration), which draft 3's body still describes as planned.
