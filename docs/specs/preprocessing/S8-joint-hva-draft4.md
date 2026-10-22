# S8 Kinematics — Joint HVA: 3D Joints from the Lifter's Body Measurements

Status: **draft 4** | Supersedes: [`S8-joint-hva-draft3.md`](S8-joint-hva-draft3.md) (feedback at its bottom) | Scope: GitHub issue [#66](https://github.com/tony-yehui-lou/PowerFlow/issues/66) (Joint HVA) under epic [#65](https://github.com/tony-yehui-lou/PowerFlow/issues/65) | Sibling: [`S8-joint-angles-draft4.md`](S8-joint-angles-draft4.md) | Pipeline: **`analyze`** ([phase split](../../superpowers/specs/2026-10-22-pipeline-phase-split-design.md)) | Updated: 2026-10-22

**Changes from draft 3:**

1. **S8 lives in the `analyze` pipeline** (user, 2026-10-22). `preprocess` ends at S4, `detect`
   holds S5–S7, and `analyze` holds S8 and S9. The stage number **S8** is settled. Every input
   is read from another stage's published tree (§0). §6's code layout moves to `analyze/`, and
   `bar_geometry` moves to `common/`.
2. **Heights only** (user, 2026-10-20). Velocity, acceleration and the Tikhonov interface
   (draft 3 §5) are removed. No null placeholder arrays are published.
3. **A height is purely calculated** (user, 2026-10-20). It is each frame's own least-squares
   solution of exactly four kinds of term: pixels, bones, wrists on the bar, ankles on the floor
   (§3.2). The weak symmetry and head-on-mid-plane priors are gone, and so is any pull towards a
   starting position.
4. **`unseen` replaces `inferred`** (§3.7). A joint S6 did not see is not solved. Its position and
   height are null, and bones ending at it are left out of the solve.
5. **No bone calibration** (§3.5). The two-pass calibration, `--kinematics-calibrate`, the
   `estimated` source and `tapeGapM` are removed. Measured, one camera only weakly observes a
   bone's length, and the report missed a 120 mm error entirely (draft 3 feedback).
6. **The implementation's measured deviations from draft 3 are now the spec:**
   - Cauchy loss (§3.2);
   - a floor-only initial guess and a tie rule for mirror solutions (§3.4);
   - per-frame `method` and `reason` in `frameStatus` (§6);
   - hands-on-bar frames read from `wrist_refinement.json` (§3.2);
   - an analytic Jacobian (§6).

Carried over from draft 3:
- the per-frame least-squares solve;
- the body-plane solve as the initial guess;
- the `athlete:` field table, its definitions and the armspan check;
- hold-one-anchor-out validation.

---

## 0. In and out

**In**, all from published trees:
- S6's `pose.json` (2D, wrists refined) and `wrist_refinement.json` (`s6_pose_output`);
- S5's `barbell.json` (`s5_barbell_output`), solved into a bar track with
  `common/bar_geometry.solve_track`;
- S4's `camera_matrix.csv` and `crop_sidecar.json` (`s4_crop_output`);
- S3's `retilt_sidecar.json`, as the floor fallback (§4);
- the capture's `metadata.yaml`: `lift.barbell_type` and the **`athlete:` block**.

S8 runs inside `powerflow analyze`. A capture with no published S6 pose is rejected with that
reason in the manifest. With no published S5 output, S8 still runs, anchored by the floor alone
(§3.6).

**Out:** `joint_hva.json` per capture (`s8_kinematics_output`): per joint per frame, the 3D
position, the height, its uncertainty, its source and its flags. The joint angles of
`S8-joint-angles-draft4.md` are written into the same file.

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
every constraint pulling together (§3). It initialises from the body-plane solve, which lands on
the right root (§3.4).

---

## 2. The body measurements

### 2.1 How a measurement is used

Every length the solve uses is a **bone**: a keypoint pair with a target length `L` and a trust
`σ`. Each bone's `L` comes from the first of these sources that exists:

1. **`measured`:** the `athlete:` field for that bone, at the measured σ.
2. **`population`:** a ratio of `height_in_cm`, at the wider population σ.
3. **`none`:** the bone is left out of the solve, and its joints rely on the other terms.

**The tape and the keypoint measure different things.** A tape measurement runs between palpated
landmarks. The solve needs the distance between the detector's keypoints, which is offset by an
amount that depends on the model (`S6-bone-constrained-reconstruction.md` §1). So a measurement
is a prior with a σ, never a hard constraint. The solve does not re-estimate the length from the
capture (draft 3 feedback: one camera observes it too weakly to correct the tape).

### 2.2 Field table

All fields are optional per capture.

| `athlete:` field | constrains (keypoint pair) | measured σ | population prior (ratio of H) | population σ |
|---|---|---|---|---|
| `tibia_in_cm` | ankle – knee, both sides | 20 mm | 0.246·H | 30 mm |
| `femur_in_cm` | knee – hip, both sides | 20 mm | 0.245·H | 30 mm |
| `torso_in_cm` | hip – shoulder (same side), both sides | 25 mm | 0.288·H | 40 mm |
| `shoulder_width_in_cm` | leftShoulder – rightShoulder | 30 mm | 0.259·H − 0.04 m† | 40 mm |
| `upper_arm_in_cm` | shoulder – elbow, both sides | 20 mm | 0.186·H | 40 mm |
| `forearm_in_cm` | elbow – wrist, both sides | 20 mm | 0.146·H | 40 mm |
| `hip_width_in_cm` | leftHip – rightHip | 25 mm | 0.10·H | 30 mm |
| `ankle_height_in_cm` | grounded ankle's height above the floor (§3.2), not a bone | 10 mm | 0.039·H | 15 mm |
| — | head – shoulder midpoint | — | 0.12·H | 50 mm |
| `height_in_cm` | scales every population prior | — | — | — |
| `armspan_in_cm` | **consistency check only** (§2.4) | — | — | — |
| `weight_in_kg`, `body_fat_percentage` | not used here (COM, #68) | — | — | — |

The ratios are segment-length ratios from Drillis–Contini and Winter. †Biacromial breadth less
the keypoint inset, an assumption.

If `height_in_cm` is also missing, a bone with no measurement has source `none`.

### 2.3 What each field means, and how to measure it

These are the definitions this spec assumes. They are written to sit near the keypoints the
detector places, and the operator's protocol must match them (open question 1).

| field | from | to | posture |
|---|---|---|---|
| `upper_arm_in_cm` | acromion | lateral epicondyle of the humerus | arm hanging, relaxed |
| `forearm_in_cm` | lateral epicondyle | radial styloid (wrist crease, thumb side) | elbow at 90° |
| `hip_width_in_cm` | greater trochanter | opposite greater trochanter (bitrochanteric breadth, as taped) | standing, feet hip-width |
| `ankle_height_in_cm` | floor | centre of the lateral malleolus | standing, in lifting shoes as filmed |

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
exceeds 50 mm, the stage warns, naming which terms were measured and which came from population
ratios. Armspan never enters the solve. It mixes shoulder breadth and hand length into one number,
and the August captures have already shown it to be inconsistent with the other fields.

### 2.5 Reading the block

S0 writes an empty `AthleteMeta`, so S8 reads the block straight from the capture's
`metadata.yaml`, the way `barbell_input.read_bar_type` reads `lift.barbell_type`.
- Each leaf is parsed **on its own**. Template placeholders (`string`, `ISO 8601 date`, `float`)
  are treated as missing, never as numbers.
- Units are taken from the field name (`_in_cm`). The value must be positive and within a
  plausibility range (0.5–2× its population prior); otherwise it is rejected with a warning and
  treated as missing.
- If the block has no usable field at all, including `height_in_cm`, the stage falls back to the
  body-plane solve for the whole capture (`method="bar_plane"`) and warns.

Plumbing the block through S0 is still the right fix later. It does not block this stage.

---

## 3. The per-frame solve

### 3.1 Unknowns

Frame `t` has **13 joints**:
- ankles, knees, hips, shoulders, elbows and wrists, both sides;
- `head`.

Each **seen** joint (§3.7) is a free 3D point `P_j` in the S4 levelled camera frame (X right, Y
down, Z forward, millimetres). An `unseen` joint is not an unknown. The clavicles are derived
afterwards, at S6's own interpolation weight between each shoulder and the shoulder midpoint.

These are full 3D points, not depths along fixed rays. That lets a noisy or misplaced keypoint
be outvoted by the bone lengths, instead of forcing the skeleton through it.

### 3.2 Residuals

Minimise `Σ ρ(r_k / σ_k)²` over exactly these four kinds of term, with
`scipy.optimize.least_squares`. The reprojection terms use a **Cauchy** loss `ρ`:

| term | per frame | residual | σ |
|---|---|---|---|
| **reprojection** | each seen joint | `project(P_j) − pixel_j` (2D) | `σ_px / confidence_j`, with `σ_px` = 3 px |
| **bone length** | each bone in §2.2 with both ends seen and a source other than `none` | `‖P_a − P_b‖ − L_ab` | the bone's σ for its source (§2.1) |
| **wrist on bar** | each wrist, hands-on-bar frames | `(P_w − C)·a ∓ g` (the measured half grip), plus distance from the bar axis beyond 60 mm | 20 mm |
| **ankle on floor** | each **grounded** ankle (§3.3) | `(Y_floor − P_y) − h_ankle` | the ankle-height σ for its source (§2.2) |

`C`, `a` and `g` come from the bar track and `wrist_refinement.json`. Hands-on-bar frames are
those where `wrist_refinement.json` reports `farCorrected`. `Y_floor` comes from §4.

**Why Cauchy, not soft-L1.** With soft-L1, a knee pixel 210 px off still dragged the ankle 85 mm.
Cauchy's influence falls away with distance. The misplaced knee itself is still not pulled back:
it can swing about the hip–ankle line without changing either bone. Fixing that needs a
knee-hinge constraint, which this spec does not have (open question 5).

There is nothing else: no symmetry prior, no head-on-mid-plane prior, no pull towards a start
and no smoothing across frames.

### 3.3 When an ankle counts as grounded

Grounded is decided in **2D**, before the solve, so the solve cannot argue itself onto the floor.
An ankle is grounded when:
- its pixel moves less than `ground_px` (default 4 px) per frame over ±3 frames;
- its confidence is at least `min_conf` (0.3).

This correctly drops the heel rise of the second pull, the jump, and a split. After the solve, a
grounded ankle whose floor residual exceeds 3σ is flagged `ground_conflict`. Measured, a wrong
ankle height does not show at the ankle: a 120 mm error spreads across the bones and the bar.
So `ground_conflict` fires only on gross errors, and the closure error (§3.6) is the real detector
(below 5 mm with the right ankle height, above 50 mm with the 120 mm error).

### 3.4 Initial guess, and choosing the right root

Least squares finds the minimum nearest where it starts.
- **Frames with the bar** start from the **body-plane solve**: each joint on a plane at its
  lateral offset from the bar mid-plane (half the shoulder width, the half grip, and half the hip
  width, each from its §2.1 source). It is closed form and independent per frame.
- **Floor-only frames** start with every joint on its ray at the ankles' depth. A floor-only frame
  has exact mirror solutions: from a cold start it landed 470 mm off, with zero residual.

Every frame is also solved a second time, starting from the previous frame's solution. Of the two
results:
- the lower-RMS one is kept;
- if they reach different minima (more than 50 mm apart) and fit equally well (RMS within 0.2),
  the frame alone cannot choose. The one nearer the previous frame's solution is kept, and the
  frame is flagged `branch_ambiguous`.

Neither rule changes any value. They only choose which of the frame's own solutions is published.

### 3.5 Bone lengths are fixed per capture

Each bone's length is fixed for the capture from §2.1. It is not re-estimated from the capture.
Draft 3's two-pass calibration was implemented and removed:
- a fit cannot measure a bone it is told to keep, so each bone had to be measured by re-solving
  without it;
- even then, a 30 mm tibia error was found only as 8.5 mm, and the rest leaked into the femur
  (+8 mm) and the ankle height (+15 mm);
- repeated rounds did not converge (tibia 428 → 432 → 422 → 429 → 415 mm);
- without the priors it found 7 mm of a 30 mm error, and nothing at all of a 120 mm one.

A bundle adjustment over all frames and lengths together is the principled upgrade (open
question 4). It is not in v1.

### 3.6 Which frames get a solve

| anchors present | what fixes the scale | `frameStatus` |
|---|---|---|
| bar **and** a grounded ankle | both, over-determined | `method="constrained"`; closure error reported |
| bar only (the jump, the catch landing) | wrists on the bar | `constrained` |
| a grounded ankle only (before the grip, after the release, bar gaps) | feet on the floor | `constrained` |
| neither | — | null, `reason="no_anchor"` |
| fewer than 6 joints `observed` | — | null, `reason="too_few_joints"` |
| solver fails to converge, or RMS > 5 | — | null, `reason="no_fit"` |

The **closure error** is the shoulder's disagreement when reached from the ankles and from the
wrists, each chain solved from the frame's own solution. It is a per-frame check and changes no
height. It should be roughly the root-sum-square of the bone σs along both chains: ~40–50 mm
with measured legs and population arms, less with every field measured.

### 3.7 Source and uncertainty, per joint per frame

- `source`:
  - `"observed"`: confidence ≥ 0.7;
  - `"weak"`: confidence between `min_conf` (0.3) and 0.7;
  - `"unseen"`: below `min_conf`, or no pixel. Not solved: position and height are null, and
    bones ending at it are left out.
- `heightSigmaM`: the height's posterior σ, from `(JᵀJ)⁻¹` at the solution, scaled by
  `max(1, residual variance)` so a near-perfect fit cannot report near-zero uncertainty.
  Heights with `σ > 50 mm` are kept but flagged `uncertain`.
- `flags`: per joint per frame, `uncertain` and `ground_conflict`.

### 3.8 Sensitivity, worked (illustrative)

- **Ankle on the floor.** A 15 mm error in ankle height (the population σ) moves the ankle's depth
  by `δh / y_n`, about 50 mm at `y_n ≈ 0.3`. Chained to a knee at `y_n ≈ 0.2`, that is ~10 mm of
  knee height. A measured `ankle_height_in_cm` (σ 10 mm) cuts both by a third.
- **Bone length.** A 20 mm tibia error, in a frame where the knee is also held by the femur, the
  hip and its own pixel, spreads across all of those terms instead of landing on the knee. The
  posterior σ reports how much reached each joint.
- **Arms.** The elbow sits between the bar-anchored wrist and the leg-anchored shoulder. With the
  arm lengths at the population σ of 40 mm, it is the least constrained interior joint. Measuring
  `upper_arm_in_cm` and `forearm_in_cm` is what most improves the elbow.

---

## 4. Floor anchor

`Y_floor` comes from the bar's rest-plate anchor (rest bar centre plus the plate radius), falling
back to S3's floor plane. The source used is reported in `floor.anchor`. It is
`common/bar_geometry.floor_anchor`, shared by the bar, the joints and S9.

---

## 5. Velocity and acceleration

Out of scope (user, 2026-10-20). `joint_hva.json` publishes heights only, with no placeholder
arrays. When velocity is specified, it gets its own spec and reads `heightM`, `heightSigmaM` and
`source` from this file.

---

## 6. Code layout and output

All paths are under `data/analyze/` (today `data/preprocess/`, until the phase split lands).

- `analyze/athlete_input.py`: reads the `athlete:` block leaf by leaf (§2.5), resolves each bone's
  source and length (§2.1), and runs the armspan check (§2.4).
- `analyze/body_plane.py`: the closed-form initial guess (§3.4).
- `analyze/skeleton_solve.py`, pure maths:
  - the residuals (§3.2), with an analytic Jacobian checked against finite differences;
  - grounded detection (§3.3);
  - the per-frame solve and the tie rule (§3.4);
  - the closure error (§3.6);
  - posterior σ (§3.7).
- `analyze/joint_hva.py`: builds the document, including the angles of
  `S8-joint-angles-draft4.md`.
- `analyze/tasks/kinematics.py`: the `@task`. It reads the §0 inputs from their published trees
  and writes `joint_hva.json` through staging.
- `common/bar_geometry.py`: the bar track and the floor anchor, shared with S6 and S9.
- **CLI:** `powerflow analyze --kinematics <root>`, reading `--pose`; `--barbell` is optional
  (§3.6).
- **Outside the stage:** `common/models.py` `AthleteMeta` carries every §2.2 field, and the
  README's `metadata.yaml` example lists them with §2.3's definitions.

**Speed:** with the analytic Jacobian, a frame needs one residual evaluation per iteration instead
of ~40.

`joint_hva.json`:
- `schemaVersion`, `captureId`, `method` (this spec's path), `frames` (S6's timing);
- `floor`: `anchor`, `yFloorM`, `restFrames`;
- `bones`, one entry per bone in §2.2, and `ankleHeight` in the same shape. Lengths are in metres,
  `null` when they do not exist:

```jsonc
"bones": {
  "<boneId>": {
    "field":       "<athlete field name> | null",   // e.g. "tibia_in_cm"; null for head
    "measuredM":   "<float> | null",                // from metadata.yaml
    "populationM": "<float> | null",                // ratio × height_in_cm
    "usedM":       "<float> | null",                // what the solve used
    "source":      "measured | population | none",
    "sigmaM":      "<float> | null"
  }
},
"ankleHeight": { /* same fields as a bone */ },
"armspanCheck": { "measuredM": "<float> | null", "impliedM": "<float>", "warn": "<bool>" }
```

- `frameStatus`, per frame: `method` (`constrained | bar_plane | null`), `reason`, `anchors`,
  `closureErrorM`, `branchAmbiguous`, `rms`;
- `joints`, per joint per frame: `positionM` (S4's levelled camera frame, metres), `heightM`
  (`y_floor − Y`), `heightSigmaM`, `source`, `flags`;
- the angle fields of `S8-joint-angles-draft4.md` §4.

Manifest `derived`:
- the source of each bone;
- frames per anchor class (§3.6);
- median and p95 closure error;
- the `branch_ambiguous` count;
- the `ground_conflict` count;
- the armspan check.

---

## 7. Validation

Captures: the seven `10 September` captures, with 50kg3 held out.

### 7.1 Synthetic (unit tests)

A known skeleton, camera and bar, rendered to pixels with noise at the measured σ:
- The solve recovers the joints to tolerance with every anchor present.
- It also recovers them with each anchor removed in turn.
- With the far side occluded, the far joints come back `unseen`, with null positions.
- In a tangent bone, a pointing bone and a split stance, the wrong root is never chosen silently:
  it is either avoided or flagged `branch_ambiguous`.
- A floor-only frame from a cold start does not land on its mirror solution.
- Each bone's source resolves correctly with:
  - every field present;
  - each field missing;
  - template placeholders;
  - out-of-range values;
  - `height_in_cm` missing.
- The armspan check fires on an inconsistent synthetic athlete and stays quiet on a consistent
  one.
- The analytic Jacobian matches finite differences.

### 7.2 Hold-one-anchor-out: the primary real-data test

Solve each frame with **only the feet** anchoring it, then predict where the wrists should be.
Compare that with the bar-derived wrist position. Then do the reverse, predicting the grounded
ankles from the bar alone. The two anchors are independent: S5 against the floor plane and S6. So
this measures the whole chain end to end without ground truth.

Acceptance: the median wrist and ankle prediction error is under 50 mm on every capture.

### 7.3 What each measurement is worth

Rerun §7.2 with each measured field in turn replaced by its population prior. The change in
closure error is what that measurement is worth. Captures of the same lifter should agree.

### 7.4 Consistency

- **Near-wrist vs bar vertical motion:** independent detectors.
- **Left-minus-right height in the pull:** median ≈ 0.
- **Height σ in the rest window:** lifter static.
- **LiDAR cross-check:** reported only. S7 Lift is the depth-based source, off by default in
  `detect`.

### 7.5 Ground truth (proposed capture)

A still standing pose at the start, with landmark heights tape-measured. Take all the `athlete:`
fields in the same session by §2.3's definitions, and record the protocol.

---

## 8. Risks

- **Measurement definitions.** If a field is measured between different landmarks than §2.3
  assumes, its prior is wrong by centimetres in every frame. With no calibration, nothing in the
  solve catches this; only §7.3 and the closure error show it.
- **A wrong length is absorbed, not reported.** The solve spreads a wrong bone across its
  neighbours (§3.5). The closure error is the only per-frame signal.
- **Shoes.** An `ankle_height_in_cm` measured barefoot, on a lift filmed in lifting shoes, puts
  every grounded ankle about 2 cm too low.
- **Unseen joints leave gaps.** The occluded far elbow is null where draft 3 would have inferred
  it. The UI must draw gaps, not interpolate across them.
- **Grounded misclassification** pins a lifted foot to the floor. That is bounded by the soft σ
  and surfaced by `ground_conflict` only when gross.
- **Local minima.** Mitigated by the initial guesses and the two-start check of §3.4, and counted
  by `branch_ambiguous`.
- **Stale upstream.** S8 now runs separately from S5 and S6. A `joint_hva.json` built on an
  earlier S6 run is not detected in v1 (phase-split design, open question 3).
- **Few bodies.** The corpus so far is one lifter. The priors, σs and thresholds are unproven for
  other proportions.

---

## Open questions

1. **How are the `athlete:` fields measured?** The operator protocol must match §2.3, and the
   existing `torso` and `shoulder_width` definitions must be confirmed. Blocking for the mapping
   in §2.2.
2. **Hip width:** bitrochanteric breadth (§2.3) is easy to tape, but it sits wider than the hip
   joint centres the keypoints track. Should the population inset be applied to the taped value?
3. **Should frames with no anchor** use bone lengths alone, with an honest σ, or stay null? Null
   in v1.
4. **A bundle adjustment** over all frames and the bone lengths (§3.5): worth trying once
   §7.3 shows how much the tape is off?
5. **A knee hinge.** A misplaced knee can swing about the hip–ankle line without changing either
   bone (§3.2). Should the knee be constrained to the leg's plane?

**Closed since draft 3:**
- **The Tikhonov interface:** velocity is out of scope (§5).
- **Two passes vs a bundle adjustment:** the two passes are removed (§3.5); the bundle adjustment
  stays open (question 4).
- **The stage's home:** `analyze` (phase-split design).
