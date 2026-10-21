# S8 Kinematics — Joint HVA: 3D Joints from the Lifter's Body Measurements

Status: **draft 2** | Supersedes: [`S8-joint-hva.md`](S8-joint-hva.md) (feedback at its bottom) | Scope: GitHub issue [#66](https://github.com/tony-yehui-lou/PowerFlow/issues/66) (Joint HVA) under epic [#65](https://github.com/tony-yehui-lou/PowerFlow/issues/65) | Updated: 2026-10-20

**Changes from draft 1:**

1. **The lifter's body measurements solve the 3D joints.** Each frame is a small least-squares
   problem. The skeleton must:
   - reproject onto S6's pixels;
   - keep the measured bone lengths;
   - hold the bar with its hands;
   - stand on the floor with its feet.

   Draft 1's body-plane solve is demoted to the **initial guess** and a weak symmetry prior.
2. **The bar is no longer required.** The feet anchor scale through the floor, so frames before the
   grip, after the release, and in 45kg1's bar gaps can now get heights.
3. **Occluded far-side joints are inferred, not dropped,** and marked `inferred`, never `observed`.
4. **Velocity and acceleration** are left to the user's Tikhonov regulariser, specified later (§5).

Carried over:
- the S4 levelled camera frame;
- the floor anchor shared with the bar;
- no LiDAR depth;
- a null with a reason over an unbacked number;
- the output arrays named to match powerflow-ui's `Joint` type.

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
  moves the depth a lot: `δZ ≈ √(2·L·δL)`, so a 20 mm femur error gives ~130 mm at tangency.
- When the bone points at the camera, the roots sit `±L` apart, and one of them is wrong.

That is why this spec does not chain greedily. It solves **all joints in a frame at once**, with
every constraint pulling together (§3). It initialises from draft 1's body-plane solve, which lands
on the right root (§3.4).

---

## 2. The body measurements

### 2.1 What the metadata provides, and how it maps to keypoints

`10 September` has one lifter across all seven captures, all filled:

| `athlete:` field | value | used as | keypoint pair | prior σ |
|---|---|---|---|---|
| `tibia_in_cm` | 39.8 | bone length, both sides | ankle – knee | 20 mm |
| `femur_in_cm` | 42.3 | bone length, both sides | knee – hip | 20 mm |
| `torso_in_cm` | 51.5 | bone length, both sides | hip – shoulder (same side) | 25 mm |
| `shoulder_width_in_cm` | 44.0 | bone length | leftShoulder – rightShoulder | 30 mm |
| `height_in_cm` | 180.0 | scales the population priors in §2.2 | — | — |
| `armspan_in_cm` | 175.6 | **sanity bound only** (§2.3) | — | — |
| `weight_in_kg`, `body_fat_percentage` | 63.0, 12.0 | not used here (COM, #68) | — | — |

**The tape and the keypoint measure different things.** A coach's tibia runs between palpated
landmarks. The solve needs the distance between the detector's knee and ankle keypoints, which is
offset by an amount that depends on the model (`S6-bone-constrained-reconstruction.md` §1). So:
- **a measurement is a prior with a σ, never a hard constraint**;
- §3.5 re-estimates every length from the capture itself, and reports how far it lands from the
  tape.

**The measurement definitions are unconfirmed.** For example: is `torso` hip-to-shoulder,
trochanter-to-acromion, or seated height? Is `shoulder_width` biacromial or bideltoid? The
mapping above is this spec's assumption. Blocking open question 1 asks for the operator's
protocol.

### 2.2 Bones the metadata does not cover

These start from population ratios of `height_in_cm` (Drillis–Contini / Winter) and are then
estimated per capture (§3.5):

| bone | prior (H = 1.80 m) | σ | why it can be estimated |
|---|---|---|---|
| upper arm (shoulder – elbow) | 0.186·H = 335 mm | 40 mm | the shoulder is pinned by the leg/torso chain, the wrist by the bar |
| forearm (elbow – wrist) | 0.146·H = 263 mm | 40 mm | as above |
| hip width (leftHip – rightHip) | 180 mm | 30 mm | the femur chains from two grounded ankles |
| head – shoulder midpoint | 0.12·H = 216 mm | 50 mm | low stakes; the head also sits near the mid-plane |
| ankle height above floor | 0.039·H = 70 mm | 15 mm | rest frames, chained down from the bar |

### 2.3 Armspan is not used as a constraint

`armspan = shoulder width + 2·(upper arm + forearm + hand)`. That gives
`(1756 − 440)/2 − hand (~0.108·H = 194 mm) ≈ 464 mm` for upper arm plus forearm. The population
figure is ~600 mm. The two are inconsistent, as the bone-reconstruction spec found on August
captures too. Armspan is reported as a cross-check against the estimated arm lengths, and nothing
more.

### 2.4 Reading the block

The pipeline does not read `athlete:` today: S0 writes an empty `AthleteMeta`. S8 reads it
straight from the capture's raw `metadata.yaml`, the way `barbell_input.read_bar_type` reads
`lift.barbell_type`.
- Each leaf is parsed **on its own**. Template placeholders (`string`, `ISO 8601 date`, `float`)
  are treated as missing, never as numbers.
- A missing measurement falls back to its population prior with the wider σ, recorded per bone.
- If the block has no usable measurement at all, the stage falls back to draft 1's body-plane
  solve for the whole capture (`method="bar_plane"`) and warns.

Plumbing the block through S0 (the bone spec's §1.1) is still the right fix later. This spec does
not depend on it.

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
| **bone length** | each bone in §2.1–2.2 | `‖P_a − P_b‖ − L_ab` | per bone, §2.1–2.2 (after §3.5, the estimate's own spread) |
| **wrist on bar** | each wrist, hands-on-bar frames | `(P_w − C)·a ∓ g` (the measured half grip), plus distance from the bar axis beyond 60 mm | 20 mm |
| **ankle on floor** | each **grounded** ankle (§3.3) | `(Y_floor − P_y) − h_ankle` | 15 mm |
| **symmetry** (weak) | hip midpoint and shoulder midpoint, hands-on-bar | distance from the bar mid-plane, `(M − C)·a` | 50 mm |
| **head on mid-plane** (weak) | head, hands-on-bar | `(P_head − C)·a` | 80 mm |

`C`, `a` and `g` come from the bar track and `wrist_refinement.json`. `Y_floor` comes from the
bar's rest-plate anchor, shared with the bar (draft 1 §2.4). The symmetry terms are weak on
purpose: they hold an occluded far-side joint in a sensible place, but they give way to the bones
and pixels whenever those disagree. That is the split-jerk case draft 1 had to reject.

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
- the offsets come from the shoulder width, half grip and hip-width prior;
- the solve is closed form and independent per frame, so one bad frame cannot carry into the next.

When the bar is absent, the start is the previous frame's solution, and failing that, the nearest
solved frame.

The frame is also solved a second time, starting from the previous frame's solution. If the two
runs reach different minima, the lower-cost one is kept and the frame is flagged
`branch_ambiguous`. That makes the two-root problem of §1 measurable instead of silent.

### 3.5 Two passes: calibrate the bones, then solve

1. **Pass 1.** Solve every frame with the measurement priors.
2. **Calibrate.** For each bone, take the median of `‖P_a − P_b‖` over well-conditioned frames:
   - both endpoints `observed` (confidence ≥ 0.7);
   - posterior σ below the median.

   Accept it if at least 30 frames survive and it lies within `3σ_prior` of the measurement.
   Otherwise keep the measurement and **report the disagreement**. Never average the two.
   Unmeasured bones (§2.2) take their estimate when accepted.
3. **Pass 2.** Solve again with the calibrated lengths, at the tighter σ of their own spread.

A bundle adjustment would solve all frames and the bone lengths together. It is the principled
upgrade, and it has a sparse Jacobian of ~16k unknowns per capture. It is deferred until the two
passes show they are not enough.

### 3.6 Which frames get a solve

| anchors present | what fixes the scale | result |
|---|---|---|
| bar **and** a grounded ankle | both, over-determined | `method="constrained"`; closure error reported |
| bar only (the jump, the catch landing) | wrists on the bar | `constrained` |
| a grounded ankle only (before the grip, after the release, bar gaps) | feet on the floor | `constrained`, **new in draft 2**; no symmetry terms |
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

- **Ankle on the floor.** A 15 mm error in ankle height moves the ankle's depth by `δh / y_n`,
  about 50 mm at `y_n ≈ 0.3`. Chained to a knee at `y_n ≈ 0.2`, that is ~10 mm of knee height.
- **Bone length.** A 20 mm tibia error, in a frame where the knee is also held by the femur, the
  hip and its own pixel, spreads across all of those terms instead of landing on the knee. The
  posterior σ reports how much reached each joint.
- **Closure.** The shoulder, reached from the ankles and from the wrists, should agree to roughly
  the root-sum-square of the bone σs along both chains, about 40–50 mm. A frame whose closure error
  is far beyond that has a bad keypoint or a wrong anchor.

---

## 4. Floor anchor

Unchanged from draft 1 §2.4. `Y_floor` comes from the bar's rest-plate anchor (rest bar centre
plus 225 mm), falling back to S3's floor plane. Both are reported. It moves into the pipeline next
to `bar_geometry.py`, so the bar and the joints share it. Ankle height `h_ankle` is calibrated in
§3.5 against it.

## 5. Velocity and acceleration (placeholder)

To be specified with the user's **Tikhonov regulariser**. This draft fixes only the interface it
receives, per joint:
- the timestamps `startEpochMs + tMs[i]`, which have VFR jitter of 16–17 ms;
- `heightM[i]` or null;
- `heightSigmaM[i]`, available as per-sample weights;
- `source[i]`, so inferred samples can be down-weighted.

It returns `velocityMps[]` and `accelMps2[]` (positive upward; kinematic, so gravity is not added).

## 6. Code layout and output

- `preprocess/athlete_input.py` (new): reads the `athlete:` block leaf by leaf (§2.4).
- `preprocess/body_plane.py` (new): draft 1's closed-form solve, now used as the initial guess
  (§3.4).
- `preprocess/skeleton_solve.py` (new), pure maths:
  - the residuals (§3.2);
  - grounded detection (§3.3);
  - the per-frame solve;
  - bone calibration (§3.5);
  - posterior σ.
- `preprocess/tasks/kinematics.py` (new `@task`). `cli.py` gains `--kinematics <root>`, which
  requires `--pose`. `--barbell` is optional now (§3.6).

`joint_hva.json` has draft 1's shape (§5 there), with these changes:
- `method` takes the values `constrained | bar_plane | null`;
- each joint adds `heightSigmaM[]`, `source[]` and `closureErrorM[]` (per frame, the same for all
  joints);
- `velocityMps[]` and `accelMps2[]` stay null until §5 is specified;
- a per-capture `bones` table replaces `lateralOffsetsM`:

```jsonc
"bones": {
  "leftTibia":  { "measuredM": 0.398, "estimatedM": 0.384, "usedM": 0.384, "source": "estimated",
                  "frames": 211, "iqrM": 0.011, "priorSigmaM": 0.020 },
  "leftUpperArm": { "measuredM": null, "priorM": 0.335, "estimatedM": 0.311, "usedM": 0.311,
                    "source": "estimated", "frames": 140, "iqrM": 0.018 },
  "...": {}
},
"armspanCheck": { "measuredM": 1.756, "impliedM": 1.701 }
```

Manifest `derived`:
- the bone sources and their tape-vs-estimate differences;
- frames per anchor class (§3.6);
- median and p95 closure error;
- the `branch_ambiguous` count;
- the `ground_conflict` count.

## 7. Validation

Captures: the seven `10 September` captures, with 50kg3 held out.

### 7.1 Synthetic (unit tests)

A known skeleton, camera and bar, rendered to pixels with noise at the measured σ:
- The solve recovers the joints to tolerance with every anchor present.
- It also recovers them with each anchor removed in turn.
- With the far side occluded, the far joints come back as `inferred` with honest σ.
- In a tangent bone, a pointing bone and a split stance, the wrong root is never chosen silently.
- Template metadata leaves (`float`, `string`) never parse as numbers.

### 7.2 Hold-one-anchor-out: the primary real-data test

Solve each frame with **only the feet** anchoring it, then predict where the wrists should be.
Compare that with the bar-derived wrist position. Then do the reverse, predicting the grounded
ankles from the bar alone. The two anchors are independent: S5 against the floor plane and S6. So
this measures the whole chain end to end without ground truth.

Acceptance: the median wrist and ankle prediction error is under 50 mm on every capture.

### 7.3 Measurements vs the capture

The tape-vs-estimate table of §3.5, per bone per capture. This is the most informative single
number in the spec, because it shows whether the metadata helps or biases. It should be stable
across the seven captures, since they share one lifter.

### 7.4 Consistency

- **Near-wrist vs bar vertical motion:** independent detectors.
- **Left-minus-right height in the pull:** median ≈ 0.
- **Height σ in the rest window:** lifter static.
- **LiDAR cross-check:** reported only.
- **Draft 1's body-plane heights vs this solve:** where they disagree, the symmetry assumption is
  the suspect.

### 7.5 Ground truth (proposed capture)

The same as draft 1 §6.5: a still standing pose at the start, with landmark heights tape-measured.
In addition, **record how each `athlete:` field was measured**, which answers open question 1.

## 8. Risks

- **Measurement definitions.** If `torso` means something else, the torso prior is wrong by
  centimetres in every frame. §3.5 catches this as a large tape-vs-estimate gap, but only after
  the fact.
- **Inferred joints look like measured ones.** This is mitigated only by `source[]` reaching the
  UI. The graphs should draw inferred stretches differently, as the bar view dashes its
  cap-missing frames.
- **Grounded misclassification** pins a lifted foot to the floor. That is bounded by the soft σ
  and surfaced by `ground_conflict`.
- **Local minima.** Mitigated by the closed-form start and the two-start check of §3.4, and
  counted by `branch_ambiguous`.
- **One lifter, one day.** Every result so far would come from a single body. The priors and
  thresholds are unproven for other proportions.

## Open questions

1. **How was each `athlete:` field measured?** Endpoints, posture and instrument. Blocking for §2.1's
   mapping.
2. **Ankle height:** the 70 mm population prior, calibrated per capture, or measured and added to
   the metadata (malleolus height)?
3. **Should frames with no anchor** use bone lengths alone, with an honest σ, or stay null? Null
   in v1.
4. **Two passes vs a bundle adjustment** (§3.5). This depends on how much the bone estimates
   scatter between frames.
5. **Should the metadata grow?** Upper arm, forearm, hip width (bi-iliac) and malleolus height
   would each turn an estimated bone into a measured one. They take about 2 minutes with a tape.
6. **The Tikhonov interface** (§5): does it take per-sample weights from `heightSigmaM` and
   `source`?

---

## Feedback (2026-10-20)

- **User:** the spec should not contain the lifter's actual measurements, only how to use them.
  The values belong in each capture's `metadata.yaml`. Draft 2 hard-coded one lifter's numbers
  in §2.1–2.3 and in the output example.
- **User:** add the four missing measurements to the metadata: upper arm, forearm, hip width and
  ankle height. Each then becomes a measured bone rather than a fitted one.
- **Incorporated in:** [`S8-joint-hva-draft3.md`](S8-joint-hva-draft3.md).
