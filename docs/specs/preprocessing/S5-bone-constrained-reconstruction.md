# S5 Pose — bone-length-constrained joint reconstruction

Status: **draft** | Relates to: `S5-rtmpose-migration.md`, `S5-pose-model-recommendation.md` (issue [#118](https://github.com/tony-yehui-lou/PowerFlow/issues/118)) | Updated: 2026-09-20

## Purpose

Recover joints that the 2D detector found but the depth lookup threw away, by solving for their
depth along the viewing ray using a known bone length to a neighbouring joint that *does* have
depth.

This is the complement of the RTMPose migration. That spec changes the detector and explicitly
does not improve what reaches `pose.json`; this one changes nothing about the detector and
targets exactly the loss it left alone:

```text
S4 Crop -> 2D detector -> depth back-projection -> [ bone-constrained fill ] -> floor frame -> PoseDocument
                ^^^^^^^^                            ^^^^^^^^^^^^^^^^^^^^^^^
            S5-rtmpose-migration.md                  this, and only this
```

The loss is measured. On `22 August/CnJ/75kgCnJ1` and `22 August/Snch/38kgSnch1`, **42% of
successful 2D joint detections are discarded** by `lookup_patch_depth_m` returning `None`.
`DepthBackedPoseModel` then nulls the pixel and zeroes the confidence too, because `JointSeries`
requires `position` and `pixelPosition` to drop out together
(`depth_pose_model.py:77-83`). A perfectly detected joint with no depth is, in the stored
document, indistinguishable from a joint that was never found.

The mechanism is understood: the depth stream is 256x192 against 1920x1440 RGB, so **one depth
pixel spans ~7.5 RGB pixels**. The `radius=2` patch (5x5 depth px ≈ 38x38 RGB px) is wider than a
forearm (~3.6 depth px) but narrower than a thigh (~6.7). That is exactly the split we see in the
bone-length medians — femur 0.35 m and tibia 0.36 m are near-plausible while upper arm reads
0.38 m, longer than the femur, which is anatomically impossible.

**In:** unchanged — `DepthBackedPoseModel.predict`'s existing arguments.
**Out:** `dict[JointId, JointSeries]` as today, plus per-joint-per-frame provenance (§5) — which
requires a schema decision, and is the one part of this spec that reaches outside S5.

## Scope

**In scope:** `depth_pose_model.py`, `pose_geometry.py`, a new bone-length estimation module, a
new reconstruction module, and the `AthleteMeta` plumbing in §1.

**Out of scope, deliberately:**

- **Any change to the 2D detector.** This must work identically behind MediaPipe and RTMPose.
- **Temporal smoothing of output positions.** Reconstruction fills gaps from geometry that holds
  in a single frame; smoothing changes values that *were* observed. Same reasoning as
  `S5-rtmpose-migration.md` §3.
- **Two-camera fusion.** See §9 — it is the better fix and this spec is not it.
- **Improving the depth sampler itself** (segmentation gating, nearest-surface instead of median,
  per-frame torso-depth gates). Those reduce the number of joints needing reconstruction and are
  complementary; they are a separate change so that each can be measured on its own.

## Assumptions

- Bone length is constant within a capture. Lifts are seconds long; soft-tissue and
  joint-centre-estimation variation is noise against the errors being corrected.
- The 2D keypoint is trustworthy where depth is not. This is the load-bearing assumption of the
  whole approach: it says the detector's *direction* is good and only its *range* is missing.
  §6.1 tests it rather than assuming it.
- Depth failures are localised, not global. If no joint in a frame has depth there is nothing to
  anchor to and the frame is left alone.
- The camera is static and rectified (S3 Retilt), so `floor_offset_m` and the intrinsics hold for
  the clip.

## 1. Where bone lengths come from

**Primary source: the capture itself. `AthleteMeta` is a prior and a validator, not the input.**

This inverts the obvious design, for four reasons:

1. **The tape measure and the keypoint measure different things.** A coach's `tibia_in_cm: 39.0`
   runs between palpated anatomical landmarks. The reconstruction needs
   `‖ankle_keypoint − knee_keypoint‖` as *that detector* places those keypoints — a different
   quantity, offset by a systematic and model-dependent amount. Feeding a landmark measurement
   into a keypoint constraint injects a bias that then propagates down every chain.
2. **Only 4 of 15 bones are covered.** `tibia`, `femur`, `torso` and `armspan` map to 6 bones
   (left and right of the first three). Upper arm and forearm are not measured separately —
   armspan confounds them with shoulder breadth *and* hand length, and the population ratios that
   would split them are not mutually consistent with the recorded armspans (a 175 cm athlete with
   a 170 cm armspan and 42 cm shoulders leaves 64 cm per side for arm + forearm + hand, against a
   Drillis–Contini prediction of ~77 cm). Hip width is not measured at all. The four
   clavicle/head bones are between *derived* joints and are not anthropometric quantities.
3. **The 11 July captures have no measurements.** Their raw `metadata.yaml` files are still
   uninstantiated templates: `tibia_in_cm: float`, `name: string`. A design that requires athlete
   measurements silently excludes half the corpus.
4. **The pipeline does not currently read them even where they exist.**
   `tasks/metadata.py:195` writes `AthleteMeta().model_dump()` — an empty model. Every stage
   output carries `tibia_in_cm: null` for the August captures whose raw files hold `39.0`.

### 1.1 Plumbing work this nonetheless requires

Even as a prior, the metadata must actually arrive:

- `AthleteMeta` gains `shoulder_width_in_cm` and `body_fat_percentage`, both present in the real
  August files and both absent from the model today.
- S0 populates the athlete block from the per-capture `metadata.yaml` when that file is *filled*,
  and continues to write nulls when it is a template. `is_template` already distinguishes them;
  a partially-filled file must be handled leaf-by-leaf, not all-or-nothing, since a value of the
  literal string `float` must never parse as a number.
- The stage-output `meta_status` currently reads `absent` for August captures that do have a
  filled raw file. **Investigate before building on it** — either the template path resolution is
  wrong or the date-level/per-capture distinction is being conflated. This spec assumes the
  measurements can be made to arrive; it does not assume they already do.
- S5 reads the athlete block from its own stage-input `metadata.yaml`, reached via
  `CameraRecord.metadata_relative`, the same way `_floor_offset_m` reaches S3's sidecar.

## 2. Estimating per-capture bone lengths

For each bone in `HUMAN_SKELETON.bones`, over all frames of one capture:

1. Collect frames where **both** endpoints have an observed position from real depth.
2. Discard observations where the two endpoint depths are mutually implausible —
   `|Z_a − Z_b|` greater than the prior bone length — since that is the signature of one endpoint
   having landed on the background.
3. Estimate the length as the **median** of what survives, and record the interquartile range.
4. **Accept** the estimate only if it has at least `min_observations` samples, its IQR is below
   `max_relative_spread` of the median, and it falls inside a population plausibility range.
5. **Reject** otherwise, and fall back in order: the `AthleteMeta` measurement for that bone if
   present; then a height-scaled population ratio if `height_in_cm` is present; then no constraint
   at all, leaving that bone's joints unreconstructed.

Plausible adult ranges, as in `S5-rtmpose-migration.md` §6.2 — femur 0.38–0.50 m, tibia
0.36–0.45, torso 0.45–0.60, upper arm 0.26–0.36. Add: forearm 0.22–0.29, hip width 0.15–0.25.

### 2.1 The circularity, stated plainly

Bone lengths estimated from observed depth inherit whatever bias observed depth has. If the patch
median systematically lands on the wall behind a forearm, the estimated forearm is too long, and
reconstruction then propagates that error into every joint it solves.

This is the central risk of the spec. Three mitigations, all required:

- **The plausibility gate (step 4) is not optional.** A 1.63 m upper arm — which the
  confidence-`0` experiment actually produced — must fail closed, not be adopted as a constraint.
- **Prefer in-plane frames.** Weight the estimate toward frames where the bone is least
  foreshortened (its 2D pixel length is largest relative to its 3D length), because that is where
  a depth error contributes least to the measured length.
- **Cross-check against `AthleteMeta` where available.** A capture-estimated tibia that disagrees
  with the measured tibia by more than a stated tolerance is a finding to surface, not a number
  to quietly use. Report both; do not average them.

A bone whose estimate fails every fallback yields *no* reconstruction for its dependent joint.
That is the correct outcome: this spec's purpose is recovering joints, but never at the price of
inventing them.

## 3. The reconstruction, per joint per frame

Let joint `a` have an observed camera-frame position `(X_a, Y_a, Z_a)` and joint `b` be its
neighbour with a 2D pixel but no depth. With `p = (u_b − cx)/fx` and `q = (v_b − cy)/fy`, the
points on `b`'s viewing ray are `(pZ, qZ, Z)`, and the bone-length constraint `‖b − a‖ = L` gives

```text
A·Z² + B·Z + C = 0
  A = p² + q² + 1
  B = −2(p·X_a + q·Y_a + Z_a)
  C = X_a² + Y_a² + Z_a² − L²
```

This is a ray–sphere intersection: the ray through `b`'s pixel against a sphere of radius `L`
centred on `a`.

### 3.1 Which intrinsics

Solve with the **RGB-resolution keypoint and RGB intrinsics**, not the depth-quantised pixel.
`backproject_pixel` currently works in depth pixel space via `depth_intrinsics`, which is right
for an observed joint (the depth reading *is* quantised to that grid) and wrong for a
reconstructed one, where the precise 2D location is the entire remaining signal and quantising it
to a 7.5x coarser grid discards what we are relying on.

This makes observed and reconstructed joints use slightly different ray definitions. That
inconsistency is acceptable for a first version and is itself an argument for moving the observed
path to RGB-resolution rays later. Note also the known half-pixel discrepancy between
`rgb_pixel_to_depth_pixel` (`row * sy`) and `depth_intrinsics` (`(cy + 0.5) * sy − 0.5`), roughly
1.4 cm at these distances; it is not introduced here but it sits in the same code path and the
docstring claiming the two match is wrong.

### 3.2 Degenerate cases

- **`B² − 4AC < 0` (ray misses the sphere).** The 2D detection and the bone length are mutually
  inconsistent. If the shortfall is small, clamp to `Z = −B/2A`, the closest approach, which is
  the least-squares answer; record it as clamped. If it is large, **reject** — emit no position
  and record the reason. Large shortfalls indict the keypoint, the bone length, or the intrinsics,
  and the per-capture rate of them is a useful canary on the whole geometry chain (§6.4).
- **`Z ≤ 0`** — behind the camera. Reject.
- **Reconstructed position outside the capture's plausible working volume** (below the floor, or
  beyond LiDAR's ~4–5 m reliable range). Reject.

## 4. Choosing the root, and disambiguating the two roots

### 4.1 Solve order

Per frame, build a spanning tree over `HUMAN_SKELETON.bones` rooted at observed joints and
expanding outward, solving each unknown from an already-known neighbour. Order matters because
error accumulates along the chain, so:

- Prefer roots on the torso. Hips and shoulders sit on body regions wider than the depth-pixel
  footprint and are the joints whose depth is most trustworthy; wrists and ankles are the least.
- Prefer the shortest path from an *observed* joint over a shorter path through an already
  *reconstructed* one. Record the hop count; a joint reconstructed at depth 3 is a much weaker
  claim than one at depth 1.
- Cap the chain at `max_chain_depth` hops. Beyond that, reject rather than extrapolate.
- The four clavicle/head bones are excluded as constraints: both endpoints of the clavicle bones
  are interpolated, not detected, so their "length" carries no independent information.

### 4.2 Disambiguation

The quadratic has two roots — `b` may be nearer or farther than `a`. The half-separation along
the ray is `√(L² − d²)`, where `d` is the perpendicular distance from `a` to the ray. This has a
convenient structure that the rules below exploit:

- **Bone perpendicular to the line of sight** (well-conditioned, unforeshortened): `d → L`, ray
  tangent, **the roots coincide**. Nothing to choose.
- **Bone pointing at or away from the camera** (fully foreshortened): `d → 0`, roots at
  `Z_a ± L` — up to 0.7 m apart for a femur.

The true solution switches branch *exactly at tangency*, where the two roots are indistinguishable.
So a continuity-based rule glides through the switch rather than having to detect it: the
ambiguity is large only where the branch is stable, and the branch changes only where the
ambiguity is small.

Apply, in order:

1. **Occlusion bound (hard constraint).** Take the *nearest* valid depth reading in the patch
   around `b`'s pixel — the minimum, not the median. A limb cannot lie behind the nearest surface
   the LiDAR sees along that ray. Discard any root beyond that bound by more than a tolerance.
   This is robust to the dominant failure mode precisely because background bleed only pushes the
   median farther, never the minimum nearer. Note the corollary: **"pick the root closest to the
   measured depth" is a biased rule here** and must not be used in its place.
2. **Temporal continuity, solved globally.** Do not choose greedily per frame. The choice is one
   binary variable per (joint, frame), with unary costs from rule 1 and any marginal depth
   reading, and pairwise costs for temporal jumps. A chain over time is a Viterbi solve, linear in
   frames, and it is what prevents one early mistake from propagating. At 30–60 fps a joint moves
   2–5 cm between frames against a root separation of up to 2L, so this is decisive by an order of
   magnitude.
3. **Kinematic plausibility**, as a tie-break: reject roots implying a hyperextended elbow or a
   backward-bending knee.

If no root survives rules 1–3, reject the joint for that frame.

## 5. Output contract and provenance

A reconstructed joint **must be distinguishable from an observed one** in the stored document. A
consumer that cannot tell them apart will compute bar-path and joint-angle metrics from
half-inferred data and report them with the same authority as measured data. The body-model viewer
(issue #120) is already such a consumer.

`JointSeries` today carries `position`, `pixel_position` and `confidence` and has no room for
this. Three options, to be decided before implementation:

| option | cost | verdict |
|---|---|---|
| **A.** Add `source: tuple[Literal["observed","reconstructed",None], ...]` to `JointSeries` | `POSE_SCHEMA_VERSION` bump; `powerflow-ui` reader update | **Recommended.** The information belongs with the value it qualifies. |
| **B.** Per-capture sidecar with reconstruction stats only | no schema change | Rejected: per-frame provenance cannot be recovered from aggregate stats. |
| **C.** Encode it in `confidence` (e.g. reconstructed joints capped below a threshold) | no schema change | Rejected: overloads a field that already means something, and is silently lossy. |

Under option A, also record per capture, as a sidecar alongside the pose document: the bone-length
table actually used with its source per bone (`estimated` / `athlete_meta` / `population` /
`none`), sample counts and spreads, the reconstruction counts by outcome
(solved / clamped / rejected-with-reason), and the chain-depth histogram.

**Confidence for a reconstructed joint** is the 2D detector's own keypoint score attenuated by the
reconstruction's quality — chain depth, bone-length spread, and root separation at that frame.
Passing the raw 2D score through unchanged would claim more than was measured. The attenuation is
a stated formula, not a magic constant, and it must leave reconstructed joints strictly below
observed ones of equal 2D score.

The `JointSeries` invariant that `position` and `pixelPosition` drop out together is preserved:
this spec *adds* positions to frames that had a pixel, which is the case the current code is
forced to throw away.

## 6. Validation

### 6.1 Hold-out accuracy — the primary test

Take joints that **do** have confident depth, discard it, reconstruct them as if it were missing,
and compare against the withheld truth. This directly tests the load-bearing assumption of §3 and
yields a real number rather than a plausibility impression.

Report, per joint and per capture: median and 90th-percentile 3D error, and the **root-choice
accuracy** — how often rule §4.2 picked the correct branch, broken out by foreshortening, since
that is where the rule is under strain.

Acceptance: reconstruction is enabled for a joint only where its hold-out median error is below a
stated threshold. A joint the method reconstructs badly is better left null.

### 6.2 Coverage

Joint-frames with a position, before and after, per reference capture. Expected to rise on
`75kgCnJ1` and `38kgSnch1` (the 42% cases). Expected to be **unchanged on
`11 July/50kg_Set3/Front`**, whose athlete sits 9–11 m out with confidence-`0` depth across the
whole body: no anchor, no measurements, no reconstruction. That capture is out of this spec's
reach and should be stated as such rather than counted as a failure.

### 6.3 Anatomical plausibility

Bone-length medians and spreads before and after, against §2's ranges. Reconstruction should
**tighten** the spread, since constrained bones are exactly length `L` by construction — so report
the distribution over *unconstrained* bone pairs too, or the metric is circular and meaningless.

### 6.4 Geometry canaries

Per capture: the discriminant-miss rate, the clamp rate, and the chain-depth histogram. A high
miss rate means the 2D keypoints, the bone lengths or the intrinsics disagree, and is a bug
signal, not a tuning knob.

### 6.5 Numerical verification

Per `CLAUDE.md` ("verify geometry numerically with synthetic fixtures; never accept a transform
merely because it ran"): synthetic fixtures with a known camera, known joint positions and exact
bone lengths, covering both roots, the tangent case, the miss case, and a chain of three joints.
Recovered positions must match the construction to floating-point tolerance.

## 7. Configuration

`PreprocessConfig` gains, all with defaults that leave current behaviour unchanged:

- `reconstruct_joints: bool = False` — the master switch. **Off by default until §6.1 passes.**
- `bone_length_source: Literal["estimated", "athlete_meta", "estimated_with_meta_prior"]`
- `min_bone_observations: int`, `max_bone_relative_spread: float`
- `max_chain_depth: int`
- `reconstruction_max_error_m: float` — the §6.1 gate

`cli.py` gains the matching flags. As with `--pose-detector`, both behaviours stay selectable and
the default flips only in a separate change that cites measurements.

## 8. Risks

- **Circularity in the bone-length estimate** (§2.1). Highest risk; mitigated but not eliminated.
- **Fabrication that looks like measurement.** Mitigated only by §5 landing properly. If the
  schema decision is deferred, this spec should not be implemented — shipping reconstruction
  without provenance is worse than shipping nothing.
- **Error propagation along chains.** Bounded by `max_chain_depth` and reported by the histogram.
- **A well-behaved method on the wrong joints.** Wrists behind a barbell and ankles at the frame
  edge are both common and both hard; §6.1's per-joint gating exists for this.
- **`AthleteMeta` plumbing (§1.1) turning out to be larger than expected.** It touches S0, which
  every stage depends on. The estimation path does not require it, so it can be sequenced second.

## 9. The alternative this spec is not

Every lift in the July corpus is filmed from **two cameras**. The depth ambiguity that §4.2
manages is *directly measured* by the other view. Solving the extrinsics between the two cameras —
which the shared floor plane from `PlaneFit` plus the shared athlete makes tractable — converts
this entire problem from an inference into a triangulation, and would also give the fusion step
`PoseDocument`'s docstring already anticipates ("two cameras of one session are not in a common
frame until a fusion step exists").

It is the better answer where it applies. It is not this spec because the August corpus is
single-camera, and because it is a larger change. If both are eventually built, this one degrades
to the single-camera fallback.

## Open questions

- **The §5 schema decision.** Recommended as option A, not yet agreed. Blocking.
- **Whether `meta_status: absent` on filled August files is a bug** (§1.1), and how large.
- **Whether the estimated and measured bone lengths agree at all**, on the August captures where
  both exist. Unknown today, and cheap to answer — this is the single most informative
  measurement available before implementation, and it may change §1's ordering.
- **Whether hold-out error is representative.** Joints that have depth are, by construction, the
  easier ones; §6.1 may be optimistic about performance on the joints that actually need help.
  No clean way around this is known.
- **Whether reconstruction should run before or after the floor-frame transform.** Bone length is
  invariant under the rigid transform so it is mathematically free, but the ray through the origin
  is only simple in camera frame. Camera frame is assumed here.
- **Interaction with the depth-sampler improvements** listed as out of scope. If nearest-surface
  sampling recovers most of the 42%, the value of this spec drops sharply. Sequencing the two is
  an open call.
