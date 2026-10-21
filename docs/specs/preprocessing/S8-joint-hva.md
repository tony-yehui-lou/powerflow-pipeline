# S8 Kinematics — Joint Height, Velocity and Acceleration (Joint HVA)

Status: **draft 1** | Scope: GitHub issue [#66](https://github.com/tony-yehui-lou/PowerFlow/issues/66) (Joint HVA) under epic [#65](https://github.com/tony-yehui-lou/PowerFlow/issues/65) (Joint) | Sibling: [#63](https://github.com/tony-yehui-lou/PowerFlow/issues/63) Barbell HVA | Updated: 2026-10-20

The stage number **S8** is a proposal: the next free number after S7 Lift. It does not depend on
S7 (§2), so it could also take another slot.

## Purpose

For every tracked joint, per frame, report:

- **H**: the joint's height above the floor (m);
- **V**: its vertical velocity (m/s, positive upward);
- **A**: its vertical acceleration (m/s², positive upward; kinematic, so gravity is not added).

These fill the `heightM[]`, `velocityMps[]` and `accelMps2[]` arrays that powerflow-ui's `Joint`
type already declares (`powerflow-ui/docs/datatypespec.md`). They are the Joint counterpart of the
bar's HVA (#63), and they feed the HVA graphs (#45, #104 and #105), COM HVA (#68) and phase
detection (#94).

Two problems have to be solved, and they are independent:

1. **Height** (§2). This needs each joint's distance from the camera. S6 only gives pixels.
2. **Velocity and acceleration** (§3). These are derivatives of a noisy height series. That is the
   same open item as #63's "choose a smoothing method", so §3 is written to serve both.

**In:**
- S6's `pose.json` (2D joints, wrists already refined);
- S6's `wrist_refinement.json`;
- S5's barbell detections, through `barbell_input.py` and `bar_geometry.solve_track`;
- S4's `camera_matrix.csv`;
- S3's `retilt_sidecar.json`, as a cross-check only;
- the declared bar type.

**Out:** `joint_hva.json` per capture (§5): H, V and A per joint per frame, with the method and
quality flags on every value.

**Not in scope:**
- COM HVA (#68);
- joint angles (#70);
- phase segmentation (#94/#95);
- the UI's joint selector and graphs (#105, #45);
- moving the bar's own HVA into this stage. That is bar-height draft 6, but §2.4 and §3 are
  written so the bar can share them.

## Assumptions

- S3 has levelled the camera, so the S4 camera frame (X right, Y **down**, Z forward) has a vertical
  Y axis. This is the frame `bar_geometry` already uses.
- **The lifter is bilaterally symmetric about the bar's mid-plane while the hands are on the
  bar.** The grip is centred on the bar, so the body's mid-sagittal plane passes through the bar
  centre and is perpendicular to the bar. This is the same assumption the wrist mirror makes
  (`S6-wrist-refinement.md` §4), applied to the whole body. §2.6 says where it breaks.
- One static, single camera per capture (the `10 September` layout), at 60 fps with VFR jitter of
  16–17 ms. Timing always comes from `startEpochMs + tMs[i]`, never `i / fps`.
- No LiDAR depth. This follows the bar-height decision of 2026-10-19 (draft 5 feedback: "drop
  depth"). That decision was made for good reason, and the same reasons apply to the body
  (§2.1).

---

## 1. What is reported

One series per `JointId`, all 15, each `frames.count` long. In order of how much a coach cares:

| group | joints | typical use |
|---|---|---|
| lower body | ankles, knees, hips | hip rise vs bar rise, the double knee bend, jump and landing |
| upper body | shoulders, elbows, wrists | early arm bend, turnover, lockout |
| derived | clavicles, head | completeness only; the clavicles are interpolated in S6 |

Every value can be `null`. A null with a reason (§5) is always better than a number the stage
cannot back up.

---

## 2. Height: placing the joints in 3D without depth

### 2.1 Methods considered

| | Method | Verdict |
|---|---|---|
| A | **S7 Lift: LiDAR patch median at each joint pixel** | **Rejected as the source; kept as a cross-check.** 42% of 2D detections get no depth. Arm bone lengths are anatomically impossible (upper arm 0.38 m against femur 0.35 m), because a 5×5 depth patch is wider than a forearm. Depth on the bar read the background or ±150 mm, which is why the bar dropped it. |
| B | A plus bone-length reconstruction (`S6-bone-constrained-reconstruction.md`) | **Rejected for now.** It inherits A's anchors and their bias (its §2.1 circularity). It is unimplemented and off by default until its hold-out test passes. |
| C | **Bar-anchored body planes:** intersect each joint's viewing ray with a vertical plane at a known lateral offset from the bar's mid-plane (§2.2) | **Recommended.** Closed form and depth-free, and built on the bar track that already exists. The only per-capture unknown, the lateral offset of each joint, is measured from the 2D pose itself (§2.3). |
| D | C with every offset set to zero: all joints on the mid-plane | **Fallback only** (§2.5). It is simple, but ignoring a shoulder's ~0.2 m lateral offset biases its height (worked in §2.2). |
| E | Monocular 3D lifters: VideoPose3D, MotionBERT, MeTRAbs | **Rejected**, as in `S6-pose-model-recommendation.md`. Their output is root-relative with no metric scale, and MeTRAbs's weights are non-commercial. |
| F | Parametric body-model fit (SMPL family) | **Rejected.** SMPL's licence is non-commercial, it adds a heavy dependency, and it still needs scale from somewhere. |
| G | Two-camera triangulation | The best method where it applies, but only the July corpus is two-camera. It is out of scope here; if built, it supersedes C for those captures. |

### 2.2 The body-plane solve (method C)

From `bar_geometry.solve_track`, frame `t` has the bar centre `C = (Cx, Cy, Cz)` and yaw `ψ`.
The bar axis is the horizontal unit vector `a = (sin ψ, 0, cos ψ)`, from cap to collar.

The **mid-plane** is `{P : (P − C)·a = 0}`. Joint `j` lives on a parallel plane at its signed lateral
offset `d_j`:

```text
(P − C) · a = d_j
```

Its pixel `(u, v)` gives the viewing ray `P = Z·r` with `r = ((u − cx)/fx, (v − cy)/fy, 1)`. Then:

```text
Z   = (C·a + d_j) / (r·a)
P   = Z · r
h_j = Y_floor − P_y              (Y is down; Y_floor is the floor's Y in camera frame, §2.4)
```

This is one division per joint per frame. It needs only that joint's own pixel, so an occluded
far-side joint does not corrupt a near-side one.

**Conditioning.** The ray must cross the plane, not run along it. With `c = |r·a| / ‖r‖`:

```text
δZ ≈ δd / (r·a)            a lateral-offset error moves the joint along its ray
δh ≈ |y_n| · δZ            and only its vertical component reaches the height (y_n = r_y)
```

On `10 September` the bar yaw is ~37° off the optical axis, so `c ≈ 0.8`, which is well
conditioned. In a pure front view (`ψ ≈ 90°`, e.g. a July `Front` camera) `c → 0` and the method
does not apply. Frames with `c < c_min` (default 0.3) are flagged `ill_conditioned` and get no
value.

**Sensitivity, worked (illustrative).** A joint 1 m below camera height at 4 m range has
`y_n ≈ 0.25`.
- **Method D** puts a shoulder that is really 0.20 m off the mid-plane on the mid-plane. That is
  `δZ ≈ 0.25 m`, so `δh ≈ 60 mm`.
- **Method C** with its offset wrong by 30 mm gives `δZ ≈ 38 mm`, so `δh ≈ 9 mm`.

Height is forgiving of range error by the factor `|y_n|`, which is why a depth-free method works
at all.

### 2.3 Measuring each joint's lateral offset `d_j`

The offsets come from three sources, in order of preference:

1. **Wrists:** `± halfGripMm` from `wrist_refinement.json`, which is already measured (461–472 mm
   on `10 September`). The sign comes from which wrist is `near`.
2. **Every left/right pair (the symmetric-pair solve).** Under the symmetry assumption, a pair's
   two points mirror each other across the mid-plane:

   ```text
   P_L = Z_L r_L,   P_R = Z_R r_R
   P_L − P_R = 2w · a                         (3 equations)
   ((P_L + P_R)/2 − C) · a = 0                (1 equation)
   ```

   That is four equations, **linear** in three unknowns `(Z_L, Z_R, w)`. Solve them by least
   squares. The residual tests the assumption on that frame: torso rotation, a split stance or a
   detector error all show up there. A frame is accepted when:
   - both joints have confidence ≥ `pair_min_conf` (default 0.7);
   - the residual is ≤ `pair_max_residual_px` when reprojected;
   - 2 px of pixel noise moves `w` by ≤ `pair_max_w_sigma_mm` (the same perturbation test the bar
     uses for yaw).

   **`d_j = ±median(w)`** over accepted frames, with at least `pair_min_frames` (default 20) of
   them.
3. **Prior.** If a pair fails the gate, use `shoulder_width_in_cm / 2` from `AthleteMeta` for the
   shoulders, and population ratios of `height_in_cm` for the rest. If neither exists, fall back to
   method D for that pair, flagged `offset_prior=none`.

`head` has `d = 0` by definition. The clavicles take S6's interpolation weight between 0 and the
shoulder offset.

One offset per capture is the v1 choice. Hips, shoulders and elbows hold their lateral offset
through a lift. Ankles and knees don't: the feet jump out between the pull and the catch, roughly
0.15 → 0.22 m half-width. By the sensitivity above, a 70 mm offset error at an ankle (`y_n ≈ 0.22`)
costs ~19 mm of ankle height, on a joint that barely rises. That is accepted for v1, and §6.4
measures it. The upgrade is a per-frame `w(t)` from the pair solve, interpolated across rejected
frames the way the bar's yaw is.

### 2.4 Floor anchor: shared with the bar

The joints and the bar must sit on **one** floor, or comparisons like "hip height minus bar height"
inherit the gap between two anchors.

`Y_floor` is the bar track's rest-plate anchor: the median bar-centre `Y` over the solved frames
in the first 60, plus the plate radius (225 mm). The fallback is S3's floor plane, exactly as in
#63. This logic lives in powerflow-ui's `build_bar_height.anchor_heights` today. It moves into the
pipeline next to `bar_geometry.py`, where it serves both the bar and the joints. S3's floor plane
is always reported alongside as a cross-check.

### 2.5 Which frames get a height

| frame state | method | flag |
|---|---|---|
| hands on bar, bar placed, joint confidence ≥ `min_conf`, `c ≥ c_min` | C | `method="bar_plane"` |
| as above, but that pair's offset fell back to no prior | D | `method="mid_plane"` |
| bar not placed (cap and collar both missing, or a gap > 30 frames) | — | null, `reason="no_bar"` |
| hands off the bar (before the grip, after the release) | — | null, `reason="off_bar"` |
| joint confidence < `min_conf` (default 0.5) | — | null, `reason="low_confidence"` |
| `c < c_min` | — | null, `reason="ill_conditioned"` |

**Hands-on-bar frames** are the frames S6 already decided, via its ±5-frame majority vote. By
`S6-wrist-refinement.md` §5, they are exactly `farCorrected` in `wrist_refinement.json`. A small
additive change exports them under an explicit `handsOnBar` key so this stage does not depend on
that coincidence.

Off-bar frames are null in v1. The lift happens with the hands on the bar. Once the bar is
dropped, the mid-plane no longer follows the body. S7's LiDAR heights could fill these frames later,
with `method="lidar"`. This spec does not do that.

**Far-side joints.** The far elbow is the weak one: confidence median 0.51 on `10 September/50kg1`,
so about half its frames fall to `low_confidence`. Option, **off by default**: fill a low-confidence
far joint with its near partner's height (symmetry gives equal heights), flagged
`method="mirrored"`. It is enabled only if §6.3's symmetry check passes on frames where both sides
are confident.

### 2.6 Where the symmetry assumption breaks

- **Split jerk.** Fore-aft is fine, since it stays in the sagittal plane. But the knees and hips
  are no longer mirror images vertically: the front knee is high and the back knee is low.
  Individual joint heights stay valid, because §2.2 uses only each joint's own pixel and a lateral
  offset. The pair solve's residual rejects those frames, so they don't pollute `d_j`.
- **Torso rotation, or an off-centre grip.** These show up in the pair residual and the
  near/far grip asymmetry. 45kg1's 544 mm half grip is the known suspect.
- **Bar tilt.** It tilts the mid-plane too. That costs ~25 mm of height per degree at the bar, and
  less at the body, which sits closer to the mid-plane. It is assumed zero, as everywhere.

---

## 3. Velocity and acceleration

### 3.1 Why raw differences are unusable

Take a 5 mm height noise at 60 fps (`Δt ≈ 16.7 ms`). Central differences give velocity noise of
about `√2·σ/Δt`. Second differences give acceleration noise of about `√6·σ/Δt²`:

```text
σ_v ≈ 0.42 m/s,   σ_a ≈ 44 m/s²
```

Peak bar velocity in a snatch is ~2 m/s, so the noise is the same size as the signal. Smoothing is
mandatory. The choice of smoother decides how much the peaks are blunted, especially the
deceleration at the catch.

### 3.2 Methods considered

| | Method | Handles VFR and gaps | Verdict |
|---|---|---|---|
| 1 | Finite differences | yes | **Rejected** (§3.1). |
| 2 | **Zero-lag Butterworth** (2nd-order low-pass run with `filtfilt`, so 4th order effective), then central differences | no: resample to a uniform grid and split at gaps | **Recommended default.** It is the biomechanics standard (Winter). Its cutoff in Hz is interpretable, and one cutoff can apply to bar and joints alike, so their curves are comparably smoothed. |
| 3 | Savitzky–Golay or local polynomial regression | uniform only; a time-weighted local-fit variant handles VFR | **Alternative.** It gives derivatives directly, but its window is less intuitive than a cutoff and its ends behave worse. |
| 4 | **Quintic smoothing spline** (Woltring GCVSPL; `scipy.interpolate.UnivariateSpline(k=5)` with weights) | yes, natively, with confidence weights | **The comparison candidate in §6.2.** It suits acceleration well. But GCV picks a different smoothing for every series, so joints and bar would not be smoothed alike. |
| 5 | Kalman / RTS smoother, constant-acceleration model | yes, natively, and gives uncertainty bands | **Deferred.** It is principled, but its process-noise tuning is one more knob without ground truth to set it against. |
| 6 | OneEuro | yes | **Rejected.** It is causal and built for real time, so it lags. This stage is offline. |

### 3.3 The recommended procedure

Per joint, on the `H` series of §2:

1. **Segment.** Bridge gaps of ≤ `max_bridge_frames` (default 6, i.e. 100 ms) by linear
   interpolation in time. Longer gaps split the series into separate segments.
2. **Resample** each segment onto a uniform grid at the capture's median frame interval.
3. **Filter** with a zero-lag Butterworth, cutoff `fc`. Pad the ends by reflection (`padtype="odd"`)
   so the filter does not pull the endpoints down.
4. **Differentiate** the filtered grid by central differences to get V, then again to get A.
5. **Sample back** at each original frame's `tMs`.
6. **Flag:**
   - bridged frames are `interpolated`; they have V and A but **no H**, so a height is never
     stated where nothing was observed;
   - the first and last `⌈fs / (2·fc)⌉` frames of each segment are `edge`;
   - a segment shorter than `min_segment_s` (default 0.5 s) keeps its H and gets no V or A.

**Cutoff.** One `fc` per capture, shared by every joint and, once draft 6 lands, the bar. The default
is **6 Hz**, Winter's figure for gross human movement. Per series, the stage also runs Winter's
residual analysis and reports the cutoff it would have chosen. If that disagrees with `fc` by more
than 2 Hz, it raises a warning rather than switching the cutoff silently. Separate cutoffs per joint
would shift each joint's peak by a different amount, and peak-timing comparisons are the point of
the graphs.

---

## 4. Code layout

Following `powerflow-pipeline/CLAUDE.md`: pure geometry and maths go in modules, with no I/O.

- `preprocess/bar_geometry.py`: gains the floor anchor (§2.4).
- `preprocess/body_plane.py` (new):
  - `solve_pair` (§2.3);
  - `lateral_offsets` (median, gates, priors);
  - `place_joint` (§2.2).
- `preprocess/kinematics.py` (new): `differentiate(t_ms, h, fc, ...)` returns H, V, A and flags
  (§3.3). It is bar-agnostic, so the bar's HVA reuses it.
- `preprocess/tasks/kinematics.py` (new `@task`): reads the inputs, writes `joint_hva.json`, and
  reports manifest values.
- `cli.py`: `--kinematics <root>`, which needs `--pose` and `--barbell`. It writes to
  `../data/s8_kinematics_output/<capture path>/`.

Thresholds are module constants, each documented where it is defined, as in `wrist_refine.py`.
Only `fc`, `min_conf`, `max_bridge_frames` and the mirrored-fill switch go into `PreprocessConfig`.

## 5. Output contract

`joint_hva.json`, written once per capture (replaced only with `--overwrite`):

```jsonc
{
  "schemaVersion": "1.0.0",
  "captureId": "10 September/50kg1",
  "method": "docs/specs/preprocessing/S8-joint-hva.md",
  "frames": { /* copied verbatim from pose.json: count, fps, startEpochMs, tMs */ },
  "floor": { "anchor": "rest_plate", "yFloorM": 1.093, "s3FloorPlaneYM": 1.121 },
  "smoothing": { "filter": "butterworth", "order": 4, "cutoffHz": 6.0,
                 "residualCutoffHz": { "rightHip": 5.4, "...": 0 } },
  "lateralOffsetsM": {
    "leftWrist":  { "value": -0.467, "source": "half_grip" },
    "leftHip":    { "value": -0.112, "source": "pair_solve", "frames": 214, "iqrM": 0.009 },
    "head":       { "value": 0.0,    "source": "midline" }
  },
  "joints": {
    "rightHip": {
      "positionM":   [[x, y, z] | null, ...],   // S4 camera frame, from §2.2; for COM and angles
      "heightM":     [float | null, ...],
      "velocityMps": [float | null, ...],
      "accelMps2":   [float | null, ...],
      "method":      ["bar_plane" | "mid_plane" | "mirrored" | null, ...],
      "flags":       [["interpolated" | "edge" | "ill_conditioned"], ...],
      "reason":      [null | "no_bar" | "off_bar" | "low_confidence" | "ill_conditioned", ...]
    }
  }
}
```

- **Array names match powerflow-ui's `Joint` type**, so the UI's lift document maps them
  straight across. The UI lists 10 joints with no shoulders. Its own note says to add shoulders
  to the list, the skeleton and the schema enum together, and this stage emits all 15.
- `positionM` is kept because COM (#68) and joint angles (#70) need more than height. Joint
  angles in particular need the sagittal coordinate, which this solve provides and the vertical
  HVA does not report.
- **Manifest `derived`:**
  - `kinematics_cutoff_hz`;
  - `kinematics_floor_anchor`;
  - `kinematics_frames_with_height` per joint;
  - `kinematics_offsets_source` per pair;
  - `kinematics_pair_frames_accepted` per pair.
- **When the stage cannot run** (no bar track, no hands-on-bar frames, no camera matrix): a
  manifest warning, with every array null and a `reason`. It never rejects the capture, and never
  falls back silently to a different method.

## 6. Validation

There is no ground truth today, so v1 is accepted on the checks below, each run on all seven
`10 September` captures. 50kg3 stays the held-out capture, as in #63.

### 6.1 Numerical (unit tests, synthetic)

Per `CLAUDE.md`: a synthetic camera, bar track and symmetric skeleton with known 3D joints.
- `place_joint` recovers them to floating-point tolerance.
- `solve_pair` recovers `w` exactly when the pair is symmetric, and its residual grows with
  injected torso rotation.
- `c → 0` raises `ill_conditioned` without dividing.
- `differentiate` recovers analytic `h(t)`, `ḣ` and `ḧ` (sinusoids at 1–3 Hz plus a bar-like
  pull-catch profile) under jittered 16/17 ms timestamps, with gaps and added noise at the
  measured σ. It reports peak-value and peak-time error against the truth.

### 6.2 Smoother choice (Butterworth vs quintic spline)

Use the synthetic profiles from §6.1, with noise at the σ measured in §6.3. The default stays
Butterworth unless the spline beats it on peak-acceleration error at the catch by a clear margin.
If it does, the default flips in a separate, small change that cites the numbers. This is the same
rollout rule as `--pose-detector`.

### 6.3 Real-data consistency checks

| check | what it measures | expectation |
|---|---|---|
| **Rest noise floor:** first 60 frames, lifter static | σ of H, V and A per joint | the σ that §6.2 uses; V ≈ 0 and A ≈ 0 |
| **Near wrist vs bar centre:** hands on bar | V(t) of the wrist against V(t) of the bar. The detectors are independent (S6 vs S5); the wrist rides the bar. | peak V within ~0.1 m/s; peak times within 1–2 frames |
| **Bilateral symmetry:** pull phase, both sides confident | left-minus-right height for hips, knees and shoulders | median ≈ 0; this gates the mirrored fill (§2.5) |
| **Integration:** `∫V dt` vs `ΔH` over each segment | that the differentiator is consistent | < 10 mm drift |
| **LiDAR cross-check:** S7 heights where its depth is confident | method C vs method A | report only; a large torso disagreement is a finding, not a failure |
| **Bone lengths:** from `positionM` | femur and tibia against `S6-rtmpose-migration.md` §6.2's ranges | inside the range, with a tighter spread than S7's |

### 6.4 Offset sensitivity

Re-run each capture with each `d_j` moved by ±30 mm, and report the change in peak height and
peak velocity per joint. Also compare per-frame `w(t)` against the per-capture constant for the
ankles and knees, which decides whether the §2.3 upgrade is needed.

### 6.5 Ground truth (proposed capture)

The lifter stands still for 3 s at the start of a capture. The operator tape-measures the
heights of the lateral malleolus, knee joint line, greater trochanter, acromion and wrist crease.
This shares the session with #63's bar-on-blocks capture. Keypoints are not anatomical
landmarks (`S6-bone-constrained-reconstruction.md` §1), so the tolerance is ±30 mm, not tape
precision.

## 7. Risks

- **The symmetry assumption is load-bearing.** The pair residual detects where it fails, but
  frames that fail it still get a height from the per-capture offset. §6.3's symmetry check and
  §6.4's sensitivity runs bound the cost.
- **The bar is a single point of failure.** No bar track means no joint heights: 45kg1 has its
  bar on only 139/367 frames. That coupling is deliberate, because the hands-on-bar frames are
  the lift, but it means bar-detection fixes now improve two features.
- **Smoothing hides real peaks.** At 6 Hz, catch impacts and the jump's foot strike are blunted.
  Peak acceleration is a lower bound, and the UI's graph labels should say "smoothed".
- **One camera geometry.** Only oblique-side views qualify (`c ≥ 0.3`). A pure front camera gets
  nothing, by design.

## Open questions

- **Stage number and home.** Is it S8 in `preprocess`, or the start of an analysis pipeline (the
  architecture's "Phase analyzer" side)? This spec assumes `preprocess` for now, next to the bar
  geometry it reuses.
- **Should off-bar frames use S7 LiDAR heights** (`method="lidar"`), or stay null? Null in v1.
- **Is a 6 Hz default right for weightlifting?** The literature on barbell velocity uses 4–10 Hz.
  §6.2 and the residual analysis decide.
- **Per-frame lateral offsets for the lower body** (§2.3). They depend on §6.4.
- **An explicit `handsOnBar` export from S6** (§2.5). It is a small additive change to
  `wrist_refinement.json`.
- **Should the bar's HVA move here in bar-height draft 6**, so bar and joints share one floor
  anchor and one smoother by construction rather than by convention? Recommended.

---

## Feedback (2026-10-20)

- **User:** velocity and acceleration will use the user's own **Tikhonov regulariser**, to be
  specified later. §3's smoother comparison (Butterworth vs spline) is dropped.
- **User:** use the lifter's **body measurements** (`metadata.yaml` `athlete:` block) to solve
  the 3D joints.
- **Checked:** all seven `10 September` captures carry the same filled athlete block: height 180,
  tibia 39.8, femur 42.3, torso 51.5, armspan 175.6 and shoulder width 44.0 cm. The pipeline does
  not read the block today: S0 writes an empty `AthleteMeta`.
- **Incorporated in:** [`S8-joint-hva-draft2.md`](S8-joint-hva-draft2.md).
