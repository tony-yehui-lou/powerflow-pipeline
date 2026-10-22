# Floor-plane retilting

Status: **draft 2** | Supersedes: `4-retilt.md` | Updated: 2026-10-21

Draft 2 drops the operator-annotated floor rectangle. The floor is found from odometry gravity
and depth alone (§1, §3), and the gravity check becomes a rejection (§7). Design and real-data
evidence: `docs/superpowers/specs/2026-10-21-retilt-gravity-floor-design.md`.

## Purpose

Add a **Retilt** stage after Orient:

```text
S0 Ingest → S1 Cut → S2 Orient → S3 Retilt → S4 Crop
```

Rotate every frame so the camera's optical axis is perpendicular to the floor and its image
plane vertical — as if the phone had been level and plumb at capture time. This establishes
invariant **I3**: *"Images are undistorted and rectified as if the camera were level (no tilt
w.r.t. vertical)."* Metric scale (I4) is carried by depth, not by this stage or any downstream
one — see `1-ingestion_orient.md` and `5-scaling.md` (retired). Common framing (I5) is S4's
concern; this stage only removes tilt and roll, never yaw, and never touches scale.

**In:** portrait RGB + depth + confidence + `camera_matrix.csv` + `odometry.csv` — Orient's
output, already time-aligned and pixel-aligned across streams. Nothing is read from
`metadata.yaml`; any `*_floor_region_*` keys left there are ignored.
**Out:** the same streams, each pixel relocated (and, for depth, revalued) by the same
rectifying rotation, plus `camera_matrix.csv` unchanged and a sidecar recording the fitted
plane and applied rotation.

If a camera's floor cannot be found or fit reliably, the camera is **rejected**, never passed through
un-retilted or retilted from a bad fit — a silently mistilted clip corrupts every downstream
geometric assumption (S4 common framing) with no visible symptom in the frame itself.

## Assumptions

- **The camera is static for the whole take** (already assumed pipeline-wide — see
  `1-ingestion_orient.md` §2). Consequently there is **one floor plane, and one rectifying
  rotation, per camera** — not one per frame. A single homography is derived once and applied
  to every frame.
- Depth is stored in **millimetres**, `uint16` (confirmed from capture:
  `../data/9 July/*/Side/depth/000000.png` is `256x192`, values ~1306–4094). Back-projection
  converts to metres.
- `odometry.csv`'s `(qx, qy, qz, qw)` columns are a per-frame pose in ARKit's
  **gravity-aligned** world frame (per `1-ingestion_orient.md` §2). This gives an
  **independent** estimate of "down" that does not depend on depth. It **selects** which
  points are floor (§1, §3) and then **validates** the fit (§7), but never sets the rotation
  itself — depth is the geometric source of truth for where the floor is in this camera's
  frame.
  Recovering "down" in *this stage's* camera frame from that quaternion:

  ```text
  g_cam = R_z(90° CW) @ R_wcᵀ @ (0, −1, 0)      # R_wc: the row's camera-to-world rotation
  up    = −mean(g_cam over sampled rows), normalised
  ```

  **Corrected in draft 2.** Draft 1 also applied ARKit's camera-axis flip `diag(1, −1, −1)`
  (camera Y up / Z backward → CV Y down / Z forward). The exported pose's camera axes are
  already CV, so that flip mirrored the tilt. It disagreed with fitted floors by 2–20° on
  every real capture, growing with tilt. Without it, 39 captures over three days agree to a
  median 1.4°, max 4.0°.
- **The floor is the lowest large horizontal surface in view** (§1, §3). Nothing real sits
  below it, and on every real capture it held 41–94% of all confident depth points. No
  operator annotation is needed or read.

## 1. Point selection

There is no region: every confident pixel of each sampled frame is a candidate, and §3 decides
which are floor.

- Never use a pixel with confidence `0` or depth `0`.
- Use **confidence `2`** pixels of the whole frame.
- If the pooled confidence-2 cloud across sampled frames has fewer than
  `retilt_min_floor_points` points, rebuild it from confidence `1` **and** `2`
  (`confidence_mode = "conf1_and_2"`).
- Pool across sampled frames (§3), then downsample with a stride to at most
  `retilt_max_pooled_points` (200 000).

**Removed in draft 2:**
- The operator rectangle, its `"(x, y)"` parsing and its role-prefixed field names.
- The "largest 4-connected confidence-2 component ≥ 1/6 of the region" rule, along with
  `retilt_conf2_area_fraction`. That rule judged whether a small rectangle was clean enough,
  and has no meaning over a whole frame.

The rectangle failed in three ways on real data:
- It landed on a PA speaker cabinet.
- It landed on a spectator's head.
- 10 September's was August's, copied over and never checked.

The lifter or bar crossing it also contaminated the fit unflagged.

## 2. Back-projection to 3D

For each selected pixel `(u, v)` with stored depth `d_mm`, using the **depth-resolution**
intrinsics `K_d` (the portrait `camera_matrix.csv` scaled by `depth_width / rgb_width` and
`depth_height / rgb_height` — depth and RGB share a portrait frame but not a resolution).
Confirmed shapes: RGB `1440x1920`, depth `192x256`, giving scale `s = 0.13333` on both axes.
Scaling is **pixel-centre**, not corner: `fx_d = fx * s`, `fy_d = fy * s`, but
`cx_d = (cx + 0.5) * s - 0.5` and `cy_d = (cy + 0.5) * s - 0.5` — scaling the principal
point by `s` alone (no half-pixel correction) biases every back-projected point by up to
half a depth pixel, which is not negligible against the floor-plane RMS tolerance (§7).

```text
Z = d_mm / 1000                    # metres
X = (u - cx_d) / fx_d * Z
Y = (v - cy_d) / fy_d * Z
```

Camera frame convention: **X right, Y down, Z forward** (out of the lens), matching ARKit's
per-frame camera-local axes. A pixel with `d_mm == 0` (no return) is never selected — it
already fails the confidence-`0` rule.

## 3. Floor search and plane fit — one plane per camera

- Sample frames every `retilt_sample_stride` (10), capped at `retilt_max_sampled_frames`
  (32). The camera is static, so more frames add redundancy, not new geometry.
- Run point selection (§1) and back-projection (§2) on each; pool into one cloud.
- **Seed.** Measure each point's height along gravity's `up`, `h_i = p_i · up`.
  1. Histogram the heights in `retilt_floor_bin_m` (1 cm) bins.
  2. Smooth the counts with a 3-bin moving average.
  3. A *significant peak* is a local maximum, strictly above its left neighbour, whose raw
     counts within ±2 bins hold ≥ `retilt_floor_min_peak_fraction` (15%) of all points.
  4. The seed height `h0` is the **lowest** significant peak.
  5. If there is none, the camera is rejected: the floor is not visible.
- **Why the lowest peak.**
  - A larger peak higher up (a bench, a stage) is not the floor.
  - Sparse specular returns below a shiny floor never reach 15%.
  - Gravity's few-degree bias smears the floor's peak, by about `D · tan ε` over a depth
    span `D`, but does not hide it.
- **Refine.** Take the points within `retilt_floor_seed_band_m` (3 cm) of `h0` and fit them
  with **`torch.linalg.lstsq`**, regressing `Y = a*X + b*Z + c`. Then re-select every point of
  the cloud within `retilt_floor_inlier_band_m` (2 cm) perpendicular of that plane and refit.
  Repeat up to `retilt_floor_refine_iterations` (3) times, or until the inlier set stops
  changing. On real data three refits are within 0.3° and 1 cm of ten.
- The refits realign the plane to the floor's own orientation, so gravity's bias never
  reaches the result. A lifter, bar or wall is not a horizontal plane at floor height, so
  none of them enters the fit.
- The unnormalized plane normal is `n = (a, -1, b)`; normalize to unit length and orient it so
  it points from the floor toward the camera (`n_y < 0` in this convention — "up" is `-Y`).
- **Ambiguity.** If another significant peak lies within `retilt_floor_ambiguity_m` (0.30 m)
  above `h0`, for example a platform or a thick mat, record both and warn. The lower surface
  is kept.
- Record the **RMS residual** of the final inliers, their count, and the fraction of the
  pooled cloud they make up. These are validation inputs (§7) and sidecar fields (§6).

## 4. Tilt and roll angles → rectifying rotation

A horizontal floor constrains exactly the two degrees of freedom of the camera's attitude
that determine "level" — roll about the optical axis and tilt about the horizontal axis. Yaw
(rotation about the vertical) is unconstrained by a single plane and is **left untouched**.

```text
roll = atan2(n_x, -n_y)              # about Z (optical axis) — levels the horizon
n'   = R_z(-roll) * n                # de-roll the normal
tilt = atan2(n'_z, -n'_y)            # about X — points the optical axis at the floor's normal

R = R_x(tilt) @ R_z(-roll)
```

**Corrected** (implementation): the final composition is `R_z(-roll)`, not `R_z(roll)` as an
earlier draft of this section had it. Verified numerically against random floor normals
(`CLAUDE.md`: "verify geometry numerically ... never accept a transform merely because it
ran") — with standard right-handed `R_x`/`R_z` rotation matrices and the de-roll step above,
`R_z(roll)` in the final line does not level the fitted normal to `(0, -1, 0)`; `R_z(-roll)`
does, exactly, to floating-point precision, for every normal the de-roll step actually
de-rolls. `roll` and `tilt` themselves are unchanged — only this composition's sign.

`R` is the rotation that, applied to the camera, makes `R @ n = (0, -1, 0)` — i.e. after
retilting, the floor's normal is exactly "up" in the rectified camera frame. `roll` and
`tilt`, in degrees, are the values the source note's step 2 asks for; they are computed here
rather than assumed, from the fitted normal, and are provenance in the sidecar (§6).

## 5. Applying the rectification

A pure rotation of the camera induces a **homography** on the image plane:

```text
H = K @ R @ K⁻¹
```

applied once per stream (`K` is that stream's own intrinsics — RGB-resolution `K` for RGB,
`K_d` for depth/confidence) via an **inverse warp**: for every *output* pixel, map through
`H⁻¹` to find the *source* pixel(s) to sample.

- **RGB:** bilinear or bicubic resampling — a photometric stream tolerates interpolation.
- **Depth and confidence:** **nearest-neighbour only**, for the same reason S1 Orient uses
  it — interpolating depth fabricates values across object edges, and interpolating
  confidence between `0` and `2` invents a `1` the sensor never reported.
- **Depth-value recomputation.** Relocating depth pixels is not sufficient: a depth value is
  the distance along the *original* optical axis, which the rotation has now tilted. For each
  sampled output pixel with source pixel `x̃ = (u, v, 1)ᵀ` and its nearest-neighbour depth
  `Z_src`, rescale:

  ```text
  Z' = Z_src * (R_row3 · K_d⁻¹ · x̃)
  ```

  where `R_row3` is the third row of `R`. This is the standard depth-reprojection-under-
  rotation correction: it accounts for the fact that a point at distance `Z_src` along the
  old axis is no longer at that distance along the new, tilted axis. Store `Z'` back as
  `uint16` millimetres. **Both position and value are corrected** — this is the rigorous
  option the owner selected over relocate-only.
- **`camera_matrix.csv` is unchanged.** A pure rotation about the optical centre does not
  change intrinsics; `K` continues to describe the (now-rectified) portrait frame. Rewriting
  it would double-apply the correction downstream, exactly the failure mode S1 Orient already
  warns against for its own rotation.
- **`odometry.csv` and `imu.csv` pass through unchanged** — they describe device pose and
  inertial samples, not image-plane geometry; nothing about them is affected by a per-frame
  image warp.
- **Border handling.** Rectification always exposes invalid border regions (no source pixel
  maps there) and, symmetrically, can push valid content outside the original frame bounds.
  This stage does **not** crop or pad — it records the **valid-content bounding box** (in
  rectified pixel coordinates) in the sidecar. The consumer is **S4 Crop, which intersects
  this box with its own motion guard band and crops once** (`6-cropping.md` §2). Deferring the
  trim to that single downstream consumer, rather than cropping here too, keeps one stage
  owning what "valid" means instead of letting S3 and S4 disagree about it.

## 6. Output contract

New stage directory, e.g. `../data/s3_retilt_output/`, mirroring the existing
`s2_orient_output` layout (per `<date>/<session>/<camera>/`):

- `rgb.mp4` — rectified, re-encoded (same `crf` convention as S1/S2).
- `depth/`, `confidence/` — rectified frame-by-frame, `uint16` / `{0,1,2}` respectively,
  same frame count and filenames as the input.
- `camera_matrix.csv` — copied unchanged from Orient's output.
- `odometry.csv`, `imu.csv` — copied unchanged.
- `retilt_sidecar.json`:
  - `tilt_deg`, `roll_deg` — the two corrected angles;
  - `floor_normal_cam` — the fitted unit normal, pre-correction camera frame;
  - `plane_rms_residual_m`, `n_floor_points`, `n_frames_sampled`;
  - `confidence_mode` — `"conf2_only"` or `"conf1_and_2"`, per §1;
  - `floor_source` — `"gravity_seeded"`, telling these sidecars apart from draft-1 ones still
    on disk;
  - `gravity_up_cam`, `floor_seed_height_m`, `floor_peaks` (every significant peak as
    `[height_m, mass_fraction]`, lowest first), `floor_inlier_fraction`,
    `floor_refine_iterations` — the floor search (§3);
  - `homography_rgb`, `homography_depth` — the applied `H` for each resolution;
  - `gravity_agreement_deg` — angle between the fitted normal and the odometry gravity
    vector (§7);
  - `gravity_check_mode` — `"reject"` (§7);
  - `valid_bounds_px` — `[x0, y0, x1, y1]` of the rectified valid-content region;
  - `depth_values_recomputed: true`, `k_rewritten: false` — explicit, machine-checkable
    provenance flags in the same spirit as S1 Orient's `k_rewritten`.

Retilt neither reads nor writes `metadata.yaml`. The fitted plane and rotation are this
camera's own derived values, written to `retilt_sidecar.json`. Draft 1's `region_normalized`,
`region_px` and `confidence_mode_per_frame` fields are gone, as is the dry run's `region_px`.

Manifest: one step per camera, `derived` carrying `tilt_deg`, `roll_deg`,
`gravity_agreement_deg`, `plane_rms_residual_m`, `floor_offset_m`, `floor_source`,
`floor_inlier_fraction`, `floor_peaks`; `warnings` for an ambiguous floor (§3); `file_ops` following the `write`/`copy`/
`publish` shape used by S1/S2.

## 7. Validation and rejection

Reject the camera (exact reason recorded, per-camera — never per-session; retilting has no
cross-camera dependency) when:

- fewer than `retilt_min_floor_points` (proposed default **500**) confident points are pooled,
  or fewer than that many end up as the final plane's inliers — too little floor to fit;
- no height peak holds `retilt_floor_min_peak_fraction` of the points (§3) — floor not visible;
- the refined plane disagrees with odometry gravity by more than
  `retilt_gravity_tolerance_deg` (**6°**) — see below;
- the plane fit's RMS residual exceeds `retilt_max_plane_rms_m` (proposed default **0.02 m**,
  i.e. 2 cm — sized against LiDAR noise on a floor patch a few metres out) — the selected
  region is not actually planar, so it is probably not the floor (e.g. selection caught a
  foot, a plate, clutter);
- the derived `tilt` or `roll` exceeds `retilt_max_tilt_deg` / `retilt_max_roll_deg`
  (proposed default **45°** each) — implausible for a handheld/tripod phone, and a sign the
  fit converged on a degenerate or wrong plane, not evidence of a real 90°-tilted phone;
- odometry shows the camera translated beyond `retilt_max_translation_m` (proposed default
  **0.05 m**) across the sampled frames — this breaks the static-camera, one-plane-per-camera
  assumption the whole stage rests on;

**The capture day's fits must agree with each other — warn-only.** For a `SINGLE_CAMERA`
group, one unmoved tripod shot every capture of the day, so their fitted planes should
match. The per-capture gates above measure how *tightly* points fit a plane, never whether
that plane is the floor: on real data a rectangle that caught a spectator's head produced a
0.85 cm RMS at −44.9° of tilt, sliding under the 45° gate by a tenth of a degree. After every
capture of a `<date>/` is fitted, the run compares each against the day's **median** tilt,
roll and camera height (a median, not a mean, so a minority of wrong fits cannot drag the
reference toward themselves and exonerate one), and raises a manifest warning past
`retilt_group_tilt_tolerance_deg` (proposed **3°**) or `retilt_group_height_tolerance_m`
(proposed **0.10 m**). Both defaults sit outside the spread measured across 27 good captures
of one real day (1.6° and 0.07 m) and well inside the failure above.

It is **warn-only** for the same reason the gravity check is: it assumes a rig that never
moved, and wants calibrating against a second single-camera day before it fails a run. It is
skipped for a two-camera session, whose cameras are *supposed* to disagree, and for a day
with fewer than three captures, where the median is not a reference worth comparing against.
The medians and each capture's deviation are recorded in the **run manifest** rather than in
`retilt_sidecar.json`: the median is not knowable when that sidecar is written, and rewriting
a published sidecar afterwards would break the staging-then-publish rule every stage follows.

**The odometry-gravity check rejects.** It is the one gate against a plane that fits cleanly
but to the wrong surface, such as a ramp, a tilted mat or a plinth side. Draft 1 kept it
warn-only because the axis-convention chain was unproven, and that chain was in fact wrong
(Assumptions). Corrected, the check measures at most 4.0° across 39 captures over three days,
so the tolerance is **6°**. A second phone model, with a different IMU-to-depth bias, should
re-check that margin.

**Exit validation**, on the staged result before publish, mirrors S1 Orient's
`_validate_exit`: depth remains `uint16`, confidence values remain within `{0, 1, 2}` (never
invented by resampling), frame counts and filenames are unchanged from the input, and RGB/
depth output dimensions match their inputs (rectification does not resize).

## Open questions

- ~~Concrete default values for: frame-sampling stride/count for the pooled fit,
  RMS-residual tolerance, gravity-agreement tolerance (degrees), plausible tilt/roll bounds,
  and the camera-translation tolerance.~~ **Resolved** — proposed defaults are now stated
  inline in §7 (`retilt_min_floor_points`, `retilt_max_plane_rms_m`,
  `retilt_gravity_tolerance_deg`, `retilt_max_tilt_deg`/`retilt_max_roll_deg`,
  `retilt_max_translation_m`), plus `retilt_sample_stride` for §3's "every ~10th frame", as
  named, tunable parameters rather than hardcoded literals. All are proposals pending
  calibration against real sessions (§7's warn-only note applies to the gravity tolerance in
  particular).
- Each camera's valid-content bounding box now has a definite consumer — S4 Crop intersects it
  with its own motion guard band and crops once, per camera (see `6-cropping.md` §2). What
  remains open is only whether these boxes should additionally be **intersected across a
  session's cameras** for I5's benefit, or whether each camera's box stays purely per-camera
  and S4 does its own common-region derivation on top of that, alone. **Still open** — no real
  S4 implementation exists yet to decide against; `6-cropping.md`'s own open question is the
  same fork stated from S4's side.
- **Platform vs floor.** When §3's ambiguity warning fires, which surface should
  `floor_offset_m` describe? The feet stand on the platform, but the warning keeps the lower
  surface. No capture so far has two peaks; S8's rest-plate anchor is the natural
  tie-breaker.
