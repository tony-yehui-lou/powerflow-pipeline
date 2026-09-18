# Floor-plane retilting

Status: **draft** | Derived from: `4-retilt.md` (source note) | Updated: 2026-08-26

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
output, already time-aligned and pixel-aligned across streams — plus the session's raw
`metadata.yaml` (§1), the same file S1 Cut reads, for this camera's operator-annotated floor
region.
**Out:** the same streams, each pixel relocated (and, for depth, revalued) by the same
rectifying rotation, plus `camera_matrix.csv` unchanged and a sidecar recording the fitted
plane and applied rotation.

If a camera's floor cannot be fit reliably, the camera is **rejected**, never passed through
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
  **independent** estimate of "down" that does not depend on the depth-fitted floor, and is
  used only to validate the fit (§7), never to derive the rotation itself — depth is the
  geometric source of truth for where the floor actually is in this camera's frame.
  Recovering "down" in *this stage's* camera frame from that quaternion requires composing
  three changes of basis, none of which this document derives independently of real data:
  world (ARKit, Y up) → ARKit camera-local axes (X right, Y up, Z **backward** — note this
  is not the X-right/Y-down/Z-forward CV convention §2 uses for back-projection) → CV camera
  axes → S2's rotated portrait frame (S2 rotates the *image*, per `orient.py`'s
  `ODOMETRY_NOTE`, without touching the odometry columns themselves). A sign error in any
  one step is not detectable from the spec text, only against real captures — so the
  gravity-agreement check (§7) ships **warn-only** until calibrated against real sessions,
  never as a rejection from day one.
- **The floor region is operator-annotated, not inferred** (§1): a person has looked at this
  specific camera's frame and drawn a rectangle around a clean patch of floor, once per
  camera per session, recorded in the session's shared `metadata.yaml`. There is no
  algorithmic fallback if it is absent.

## 1. Floor-region selection

The selection region is **operator-annotated per capture**, not inferred. Which fields to
read is decided by the capture's **role**, resolved once by discovery along with the path of
the `metadata.yaml` that governs it — the same file S1 Cut reads for the lift window.

| role | keys read from the `video:` block |
|---|---|
| `front` | `front_floor_region_bottom_left_in_pixels`, `front_floor_region_top_right_in_pixels` |
| `side` | `side_floor_region_*` |
| `single` | **unprefixed** `floor_region_*`, accepting the `side_`-prefixed spelling as an alias |

The `side_` alias exists because a single-camera capture's lift window is already spelled
`..._side_in_ms`; whichever the operator types works. There is **no `front_` alias for
`single`**, and `FloorRegion` does **not** swap an inverted `y` pair: real captures of that
shape pair a `front_`-prefixed bottom-left with an unprefixed top-right *and* invert `y`, and
both are authoring mistakes rather than alternate spellings. Accepting either would wave
through a rectangle nobody finished checking — on the real data, one that sat on a PA speaker
cabinet rather than the floor. The rejection names the exact field that was missing.

Two cameras of a session share one `metadata.yaml` at `<date>/<session>/metadata.yaml`, so
their region fields are **prefixed by view** to tell them apart, under a `video:` block:

```yaml
video:
  front_floor_region_bottom_left_in_pixels: (x0, y0)
  front_floor_region_top_right_in_pixels: (x1, y1)
  side_floor_region_bottom_left_in_pixels: (x0, y0)
  side_floor_region_top_right_in_pixels: (x1, y1)
```

Retilt reads the pair matching **this capture's own** role — a plain lookup key into the
governing file, not a geometric rule: nothing about the rectangle itself is
derived from the camera's name, only which of the four fields to read.

Coordinates are normalized to `[0, 1]` in **standard image convention**: `x` left→right,
`y` **top→bottom** (`y=0` is the top of the frame, `y=1` is the bottom). This is the
*opposite* vertical convention from the ARKit/CV camera-frame axes used from §2 onward in
this document (`Y` down there means a 3D direction, not a 2D screen position) — do not
conflate the two. "`bottom_left`"/"`top_right`" name the rectangle's corners by their
on-screen position, not a fixed geometric rule: a front-facing capture's region can span
nearly the full width while a side-facing one covers less than half, entirely at the
operator's discretion once they have looked at the frame. (This replaces an earlier draft of
this section that derived the region algorithmically from the camera's `Front`/`Side` role;
real annotated regions contradicted that rule's assumptions, so it has been dropped.)

**On-disk representation.** Real captures write these four fields as **parenthesised
strings**, e.g. `front_floor_region_bottom_left_in_pixels: (0, 1)` — `yaml.safe_load` returns
the literal string `"(0, 1)"`, not a YAML sequence or a tuple. A reader must parse
`"(x, y)"` explicitly (strip parens, split on `,`, `float()` each part); treating the value
as already-numeric, or applying a numeric-type check before parsing, rejects every real
region. This mirrors `2-cut.md`'s `read_lift_window`, which validates *after* it has a
number in hand, not before.

**The `_in_pixels` suffix is legacy and wrong**: every real value observed is a normalized
`[0, 1]` fraction (e.g. `0.75093`, `0.40268`), consistent with the rest of this section, not
a pixel coordinate. Treat the field names as fixed identifiers and the suffix as
non-authoritative; a value outside `[0, 1]` after parsing is rejected as out-of-bounds (§7),
not reinterpreted as pixels.

Converted to pixel row/column ranges for a `width x height` portrait image (0-indexed rows,
top→bottom):

```text
col_start = round(width * x0)      row_start = round(height * y1)
col_end   = round(width * x1)      row_end   = round(height * y0)
```

Within that region, restated from the source note with the threshold denominator the owner
has fixed at **one-sixth**:

- Never use depth coordinates with confidence `0`.
- Find the **largest 4-connected region** of confidence-`2` pixels within the annotated
  region. If it covers **at least one-sixth of the region's area**, use **only**
  confidence-`2` coordinates from it.
- **Otherwise**, use coordinates with confidence `1` **and** `2` from it (still excluding
  `0`).

This selection runs per sampled frame (§3); the pooled union across sampled frames is what
the plane is fit to. A camera whose view-prefixed floor-region fields are missing from the
session's `metadata.yaml`, or whose region is degenerate or out of bounds, is **rejected**
rather than guessed at (§7).

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

## 3. Plane fit — one plane per camera

- Sample **N frames** spread across the camera's retained span (parameter, default proposed:
  every ~10th frame, capped at a few hundred points total) rather than every frame — the
  camera is static, so additional frames add redundancy, not new geometric information.
- Run confidence-based selection (§1) and back-projection (§2) on each sampled frame; pool all
  points into one point cloud `{(X_i, Y_i, Z_i)}`.
- Fit the plane with **`torch.linalg.lstsq`** (the "PyTorch linear regression" of the source
  note), regressing `Y = a*X + b*Z + c` over the pooled cloud. This is well-conditioned
  because a floor viewed by a roughly-level, roughly-plumb phone is close to the `X`-`Z`
  plane in camera coordinates, never close to vertical.
- The unnormalized plane normal is `n = (a, -1, b)`; normalize to unit length and orient it so
  it points from the floor toward the camera (`n_y < 0` in this convention — "up" is `-Y`).
- Record the **RMS residual** of the fit (perpendicular distance from each point to the
  plane) and the pooled point count; both are validation inputs (§7) and sidecar fields (§6).

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
  - `homography_rgb`, `homography_depth` — the applied `H` for each resolution;
  - `gravity_agreement_deg` — angle between the fitted normal and the odometry gravity
    vector (§7);
  - `gravity_check_mode` — `"warn"` while the gravity check is unpromoted (§7), the run-time
    counterpart of the same field's `"reject"` value once promoted;
  - `valid_bounds_px` — `[x0, y0, x1, y1]` of the rectified valid-content region;
  - `depth_values_recomputed: true`, `k_rewritten: false` — explicit, machine-checkable
    provenance flags in the same spirit as S1 Orient's `k_rewritten`.

`metadata.yaml` gains no new required fields (unlike Cut) — retilt *reads* the operator's
floor region from it (§1) but does not add anything to it; the fitted plane and rotation are
this camera's own derived values, written to `retilt_sidecar.json` instead.

Manifest: one step per camera, `derived` carrying `tilt_deg`, `roll_deg`,
`gravity_agreement_deg`, `plane_rms_residual_m`; `warnings` for a near-threshold gravity
disagreement that did not quite trigger rejection; `file_ops` following the `write`/`copy`/
`publish` shape used by S1/S2.

## 7. Validation and rejection

Reject the camera (exact reason recorded, per-camera — never per-session; retilting has no
cross-camera dependency) when:

- fewer than `retilt_min_floor_points` (proposed default **500**) floor points are selected
  across all sampled frames — too little confident floor to fit anything;
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
- the governing `metadata.yaml` is missing this capture's role-appropriate floor-region fields,
  fails to parse as `"(x, y)"` (§1), or the parsed region is degenerate (`x0 >= x1` **or**
  `y1 >= y0` — note `y1 < y0` is the *valid* case under §1's top→bottom, bottom-left/
  top-right naming, since `bottom_left.y` is numerically larger) or outside `[0, 1]` bounds.

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

**The odometry-gravity check is warn-only, not a rejection, until calibrated.** The fitted
normal disagreeing with the **odometry gravity vector** by more than
`retilt_gravity_tolerance_deg` (proposed default **5°**) is recorded as
`gravity_agreement_deg` in the sidecar and raised as a manifest warning past tolerance — this is the check
`1-ingestion_orient.md` anticipated reusing "S3 machinery" for, and is intended as the
primary defense against a plane fit that is geometrically clean but simply fit to the wrong
surface. It does not reject on its own yet because the axis-convention chain it depends on
(assumptions, above) cannot be proven correct from this document alone. A follow-up change
promotes it to a rejection once real sessions confirm the computed agreement is
consistently small.

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
- The exact schema key for `lift_start_time_side_in_ms`/`lift_end_time_side_in_ms` — real
  captures now nest them under a new `video:` block, but `2-cut.md`'s current text and Cut's
  shipped `read_lift_window` still expect them directly under `lift:`. This document assumes
  whatever S1 Cut resolves that to; retilt reads the same session `metadata.yaml` file.
  **Still open**, and orthogonal to the floor-region fields this document depends on, which
  live under `video:` unambiguously in both real sessions inspected.
