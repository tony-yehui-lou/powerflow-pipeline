# Floor-plane retilting

Status: **draft** | Derived from: `4-retilt.md` (source note) | Updated: 2026-07-15

## Purpose

Add a **Retilt** stage after Orient:

```text
S0 Ingest → S1 Cut → S2 Orient → S3 Retilt → S4 Scaling
```

Rotate every frame so the camera's optical axis is perpendicular to the floor and its image
plane vertical — as if the phone had been level and plumb at capture time. This establishes
invariant **I3**: *"Images are undistorted and rectified as if the camera were level (no tilt
w.r.t. vertical)."* Scale (I4) and common framing (I5) are downstream concerns for S4/S5; this
stage only removes tilt and roll, never yaw, and never touches scale.

**In:** portrait RGB + depth + confidence + `camera_matrix.csv` + `odometry.csv` — Orient's
output, already time-aligned and pixel-aligned across streams. **Out:** the same streams,
each pixel relocated (and, for depth, revalued) by the same rectifying rotation, plus
`camera_matrix.csv` unchanged and a sidecar recording the fitted plane and applied rotation.

If a camera's floor cannot be fit reliably, the camera is **rejected**, never passed through
un-retilted or retilted from a bad fit — a silently mistilted clip corrupts every downstream
geometric assumption (S4 scale, S5 common framing) with no visible symptom in the frame
itself.

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

## 1. Confidence-based floor-point selection

The selection ROI is **view-dependent**: a front-facing shot has the barbell plates at the
horizontal extremes, and a side-facing shot has the lifter's body dominating its left half.
The camera's view is not inferred — it is the capture's own `Front`/`Side` directory name
(`record.camera`), carried through S0–S2 unchanged.

In bottom-left-origin normalized coordinates (`x` left→right, `y=0` at the bottom of the
portrait frame, increasing upward), the ROI rectangle `(x0, y0)`–`(x1, y1)` is:

| View | Rectangle | Meaning |
|---|---|---|
| Front | `(1/6, 0)` – `(5/6, 1/4)` | middle two-thirds of the width, bottom quarter of the height |
| Side | `(1/2, 0)` – `(1, 1/3)` | right half of the width, bottom third of the height |

Converted to pixel row/column ranges for a `width x height` portrait image (0-indexed rows,
top→bottom):

```text
Front: col in [round(width/6), round(width*5/6)), row in [height - round(height/4), height)
Side:  col in [round(width/2), width),             row in [height - round(height/3), height)
```

Within that view-specific ROI, restated from the source note with the threshold denominator
the owner has fixed at **one-sixth**:

- Never use depth coordinates with confidence `0`.
- Find the **largest 4-connected region** of confidence-`2` pixels within the ROI. If it
  covers **at least one-sixth of the ROI's area**, use **only** confidence-`2` coordinates
  from that ROI.
- **Otherwise**, use coordinates with confidence `1` **and** `2` from the ROI (still
  excluding `0`).

This selection runs per sampled frame (§3); the pooled union across sampled frames is what
the plane is fit to. A camera whose name is neither `Front` nor `Side` has no defined ROI
and is **rejected** rather than guessed at (§7).

## 2. Back-projection to 3D

For each selected pixel `(u, v)` with stored depth `d_mm`, using the **depth-resolution**
intrinsics `K_d` (the portrait `camera_matrix.csv` scaled by `depth_width / rgb_width` and
`depth_height / rgb_height` — depth and RGB share a portrait frame but not a resolution):

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

R = R_x(tilt) @ R_z(roll)
```

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
  rectified pixel coordinates) in the sidecar. Cropping to a common region across cameras is
  S5's job (`5-scaling.md` already scales to "the common output size"); duplicating that
  logic here would let the two stages disagree about what "valid" means.

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
  - `valid_bounds_px` — `[x0, y0, x1, y1]` of the rectified valid-content region;
  - `depth_values_recomputed: true`, `k_rewritten: false` — explicit, machine-checkable
    provenance flags in the same spirit as S1 Orient's `k_rewritten`.

`metadata.yaml` gains no new required fields (unlike Cut) — retilting is derived entirely
from this camera's own streams, nothing cross-camera or operator-declared.

Manifest: one step per camera, `derived` carrying `tilt_deg`, `roll_deg`,
`gravity_agreement_deg`, `plane_rms_residual_m`; `warnings` for a near-threshold gravity
disagreement that did not quite trigger rejection; `file_ops` following the `write`/`copy`/
`publish` shape used by S1/S2.

## 7. Validation and rejection

Reject the camera (exact reason recorded, per-camera — never per-session; retilting has no
cross-camera dependency) when:

- fewer than a minimum number of floor points are selected across all sampled frames (too
  little confident floor to fit anything);
- the plane fit's RMS residual exceeds tolerance — the selected region is not actually
  planar, so it is probably not the floor (e.g. selection caught a foot, a plate, clutter);
- the fitted normal disagrees with the **odometry gravity vector** by more than a few degrees
  — the real check `1-ingestion_orient.md` anticipated reusing "S3 machinery" for. This is
  the primary defense against a plane fit that is geometrically clean but simply fit to the
  wrong surface;
- the derived `tilt` or `roll` exceeds an implausible-for-a-handheld/tripod-phone bound (a
  sign the fit converged on a degenerate or wrong plane, not evidence of a real 90°-tilted
  phone);
- odometry shows the camera translated beyond tolerance across the sampled frames — this
  breaks the static-camera, one-plane-per-camera assumption the whole stage rests on;
- the camera's name is neither `Front` nor `Side`, so its selection ROI (§1) is undefined.

**Exit validation**, on the staged result before publish, mirrors S1 Orient's
`_validate_exit`: depth remains `uint16`, confidence values remain within `{0, 1, 2}` (never
invented by resampling), frame counts and filenames are unchanged from the input, and RGB/
depth output dimensions match their inputs (rectification does not resize).

## Open questions

- Concrete default values for: frame-sampling stride/count for the pooled fit, RMS-residual
  tolerance, gravity-agreement tolerance (degrees), plausible tilt/roll bounds, and the
  camera-translation tolerance. Proposed as named, tunable parameters rather than hardcoded.
- Whether the valid-content bounding box recorded here should additionally be **intersected**
  across a session's cameras before S4/S5, or whether each camera's box is purely informative
  and S5 does its own common-region derivation.
