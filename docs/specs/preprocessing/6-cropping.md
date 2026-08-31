# Residual-motion common-region cropping

Status: **draft** | Derived from: `6-cropping.md` (source note) | Updated: 2026-08-27

## Purpose

Add a **Crop** stage after Retilt:

```text
S0 Ingest → S1 Cut → S2 Orient → S3 Retilt → S4 Crop
```

For each camera independently, crop every retained frame to one fixed rectangle that is both
(a) free of the warp-invalid border S3 Retilt's rotation leaves behind, and (b) conservative
against small residual camera translation. This is the per-camera operational form of
invariant **I5** as applied *within* one camera: *every frame within one camera has the same
stable image region and dimensions.* It does not attempt to make Front and Side share a field
of view, nor does it claim exact 3D temporal registration — **see the open question below**:
`1-ingestion_orient.md`'s invariant table states I5 as "every image has identical pixel
dimensions," which reads as a cross-clip guarantee that this per-camera crop does not, by
itself, provide.

Cropping is deliberately a guard-band operation, not stabilization. It never shifts, warps, or
reprojects pixels. Those operations would need depth-aware reprojection to be geometrically
correct under translation, and are outside this stage's scope.

**In:** one camera's **S3-retilted** RGB, depth, confidence, `camera_matrix.csv`,
`odometry.csv`, and `retilt_sidecar.json` (for `valid_bounds_px`). Unlike an earlier draft of
this pipeline, there is no intervening scaling stage — S4 is retilt's direct consumer, and it
is the sole owner of removing S3's warp-invalid border; see §2. **Out:** the same streams
cropped to one fixed rectangle, with intrinsics updated for the rectangle's new origin and a
sidecar that records the measured motion, the valid-content bounds, and the crop actually
applied.

## Assumptions and scope

- The camera is static apart from small residual translation. S3 Retilt already rejects motion
  that violates the static-camera assumption; S4 measures the remaining translation again and
  applies a stricter crop-specific bound.
- S3 owns floor alignment and tilt/roll removal. S4 does not correct per-frame orientation
  changes; a capture with material rotational motion is outside this stage's contract.
- The crop is **per camera**. Front and Side have different viewpoints, so intersecting their
  image rectangles would discard useful data without creating a physically meaningful common
  view.
- A fixed rectangle is applied to every frame of one camera. Per-frame crop windows would change
  the image coordinate system over time and would be stabilization, not cropping.
- This stage makes the central region conservative and stable under the static-camera
  approximation. It does **not** guarantee that a pixel represents exactly the same 3D point in
  every frame; exact temporal alignment under translation requires depth-aware reprojection.

## 1. Residual translation from odometry

Use frame `0` as the camera's reference pose. Let `p_0` and `p_i` be its position and the position
at retained frame `i`, expressed in the common ARKit world frame. Express the displacement in the
reference camera frame:

```text
Δp_i = R_0ᵀ (p_i - p_0) = (ΔX_i, ΔY_i, ΔZ_i)
```

where `R_0` is the frame-0 camera-to-world rotation from `odometry.csv`. Record the maximum
translation magnitude:

```text
max_translation_m = max_i ||Δp_i||₂
```

The stage rejects a camera when `max_translation_m` exceeds the configurable
`max_residual_translation_m`. Odometry timestamps and rows are already frame-aligned by S1 Cut;
S4 uses the first `n_frames` rows in that established order.

## 2. The valid-content region from S3

S3 Retilt's rectifying rotation always exposes a border with no source pixel (its own §5
explicitly does not crop it) and records the surviving valid-content bounding box as
`valid_bounds_px` in `retilt_sidecar.json`, in rectified RGB pixel coordinates — the same space
this stage's own margins (§3) are computed in.

This matters more than a minor border: on the four cameras measured in `data/raw/11 July`, the
invalid band was **238–268 rows out of 1920** (12–14% of frame height), and its size **varies
per camera** — 238 / 268 / 241 / 246 rows respectively. A border whose size correlates with
which camera produced it is a camera-identity signal any model trained on these frames could
exploit as a shortcut, independent of the pixels it wastes.

S4 is the single stage responsible for removing this border. Read `valid_bounds_px =
[x0, y0, x1, y1]` from the input camera's `retilt_sidecar.json` before computing anything else
in this document. Reject the camera if `retilt_sidecar.json` is missing or unreadable, or if
`valid_bounds_px` is absent or degenerate (`x0 >= x1` or `y0 >= y1`).

## 3. Conservative pixel guard band

Translation has depth-dependent parallax: a nearby object moves farther in image coordinates than
a distant object for the same camera movement. S4 does not correct that parallax. Instead, it
uses a conservative near-depth guard value, `Z_guard_m`, and removes enough border pixels to
cover the largest predicted image displacement.

`Z_guard_m` is the configurable lower quantile of positive, confidence-nonzero depth values across
the retained camera span. A raw minimum is not used because isolated invalid or noisy values would
make the crop unnecessarily large. The selected quantile and resulting depth are provenance.

`crop_safety_px` (below) is a guard against odometry and depth-estimation error, expressed in
each camera's own native pixels — there is no cross-camera normalized pixel space upstream of
this stage (an earlier draft of this pipeline included a scaling stage for that purpose; it was
retired, see `5-scaling.md`). Its default therefore needs calibration against representative
captures per camera resolution, same as every other tunable in the Configuration section below.

For a stream of width `W`, height `H`, intrinsics `(fx, fy, cx, cy)`, and the valid-content
region `[x0, y0, x1, y1]` from §2, define the farthest pixel radii from the principal point
**within the valid region** — pixels outside it are discarded regardless of translation, so
bounding the radius by the full frame would needlessly inflate the margin:

```text
r_x = max(cx - x0, x1 - 1 - cx)
r_y = max(cy - y0, y1 - 1 - cy)
```

On real captures, measured residual translation spans were 1–5 mm, so the `r · |ΔZ| / Z_guard`
term this feeds is small in practice; restricting `r_x`/`r_y` to the valid region is a precision
refinement, not a load-bearing correction.

For every retained odometry displacement, bound the horizontal and vertical pixel movement:

```text
m_x,i = fx * |ΔX_i| / Z_guard_m + r_x * |ΔZ_i| / Z_guard_m
m_y,i = fy * |ΔY_i| / Z_guard_m + r_y * |ΔZ_i| / Z_guard_m
```

The motion-only margins are:

```text
left = right = ceil(max_i m_x,i) + crop_safety_px
top  = bottom = ceil(max_i m_y,i) + crop_safety_px
```

giving the motion guard-band rectangle `[left, W - right) × [top, H - bottom)`. The symmetric
margins deliberately favor a simple, auditable rectangle over recovering a few extra edge
pixels.

**The final RGB crop rectangle is the intersection of the motion guard band and the §2
valid-content region:**

```text
crop_x0 = max(x0, left)          crop_x1 = min(x1, W - right)
crop_y0 = max(y0, top)           crop_y1 = min(y1, H - bottom)
```

Depth and confidence use the corresponding rectangle scaled from this **intersected** RGB box
to their own resolution and rounded **inward**, so all retained depth/confidence pixels lie
inside the RGB camera region. For RGB bounds `[crop_x0, crop_y0, crop_x1, crop_y1)` and depth
dimensions `(W_d, H_d)`:

```text
crop_x0_d = ceil(crop_x0 * W_d / W)     crop_x1_d = floor(crop_x1 * W_d / W)
crop_y0_d = ceil(crop_y0 * H_d / H)     crop_y1_d = floor(crop_y1 * H_d / H)
```

Confidence always uses exactly the same depth rectangle.

## 4. Applying the crop

- **RGB:** crop every decoded frame to the final intersected RGB rectangle and re-encode it
  using the pipeline's configured CRF. Preserve the zero-based, odometry-derived video timeline
  established by S1 Cut.
- **Depth and confidence:** crop every frame to the corresponding depth rectangle. Do not
  interpolate, rescale, or alter values. Depth remains `uint16`; confidence remains in
  `{0, 1, 2}`.
- **`camera_matrix.csv`:** rewrite the authoritative RGB intrinsics for the new image origin —
  one combined shift, since this is a pure crop with no resampling:

  ```text
  fx' = fx              fy' = fy
  cx' = cx - crop_x0    cy' = cy - crop_y0
  ```

  The crop does not change focal lengths, scale, or distortion. The output matrix must describe
  the cropped RGB buffer, so leaving its principal point unchanged would be incorrect.
- **`odometry.csv` and `imu.csv`:** copy unchanged. Their pose and inertial measurements remain
  in physical camera coordinates; `camera_matrix.csv` is the authoritative image intrinsics after
  S2 Orient and this stage.

No image stream is shifted to compensate for translation. All frames use the same rectangle and
therefore retain one stable pixel coordinate system.

## 5. Output contract

Publish to a new S4 root, for example `../data/s4_crop_output/`, using the existing
`<date>/<session>/<camera>/` layout:

- `rgb.mp4` — cropped and re-encoded; same frame count and timeline as input.
- `depth/`, `confidence/` — cropped losslessly; same frame counts and filenames as input.
- `camera_matrix.csv` — cropped-frame intrinsics.
- `odometry.csv`, `imu.csv` — byte-for-byte copies.
- `crop_sidecar.json`, containing:
  - `reference_frame: 0`, `max_translation_m`, and `max_residual_translation_m`;
  - `z_guard_m`, the depth quantile used, and `crop_safety_px`;
  - `valid_bounds_px: [x0, y0, x1, y1]` — the S3 valid-content region consumed from
    `retilt_sidecar.json` (§2);
  - `motion_bounds_px: [left, top, right_exclusive, bottom_exclusive]` — the motion-only guard
    band before intersecting with `valid_bounds_px` (§3);
  - `crop_bounds_px: [crop_x0, crop_y0, crop_x1, crop_y1]` — the rectangle actually applied,
    i.e. the intersection of the two above;
  - `bound_source: {left, top, right, bottom}`, each `"valid"` or `"motion"` — which of the two
    input rectangles determined that edge of `crop_bounds_px`, so a large or unexpected crop is
    explainable from the sidecar alone;
  - `depth_crop_bounds_px` and `confidence_crop_bounds_px` in their native resolution;
  - `rgb_input_size`, `rgb_output_size`, `depth_input_size`, and `depth_output_size`;
  - `camera_translation_corrected: false`, `pixels_shifted: false`, and
    `k_rewritten: true` as explicit provenance flags.

`metadata.yaml` gains no required fields. The run manifest records per-camera crop bounds,
`max_translation_m`, `z_guard_m`, and output dimensions in `derived`; it records the exact
rejection reason for a camera that cannot produce a valid common region.

## 6. Validation and rejection

Reject one camera, never a whole session, when any of the following is true:

- `retilt_sidecar.json` is missing, unreadable, or its `valid_bounds_px` is absent or
  degenerate (§2);
- `odometry.csv` lacks enough aligned pose rows to cover the retained image frames;
- a pose cannot be parsed into a finite position and rotation;
- `max_translation_m` exceeds `max_residual_translation_m`;
- no positive confidence-nonzero depth samples are available to derive `Z_guard_m`;
- the intersection of the valid-content region and the motion guard band (§3) has zero or
  negative width or height, for either the computed RGB or depth crop;
- the crop would remove more than the configurable maximum fraction of either image dimension;
- cropped depth or confidence no longer has the input's frame count, matching filenames, native
  dtype, or valid confidence domain; or
- the rewritten principal point falls outside the cropped RGB image.

Before publishing staged output, verify that all RGB/depth/confidence streams preserve their
input frame counts; all RGB frames share the declared cropped dimensions; depth and confidence
share their declared cropped dimensions and filenames; the published `crop_bounds_px` lies
entirely inside `valid_bounds_px`; and the output camera matrix has positive focal lengths with
its principal point inside the output image.

## Configuration

The following named parameters are required rather than hardcoded:

- `max_residual_translation_m` — maximum tolerated pose drift from frame 0;
- `depth_guard_quantile` — lower quantile used to derive `Z_guard_m` from valid depth;
- `crop_safety_px` — extra symmetric pixel guard band, in each camera's own native pixels (no
  cross-camera normalized pixel space exists upstream of this stage — see §3);
- `max_crop_fraction` — largest permitted cropped fraction of width or height.

Default values require calibration against representative captures before this stage is enabled.

## Non-goals

- Cross-camera cropping or a shared Front/Side rectangle.
- Per-frame shifts, camera-motion stabilization, or optical-flow correction.
- Depth-aware reprojection, occlusion handling, or restoration of parallax-distorted content.
- Rescaling, interpolation, or modification of depth/confidence values.

## Open questions

- **I5's scope: per-camera or cross-clip.** `1-ingestion_orient.md`'s invariant table states
  I5 as *"every image has identical pixel dimensions and covers the same physical region"* with
  no per-camera qualifier — read plainly, that is a promise across the whole session (Front and
  Side, every clip). This document's Purpose (above) and its Non-goals both explicitly scope
  I5 down to *within one camera*, and rule out a shared Front/Side rectangle as out of scope.

  **Resolved (implemented per-camera, exactly as §3 specifies.)** Verified against the four
  real cameras in `data/raw/11 July` (S4 run 2026-08-30, against the existing S3 output): each
  camera keeps its own rectangle and its own output size —

  | camera | `crop_bounds_px` | RGB output size |
  |---|---|---|
  | 30kg_Set1/Front | `[10, 238, 1430, 1910]` | 1420×1672 |
  | 30kg_Set1/Side | `[18, 268, 1422, 1908]` | 1404×1640 |
  | 50kg_Set3/Front | `[10, 241, 1430, 1909]` | 1420×1668 |
  | 50kg_Set3/Side | `[21, 246, 1419, 1908]` | 1398×1662 |

  Four different output sizes, so `1-ingestion_orient.md`'s invariant table wording ("every
  image has identical pixel dimensions") is *not* satisfied by S4 alone — that would require a
  separate cross-camera reconciliation step, applied to the two cameras' `crop_bounds_px`,
  which does not exist in any spec. This was a deliberate scope decision (Non-goals, above),
  not an oversight; `4-retilt.md`'s matching open question about `valid_bounds_px` should be
  updated to point here rather than carrying its own separate resolution.
