# Residual-motion common-region cropping

Status: **draft** | Derived from: `6-cropping.md` (source note) | Updated: 2026-07-15

## Purpose

Add a **Cropping** stage after Scaling:

```text
S0 Ingest → S1 Cut → S2 Orient → S3 Retilt → S4 Scaling → S5 Cropping
```

For each camera independently, remove a fixed conservative border from every retained frame so
the output excludes edge content that could differ because of small residual camera translation.
This is the per-camera operational form of invariant **I5**: *every frame within one camera has
the same stable image region and dimensions.* It does not attempt to make Front and Side share a
field of view, nor does it claim exact 3D temporal registration.

Cropping is deliberately a guard-band operation, not stabilization. It never shifts, warps, or
reprojects pixels. Those operations would need depth-aware reprojection to be geometrically
correct under translation, and are outside this stage's scope.

**In:** one camera's S4-scaled RGB, depth, confidence, `camera_matrix.csv`, and
`odometry.csv`. **Out:** the same streams cropped to one fixed rectangle, with intrinsics updated
for the rectangle's new origin and a sidecar that records the measured motion and crop bounds.

## Assumptions and scope

- The camera is static apart from small residual translation. S3 Retilt already rejects motion
  that violates the static-camera assumption; S5 measures the remaining translation again and
  applies a stricter crop-specific bound.
- S3 owns floor alignment and tilt/roll removal. S5 does not correct per-frame orientation
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
S5 uses the first `n_frames` rows in that established order.

## 2. Conservative pixel guard band

Translation has depth-dependent parallax: a nearby object moves farther in image coordinates than
a distant object for the same camera movement. S5 does not correct that parallax. Instead, it
uses a conservative near-depth guard value, `Z_guard_m`, and removes enough border pixels to
cover the largest predicted image displacement.

`Z_guard_m` is the configurable lower quantile of positive, confidence-nonzero depth values across
the retained camera span. A raw minimum is not used because isolated invalid or noisy values would
make the crop unnecessarily large. The selected quantile and resulting depth are provenance.

For a stream of width `W`, height `H`, and intrinsics `(fx, fy, cx, cy)`, define the farthest
pixel radii from the principal point:

```text
r_x = max(cx, W - 1 - cx)
r_y = max(cy, H - 1 - cy)
```

For every retained odometry displacement, bound the horizontal and vertical pixel movement:

```text
m_x,i = fx * |ΔX_i| / Z_guard_m + r_x * |ΔZ_i| / Z_guard_m
m_y,i = fy * |ΔY_i| / Z_guard_m + r_y * |ΔZ_i| / Z_guard_m
```

The fixed RGB margins are:

```text
left = right = ceil(max_i m_x,i) + crop_safety_px
top  = bottom = ceil(max_i m_y,i) + crop_safety_px
```

`crop_safety_px` is a configurable non-negative guard against odometry and depth-estimation error.
The symmetric margins deliberately favor a simple, auditable rectangle over recovering a few extra
edge pixels. The RGB crop rectangle is the closed-open pixel box:

```text
[left, W - right) × [top, H - bottom)
```

Depth and confidence use the corresponding rectangle scaled from RGB coordinates to their own
resolution and rounded **inward**, so all retained depth/confidence pixels lie inside the RGB
camera region. For RGB bounds `[left, top, right, bottom)` and depth dimensions `(W_d, H_d)`:

```text
left_d   = ceil(left   * W_d / W)     right_d  = floor(right  * W_d / W)
top_d    = ceil(top    * H_d / H)     bottom_d = floor(bottom * H_d / H)
```

Confidence always uses exactly the same depth rectangle.

## 3. Applying the crop

- **RGB:** crop every decoded frame to the fixed RGB rectangle and re-encode it using the
  pipeline's configured CRF. Preserve the zero-based, odometry-derived video timeline established
  by S1 Cut.
- **Depth and confidence:** crop every frame to the fixed depth rectangle. Do not interpolate,
  rescale, or alter values. Depth remains `uint16`; confidence remains in `{0, 1, 2}`.
- **`camera_matrix.csv`:** rewrite the authoritative RGB intrinsics for the new image origin:

  ```text
  fx' = fx        fy' = fy
  cx' = cx - left cy' = cy - top
  ```

  The crop does not change focal lengths, scale, or distortion. The output matrix must describe
  the cropped RGB buffer, so leaving its principal point unchanged would be incorrect.
- **`odometry.csv` and `imu.csv`:** copy unchanged. Their pose and inertial measurements remain
  in physical camera coordinates; `camera_matrix.csv` is the authoritative image intrinsics after
  S2 Orient and this stage.

No image stream is shifted to compensate for translation. All frames use the same rectangle and
therefore retain one stable pixel coordinate system.

## 4. Output contract

Publish to a new S5 root, for example `../data/s5_crop_output/`, using the existing
`<date>/<session>/<camera>/` layout:

- `rgb.mp4` — cropped and re-encoded; same frame count and timeline as input.
- `depth/`, `confidence/` — cropped losslessly; same frame counts and filenames as input.
- `camera_matrix.csv` — cropped-frame intrinsics.
- `odometry.csv`, `imu.csv` — byte-for-byte copies.
- `crop_sidecar.json`, containing:
  - `reference_frame: 0`, `max_translation_m`, and `max_residual_translation_m`;
  - `z_guard_m`, the depth quantile used, and `crop_safety_px`;
  - `rgb_crop_bounds_px: [left, top, right_exclusive, bottom_exclusive]`;
  - `depth_crop_bounds_px` and `confidence_crop_bounds_px` in their native resolution;
  - `rgb_input_size`, `rgb_output_size`, `depth_input_size`, and `depth_output_size`;
  - `camera_translation_corrected: false`, `pixels_shifted: false`, and
    `k_rewritten: true` as explicit provenance flags.

`metadata.yaml` gains no required fields. The run manifest records per-camera crop bounds,
`max_translation_m`, `z_guard_m`, and output dimensions in `derived`; it records the exact
rejection reason for a camera that cannot produce a valid common region.

## 5. Validation and rejection

Reject one camera, never a whole session, when any of the following is true:

- `odometry.csv` lacks enough aligned pose rows to cover the retained image frames;
- a pose cannot be parsed into a finite position and rotation;
- `max_translation_m` exceeds `max_residual_translation_m`;
- no positive confidence-nonzero depth samples are available to derive `Z_guard_m`;
- the computed RGB or depth crop has zero or negative width or height;
- the crop would remove more than the configurable maximum fraction of either image dimension;
- cropped depth or confidence no longer has the input's frame count, matching filenames, native
  dtype, or valid confidence domain; or
- the rewritten principal point falls outside the cropped RGB image.

Before publishing staged output, verify that all RGB/depth/confidence streams preserve their
input frame counts; all RGB frames share the declared cropped dimensions; depth and confidence
share their declared cropped dimensions and filenames; and the output camera matrix has positive
focal lengths with its principal point inside the output image.

## Configuration

The following named parameters are required rather than hardcoded:

- `max_residual_translation_m` — maximum tolerated pose drift from frame 0;
- `depth_guard_quantile` — lower quantile used to derive `Z_guard_m` from valid depth;
- `crop_safety_px` — extra symmetric pixel guard band;
- `max_crop_fraction` — largest permitted cropped fraction of width or height.

Default values require calibration against representative captures before this stage is enabled.

## Non-goals

- Cross-camera cropping or a shared Front/Side rectangle.
- Per-frame shifts, camera-motion stabilization, or optical-flow correction.
- Depth-aware reprojection, occlusion handling, or restoration of parallax-distorted content.
- Rescaling, interpolation, or modification of depth/confidence values.
