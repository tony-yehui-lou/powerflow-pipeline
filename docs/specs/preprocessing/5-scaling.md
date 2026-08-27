# View-dependent frame scaling

Status: **draft** | Updated: 2026-08-26

## Purpose

Add a **Scaling** stage after Retilt:

```text
S0 Ingest → S1 Cut → S2 Orient → S3 Retilt → S4 Scaling → S5 Cropping
```

Scale each camera so that the same real-world distance is represented by the same pixel
distance across clips. This establishes invariant **I4**: *"Scale is constant: one pixel means
the same physical distance in every clip."*

**This stage does not produce common pixel dimensions.** That is invariant **I5**, owned by S5
(`6-cropping.md`) — see the invariant table in `1-ingestion_orient.md`. Because the scale
factor is derived per camera from a per-camera reference (barbell length or plate size), the
scaled output width and height differ from camera to camera by design. Do not read "scale to a
common size" anywhere below as a claim about output dimensions — it means only that px_per_mm
becomes common.

**In:** one camera's S3-retilted RGB, depth, confidence, `camera_matrix.csv`, `odometry.csv`,
and `retilt_sidecar.json` (for `valid_bounds_px`). **Out:** the same streams, trimmed to S3's
valid region and then uniformly scaled, with intrinsics rewritten for the new origin and scale,
and a sidecar recording the reference used and the derived factor.

## Step 0 — trim the rectified valid region, before any resampling

S3 Retilt's rotation leaves a border with no source pixel (`valid_bounds_px` in
`retilt_sidecar.json`, `4-retilt.md` §5) and deliberately does not crop it. This stage trims to
that box **first, before any scaling resampling runs**:

- Resampling blends neighbouring pixels. Scaling across the valid/invalid boundary before
  trimming would blend undefined border content into real footage along the edge.
- The scale reference itself (barbell length, plate size) must be measured in a frame that is
  entirely valid content — measuring it across a partially-blank frame is meaningless.

RGB is sliced losslessly to `valid_bounds_px` — no interpolation. Depth and confidence are
sliced to the same rectangle mapped to their own resolution and rounded **inward**, so every
retained depth/confidence pixel lies inside the RGB region — the identical formula S5 Cropping
uses for its own crop (`6-cropping.md` §2, rounding rule at lines 105–112): for RGB bounds
`[x0, y0, x1, y1]` and depth dimensions `(W_d, H_d)`,

```text
x0_d = ceil(x0 * W_d / W)     x1_d = floor(x1 * W_d / W)
y0_d = ceil(y0 * H_d / H)     y1_d = floor(y1 * H_d / H)
```

Reusing this formula rather than restating it independently keeps S4's trim and S5's crop from
disagreeing about how RGB bounds map to depth bounds.

## Procedure

1. Identify whether the video is front-facing or side-facing.
2. Select the scale reference for that view:
   - For a front-facing video, use the barbell length between the plates.
   - For a side-facing video, use the plate size.
3. Derive the scale factor `s` from the measured reference, in the **trimmed** frame from Step
   0.
4. Resample every frame — RGB and depth/confidence alike — by `s`.

Locating the reference (the barbell/plate detection itself) is undesigned and out of scope for
this document.

## Resampling rules

- **RGB:** bilinear or bicubic — a photometric stream tolerates interpolation.
- **Depth and confidence:** **nearest-neighbour only**, the same rule S1 Orient and S3 Retilt
  already use (`4-retilt.md` §5): interpolating depth fabricates values across object edges,
  and interpolating confidence between `0` and `2` invents a `1` the sensor never reported.
  Depth stays `uint16`; confidence stays `⊆ {0, 1, 2}`.
- Scaling relocates depth pixels but must **not** rescale the millimetre depth *values* —
  resizing the image does not change metric distance to the floor or the subject.

## `camera_matrix.csv` rewrite

Trim (Step 0), then scale, applied to intrinsics in that same order. Reuse the pixel-centre
convention already fixed for S3 (`4-retilt.md` §2):

```text
fx' = fx * s                        fy' = fy * s
cx' = ((cx - x0) + 0.5) * s - 0.5   cy' = ((cy - y0) + 0.5) * s - 0.5
```

where `(x0, y0)` is the trimmed region's origin from Step 0. Unlike S3 Retilt — a pure rotation
that leaves intrinsics unchanged — this stage's trim-and-scale genuinely changes the geometry
the matrix describes, so it must set `k_rewritten: true`.

## Output contract

New stage directory, e.g. `../data/s4_scale_output/`, mirroring the existing
`<date>/<session>/<camera>/` layout:

- `rgb.mp4` — trimmed, scaled, re-encoded.
- `depth/`, `confidence/` — trimmed and scaled frame-by-frame, `uint16` / `{0,1,2}`
  respectively, same frame count and filenames as the input.
- `camera_matrix.csv` — trimmed-and-scaled-frame intrinsics.
- `odometry.csv`, `imu.csv` — copied unchanged; they describe physical pose, not image-plane
  geometry.
- `scale_sidecar.json`, containing at minimum:
  - `view: "front" | "side"` and which reference was used;
  - the measured reference value and the derived scale factor `s`;
  - resulting `px_per_mm` (or equivalent);
  - `valid_bounds_px` consumed from S3's sidecar;
  - `rgb_input_size`, `rgb_output_size`, `depth_input_size`, `depth_output_size` — **output
    dimensions vary per camera by design**, not a bug to reconcile here;
  - `k_rewritten: true`.

`metadata.yaml` gains no required fields.

## Non-goals

- **Common pixel dimensions across cameras** — that is I5, owned by S5 Cropping.
- **Motion guard-band cropping** — also S5's job; S5 runs after this stage precisely so that
  its `crop_safety_px` guard means the same physical margin on every camera, once px_per_mm is
  common (see `6-cropping.md` §2).
- Rescaling, reinterpreting, or otherwise modifying depth/confidence *values* — only their
  pixel positions change.
- Detecting the barbell or plate reference itself.
