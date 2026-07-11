# Preprocessing Spec — v1

Status: **draft** | Derived from: `Preprocessing.md` (raw notes) | Updated: 2026-07-11

> Scope: everything that happens between **raw multi-camera capture** and the **tensor handed to the
> CNN**. Nothing about model architecture, training, or inference lives here.
>
> Raw notes are terse, so anything I inferred is marked **(assumption)** and anything I could not
> infer is in **Open questions**. Do not treat assumptions as settled.

---

## 1. Purpose & contract

The CNN sees lifts filmed by several phone/tablet cameras, from different angles, at different
distances, tilts, and start times. It must not have to learn away any of that nuisance variation.

**Preprocessing therefore guarantees the following invariants** on every frame it emits. These are
the acceptance criteria for the whole pipeline — each stage below exists to establish exactly one of
them.

| # | Invariant | Established by |
|---|---|---|
| I1 | Every image is **portrait**-oriented. | Stage 1 |
| I2 | Frame `k` of **every** stream (RGB, depth, odometry, IMU) refers to the **same instant** of the lift, and `t=0` is a common origin. | Stage 2 |
| I3 | Images are **undistorted** and rectified as if the camera were level (no tilt w.r.t. vertical). | Stage 3 |
| I4 | **Scale is constant**: one pixel means the same physical distance in every clip. | Stage 4 |
| I5 | Every image has **identical pixel dimensions** and covers the **same physical region** of the platform. | Stage 5 |

If a stage cannot establish its invariant for a given clip, the clip is **rejected**, not passed
through degraded. A silently mis-scaled or mis-synced clip is worse than a missing one — it teaches
the CNN the wrong thing.

---

## 2. Inputs

Per capture session, per camera (**assumption**: StrayScanner-style capture on iOS, per `Notes.md`):

- `rgb/` — frame sequence, landscape, with per-frame timestamps.
- `depth/` — per-frame depth map (LiDAR), and its own resolution/timestamps.
- `confidence/` — **one confidence map per depth frame**, pixel-aligned with it, valued in
  `{0, 1, 2}`: `0` = unconfident, `1` = medium, `2` = most confident. Depth is **never** consumed
  without its confidence map (see S3 for the gating rule).
- `intrinsics` — camera matrix `K` (`fx, fy, cx, cy`) and distortion coefficients.
- `odometry` — per-frame 6-DoF camera pose (ARKit world frame).
- `imu` — per-frame inertial samples.
- **A stopwatch on an iPad, in view of every camera** — the shared clock used for sync (per `Notes.md`;
  chosen over a $300 Tentacle Sync box).

Capture-side assumptions this pipeline relies on: single lifter, single barbell, cameras **static**
for the duration of a take, ~60 fps, and the barbell plate face visible to at least one camera.

---

## 3. Pipeline

Stages are **strictly ordered** — each depends on the invariants of the previous one. Notably:
undistort (S3) **before** measuring plate size (S4), or the measured scale is wrong; and normalize
scale (S4) **before** cropping to a fixed pixel size (S5), or the crop covers a different physical
region per camera.

```
S0 Ingest  →  S1 Orient  →  S2 Sync  →  S3 Undistort/De-tilt  →  S4 Scale  →  S5 Crop/Register  →  S6 Tensorize
```

---

### S0 · Ingest & validate

**Out:** a normalized in-memory record per camera: frames, depth, confidence, `K`, distortion, poses,
timestamps.

Reject the session if: a stream is missing, RGB / depth / **confidence** frame counts disagree beyond
tolerance, a depth frame has no matching confidence map (or the two differ in resolution),
intrinsics are absent, or the stopwatch is not legible in any camera.

---

### S1 · Orientation normalization

> *Note: "Rotate all images 90 degrees so that they are portrait mode (rather than current landscape mode)."*

**In:** landscape RGB + depth + confidence. **Out:** portrait RGB + depth + confidence, with
**corrected intrinsics**.

1. Rotate every RGB frame by 90°.
2. Rotate the **depth map and its confidence map by the same 90°** — the notes only say "images";
   both must follow or they desynchronize from RGB. The confidence map is a **label map**, so rotate
   it with nearest-neighbour only: interpolating between confidence `0` and `2` would invent a `1`
   that the sensor never reported.
3. <mark>**Rewrite the camera matrix.**</mark> A rotation is not a free operation: for a 90° rotation of a `W×H`
   image, `fx ↔ fy` swap and the principal point becomes `cx' = H - 1 - cy`, `cy' = cx` (sign
   depends on rotation direction — pick one and pin it in code). Skipping this quietly corrupts
   every downstream geometric step (S3, S4, S5).

**Params:** rotation direction (CW vs CCW) — must be **fixed and identical across all cameras**, and
chosen so the lifter is upright, not inverted.

**Validation:** output aspect ratio is portrait (`H > W`); re-projecting a known point with `K'`
lands where it visually is.

---

### S2 · Temporal synchronization & trim

> *Note: "Based on the videos, crop until the millisecond on the iPad is same, and calculate the
> corresponding frames for all other data points (odometry, IMU, confidence, depth, rgb) and delete
> everything before that."*

**In:** per-camera streams with independent, unaligned start times. **Out:** all streams sharing a
common `t=0`; everything before it deleted. → establishes **I2**.

1. In each camera's RGB, **read the iPad stopwatch** to recover the wall-clock time of a frame.
   (**Assumption**: OCR of the stopwatch digits; alternatively a manual/labelled anchor frame.)
2. Pick the sync instant `T₀` = the **latest** stopwatch time that is visible in *all* cameras
   (you can only start where every camera already has data).
3. For each camera, find the frame whose stopwatch reading is nearest `T₀` → this becomes frame 0.
4. Map `T₀` onto **every other stream** — the notes enumerate them: **odometry, IMU, confidence,
   depth, RGB** — via their own timestamps, and drop everything earlier. Every modality is trimmed,
   not just RGB. Depth and confidence are trimmed **identically**, so that frame `k` of depth always
   has frame `k` of confidence.

**Residual error budget:** sub-frame (<1 frame at 60 fps ≈ 16.7 ms). The notes accept ~1 ms of
stopwatch-reading error as insignificant; that holds — **but the dominant error is frame
quantization, not the stopwatch**. Cameras are not genlocked, so two cameras' frame 0 can still be up
to half a frame period apart. Record the per-camera residual offset alongside the clip.

**Why it matters:** the second pull lasts 150–250 ms — only ~15 frames at 60 fps (`Notes.md`). A
one-frame sync error is several percent of the entire phase.

**Validation:** after trimming, an event visible to multiple cameras (bar leaves floor) falls on the
same frame index in all of them.

---

### S3 · Undistortion & tilt rectification

> *Note: "First, determine which depth is useable. We dispose of all depth coordinates that have
> confidence coordinate value 0. Now, if there exists a continuous region of depth values that are
> confidence 2 and the region is at least 1 ninth of the image, calculate directly using those depths
> as suggested below. Else, incorporate confidence 1 points in the calculation. Calculate, using
> depth, for the angle in which the camera is angled at with respect to the vertical. Using the camera
> matrix and focal length with cv2 in python, distort the RGB back to reality."*

**In:** portrait RGB + depth + confidence + `K'`. **Out:** undistorted, level-rectified RGB. →
establishes **I3**.

Three steps, in this order. **Depth is never trusted uniformly** — step 1 decides which depth pixels
are allowed to vote before any geometry is computed from them.

1. **Depth confidence gating.** Produce the *usable depth mask* for the frame:

   a. **Discard confidence `0` unconditionally.** These pixels are dropped from every downstream
      computation and are never reinstated, regardless of what the rest of this step decides.

   b. **Try the high-confidence path.** Find connected regions of confidence-`2` pixels. If any single
      **contiguous** region covers **≥ 1/9 of the image**, the mask is *that region alone* —
      confidence-`1` pixels are **excluded**, even though they survived (a).

   c. **Otherwise, fall back.** If no confidence-`2` region is large enough, admit confidence-`1`
      pixels into the mask alongside the `2`s.

   The threshold is a **contiguity** test, not a pixel count: 1/9 of the image scattered as isolated
   confident pixels does **not** qualify. This is what makes the mask usable for plane fitting — a
   large connected patch of confident depth is overwhelmingly likely to *be* a real planar surface
   (the floor), whereas the same number of scattered points is just noise that happens to be trusted.

   Record which path (b or c) was taken, per frame. A clip that mostly falls back to (c) is a clip
   whose tilt estimate deserves less trust downstream.

2. **Lens undistortion.** Remove radial/tangential distortion with `cv2.undistort` (or
   `initUndistortRectifyMap` + `remap`, which is the right call here since the maps are constant per
   camera and reused across every frame). Apply the **same** maps to depth **and to the confidence
   map**, both with nearest-neighbour interpolation (bilinear on a depth map fabricates depths across
   edges; on a confidence map it fabricates confidence levels).

3. **Tilt estimation and rectification.** Recover the camera's tilt w.r.t. vertical **using only the
   masked depth from step 1**: fit the dominant **ground plane** to the back-projected point cloud
   (RANSAC), take its normal, and measure the angle between the camera's optical axis and that normal.
   Then warp the image by the homography that maps the tilted image plane to a **vertical**
   one, so that "up" in the image is "up" in the world.

   > **(Assumption)** the notes say "distort the RGB back to reality" without saying how far to take
   > it. I read this as: rectify tilt only (a homography), **not** a full novel-view synthesis to a
   > canonical camera pose. Full re-rendering would need dense depth everywhere and would introduce
   > holes. **Cross-check:** ARKit odometry already carries a gravity-aligned world frame — its
   > gravity vector is a free, independent estimate of tilt. Use it to validate the depth-derived
   > angle; large disagreement (>2°, assumption) should reject the clip.

**Why it matters:** an un-corrected tilt turns a *vertical* bar path into a *slanted* one. Bar-path
verticality is a core signal — this stage protects the label, not just the pixels.

**Validation:** a known-vertical reference in scene (squat rack upright, door frame) is vertical in
the output within ~1°.

**Rejection:** the notes stop at the fallback (c), but the fallback can *also* fail — a frame may have
no usable depth at all once the `0`s are gone. If the mask after (c) still cannot support a stable
plane fit (**assumption**: same 1/9 contiguity bar applied to the combined `1`+`2` mask), the frame
has no depth-derived tilt. Fall back to the ARKit gravity vector for that frame, or reject the clip —
**Open question**, but the pipeline must not fit a plane to a handful of scattered points and report
the resulting angle as if it were measured.

---

### S4 · Scale normalization

> *Note: "Based on the size of the barbell circle, scale all videos such that the barbell circle is
> same size."*

**In:** rectified RGB. **Out:** RGB resampled to a **constant pixels-per-metre**. → establishes **I4**.

1. Detect the barbell plate face — a circle (an **ellipse** in any non-frontal view; the notes say
   "circle", which is only true for the camera facing the plate). Use Hough circle / ellipse fit on
   the plate.
2. The plate's physical diameter is **known and standard**: 450 mm for a competition bumper plate.
   That is what makes it a valid ruler.
3. Resample each clip so the plate's **major axis** measures a fixed target in pixels →
   `px_per_mm` is now identical across cameras, distances, and sessions.
4. Persist `px_per_mm` per clip. **This is the constant that converts CNN-space pixels back to
   metres/second** — without it, no velocity output is in real units.

**Design note (assumption):** prefer estimating the scale factor **once per clip** (cameras are
static) from a robust aggregate — e.g. the median plate size over frames — rather than per-frame.
Per-frame rescaling would make the plate constant in size while making everything *else* jitter,
which is exactly the kind of nuisance motion this pipeline exists to remove.

**Fallback:** for a camera that never sees the plate face (e.g. a pure side camera sees the plate
edge-on), derive scale from **depth + intrinsics** instead — `px_per_mm = fx / distance` — or from a
camera whose scale is known plus the odometry-known relative pose. If this fallback is used, the
`distance` it reads **must come from confidence-gated depth** (S3 step 1) — a scale factor derived
from a confidence-`0` depth pixel would silently mis-scale the entire clip, and scale error
propagates straight into reported bar velocity.

**Validation:** the barbell shaft's known length spans the expected pixel count.

---

### S5 · Spatial crop & registration

> *Note: "Trim all images using odometry till the cropped images are the same size and represent same
> region."*

**In:** scale-normalized RGB. **Out:** fixed-size crops covering the **same physical region**. →
establishes **I5**.

1. Use **odometry poses** (each camera's 6-DoF pose in a shared world frame) to determine what
   physical region each camera's image covers.
2. Define a **canonical lift volume** in world coordinates — the region the lift actually occupies
   (**assumption**: the platform's footprint, extended from floor to the bar's overhead lockout,
   plus margin).
3. Project that volume into each camera and crop to it.
4. Resample to a **fixed pixel size** `H×W` shared by all cameras (required: the CNN needs a fixed
   input shape).

**Caveat this stage rests on:** the cameras' odometry must live in a **common world frame** for their
crops to agree. ARKit gives each device its **own** origin. So this stage requires a cross-camera
extrinsic calibration — see Open questions; it is the biggest unstated dependency in the notes.

**Validation:** back-project a corner of the canonical volume in two cameras — they must land on the
same physical point.

---

### S6 · Tensorization

**Out:** the CNN's actual input. Not covered by the notes; recorded here so the contract is complete.
See Open questions for the parts that are still yours to decide (channels, clip length, augmentation).

---

## 4. Cross-cutting requirements

- **Determinism.** Same input → byte-identical output. Seed anything stochastic (RANSAC in S3).
  Training-set reproducibility depends on this.
- **Cache per stage.** S3's undistort maps and S4's scale factor are constant per camera/clip; compute
  once, reuse for every frame. The pipeline is otherwise dominated by redundant per-frame work.
- **Carry the parameters forward.** Every clip ships with a sidecar recording `K'`, tilt angle,
  `px_per_mm`, `T₀`, sync residual, crop bounds, and **which confidence path (S3.1b or S3.1c) each
  frame took**. **Without `px_per_mm` and the crop bounds, a CNN prediction cannot be converted back
  into a real-world measurement.**
- **Reject, don't degrade.** Any stage that fails its validation drops the clip and logs the reason.
- **Depth and confidence follow RGB, together.** Every geometric operation applied to RGB (S1, S3,
  S5) is applied to the depth map **and** its confidence map, both with **nearest-neighbour**
  interpolation. They must stay pixel-aligned with each other at every stage — a confidence map that
  has drifted from its depth map is worse than no confidence map, because it silently certifies the
  wrong pixels.
- **No depth without confidence.** Every consumer of depth in this pipeline (S3 tilt, S4 fallback
  scale, S5 registration, any depth channel in S6) reads through the S3.1 usable-depth mask.
  Confidence-`0` depth is never used anywhere, for anything.

---

## 5. Open questions

- [ ] **Cross-camera extrinsics.** S5 needs all cameras in one world frame; ARKit gives each device
      its own origin. How are they registered — a shared visual marker, a chessboard, or the barbell
      itself (`Notes.md` mentions "align barbell with cameras with markings")? **Blocking for S5.**
- [ ] **Stopwatch reading.** OCR'd automatically, or a human-labelled anchor frame per clip? Digital
      readout, or does it need to be legible at 60 fps in every camera's view?
- [ ] **Camera count/placement.** `Notes.md` reasons toward 4 (front/back/left/right) but notes 2 may
      suffice by symmetry. The pipeline is written camera-agnostic; S5's canonical volume needs the
      final answer.
- [ ] **How far does "distort back to reality" go?** Tilt-rectifying homography only (my reading), or
      full reprojection to a canonical camera pose?
- [ ] **Depth-derived tilt vs. ARKit gravity vector** — which is authoritative when they disagree?
- [ ] **"Continuous region" (S3.1b)** — connected component under 4- or 8-connectivity? 8 is the
      looser test and will admit thin diagonal bridges between patches; 4 is stricter. Pick one and
      pin it, since it decides which clips take the fallback path.
- [ ] **"1 ninth of the image" (S3.1b)** — one ninth of the *full frame* pixel count, or of the
      *valid* (non-zero-confidence) pixels? These diverge badly on a frame where most depth is
      already discarded. **(assumption: full frame)**
- [ ] **Per-frame or per-clip gating?** The rule reads per-frame, but the cameras are static, so the
      floor patch barely moves. Deciding the mask **once per clip** would be cheaper and stop the
      tilt estimate from flickering between the (b) and (c) paths frame to frame — at the cost of
      ignoring genuine per-frame dropouts (e.g. the lifter occluding the floor mid-lift). Which?
- [ ] **When even the fallback has no usable depth** — reject the clip, or fall back to the ARKit
      gravity vector for tilt? (See S3 Rejection.)
- [ ] **Is confidence itself a CNN input?** It is a per-pixel reliability map; if S6 feeds a depth
      channel, the confidence map is the natural mask/4th channel to feed with it.
- [ ] **Scale target.** What is the fixed plate diameter in pixels, and what `H×W` does the CNN take?
      These two together fix the field of view in metres.
- [ ] **Non-competition plates.** The 450 mm ruler assumes bumper plates. Are training clips ever
      filmed with smaller/steel plates? If so, plate diameter cannot be assumed.
- [ ] **S6 entirely:** input channels (RGB? RGB-D? stacked multi-camera?), clip length / temporal
      window, per-channel normalization, and which augmentations are legal *after* a pipeline whose
      whole purpose was removing geometric variation.

---

## 6. Feedback for v2

We do NOT need to rotate the camera matrix, as the rgb is rotated due to our input app's issue, but the camera matrix is directly from phone data