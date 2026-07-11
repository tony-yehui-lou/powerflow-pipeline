# Preprocessing Spec — v2

Status: **draft** | Derived from: `Preprocessing.md` (raw notes) + feedback on `preprocessing-spec-v1.md` | Updated: 2026-07-11

> Scope: everything that happens between **raw multi-camera capture** and the **tensor handed to the
> CNN**. Nothing about model architecture, training, or inference lives here.
>
> Anything inferred rather than stated is marked **(assumption)**; anything undecided is in
> **Open questions**. Do not treat assumptions as settled.

## Change log

- **v2 — S1: the camera matrix is NOT rewritten.** Per feedback: the RGB arrives landscape because of
  a **capture-app bug**, while `K` comes directly from the phone. The 90° rotation therefore *repairs*
  the image back into the frame `K` already describes. Rewriting `K` on top of that would double-apply
  the rotation. v1 had this exactly backwards. See S1, which now states the condition this rests on
  and how to verify it once.
- **v2 — S3 is reordered and fused.** Lens undistortion now precedes tilt rectification (v1 left the
  order implicit and listed gating first), and the two are applied as a **single composed remap**
  rather than two warps. Tilt is estimated on *coordinates*, not on a resampled depth image.
- **v2 — accuracy measures added** to S3 (§3.3.4) and a new rolling-shutter item raised in Open
  questions.
- **v2 — S1 rotates all three image streams.** Confirmed: the capture app's rotation bug affects RGB,
  depth **and** confidence, so all three are rotated together (depth/confidence nearest-neighbour).
  Closes the blocking open question this draft raised.

---

## 1. Purpose & contract

The CNN sees lifts filmed by several phone/tablet cameras, from different angles, at different
distances, tilts, and start times. It must not have to learn away any of that nuisance variation.

**Preprocessing guarantees the following invariants** on every frame it emits. These are the
acceptance criteria for the whole pipeline — each stage exists to establish exactly one of them.

| # | Invariant | Established by |
|---|---|---|
| I1 | Every image is **portrait**-oriented, in the frame its intrinsics describe. | S1 |
| I2 | Frame `k` of **every** stream (RGB, depth, confidence, odometry, IMU) refers to the **same instant**, and `t=0` is a common origin. | S2 |
| I3 | Images are **undistorted** and rectified as if the camera were **level** (no tilt w.r.t. vertical). | S3 |
| I4 | **Scale is constant**: one pixel means the same physical distance in every clip. | S4 |
| I5 | Every image has **identical pixel dimensions** and covers the **same physical region**. | S5 |

If a stage cannot establish its invariant for a clip, the clip is **rejected**, not passed through
degraded. A silently mis-scaled or mis-synced clip is worse than a missing one — it teaches the CNN
the wrong thing.

---

## 2. Inputs

Per capture session, per camera (**assumption**: StrayScanner-style capture on iOS, per `Notes.md`):

- `rgb/` — frame sequence, **stored landscape** (capture-app bug — see S1), with per-frame timestamps.
- `depth/` — per-frame LiDAR depth map, with its own resolution and timestamps. **Also stored
  landscape** — the app's rotation bug affects every image stream, not just RGB.
- `confidence/` — **one confidence map per depth frame**, pixel-aligned with it, valued `{0, 1, 2}`:
  `0` = unconfident, `1` = medium, `2` = most confident. Also stored landscape. Depth is **never**
  consumed without it.
- `intrinsics` — camera matrix `K` (`fx, fy, cx, cy`) **as reported by the phone**, plus distortion
  coefficients **if the capture app exports them** (see Open questions — it may not).
- `odometry` — per-frame 6-DoF camera pose in ARKit's **gravity-aligned** world frame.
- `imu` — per-frame inertial samples.
- **A stopwatch on an iPad, in view of every camera** — the shared clock (per `Notes.md`; chosen over
  a $300 Tentacle Sync box).

Capture-side assumptions: single lifter, single barbell, cameras **static** for the take, ~60 fps,
barbell plate face visible to at least one camera.

---

## 3. Pipeline

Stages are **strictly ordered**; each depends on the invariants of the last.

```
S0 Ingest → S1 Orient → S2 Sync → S3 Undistort+De-tilt → S4 Scale → S5 Crop/Register → S6 Tensorize
```

The three orderings that are **load-bearing**, and why:

1. **Undistort before de-tilt** (inside S3) — see §3.3, this is a correctness requirement, not a
   preference.
2. **S3 before S4** — measuring the plate on a distorted image measures a bent ruler.
3. **S4 before S5** — cropping to a fixed pixel size before scale is normalized means each camera's
   crop covers a different physical region.

---

### S0 · Ingest & validate

**Out:** a normalized record per camera: RGB, depth, confidence, `K`, distortion, poses, timestamps.

Reject the session if a stream is missing, if RGB / depth / confidence frame counts disagree beyond
tolerance, if a depth frame has no matching confidence map (or they differ in resolution), if
intrinsics are absent, or if the stopwatch is illegible in any camera.

---

### S1 · Orientation repair

> *Note: "Rotate all images 90 degrees so that they are portrait mode (rather than current landscape mode)."*
>
> *Feedback (v1 → v2): "We do NOT need to rotate the camera matrix, as the rgb is rotated due to our
> input app's issue, but the camera matrix is directly from phone data."*

**In:** landscape RGB + depth + confidence. **Out:** all three portrait. **`K` is passed through
untouched.** → establishes **I1**.

This stage is a **repair, not a transformation.** The phone filmed in portrait; the capture app wrote
the pixels out landscape. `K` was never landscape — it came from the phone and has always described
the portrait frame. Rotating the pixels back to portrait therefore *restores* agreement between the
image and `K`. **Do not rewrite the camera matrix.** Rewriting it would apply the rotation a second
time, in the opposite direction, and every geometric step downstream (S3, S4, S5) would be silently
wrong — with no visible symptom in the images themselves.

1. Rotate every RGB frame by 90°, so the lifter is upright.
2. **Rotate the depth map and the confidence map by the same 90°.** The app's bug affects *all three*
   streams, so all three are repaired together. Use **nearest-neighbour** for both: interpolating a
   depth map fabricates depths across edges, and interpolating a confidence map between `0` and `2`
   invents a `1` the sensor never reported.
3. **Leave `K` exactly as the phone reported it.**

**Params:** rotation direction (CW vs CCW) — fixed, identical across all cameras and **identical
across all three streams**, and whichever one makes the lifter upright rather than inverted. Rotating
RGB one way and depth the other would leave them mirror-misaligned while both individually looking
correct.

**The condition this rests on, and how to confirm it once.** The rule "rotate pixels, keep `K`" is
correct if and only if `K` is expressed in the **portrait** frame — i.e. the orientation *after* the
repair. That is the stated premise, but the failure mode if it's ever false is *silent*, so verify it
a single time per device rather than trusting it forever:

- **Cheap check:** the principal point `(cx, cy)` should sit near the centre of the **portrait**
  image, and `cx < W_portrait`, `cy < H_portrait`. If `(cx, cy)` only makes sense against the
  landscape dimensions, the premise is false and `K` *does* need the rewrite.
- **Real check (reuses S3 machinery):** back-project the depth map through `K` into 3D and fit the
  floor plane. Compare its normal to **ARKit's gravity vector**. If `K` is being applied in the wrong
  orientation, `fx`/`fy` and `cx`/`cy` are effectively transposed, the point cloud comes out sheared,
  and the floor will not agree with gravity. Agreement to within a degree or two confirms the premise.

**Validation:** all three streams are portrait (`H > W`) and agree pixel-for-pixel — a landmark
visible in RGB sits at the same coordinates in depth and confidence.

---

### S2 · Temporal synchronization & trim

> *Note: "Based on the videos, crop until the millisecond on the iPad is same, and calculate the
> corresponding frames for all other data points (odometry, IMU, confidence, depth, rgb) and delete
> everything before that."*

**In:** streams with independent start times. **Out:** all streams sharing a common `t=0`. → **I2**.

1. In each camera's RGB, **read the iPad stopwatch** to recover a frame's wall-clock time.
   (**Assumption**: OCR of the digits; alternatively a hand-labelled anchor frame.)
2. `T₀` = the **latest** stopwatch time visible in *all* cameras — you can only start where every
   camera already has data.
3. Per camera, the frame whose stopwatch reading is nearest `T₀` becomes frame 0.
4. Map `T₀` onto **every other stream** — *odometry, IMU, confidence, depth, RGB* — via their own
   timestamps, and delete everything earlier. Depth and confidence are trimmed **identically**, so
   frame `k` of depth always keeps frame `k` of confidence.

**Error budget:** the ~1 ms stopwatch-reading error the notes accept is indeed negligible — but it is
**not the dominant term**. The cameras aren't genlocked, so frame quantization leaves two cameras'
frame 0 up to half a frame period (~8 ms at 60 fps) apart. Record the per-camera residual offset.

**Why it matters:** the second pull lasts 150–250 ms — ~15 frames at 60 fps (`Notes.md`). One frame
of sync error is several percent of the entire phase.

**Validation:** an event visible to multiple cameras (bar leaves floor) lands on the same frame index
in all of them.

---

### S3 · Undistortion & tilt rectification

> *Note: "First, determine which depth is useable. We dispose of all depth coordinates that have
> confidence coordinate value 0. Now, if there exists a continuous region of depth values that are
> confidence 2 and the region is at least 1 ninth of the image, calculate directly using those depths
> as suggested below. Else, incorporate confidence 1 points in the calculation. Calculate, using depth,
> for the angle in which the camera is angled at with respect to the vertical. Using the camera matrix
> and focal length with cv2 in python, distort the RGB back to reality."*

**In:** portrait RGB + depth + confidence + `K`. **Out:** undistorted, level-rectified RGB. → **I3**.

**Why it matters:** an uncorrected tilt turns a *vertical* bar path into a *slanted* one. Bar-path
verticality is a core signal — this stage protects the label, not just the pixels.

#### 3.3.1 · Order: undistortion strictly precedes tilt rectification

Image formation applies operations in a fixed physical order: world → perspective projection through
the pinhole → **lens distortion bends the result** → pixels. Distortion is applied *last* by the
physics, so inverting the chain means removing it **first**.

De-tilting first is not merely less accurate, it is **unrecoverable**. Tilt rectification is a
homography, which assumes a *linear projective* camera — which a distorted image is not, so the warp
fits the wrong model. And after that warp, the distortion centre has moved and its radial symmetry is
destroyed, so the standard `k1,k2,p1,p2` model can no longer undo it **at all**. Undistort-then-tilt
has no such problem: the homography then acts on a clean pinhole image, exactly as it assumes.

#### 3.3.2 · The two corrections are applied as ONE remap

Every `remap` resamples, and resampling twice compounds interpolation blur — softening the very plate
edges S4 must measure. Compose them instead. Tilt rectification is nothing more than **rotating the
camera about its own optical centre**, which OpenCV takes as an argument to the undistortion map:

```python
# R    = rotation that levels the camera (from 3.3.3)
# newK = intrinsics of the virtual, level camera
map1, map2 = cv2.initUndistortRectifyMap(K, dist, R, newK, size, cv2.CV_32FC1)
rgb_rect   = cv2.remap(rgb, map1, map2, cv2.INTER_CUBIC)   # ONE resample, both corrections
```

The cameras are static, so the maps are **constant for the whole clip**: build once, reuse per frame.

**The principle that makes this safe, and that bounds what "back to reality" can mean:** a pure
**rotation** of the camera about its optical centre is exact for every pixel *regardless of scene
depth* — it induces no parallax, needs no depth, and leaves no holes. A **translation** of the virtual
camera does not have this property: it would require dense per-pixel depth and would tear holes
wherever the scene was occluded. So tilt correction is free and exact; full viewpoint normalization
(reprojecting all cameras to one canonical pose) is neither, and is **out of scope**. "Distort back to
reality" = **undistort + de-rotate**, nothing more.

#### 3.3.3 · Estimating the tilt

**Step 1 — Confidence-gate the depth.** Produce the *usable depth mask*:

  a. **Discard confidence `0` unconditionally.** Never reinstated, whatever the rest of this step
     decides.
  b. **High-confidence path.** Find connected regions of confidence-`2` pixels. If any single
     **contiguous** region covers **≥ 1/9 of the image**, the mask is *that region alone* —
     confidence-`1` pixels are excluded, even though they survived (a).
  c. **Fallback.** If no confidence-`2` region is large enough, admit confidence-`1` pixels alongside
     the `2`s.

  The threshold is a **contiguity** test, not a pixel count: 1/9 of the image scattered as isolated
  confident pixels does not qualify. That's what makes the mask usable for plane fitting — a large
  *connected* patch of confident depth is overwhelmingly likely to actually **be** a planar surface
  (the floor), whereas the same number of scattered points is just noise that happens to be trusted.
  Record which path each frame took; a clip that mostly falls back to (c) has a tilt estimate that
  deserves less trust downstream.

**Step 2 — Back-project without resampling depth.** Take the **pixel coordinates** of the masked depth
pixels, push them through `cv2.undistortPoints` to get undistorted normalized rays, and multiply by
depth to get 3D points. This undistorts a few thousand *coordinates* exactly, instead of resampling a
256×192 depth image and interpolating fabricated depths across its edges. **Do not warp the depth map
to estimate tilt.**

**Step 3 — Fit the floor plane and recover `R`.** RANSAC for outlier rejection, then refit by
total-least-squares/PCA on the inliers. Weight each point by **confidence** (`2` over `1`) and by
**inverse depth** — LiDAR error grows roughly with the square of range, so distant floor pixels must
not get an equal vote. The plane normal, against the camera's optical axis, gives the tilt; `R` is the
rotation that takes the camera to level.

**Step 4 — Cross-check against gravity.** ARKit's world frame is gravity-aligned *by construction*, so
the odometry pose yields a second, wholly independent tilt estimate. See §3.3.4 — it is arguably the
**better** one.

#### 3.3.4 · Accuracy measures beyond stock `cv2`

Ranked by expected payoff:

1. **Confirm you actually have distortion coefficients.** ARKit reports `K` under a *pinhole* model and
   the capture app may export **no distortion coefficients at all**. If `dist` is all zeros,
   `cv2.undistort` is a **no-op** and this stage is decorative. Check first. If they're missing,
   calibrate each device once with a **ChArUco board** (more robust than a plain chessboard: it works
   from partial views and gives sub-pixel corners); target reprojection error < ~0.3 px. **Better
   parameters beat better warping code.** → Open question.
2. **Prefer ARKit gravity as the *primary* tilt estimate; demote the plane fit to cross-check.**
   Gravity is IMU-fused, typically well under 1°, needs no visible floor, no RANSAC, and no confidence
   gating. Apple's LiDAR is 256×192 and noisy at the centimetre scale, so the plane fit is the
   *shakier* of the two. This also relegates the whole confidence-gating rule to a fallback path
   rather than the critical path — a far more comfortable place for it to sit. → **decision pending**,
   Open questions.
3. **Estimate tilt once per clip, not per frame.** The cameras are static, so tilt is a constant.
   Per-frame estimation only injects noise and makes the rectification flicker. Take a **temporal
   median of the depth maps first** (static camera → the median across frames kills most LiDAR noise),
   then fit one plane.
4. **Plumb-line calibration**, if a calibration board never happens. A gym is full of known-vertical,
   known-straight edges (rack uprights, door frames). The classical plumb-line method recovers
   distortion coefficients from the constraint that straight world lines must be straight in the
   image — no board needed — and its vertical vanishing point gives a third independent tilt estimate.
5. **Interpolation hygiene.** Use `CV_32FC1` maps, not fixed-point `CV_16SC2` (which quantizes to
   1/32 px). `INTER_CUBIC`/`INTER_LANCZOS4` for the RGB warp. And `INTER_AREA` whenever S4
   **downscales** — bilinear downscaling aliases, and aliased plate edges corrupt the very measurement
   S4 exists to make.

#### 3.3.5 · Validation & rejection

**Validation:** a known-vertical reference in scene (rack upright, door frame) is vertical in the
output to within ~1°. Depth-derived tilt and ARKit gravity agree to within ~2° (**assumption**).

**Rejection:** the notes stop at fallback (c), but **(c) can itself fail** — after the `0`s are gone a
frame may have no usable depth at all. If the combined `1`+`2` mask still cannot support a stable
plane fit (**assumption**: the same 1/9 contiguity bar), that frame has no depth-derived tilt: fall
back to gravity, or reject. The pipeline **must not** fit a plane to a handful of scattered points and
report the resulting angle as though it were measured.

---

### S4 · Scale normalization

> *Note: "Based on the size of the barbell circle, scale all videos such that the barbell circle is
> same size."*

**In:** rectified RGB. **Out:** RGB at a **constant pixels-per-metre**. → **I4**.

1. Detect the barbell plate face — a circle head-on, an **ellipse** from any other angle (the notes say
   "circle", which only holds for the camera facing the plate). Hough circle / ellipse fit.
2. Its physical diameter is **known and standard** — 450 mm for a competition bumper plate. That is
   what makes it a valid ruler.
3. Resample so the plate's **major axis** measures a fixed pixel target → `px_per_mm` becomes identical
   across cameras, distances, and sessions.
4. **Persist `px_per_mm`.** This is the constant that converts CNN-space pixels back into metres per
   second. Without it, no velocity output is in real units.

**Once per clip, not per frame (assumption).** The cameras are static, so take a robust aggregate (e.g.
the median plate size across frames). Per-frame rescaling would hold the plate constant while making
*everything else* jitter — precisely the nuisance motion this pipeline exists to remove.

**Fallback:** a camera that never sees the plate face (a side camera sees it edge-on) gets its scale
from **depth + intrinsics** — `px_per_mm = fx / distance` — or from a camera whose scale is known plus
the odometry-known relative pose. That `distance` **must come from confidence-gated depth** (§3.3.3): a
scale factor read off a confidence-`0` pixel would silently mis-scale the whole clip, and scale error
propagates straight into reported bar velocity.

**Validation:** the barbell shaft's known length spans the expected pixel count.

---

### S5 · Spatial crop & registration

> *Note: "Trim all images using odometry till the cropped images are the same size and represent same
> region."*

**In:** scale-normalized RGB. **Out:** fixed-size crops over the same physical region. → **I5**.

1. Use **odometry poses** to determine what physical region each camera covers.
2. Define a **canonical lift volume** in world coordinates (**assumption**: the platform footprint,
   floor to overhead lockout, plus margin).
3. Project that volume into each camera; crop to it.
4. Resample to a **fixed `H×W`** shared by all cameras — the CNN needs a fixed input shape.

**The dependency this rests on:** the cameras' odometry must live in a **common world frame** for their
crops to agree, and ARKit gives each device its **own** origin. This stage therefore requires
cross-camera extrinsic calibration. It remains the **biggest unstated dependency** in the notes.

**Validation:** back-project a corner of the canonical volume in two cameras — it must land on the same
physical point.

---

### S6 · Tensorization

**Out:** the CNN's actual input. Not covered by the notes; listed so the contract is complete. Channels,
clip length, normalization, and legal augmentations are all open (see Open questions).

---

## 4. Cross-cutting requirements

- **Warp once.** Compose every geometric operation into a **single resample** per stream per stage
  (§3.3.2). Each additional `remap` is irreversible blur, and it lands on the plate edges and bar
  outline that everything downstream measures.
- **Compute geometry on coordinates, not on resampled pixels.** Where a quantity can be derived by
  transforming a handful of *points* (`undistortPoints`) instead of warping a whole *image*, do that —
  it is exact, and free.
- **Depth and confidence move together.** Any geometric operation applied to depth is applied
  identically to its confidence map, both **nearest-neighbour**. A confidence map that has drifted from
  its depth map is worse than none, because it silently certifies the wrong pixels.
- **No depth without confidence.** Every consumer of depth (S3 tilt, S4 fallback scale, S5
  registration, any S6 depth channel) reads through the §3.3.3 usable-depth mask. Confidence-`0` depth
  is never used anywhere, for anything.
- **Determinism.** Same input → byte-identical output. Seed RANSAC. Training-set reproducibility
  depends on it.
- **Cache what is constant.** The S3 remap maps and the S4 scale factor are per-camera/per-clip
  constants. Compute once; the pipeline is otherwise dominated by redundant per-frame work.
- **Carry the parameters forward.** Each clip ships a sidecar with `K`, `R`/tilt angle, `newK`,
  `px_per_mm`, `T₀`, per-camera sync residual, crop bounds, and the confidence path each frame took.
  **Without `px_per_mm` and the crop bounds, a CNN prediction cannot be turned back into a real-world
  measurement.**
- **Reject, don't degrade.** Any stage failing its validation drops the clip and logs why.

---

## 5. Open questions

**Blocking:**

- [ ] **Do we have distortion coefficients at all?** (§3.3.4-1) If the app exports none and `dist` is
      zeros, S3's undistortion does **nothing**. Determines whether a ChArUco calibration is needed.
- [ ] **Cross-camera extrinsics.** (S5) ARKit gives each device its own origin. Shared marker,
      chessboard, or the barbell itself (`Notes.md`: "align barbell with cameras with markings")?

**Decisions pending:**

- [ ] **Is ARKit gravity the primary tilt source, with the depth plane fit as cross-check?**
      (§3.3.4-2 recommends yes.) If yes, the confidence-gating rule becomes a fallback path rather than
      the critical path.
- [ ] **Per-frame or per-clip gating/tilt?** Static cameras argue for once-per-clip (§3.3.4-3) — cheaper,
      no flicker — at the cost of ignoring genuine per-frame dropouts (the lifter occluding the floor
      mid-lift).
- [ ] **"Continuous region"** — 4- or 8-connectivity? 8 admits thin diagonal bridges between patches; 4
      is stricter. It decides which clips take the fallback path, so pin it.
- [ ] **"1 ninth of the image"** — of the *full frame*, or of the *valid* (non-zero-confidence) pixels?
      These diverge badly on a frame where most depth was discarded. **(assumption: full frame)**
- [ ] **When even the fallback has no usable depth** — reject the clip, or fall back to gravity?
- [ ] **Scale target & CNN input size** — plate diameter in pixels, and `H×W`. Together these fix the
      field of view in metres.

**Raised in v2:**

- [ ] **Rolling shutter — possibly a larger error source than lens distortion, for *this* application.**
      Phone sensors read row-by-row over ~10–30 ms. The bar moves at 1.8–2.2 m/s through the second pull
      (`Notes.md`), and it moves **vertically** — i.e. *along* the readout direction of a portrait frame.
      Top and bottom of the bar are therefore captured at genuinely different instants, biasing its
      apparent position, and that bias lands directly in the velocity numbers that are the whole point of
      the tool. **No amount of undistortion fixes this.** Measure each device's readout time (film an
      object of known speed), then decide: model it, or accept and document it as a known bias.
- [ ] **Is confidence itself a CNN input?** It's a per-pixel reliability map; if S6 feeds a depth
      channel, the confidence map is its natural companion mask.
- [ ] **Non-competition plates.** The 450 mm ruler assumes bumper plates. Will any clips be filmed with
      steel/smaller plates? If so, plate diameter can't be assumed.
- [ ] **Stopwatch reading** — OCR, or a hand-labelled anchor frame per clip?
- [ ] **Camera count/placement.** `Notes.md` reasons toward 4 (front/back/left/right) but notes 2 may
      suffice by symmetry. S5's canonical volume needs the final answer.
- [ ] **S6 entirely:** channels (RGB? RGB-D? stacked multi-camera?), clip length, per-channel
      normalization, and which augmentations are even legal after a pipeline whose whole purpose was
      removing geometric variation.

---

## 6. Feedback for v3

<!-- Append feedback below; incorporate into preprocessing-spec-v3.md rather than editing above. -->
