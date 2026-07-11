# Preprocessing Implementation Manual — v1

Companion to `preprocessing-spec-v2.md`. The spec says **what must be true**; this manual says
**what to do**. Where they disagree, the spec wins and this manual is a bug.

Audience: an AI agent (or a human) implementing the pipeline end to end.

---

## 0 · Rules of engagement

Read these before writing a line of code. They are what keep a plausible-looking pipeline from
silently producing garbage.

1. **Preflight before pipeline.** §2 resolves four unknowns that change what the code *is*, not just
   how it's tuned. Do not write stage code before §2 is done. Two of them (P1, P2) can delete an
   entire stage.
2. **Every stage has a validation gate.** A stage is not "done" when it runs without exceptions. It is
   done when its gate passes on real data. Gates are numbered `G1`–`G5` and are not optional.
3. **Reject, never degrade.** On gate failure: log the reason, mark the clip rejected, move on. Never
   emit a clip with a fallback value you invented so the batch would finish.
4. **You cannot eyeball images.** This is the single biggest hazard for an agent implementing computer
   vision. A tilted, mis-scaled, or half-mirrored image looks *completely fine* as an array of numbers.
   So: **every geometric step is verified against synthetic data with known ground truth** (§8). Build
   the synthetic fixtures *first*. Do not trust a geometric transform you have only run on real data.
5. **Sign and convention errors are the default outcome**, not an edge case. ARKit and OpenCV disagree
   about which way `y` points (§1.2), and `cv2`'s rectification `R` is easy to invert. Do not reason
   these out and move on — pin each one with a unit test that fails if it's backwards.
6. **Log every derived parameter.** Every clip emits a sidecar (§7). A number that isn't in the sidecar
   didn't happen.

---

## 1 · Conventions

Fix these once. Most pipeline bugs are two components disagreeing about one of them.

### 1.1 Arrays, dtypes, units

| Stream | dtype | Shape | Units | Notes |
|---|---|---|---|---|
| `rgb` | `uint8` | `(H, W, 3)` | — | RGB order, not BGR. `cv2.imread` gives BGR — convert on load, once. |
| `depth` | `float32` | `(Hd, Wd)` | **metres** | Native ARKit LiDAR ≈ 256×192. `0` / `NaN` = no return. |
| `confidence` | `uint8` | `(Hd, Wd)` | `{0,1,2}` | Same shape as depth, always. |
| `K` | `float64` | `(3,3)` | pixels | Pinhole. See §1.3 — **depth needs its own `K`**. |
| `pose` | `float64` | `(4,4)` | metres | Camera→world, ARKit convention. See §1.2. |

All angles in **radians** internally; degrees only in logs and gates.

### 1.2 Coordinate frames — the ARKit/OpenCV trap

**ARKit camera:** `x` right, `y` **up**, `z` **backward** (right-handed, looks down `−z`).
**OpenCV camera:** `x` right, `y` **down**, `z` **forward** (looks down `+z`).

They differ by a 180° rotation about `x`. Every ARKit pose must be converted before it touches any
`cv2` call:

```python
FLIP = np.diag([1.0, -1.0, -1.0])          # ARKit cam → OpenCV cam

def arkit_pose_to_cv(T_wc_arkit):          # (4,4) camera→world, ARKit
    R = T_wc_arkit[:3, :3] @ FLIP          # world→? no: see test below
    t = T_wc_arkit[:3, 3]
    return R, t                            # camera→world, OpenCV cam axes
```

**ARKit world:** gravity-aligned, `+y` is **up**. So `up_world = (0, 1, 0)` — this is what makes the
gravity cross-check in §4 free.

> **Do not trust the snippet above on inspection.** Pin it with the test in §8.2: take a point known to
> be *above* the camera in the world, transform it into the camera frame, and assert its OpenCV `y` is
> **negative** (up = −y in OpenCV). If it's positive, your flip is on the wrong side of the multiply.

### 1.3 Depth has its own camera matrix

The depth map is ~256×192 while RGB is ~1920×1440. ARKit's depth is registered to the RGB camera's
optical frame, but it is **not the same pixel grid**, so it does not share `K`:

```python
def scale_K(K, from_size, to_size):        # (W,H) tuples
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    Kd = K.copy()
    Kd[0, 0] *= sx;  Kd[0, 2] *= sx        # fx, cx
    Kd[1, 1] *= sy;  Kd[1, 2] *= sy        # fy, cy
    return Kd
```

Back-projecting depth with the **RGB** `K` is a silent, catastrophic error: the point cloud comes out
sheared, the floor is not planar, and your tilt estimate is confidently wrong. Use `K_depth`.

### 1.4 Naming

Frames are `(clip_id, camera_id, frame_idx)`. `frame_idx` is **post-sync** (S2), i.e. index 0 is `T₀`.
Never let a raw file index leak downstream — that's how off-by-one sync bugs survive.

---

## 2 · Preflight: resolve these before writing stage code

Each is a small, self-contained script. Each **changes the pipeline**, not just its parameters.

### P1 · Is there any lens distortion at all? *(may delete S3's undistortion)*

The distortion coefficients are dimensionless and describe **straightness**, not size — your known bar
length and plate diameter are irrelevant here (they calibrate *scale*, in S4). Under a pinhole camera a
straight world line is *always* straight in the image, so any curvature **is** distortion, and you can
measure it directly.

```
1. Pick 3–5 frames containing a long straight edge that runs NEAR THE FRAME BORDER
   (rack upright, door frame, floor-wall junction, platform edge).
   The centre of the image tells you nothing — distortion grows with r².
2. Sample points along each edge (Canny + fit, or hand-click; hand-clicking 20 points is fine).
3. Fit a line through the two endpoints.
4. Record max perpendicular deviation of the sampled points from that line, in pixels.
```

| Result | Action |
|---|---|
| **< 1 px** | **No meaningful distortion.** Set `dist = zeros`, skip undistortion, keep only the tilt rotation in S3. Record the measurement in the sidecar and move on. |
| **1–3 px** | Marginal. Calibrate via P1b; the correction is small but cheap. |
| **> 3 px** | Real. Calibrate via P1b or P1c before doing anything else. |

**Check first whether the capture app even exports `dist`.** If it exports nothing, or all zeros, then
`cv2.undistort` is a **no-op** and any "undistortion" in your pipeline is decorative. ARKit reports `K`
under a pinhole model and may not give you coefficients at all.

#### P1b · Plumb-line calibration (no board, no new footage)

**Your barbell is a calibration target** — not because you know its length, but because you know it is
**straight**. Across a lift it sweeps through many positions and orientations, including out toward the
frame edges where distortion actually bites. Harvest it from footage you already have, plus the gym's
fixed straight edges.

```python
from scipy.optimize import least_squares

# lines: list of arrays, each (N_i, 2) of sampled pixel points along ONE straight world edge.
# Sample MANY edges, spread across the frame, weighted toward the corners.

def residuals(params, lines, K):
    dist = np.array([params[0], params[1], 0., 0., 0.])   # k1, k2 only
    out = []
    for pts in lines:
        und = cv2.undistortPoints(pts.reshape(-1,1,2).astype(np.float64), K, dist, P=K)
        und = und.reshape(-1, 2)
        # perpendicular residual of each point from the best-fit line through this edge
        c = und.mean(0)
        u, s, vt = np.linalg.svd(und - c)
        normal = vt[1]                                     # minor axis = line normal
        out.append((und - c) @ normal)
    return np.concatenate(out)

fit = least_squares(residuals, x0=[0.0, 0.0], args=(lines, K))
k1, k2 = fit.x
```

Fit **`k1` alone first**; add `k2` only if the residual demands it. Fitting more coefficients than the
data supports produces a model that confidently "corrects" noise. `K` stays **fixed** at the phone's
value throughout — you are solving for distortion only.

#### P1c · ChArUco calibration (the rigorous option, ~10 min/device)

Print a ChArUco board, film ~20 views per device, `cv2.aruco` + `cv2.calibrateCamera` with
`CALIB_FIX_FOCAL_LENGTH | CALIB_FIX_PRINCIPAL_POINT` so you solve for `dist` only. Target reprojection
error < 0.3 px.

> **The classic silent failure:** calibrate at the *exact* resolution, capture format, and **lens**
> (wide vs ultrawide) that StrayScanner records with. Intrinsics do not transfer across any of those.
> A calibration from a different capture mode will look perfectly valid and be perfectly wrong.

### P2 · Is `K` really in the portrait frame? *(validates S1)*

The spec's rule — *rotate the pixels, leave `K` alone* — is correct **iff** `K` describes the
post-rotation portrait frame. If that's ever false, every geometric stage is wrong with **no visible
symptom**. Confirm once per device:

```
Cheap:  assert cx < W_portrait and cy < H_portrait, and (cx, cy) sits near the portrait centre.
        If (cx, cy) only makes sense against the LANDSCAPE dimensions, K needs rotating after all.

Real:   back-project the depth map through K_depth into 3D, fit the floor plane (§4.3),
        and compare its normal to the ARKit gravity vector.
        Wrong orientation  ⇒  fx/fy and cx/cy are effectively transposed
                           ⇒  the cloud is sheared, and floor ≠ gravity by a lot.
        Agreement within ~2° confirms the premise.
```

### P3 · Cross-camera extrinsics *(blocks S5 entirely)*

ARKit gives **each device its own world origin**. S5 cannot crop cameras to a common physical region
until they share a frame. Decide and implement the registration method (shared ArUco marker on the
platform is the least-effort option: each camera sees it, `solvePnP` gives each camera's pose relative
to the marker, and the marker becomes the shared origin). **Until this exists, S5 is unimplementable —
build S0–S4, and stub S5.**

### P4 · Rolling shutter *(quantify; may not be fixable, must be known)*

Phone sensors expose row by row over ~10–30 ms. The bar moves at 1.8–2.2 m/s through the second pull,
**vertically** — i.e. *along* the readout direction of a portrait frame. Top and bottom of the bar are
therefore captured at different instants, biasing its apparent position, and that bias lands directly in
the velocity numbers the whole tool exists to produce. **No amount of undistortion fixes this.**

Measure the readout time (film an object of known speed crossing the frame; the skew gives you the
row delay), then decide: model it, or document it as a known bias. Either is acceptable. Not knowing it
is not.

---

## 3 · S1 · Orientation repair

**Goal (I1):** RGB, depth, and confidence all portrait, all mutually aligned, `K` untouched.

```python
ROT = cv2.ROTATE_90_CLOCKWISE          # pick ONE; identical for every camera and every stream

rgb  = cv2.rotate(rgb,  ROT)                                   # any interpolation is fine (no resample)
dep  = cv2.rotate(dep,  ROT)                                   # cv2.rotate is a transpose+flip:
conf = cv2.rotate(conf, ROT)                                   # exact, no interpolation. Good.
# K is NOT touched.
```

`cv2.rotate` by 90° is a **transpose + flip** — a pure memory reindex, not a resample. There is no
interpolation and therefore no interpolation error. (If you ever implement the rotation with
`warpAffine` instead, you *must* force `INTER_NEAREST` on depth and confidence: interpolating depth
fabricates surfaces across object edges, and interpolating confidence invents a `1` between a `0` and a
`2` — a confidence value the sensor never reported, sitting on a depth value you would then trust.)

**Rotation direction must be identical across all three streams.** Rotating RGB one way and depth the
other leaves them **mirror-misaligned while each looks individually correct**. You will not see this. It
will surface as inexplicable noise in the tilt estimate three stages later.

> **`G1` — Gate.** (a) All three streams are portrait, `H > W`. (b) A landmark localized in RGB lands on
> the same pixel in depth (up to the §1.3 resolution scaling). Verify by rendering a synthetic marker
> into all three streams (§8.1) and asserting coordinate agreement — **not** by looking at them.

---

## 4 · S3 · Undistortion + tilt rectification

**Goal (I3):** undistorted, level-rectified RGB, produced by **one** resample.

*(S2, temporal sync, is stage-ordered before this but is procedurally independent; see §6.)*

### 4.1 Order is a correctness requirement

Physics applies: world → pinhole projection → **lens bends it** → pixels. Distortion is applied *last*,
so it is removed **first**. De-tilting first is not merely worse — it is **unrecoverable**: the
homography assumes a linear projective camera (which a distorted image is not), and after that warp the
distortion centre has moved and its radial symmetry is destroyed, so the `k1,k2,p1,p2` model can no
longer undo it *at all*.

### 4.2 Confidence-gate the depth

```python
def usable_depth_mask(depth, conf, connectivity=8):
    valid = (conf > 0) & np.isfinite(depth) & (depth > 0)      # (a) conf 0 dropped, forever

    high = (conf == 2) & valid
    n, labels, stats, _ = cv2.connectedComponentsWithStats(high.astype(np.uint8), connectivity)
    thresh = depth.size / 9.0                                  # 1/9 of the FULL depth frame
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= thresh]

    if big:                                                    # (b) high-confidence path
        mask = np.isin(labels, big)
        path = "b"
    else:                                                      # (c) fallback: admit conf==1
        mask = valid
        path = "c"
    return mask, path
```

The 1/9 test is a **contiguity** test, not a pixel count: 1/9 of the image scattered as isolated
confident pixels does **not** qualify. That is precisely what makes the mask usable for plane fitting —
a large *connected* patch of confident depth is overwhelmingly likely to actually **be** a planar
surface (the floor); the same number of scattered points is just noise that happens to be trusted.

Record `path` per frame. A clip that mostly runs (c) has a tilt estimate that deserves less trust.

### 4.3 Estimate the tilt — as a single `up_cam` vector

**Key simplification:** every tilt estimator, whether from gravity or from the floor, produces the same
thing — **the world's up direction expressed in camera coordinates**. Write one function that consumes
`up_cam`, and the estimators become interchangeable.

**Estimator A — ARKit gravity (recommended primary).** Free, IMU-fused, typically well under 1°,
needs no visible floor, no RANSAC, no confidence gating:

```python
def up_cam_from_gravity(R_wc_cv):                 # camera→world rotation, OpenCV axes
    return R_wc_cv.T @ np.array([0., 1., 0.])     # ARKit world: +y is up
```

**Estimator B — LiDAR floor plane (cross-check).** Apple's LiDAR is 256×192 and noisy at the centimetre
scale, so this is the *shakier* of the two — which is exactly why it belongs in the cross-check role:

```python
def up_cam_from_floor(depth, conf, K_depth, seed=0):
    mask, path = usable_depth_mask(depth, conf)
    v, u = np.nonzero(mask)
    pts_px = np.stack([u, v], -1).astype(np.float64)

    # Back-project COORDINATES, never a resampled depth image:
    rays = cv2.undistortPoints(pts_px.reshape(-1,1,2), K_depth, DIST).reshape(-1, 2)
    z = depth[v, u]
    P = np.stack([rays[:,0]*z, rays[:,1]*z, z], -1)            # (N,3) camera-frame points

    n, inl = ransac_plane(P, thresh=0.02, iters=1000, seed=seed)   # 2 cm inlier band, FIXED seed
    w = np.where(conf[v,u][inl] == 2, 1.0, 0.25) / z[inl]**2       # confidence × inverse-depth²
    n = weighted_pca_normal(P[inl], w)                             # total-least-squares refit

    if n @ np.array([0., -1., 0.]) < 0:                            # up is −y in OpenCV
        n = -n
    return n / np.linalg.norm(n), path
```

Two weights, both load-bearing. **Confidence** (`2` outvotes `1`). And **inverse depth squared** —
LiDAR error grows roughly with the square of range, so distant floor pixels must not get an equal vote.

**Cross-check:** `angle(up_gravity, up_floor) < 2°`. On disagreement, trust gravity and flag the clip;
a large disagreement usually means the "floor" you fitted is actually a wall, a platform edge, or the
lifter's back.

**Once per clip, not per frame.** The cameras are static, so tilt is a **constant**. Per-frame estimation
only injects noise and makes the rectification flicker. Take a **temporal median of the depth maps
first** (static camera ⇒ the median across frames kills most LiDAR noise), then fit one plane.

### 4.4 Build the leveling rotation `R`

Level = the camera's `y` axis points along world-down, keeping the original azimuth (we de-tilt; we do
not pan):

```python
def leveling_rotation(up_cam, fwd_cam=np.array([0., 0., 1.])):
    y_v = -up_cam / np.linalg.norm(up_cam)                  # OpenCV camera y = DOWN
    z_v = fwd_cam - (fwd_cam @ y_v) * y_v                   # forward, with tilt projected out
    z_v /= np.linalg.norm(z_v)
    x_v = np.cross(y_v, z_v)                                # right-handed: x = y × z
    return np.stack([x_v, y_v, z_v])                        # rows ⇒ R maps original cam → virtual cam
```

### 4.5 Apply BOTH corrections in ONE remap

Every `remap` resamples, and resampling twice compounds blur — softening the exact plate edges S4 must
measure. Tilt rectification is nothing more than **rotating the camera about its own optical centre**,
which is precisely what `cv2`'s rectification argument does:

```python
newK = cv2.getOptimalNewCameraMatrix(K, DIST, (W,H), alpha=1)[0]   # alpha=1 keeps all pixels
map1, map2 = cv2.initUndistortRectifyMap(K, DIST, R, newK, (W,H), cv2.CV_32FC1)
rgb_rect   = cv2.remap(rgb, map1, map2, cv2.INTER_CUBIC)           # ONE resample, BOTH corrections
```

- `CV_32FC1` maps, **not** fixed-point `CV_16SC2` (which quantizes to 1/32 px).
- `INTER_CUBIC` (or `LANCZOS4`) for RGB; `INTER_NEAREST` for depth/confidence if you warp them at all.
- Cameras are static ⇒ **the maps are constant for the whole clip.** Build once, reuse every frame.
- `alpha=1` retains all pixels (black borders). `alpha=0` crops to valid pixels only — which may cut off
  the overhead lockout or the feet. Prefer `alpha=1` and let S5 do the cropping deliberately.
- **`newK` is now the camera matrix of the rectified image.** Record it; everything downstream projects
  with `newK`, not `K`.

> **The `R` convention is a coin-flip you must not guess.** OpenCV applies `R⁻¹` to the *destination*
> ray. Whether your `R` should be `cam→virtual` or its transpose is exactly the kind of thing that
> produces a plausible-looking image tilted the **wrong way by twice the angle**. Pin it with §8.3.

> **The principle that bounds this stage:** a pure **rotation** about the optical centre is exact for
> every pixel *regardless of scene depth* — no parallax, no depth needed, no holes. A **translation** of
> the virtual camera has none of those properties: it needs dense per-pixel depth and tears holes where
> the scene was occluded. So tilt correction is free and exact; reprojecting all cameras to one
> canonical viewpoint is neither, and is **out of scope**. "Distort back to reality" = *undistort +
> de-rotate*, nothing more.

> **`G3` — Gate.** (a) A known-vertical scene edge (rack upright) is vertical in the output to within
> 1°. (b) `angle(up_gravity, up_floor) < 2°`. (c) On synthetic data with an injected tilt, the recovered
> angle is within 0.2° of truth (§8.3).

**Rejection.** Fallback (c) can *itself* fail: after the `0`s are gone a frame may have no usable depth
at all. If the combined `1`+`2` mask cannot support a stable plane fit, that frame has no depth-derived
tilt — fall back to gravity, or reject. **Never** fit a plane to a handful of scattered points and report
the angle as though it were measured.

---

## 5 · S4 · Scale normalization

**Goal (I4):** constant pixels-per-metre across every camera, distance, and session.

```
1. Detect the plate face. Head-on it is a circle; from any other angle it is an ELLIPSE.
   Use cv2.fitEllipse on the plate contour (HoughCircles only works for the frontal camera).
2. Take the MAJOR axis — under projection, the major axis of the ellipse preserves the true diameter.
3. Known truth: competition bumper plate = 450 mm. That is what makes it a valid ruler.
4. s = target_major_axis_px / measured_major_axis_px
5. Resample the clip by s.
```

**Once per clip, from a robust aggregate** (median major-axis over frames where the plate is confidently
detected). The cameras are static. Per-frame rescaling would hold the plate constant in size while
making *everything else* jitter — precisely the nuisance motion this pipeline exists to remove.

**Rescaling changes the camera matrix.** Scaling an image by `s` scales every entry of `K`:

```python
K_final = newK.copy()
K_final[:2, :] *= s          # fx, fy, cx, cy all scale with s
```

Record `K_final`. S5 projects with it. Forgetting this makes S5's crop wrong by exactly the factor you
just applied — an error that looks like "the crop is slightly off" rather than like a bug.

**Persist `px_per_mm`.** It is the constant that converts CNN-space pixels back into metres per second.
Without it, no velocity output the model ever produces is in real units.

**Fallback** (camera never sees the plate face — a side camera sees it edge-on): derive scale from depth
and intrinsics, `px_per_mm = fx / distance_mm`. That `distance` **must** come from confidence-gated
depth (§4.2). A scale factor read off a confidence-`0` pixel silently mis-scales the entire clip, and
scale error propagates straight into reported bar velocity.

**Downscaling aliases.** When `s < 1`, use `INTER_AREA`. Bilinear downscaling aliases, and aliased plate
edges corrupt the very measurement this stage exists to make.

> **`G4` — Gate, and this is the good one:** after undistortion and scaling, the plate's apparent
> diameter must read **450 mm regardless of where in the frame it sits**. Measure it near the centre and
> near a corner. If the corner reading is systematically larger or smaller, your **distortion
> coefficients are wrong** — and the sign of the discrepancy tells you which way. This is a single
> end-to-end check on `K`, `dist`, *and* depth, using only objects you already own. Same trick with the
> bar's known length.

---

## 6 · S2 · Temporal sync (procedurally independent)

```
1. OCR the iPad stopwatch in each camera's RGB (or hand-label an anchor frame — for a first dataset,
   hand-labelling one frame per clip is entirely acceptable and removes an OCR failure mode).
2. T₀ = the LATEST stopwatch time visible in ALL cameras (you can only start where every camera
   already has data).
3. Per camera: frame nearest T₀ becomes frame 0.
4. Trim EVERY stream — odometry, IMU, confidence, depth, RGB — by its own timestamps.
   Depth and confidence are trimmed IDENTICALLY: frame k of depth always keeps frame k of confidence.
5. Record the per-camera residual offset (frame 0's true time minus T₀).
```

**The ~1 ms stopwatch-reading error is not your error budget.** The cameras aren't genlocked, so frame
quantization leaves two cameras' frame 0 up to **half a frame period (~8 ms at 60 fps)** apart — an
order of magnitude worse. The second pull lasts 150–250 ms, i.e. ~15 frames at 60 fps, so one frame of
sync error is several percent of the entire phase.

> **`G2` — Gate.** An event visible to multiple cameras (bar leaves the floor) lands on the **same frame
> index** in all of them.

---

## 7 · Orchestration

### 7.1 The sidecar (mandatory, one per clip per camera)

```json
{
  "clip_id": "...", "camera_id": "...",
  "rotation": "ROTATE_90_CLOCKWISE",
  "K": [[...]], "dist": [k1, k2, p1, p2, k3], "dist_source": "charuco|plumbline|none",
  "newK": [[...]], "K_final": [[...]],
  "up_cam_gravity": [x, y, z], "up_cam_floor": [x, y, z], "gravity_floor_angle_deg": 0.7,
  "R_level": [[...]], "tilt_deg": 4.2,
  "confidence_path_per_frame": ["b", "b", "c", ...],
  "T0_ms": 0.0, "sync_residual_ms": 3.1,
  "px_per_mm": 0.0, "plate_major_axis_px": 0.0,
  "crop_bounds": [x0, y0, x1, y1],
  "gates": {"G1": "pass", "G2": "pass", "G3": "pass", "G4": "pass", "G5": "skip"},
  "rejected": false, "reject_reason": null,
  "opencv_version": "4.x.y", "pipeline_git_sha": "..."
}
```

**Without `px_per_mm` and `crop_bounds`, a CNN prediction cannot be converted back into a real-world
measurement.** The sidecar is not logging; it is part of the output.

### 7.2 Determinism

Same input ⇒ byte-identical output. **Seed the RANSAC.** Pin the OpenCV version (interpolation kernels
have changed between releases). Training-set reproducibility depends on both.

### 7.3 Caching

The S3 remap maps and the S4 scale factor are **per-camera, per-clip constants**. Compute once, reuse
for every frame. Recomputing them per frame is the difference between a pipeline that runs overnight and
one that doesn't.

---

## 8 · Verification (build this FIRST)

An agent cannot look at an image and notice it is tilted 4° or mirrored. Therefore **synthetic
fixtures with known ground truth are not optional** — they are the only thing standing between you and
a pipeline that runs cleanly and outputs nonsense.

### 8.1 The synthetic scene

Render, with known parameters: a **floor plane** at a known orientation; a **circle** of exactly 450 mm
at a known position; several **straight lines** across the frame. Project with a known `K`, apply a
known `dist`, and place the camera at a known tilt `θ`. Emit matching fake depth + confidence maps.

### 8.2 Convention tests (§1.2)

- Point known to be **above** the camera in the world ⇒ its OpenCV camera-frame `y` is **negative**.
- `undistortPoints(distortPoints(p)) ≈ p` to < 0.01 px.

### 8.3 Recovery tests (the important ones)

| Inject | Assert |
|---|---|
| Known tilt `θ` | Recovered tilt within **0.2°**, and the rectified straight lines are vertical to within 0.1°. Run θ = ±2°, ±5°, ±15°. **If the recovered angle is ≈ −θ or ≈ 2θ, your `R` convention is inverted (§4.5).** |
| Known `k1, k2` | Plumb-line fit (P1b) recovers them; residual straightness < 0.5 px. |
| Known plate distance | S4 recovers `px_per_mm` within **0.5%**. |
| Plate at frame centre **and** at a corner | Both measure 450 mm within 1% *after* undistortion. This is `G4`, run on data where you know the answer. |

### 8.4 Real-data smoke test

One clip, all gates, sidecar written, no rejections. Then ten clips. Then the set.

---

## 9 · Unsolved issues

### 9.1 Blocking — cannot implement the affected stage without an answer

| # | Issue | Blocks | How to resolve |
|---|---|---|---|
| **U1** | **Do we have distortion coefficients at all?** If the app exports none, `cv2.undistort` is a no-op and S3's undistortion is decorative. | S3 | Preflight **P1**. Measure edge curvature; if > 1 px, calibrate via P1b (plumb-line, free) or P1c (ChArUco). |
| **U2** | **Cross-camera extrinsics.** ARKit gives each device its own world origin, so cameras cannot be cropped to a common physical region. This remains **the biggest unstated dependency in the original notes.** | **S5 entirely** | Preflight **P3**. Shared ArUco marker on the platform is the least-effort path. Build S0–S4 and stub S5 meanwhile. |
| **U3** | **Plate detection reliability.** S4's whole ruler depends on fitting an ellipse to the plate. Nobody has established that this works on real gym footage (motion blur at 2 m/s, occlusion by the lifter, glare, non-frontal cameras seeing the plate edge-on). | S4 | Test the detector on 20 real frames before trusting the stage. If it's unreliable, fall back to depth-derived scale (§5) and demote the plate to a cross-check. |

### 9.2 Decisions pending — implementable either way, but pick before you build

| # | Issue | Recommendation |
|---|---|---|
| **U4** | **Is ARKit gravity the primary tilt source, with the LiDAR plane fit as cross-check?** | **Yes** (§4.3). Gravity is more accurate, needs no visible floor, and this relegates the whole confidence-gating rule to a fallback path rather than the critical path — a far more comfortable place for it. **This inverts the emphasis of the original notes.** |
| **U5** | **Per-frame or per-clip gating and tilt?** | **Per-clip.** Cameras are static ⇒ tilt is a constant; per-frame estimation only adds noise and flicker. Cost: ignores genuine per-frame dropouts (lifter occluding the floor mid-lift). |
| **U6** | **"Continuous region" — 4- or 8-connectivity?** | Pick one and pin it; it decides which clips take the fallback path. 8 admits thin diagonal bridges between patches; 4 is stricter. Manual defaults to **8**. |
| **U7** | **"1 ninth of the image" — of the full frame, or of valid (non-zero-confidence) pixels?** | Manual assumes **full frame**. These diverge badly on a frame where most depth was already discarded. |
| **U8** | **Confidence weights in the plane fit.** Manual assumes `conf 2 → 1.0`, `conf 1 → 0.25`. Unjustified — the ratio is a guess. | Tune, or justify from Apple's confidence semantics. |
| **U9** | **When even the fallback has no usable depth** — reject the clip, or fall back to gravity? | Falls out of U4: if gravity is primary, this stops mattering. |
| **U10** | **Scale target & CNN input size** — plate diameter in pixels, and `H×W`. | Together these fix the field of view in metres. Must be chosen before S4/S5 emit anything. |

### 9.3 Known unknowns — will bite later, not now

| # | Issue | Note |
|---|---|---|
| **U11** | **Rolling shutter.** Possibly a **larger error source than lens distortion** for this application: the bar moves vertically, i.e. *along* the readout direction of a portrait frame, so its apparent position is biased — and that bias lands directly in the velocity numbers the tool exists to produce. Undistortion does not touch it. | Preflight **P4**. Measure the readout time; then either model it or document it as a known bias. |
| **U12** | **Non-competition plates.** The 450 mm ruler assumes bumper plates. Any clip filmed with steel/smaller plates silently mis-scales. | Either constrain capture to bumpers, or record plate diameter per session. |
| **U13** | **Stopwatch legibility at 60 fps**, across every camera, at distance, under motion blur. OCR may simply not work. | Hand-labelling one anchor frame per clip is a perfectly good v1. |
| **U14** | **Camera count/placement.** `Notes.md` reasons toward 4 (front/back/left/right) but notes 2 may suffice by symmetry. | S5's canonical volume needs the final answer. |
| **U15** | **S6 (tensorization) is entirely unspecified.** Channels (RGB? RGB-D? stacked multi-camera?), clip length, per-channel normalization — and **which augmentations are even legal** after a pipeline whose entire purpose was removing geometric variation. | Rotation/scale augmentation would re-introduce exactly the nuisance variation S1–S5 spent all this effort deleting. Think carefully. |
| **U16** | **Is confidence itself a CNN input?** It is a per-pixel reliability map; if S6 feeds a depth channel, confidence is its natural companion mask. | Open. |

---

## 10 · Feedback for v2

<!-- Append feedback below; incorporate into preprocessing-manual-v2.md rather than editing above. -->

