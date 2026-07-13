Status: **draft** | Derived from: `Preprocessing.md` (raw notes) + feedback on `preprocessing-spec-v1.md` | Updated: 2026-07-11

  

> Scope: everything that happens between **raw multi-camera capture** and the **tensor handed to the

> CNN**. Nothing about model architecture, training, or inference lives here.

>

> Anything inferred rather than stated is marked **(assumption)**; anything undecided is in

> **Open questions**. Do not treat assumptions as settled.

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

There is one folder with date as the name (called date folder from this point onwards) and two positional folders inside the date folder (named Front and Side) and one `metadata.yaml` inside the data folder. The below describes the content in each positional folder.

Per capture session, per camera (**assumption**: StrayScanner-style capture on iOS):

  

- `rgb.mp4` — frame sequence, **stored landscape** (capture-app bug — see S1), with per-frame timestamps.

- `depth/` — per-frame LiDAR depth map, with its own resolution and timestamps. **Also stored

landscape** — the app's rotation bug affects every image stream, not just RGB.

- `confidence/` — **one confidence map per depth frame**, pixel-aligned with it, valued `{0, 1, 2}`:

`0` = unconfident, `1` = medium, `2` = most confident. Also stored landscape. Depth is **never**

consumed without it.

- `camera_matrix.csv` — camera matrix `K` (`fx, fy, cx, cy`) **as reported by the phone**, plus distortion

coefficients **if the capture app exports them** (see Open questions — it may not).

- `odometry.csv` — per-frame 6-DoF camera pose in ARKit's **gravity-aligned** world frame.

- `imu.csv` — per-frame inertial samples.

- **A stopwatch on an iPad, in view of every camera** — the shared clock.

  

Capture-side assumptions: single lifter, single barbell, cameras **static** for the take, ~60 fps,

barbell plate face visible to at least one camera.

  

---

  

## 3. Pipeline

  

Stages are **strictly ordered**; each depends on the invariants of the last.

  

```

S0 Ingest → S1 Orient
```

  

---

  

### S0 · Ingest, validate & update


Read the metadata of `rgb.mp4`, and check with the metadata in `metadata.yaml` file inside the date folder. Update all information into `metadata.yaml`.

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

