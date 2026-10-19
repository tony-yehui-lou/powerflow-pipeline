# S7 Barbell — Bar Centre Track per Frame: Height and Yaw

Status: **draft 4** | Supersedes: [`S7-bar-height-draft3.md`](S7-bar-height-draft3.md) (feedback at
its bottom) | Scope: GitHub issue [#63](https://github.com/tony-yehui-lou/PowerFlow/issues/63) (Barbell HVA) under epic [#64](https://github.com/tony-yehui-lou/PowerFlow/issues/64) | Updated: 2026-10-19

**Changes from draft 3:**

1. **The tilt decision is made.** Two solves run side by side on every frame: **A, level bar**
   (θ = 0) and **B, known depth** (the centre's Z held at its rest value). Their disagreement is
   the tilt proxy.
2. **Collar depth arbitrates.** Prototyping showed the A–B disagreement also fires when the bar
   moves toward or away from the camera, so on its own it is not a pure tilt signal. The collar
   depth residual tells the two causes apart (§5).

Carried over from draft 3: the tracked point is the **bar centre (COM)**; the cap and collar only
find the bar; yaw gaps are filled by **linear interpolation** between the flanking measured frames.

**In:** per-frame cap and collar boxes (`predictions.json`, see
[`S7-barbell-landmark-detection.md`](S7-barbell-landmark-detection.md)), S4's rewritten
`camera_matrix.csv` and `depth/`, S3's `retilt_sidecar.json`, and the declared bar type.
**Out:** per frame, the bar centre's height above the floor (mm), yaw (deg), the tilt proxy, and
flags.

Measured numbers are from a read-only prototype on `10 September/50kg3`, the held-out test capture:
433 frames, 413 with both landmarks.

---

## 1. What is tracked

The **bar centre** is the midpoint between the caps, which is also the COM under symmetric
loading. It sits on the axis, `C` from the visible cap:

| | `C` (cap → centre) | `L` (cap → opposite collar) |
|---|---|---|
| Men's IWF | 1100 mm | 1755 mm |
| Women's | 1005 mm | 1660 mm |

## 2. Geometry

S3 makes the camera level. In S4's camera frame (X right, Y down, Z forward), `Y = (v − cy)·Z/f`.

A bar pose has five degrees of freedom: centre (3), yaw ψ, and tilt θ (+ means the collar end is
lower). One cap and one collar give four numbers. Each solve supplies the fifth constraint
differently:

| Solve | Fifth constraint | Unknowns solved | Fails when |
|---|---|---|---|
| **A — level bar** | θ = 0 | centre X, Y, Z; ψ | the bar is tilted; and ψ near camera height (§4) |
| **B — known depth** | centre Z = `Z_rest` | centre X, Y; ψ; θ | the centre moves in Z |

Neither assumption is checked by its own solve: each fits the four observations exactly. That is
why §5 needs a fifth observation from outside both of them.

**Yaw convention.** ψ is measured in the horizontal plane from +Z towards +X, along cap → collar.
The stage also emits the unit axis vector in the level camera frame.

## 3. The two solves

**A, closed form.** With both ends at one `Y`, `Z_i = f·Y/(v_i − cy)`:

```
P_i = Y · ( (u_i − cx)/(v_i − cy),  1,  f/(v_i − cy) )
|P_cap − P_collar| = L   ⇒   |Y| = L / |d|,   sign(Y) = sign(v − cy)
ψ = atan2(X_collar − X_cap, Z_collar − Z_cap),   centre = P_cap + (C/L)(P_collar − P_cap)
```

**B, four equations in four unknowns.** Parameterise by centre `(X, Y, Z_rest)`, ψ and θ. The cap
sits at `centre − C·a` and the collar at `centre + (L − C)·a`, where
`a = (sin ψ cos θ, sin θ, cos ψ cos θ)`. Solve so that both project onto their observed pixels
(Gauss–Newton, warm-started from the previous frame). On 50kg3 the residual reaches machine
precision on every frame.

**`Z_rest`** is the median of A's centre Z over the rest window, where θ = 0 holds physically
because both plates are on the floor. On 50kg3 it is **2777 mm, spread 5 mm**.

## 4. Yaw

Yaw comes from A. A frame is *measured* when `σψ ≤ ψ_max`, with σψ propagated from 2 px of landmark
noise. Each run of unmeasurable frames is filled by **linear interpolation** between the last
measured frame before it and the first after it, on the unwrapped angle. A run that touches either
end of the clip holds its one flank. Those frames are flagged `yaw_interpolated` with their flank
frames.

On 50kg3 the run covers about frames 205–255, the turnover. **B's yaw is reported alongside as a
check:** it stays at 37.8–38.7° through that window, while the interpolation runs from ~38° to
~43–45°. A large gap between the two inside an interpolated run is flagged `yaw_check_disagrees`.
The interpolated value stays the output.

On interpolated frames A's centre is recomputed with the interpolated ψ (the fixed-yaw form,
draft 3 §4).

## 5. Tilt proxy, and arbitration

**Proxy.** `Δ = Y_B − Y_A`, the difference in centre height. A real tilt pushes the two solves
apart, so Δ is the tilt proxy. B's θ is reported as the tilt estimate *when B is selected*.

**Why it needs an arbiter.** Δ fires on a broken assumption in *either* solve:

| 50kg3 window | median `|Δ|` | B's θ | Actual cause |
|---|---|---|---|
| Rest (< 140) | 1 mm (max 7) | 0.00° (−0.26..0.17) | neither assumption broken |
| Lift (150–370) | 34 mm (max 76) | −2..+3° | mixed, see below |
| After drop (380+) | 330–400 mm | −14..−19° | the bar moved ~0.8 m toward camera; B's Z is false |

**Arbiter: collar depth.** Sample the depth map at the collar centre (3×3 median). Subtract its
**rest offset**, the median of (model − depth) over the rest window: 127 mm on 50kg3, the same
for both models, MAD 37. Then compare each model's predicted collar Z with the corrected depth:

| 50kg3 window | A residual | B residual | Selected |
|---|---|---|---|
| Rest | ~0 | ~0 | A = B |
| Turnover (205–255) | +356 mm | +49 mm | **B**: A's yaw is unobservable there |
| Overhead hold (290–370) | −53 mm | +133 mm | **A**: the bar drifted toward the camera |
| After drop (380+) | −110 mm | +559 mm | **A** |

**Per-frame rule:**

1. If `|Δ| ≤ Δ_agree`, the models agree. Report A's centre and flag nothing.
2. Otherwise, select the model with the smaller absolute depth residual, if it is smaller by at
   least `r_margin`. Set `selected` to `A` or `B`.
   - If B is selected, A's level assumption failed: report B's θ as the tilt estimate and flag
     `tilted`.
   - If A is selected, the centre moved in Z: flag `z_drift`. Δ says nothing about tilt here.
3. If neither residual wins by `r_margin`, flag `ambiguous` and report A, with σ inflated to
   `|Δ|`.

`Δ_agree` and `r_margin` are to be calibrated (§8). Collar depth noise (MAD ~40–70 mm) makes
`r_margin` ≈ 150 mm a starting point. That is enough to separate the 50kg3 windows above, but not
a 1° tilt (~28 mm of height, a few tens of mm of collar Z).

**What this does and does not buy.** Large failures are caught and attributed: the turnover yaw,
the drop, and the hold drift. Small tilts of 1–2° in mid-pull leave Δ at 25–55 mm, inside the
arbiter's blind zone. They show up as Δ but cannot be told apart from small Z drift.

## 6. From "below camera" to "above floor"

These are unchanged from draft 3, and apply to the selected centre `Y`:

- **Primary, rest-plate anchor:** `height = 225 + (Y_rest − Y)`, assuming 450 mm bumpers.
- **Check, floor-plane anchor:** `height = floor_offset − Y`. It is 31 mm lower on 50kg3 at rest.
  Report both and their gap.

## 7. Output

`bar_pose.json` per capture:

- **Per frame:**
  - `centre_height_mm` and `centre_height_sigma_mm` (from the selected model);
  - `centre_cam_mm`, `yaw_deg`, `yaw_sigma_deg`, `axis_cam`;
  - from both solves: `a: {centre_cam_mm, yaw_deg}` and `b: {centre_cam_mm, yaw_deg, tilt_deg}`;
  - `delta_mm` (the tilt proxy), `collar_depth_residual_mm: {a, b}`, and `selected` (`A`, `B` or
    `agree`);
  - `flags`: `yaw_interpolated` (with flank frames), `yaw_check_disagrees`, `tilted`, `z_drift`,
    `ambiguous`, `collar_only`, `axis_along_ray`, `missing`.
- **Provenance:** `L`, `C`, landmark σ, `ψ_max`, rest window, `Z_rest` and its spread, the collar
  depth rest offset, `Δ_agree`, `r_margin`, both height anchors, and the checkpoint behind
  `predictions.json`.
- **Height is not smoothed.** Velocity and acceleration choose their own filter.

On collar-only frames A and B cannot run. Carry Z and ψ from the nearest solved frame, place the
centre from the collar, and flag `collar_only`.

## 8. Open validation items

1. **Calibrate `Δ_agree`, `r_margin` and `ψ_max`** across all seven captures. Report how many
   frames land in each `selected` class and in `ambiguous`.
2. **The depth-arbiter's rest offset** (127 mm) must be stable within a capture. Is its cause a
   landmark that sits off the plate face, or depth bias? It is re-estimated per capture rather than
   fixed.
3. **Ground truth for height.** The overhead hold reads ~1875 mm (A) or ~1915 mm (B), against a
   rough 2.0–2.1 m estimate. A tape-measured reference capture is needed.
4. **Ground truth for tilt.** A capture with a deliberately tilted bar, e.g. one end on a 25 mm
   plate at rest, would check B's θ and the arbiter together.
5. **Ground truth for yaw**, including whether linear interpolation or B's yaw is closer through
   the turnover.
6. **Floor vs gravity disagree by 10°** (S3, warn-only). This defines "level" for both solves.
7. **Plate diameter and bar type must be declared.**

## 9. Implementation outline

Pure geometry is kept separate from I/O and built test-first:

- **`solve_level(obs, K, L, C)` (A) and `solve_known_depth(obs, K, L, C, z, x0)` (B).** Synthetic
  fixtures assert:
  - exact recovery for a level bar, by both solves;
  - for a tilted bar at fixed Z, B recovers θ exactly and A shows draft 3 §5.1's bias;
  - for a level bar moved in Z, A is exact and B is biased;
  - a bar at camera height, and a bar along a viewing ray, are flagged with no divide.
- **`interpolate_yaw(frames, yaw, measured_mask)`.** Interior gap, edge gap (held), 0°/360° wrap,
  and back-to-back gaps.
- **`arbitrate(delta, resid_a, resid_b, delta_agree, r_margin)`.** A table-driven test covering
  each branch of §5.
- **A task that reads `predictions.json`, S4's K and depth, and S3's sidecar, and writes
  `bar_pose.json`.** The manifest reports:
  - rest height under both anchors;
  - `Z_rest` and its spread, and the depth rest offset;
  - counts per `selected` class and per flag;
  - interpolated runs;
  - peak centre height.

---

## Feedback (2026-10-19)

- **User:** prototype B with a per-frame plate depth in place of the held `Z_rest`, run it
  alongside A, and use plate depth as the arbiter.
- **How plate depth is measured.** The far plate's inner face is centred on the collar and shows
  as a clean, separate blob in the depth map.
  - Region-grow from the collar's depth pixel (4-connected, within ±250 mm of the seed), clip to
    1.3 × the projected plate radius, then take the median.
  - That gives ~480 depth pixels per frame, against 9 for the 3×3 collar sample.
  - Frame-to-frame step at rest is 7 mm. But the reading **drifts slowly by ±60–130 mm while the
    bar is static** (rest std 51 mm against A), with a rest offset of −126 mm.
  - The depth map is too smoothed to give the plate's *orientation*. A plane fit returns a
    camera-facing normal (yaw ≈ 0 ± 9°) where the bar's yaw is ~39°.
- **B with plate depth as its fifth constraint** (5 unknowns; 4 pixel observations at σ 2 px plus
  collar Z at σ 60 mm), measured on 50kg3:

  | Window | median `|Δ|` | B's tilt | B's yaw | A's yaw |
  |---|---|---|---|---|
  | Rest < 140 | 24 mm (max 86) | 0.00° ± 1.12 | 38.5° ± 2.1 | 38.6° ± 0.4 |
  | Pull 150–200 | 50 mm | −1.9° | 41.7° ± 4.7 | 36.7–38.5° |
  | Turnover 205–255 | 5 mm | +1.5° | **88°** | unmeasurable |
  | Overhead hold 290–370 | 166 mm | **+13.7°** | **90°** | 36–43° |

  - **At rest** (truth: tilt 0, bar static), B is noisier than A in every output. The depth drift
    goes straight into tilt, yaw and height.
  - **In the hold and turnover, B is contradicted by the video.** A 90° yaw puts the plate
    edge-on, but frame 330's plate is clearly seen face-on, as an ellipse. The solve absorbs a
    ~100 mm depth error by rotating the bar ~50° and tilting it 14°. Because yaw and tilt trade
    off, a single depth reading at the collar cannot pin them down.
  - **Overhead height:** A gives 1869 mm and B 2030 mm. B lands nearer the anthropometric guess,
    but on a pose the video rules out, so it is not evidence for B.
- **A constraint cannot also be the arbiter.** If B is fitted to plate depth, it matches plate
  depth by construction, so plate depth can then only test A.
- **Plate depth as arbiter between A and B (held `Z_rest`).** The blob is a better sample than
  draft 4's 3×3 window.
  - Turnover: A's residual is +304 mm and B's ~±50, so B is selected, which is right.
  - After the drop: A −153 against B ~+600, so A is selected, which is right.
  - Overhead hold: A −91 against B ~−50. That is inside a 150 mm margin, so the frames are
    `ambiguous`; the single-pixel sample of draft 4 had (wrongly confidently) picked A.
- **Correction, incorporated in [`S7-bar-height-draft5.md`](S7-bar-height-draft5.md).** The
  overhead-hold B residual above ("~−50") was read off a few frames by eye. With B's own rest
  offset measured and subtracted, the median over 290–370 is **+85 mm** (A: −91). The two are
  symmetric, which is why draft 5 calls the hold ambiguous.
