# S7 Barbell — Bar Centre Track per Frame: Height and Yaw

Status: **draft 3** | Supersedes: [`S7-bar-height-draft2.md`](S7-bar-height-draft2.md) (feedback at
its bottom) | Scope: GitHub issue [#63](https://github.com/tony-yehui-lou/PowerFlow/issues/63) (Barbell HVA) under epic [#64](https://github.com/tony-yehui-lou/PowerFlow/issues/64) | Updated: 2026-10-19

**Changes from draft 2:**

1. **The tracked point is the bar centre (COM), not either landmark.** The cap and collar are only
   there to *find* the bar, and their own heights are never reported.
2. **Yaw gaps are filled by linear interpolation** between the two measured frames flanking each
   gap. There is no smoother.
3. **Tilt is treated as an open decision (§5), not a stated assumption.** The simulation reported
   in draft 2's feedback shows that tracking the centre does not make tilt matter less with this
   landmark pair. §5 gives the numbers and the options.

**In:** per-frame cap and collar boxes (`predictions.json`, see
[`S7-barbell-landmark-detection.md`](S7-barbell-landmark-detection.md)), S4's rewritten
`camera_matrix.csv`, S3's `retilt_sidecar.json`, and the declared bar type.
**Out:** per frame, the bar centre's height above the floor (mm), the bar's yaw (deg), a 1σ value
for each, and flags.

Measured numbers are from a read-only prototype on `10 September/50kg3`, the held-out test capture:
433 frames, 413 with both landmarks.

---

## 1. What is tracked

The **bar centre** is the midpoint of the bar between its two caps. With symmetric loading it is
also the COM of the bar and plates. It lies on the bar axis at:

| | Cap → centre | Cap → opposite collar (`L`) |
|---|---|---|
| Men's IWF | 1100 mm | 1755 mm |
| Women's | 1005 mm | 1660 mm |

The centre is never visible: it sits between the lifter's hands. It is placed by solving the bar's
pose from the landmarks and stepping along the axis.

## 2. Geometry

S3 makes the camera level, so in S4's camera frame (X right, Y down, Z forward) image `v` is
aligned with gravity. A point at depth `Z` lies `Y = (v − cy)·Z/f` below the camera.

A bar pose has **five degrees of freedom**: centre (3), yaw ψ about the vertical, and tilt θ
(one end higher). One cap and one collar give **four numbers**. Any solve from these two landmarks
therefore fixes one degree of freedom from outside the image. §5 decides which one.

## 3. Per-frame solve (with θ = 0, pending §5)

If both ends share one `Y`, then `Z_i = f·Y/(v_i − cy)` and

```
P_i = Y · ( (u_i − cx)/(v_i − cy),  1,  f/(v_i − cy) )
|P_cap − P_collar| = L   ⇒   |Y| = L / |d|,   sign(Y) = sign(v − cy)
ψ      = atan2(X_collar − X_cap,  Z_collar − Z_cap)
centre = P_cap + (C/L)·(P_collar − P_cap),   C = cap → centre distance
```

The solve is closed form and stateless.

**Yaw convention.** ψ is measured in the horizontal plane from the optical axis (+Z) towards +X,
along cap → collar. ψ = 90° means fronto-parallel. The stage also emits the unit axis vector in the
level camera frame.

## 4. Yaw gaps: linear interpolation

Each frame's σψ is propagated from landmark noise (σ = 2 px per coordinate, the detector's median
sharp-box centre error). A frame's yaw is **measured** when `σψ ≤ ψ_max`, and **unmeasurable**
otherwise. That happens when both landmarks are near camera height, where yaw rests on a few
pixels of `v` difference.

**Filling rule.** For each maximal run of unmeasurable frames:

- Take the last measured frame before the run (`a`, `ψ_a`) and the first after it (`b`, `ψ_b`).
- Set `ψ(t) = ψ_a + (ψ_b − ψ_a)·(t − a)/(b − a)`.
- Interpolate on the unwrapped angle, so a pair like 359° → 1° crosses 0° rather than sweeping back
  through 180°.
- If the run touches the start or end of the capture, so only one flank exists, hold that flank's
  value.
- Flag each frame `yaw_interpolated` and record `a` and `b`.

On 50kg3 the run is roughly frames 205–255 (~0.8 s, covering the turnover and catch). Its flanks
measure 36–39° (σ ≈ 0.4–1.1°) and ~43–45° (σ ≈ 1°), so the interpolated window spans about 6°.

**Centre on interpolated frames.** Recompute the centre with the interpolated ψ:

```
x_i = (u_i − cx)/f,  a = (sin ψ, cos ψ)
Z_cap = L·(a_x − x_collar·a_z)/(x_collar − x_cap),  Z_collar = Z_cap + L·a_z
Y_i = (v_i − cy)·Z_i/f,   centre = P_cap + (C/L)·(P_collar − P_cap)
```

This degenerates when the axis lies along a viewing ray (`x_collar ≈ x_cap`). Such frames keep the
per-frame solve and are flagged `axis_along_ray`.

## 5. Tilt — decision needed

### 5.1 Measured: centre height is *not* tilt-insensitive with this pair

The premise was that tracking the centre should shrink tilt to a lever-arm effect about it. The
simulation does not bear that out. A synthetic bar at 50kg3's geometry (ψ = 38.6°) was tilted
about its centre, projected through S4's K, and solved with §3:

| Phase | Centre height error / ° of tilt | Yaw error / ° of tilt | Centre depth error / ° |
|---|---|---|---|
| Rest (1150 mm below camera) | +28 mm | +1.6° | +57 mm |
| Pull (600 mm below) | +28 mm | +3.2° | +110 mm |
| Overhead (500 mm above) | +24 mm | −3.1° | −101 mm |
| Near camera height (150 mm above) | +25 mm | −9.6° | −380 mm |

These hold roughly linearly to ~2°, then grow, reaching 130–290 mm at 10°.

**Why.** The solve reads the ends' extra vertical image offset as perspective, a depth difference
between them. That mis-scales the whole bar, and the centre with it. The error is not the centre's
own vertical motion, which is zero by construction. It is a wrong depth, and therefore a wrong
scale. The centre comes out ~25 mm/deg wrong, which exceeds the cap end's own 19 mm/deg geometric
drop. Tracking the centre is still the right *output*, because it is the COM. It does not by
itself make the estimate tilt-robust.

### 5.2 Options

| Option | Fifth constraint | Cost |
|---|---|---|
| **A. Level bar** (as §3) | θ = 0 | ~25 mm of centre height per degree of real tilt; yaw corrupted near camera height |
| **B. Known depth** | centre Z from the rest window (where θ = 0 holds, plates on floor), held or slowly varying | centre height error ≈ `Y·δZ/Z`: ~0.4 mm per mm of depth error at rest-level heights, ~0.2 overhead. Sagittal bar drift of ±10 cm shows as ±6 cm in Z at this yaw, so ~±25 mm. Tilt becomes an output, but a noisy one |
| **C. Third landmark** | far cap, a second collar, or the plate ellipse's axis | removes the ambiguity; needs a detector change |
| **D. Second camera** | stereo | removes the ambiguity; needs a capture change |

Collar depth from the depth map as the fifth constraint was ruled out in draft 2's feedback. Its
±150 mm noise is worth ~±5° of tilt.

**Recommendation until decided:** run A and B side by side, and report their per-frame
disagreement as an observable proxy for tilt. A real tilt pushes them apart; a level bar keeps
them together.

## 6. From "below camera" to "above floor"

These are unchanged from draft 2:

- **Primary, rest-plate anchor:** `height = 225 + (Y_rest − Y)`, using the centre's `Y` and
  assuming 450 mm bumpers.
- **Check, floor-plane anchor:** `height = floor_offset − Y`, using S3's `floor_offset_m`.
- On 50kg3 they differ by 31 mm at rest. Report both and their gap.

## 7. Missing landmarks

On collar-only frames (the 20 cap-missed frames on 50kg3, all after the lift), carry the collar's Z
and ψ from the nearest solved frame, place the centre from the collar alone, and flag
`collar_only` with inflated σ. Depth at the cap is unusable: it reads the background behind the
thin sleeve end.

## 8. Output

`bar_pose.json` per capture:

- **Per frame:** `centre_height_mm`, `centre_height_sigma_mm`, `centre_cam_mm` (X, Y, Z),
  `yaw_deg`, `yaw_sigma_deg`, `axis_cam` (unit, cap → collar), and `flags` (`yaw_interpolated`
  with flank frames, `collar_only`, `axis_along_ray`, `missing`). Once §5 is decided, also the
  tilt-option residual.
- **Provenance:** `L`, `C`, landmark σ, `ψ_max`, rest window, both anchors and their gap, the §5
  option, and the checkpoint behind `predictions.json`.
- **Height is not smoothed.** Velocity and acceleration choose their own filter.

## 9. Open validation items

1. **The §5 decision.**
2. **Ground truth for height.** The overhead hold reads ~1875 mm against a rough 2.0–2.1 m
   anthropometric estimate. Note that 3° of real tilt alone would account for ~75 mm (§5.1). A
   tape-measured reference capture is needed.
3. **Ground truth for yaw.** A bar set at a known angle.
4. **`ψ_max`, and the length of each interpolated run, across all seven captures.** Linear
   interpolation is only as good as the assumption that the bar rotates steadily through the
   turnover. Long runs should be reported, not hidden.
5. **Floor vs gravity disagree by 10°** (S3 `gravity_agreement_deg`, warn-only). This defines both
   "level" and the bar's height axis.
6. **The 31 mm gap between the two anchors.**
7. **Plate diameter and bar type must be declared.** Plate diameter has no field, and
   `barbell_type` is dropped before S4.

## 10. Implementation outline

Pure geometry is kept separate from I/O and built test-first:

- **`solve_bar_level` and `bar_centre_fixed_yaw`.** Synthetic fixtures assert exact recovery of the
  centre and ψ for a level bar, behaviour at camera height (height finite, yaw flagged), and a bar
  along a viewing ray (flagged, no divide).
- **A tilt fixture.** It encodes §5.1's sensitivity, so a change to the solve that alters it is
  caught.
- **`interpolate_yaw(frames, yaw, measured_mask)`.** Fixtures cover an interior gap (exact linear
  values), a gap at either end of the clip (held), wrap-around across 0°/360°, and back-to-back
  gaps.
- **A task that reads `predictions.json`, S4's K and S3's sidecar, and writes `bar_pose.json`.**
  The manifest reports: rest centre height under both anchors, yaw mean and spread at rest,
  interpolated runs (count, longest, flank values), `collar_only` count, and peak centre height.

---

## Feedback (2026-10-19) — incorporated in [`S7-bar-height-draft4.md`](S7-bar-height-draft4.md)

- **User:** adopt Options A (level bar) and B (known depth from rest) side by side, with their
  disagreement as the tilt proxy.
- **Prototype follow-up on 50kg3** (B fixes the centre's Z at the rest median, 2777 mm, spread
  5 mm, and solves X, Y, ψ, θ per frame):
  - **Rest:** A and B agree to ≤ 7 mm (median 1), and B reads tilt 0.00° (range −0.26..0.17).
    Both are correct, as expected.
  - **Lift (150–370):** they disagree by a median of 34 mm (max 76), with B's tilt between −2° and
    +3°.
  - **After the drop (380+):** they disagree by 330–400 mm, with B's tilt at −14..−19°. That is
    not tilt: the bar has moved ~0.8 m toward the camera, so B's held Z is false.
  - **The disagreement is not a pure tilt proxy.** It fires on real tilt *or* on the centre moving
    in Z, and the two cannot be told apart from A and B alone.
  - **Collar depth can arbitrate.** Subtract its rest-window offset (both models read ~127 mm
    deeper than the depth map at rest, MAD 37). Then the remaining residual picks the model whose
    assumption held:

    | Window | A residual | B residual | Model that held |
    |---|---|---|---|
    | Turnover | +356 mm | +49 mm | B (A's yaw is unobservable there) |
    | Overhead hold | −53 mm | +133 mm | A (the bar drifted toward the camera) |
    | After drop | −110 mm | +559 mm | A |

  - **B's yaw stays at 37.8–38.7° through the turnover,** where A's is unmeasurable. That is an
    independent check on the linearly interpolated yaw.
