# S7 Barbell — Bar Centre Track per Frame: Height and Yaw

Status: **draft 5** | Supersedes: [`S7-bar-height-draft4.md`](S7-bar-height-draft4.md) (feedback at
its bottom) | Scope: GitHub issue [#63](https://github.com/tony-yehui-lou/PowerFlow/issues/63) (Barbell HVA) under epic [#64](https://github.com/tony-yehui-lou/PowerFlow/issues/64) | Updated: 2026-10-19

**Changes from draft 4:**

1. **The arbiter measures the plate, not the collar pixel.** It uses the median depth of the far
   plate's face (~480 depth pixels per frame) instead of a 3×3 sample at the collar (9 pixels).
2. **B keeps the held `Z_rest`.** B with per-frame plate depth as its fifth constraint was
   prototyped and rejected (§6).
3. **The overhead-hold verdict changes from "A" to "ambiguous".** Draft 4's single-pixel sample
   picked A there. The plate measurement shows the evidence is split, so the stage no longer
   claims to know.

Carried over: the tracked point is the **bar centre (COM)**; the cap and collar only find the bar;
yaw gaps are filled by **linear interpolation** between the flanking measured frames; **A (level
bar)** and **B (known depth)** run side by side, and their disagreement is the tilt proxy.

**In:** per-frame cap and collar boxes (`predictions.json`, see
[`S7-barbell-landmark-detection.md`](S7-barbell-landmark-detection.md)), S4's rewritten
`camera_matrix.csv` and `depth/`, S3's `retilt_sidecar.json`, and the declared bar type.
**Out:** per frame, the bar centre's height above the floor (mm), yaw (deg), the tilt proxy, which
solve was selected, and flags.

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
lower). One cap and one collar give four numbers, so each solve supplies a fifth constraint:

| Solve | Fifth constraint | Solves for | Fails when |
|---|---|---|---|
| **A — level bar** | θ = 0 | centre X, Y, Z; ψ | the bar is tilted; ψ near camera height (§4) |
| **B — known depth** | centre Z = `Z_rest` | centre X, Y; ψ; θ | the centre moves in Z |

Each solve fits its four observations exactly, so neither can check its own assumption. The
arbiter (§5) brings in a fifth observation from outside both.

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
(Gauss–Newton, warm-started from the previous frame). On 50kg3 the residual is at machine precision
on every frame.

**`Z_rest`** is the median of A's centre Z over the rest window, where θ = 0 holds physically
because both plates are on the floor. On 50kg3 it is 2777 mm, spread 5 mm.

## 4. Yaw

Yaw comes from A. A frame is *measured* when `σψ ≤ ψ_max`, with σψ propagated from 2 px of landmark
noise. Each run of unmeasurable frames is filled by **linear interpolation** between the last
measured frame before it and the first after it, on the unwrapped angle. A run that touches either
end of the clip holds its one flank. Those frames are flagged `yaw_interpolated` with their flank
frames.

B's yaw is reported alongside as a check. On 50kg3 it stays at 37.8–38.7° through the turnover
(~205–255), where A is unmeasurable. A large gap between the two inside an interpolated run is
flagged `yaw_check_disagrees`.

On interpolated frames A's centre is recomputed with the interpolated ψ (the fixed-yaw form,
draft 3 §4).

## 5. Tilt proxy and the plate-depth arbiter

**Proxy.** `Δ = Y_B − Y_A`, the difference in centre height. Δ fires on a broken assumption in
either solve: real tilt breaks A, and Z movement breaks B. The arbiter decides which.

### 5.1 Plate depth

The far plate's inner face is a ~450 mm disc centred on the collar. In the depth map it is a clean
blob, separate from the background, on rest, turnover and overhead frames alike. Measured per
frame:

1. **Seed:** the median of the 3×3 depth window at the collar centre (depth pixel = RGB pixel ×
   188/1416).
2. **Grow:** take the 4-connected component of pixels within ±250 mm of the seed.
3. **Clip:** keep pixels within 1.3 × the plate's projected radius, `f·225/seed`, of the collar.
4. **Measure:** the median depth of what remains. Report the pixel count too.

On 50kg3 this gives ~480 pixels per frame (231–790), and the frame-to-frame step at rest is 7 mm.

**It measures distance only.** The depth map is too smoothed to show the plate's orientation. A
plane fit returns a camera-facing normal (yaw ≈ 0 ± 9°) where the bar's yaw is ~39°.

**Rest offset.** Each solve's collar Z reads deeper than the plate median at rest by about the same
amount: 126 mm for A, 122 mm for B. The offset is estimated per capture and per solve over the rest
window, and subtracted. The rest residual still has **σ ≈ 50 mm, as slow drift rather than noise**.
That residual σ is the floor below which the arbiter cannot decide.

### 5.2 Per-frame rule

With `r_A` and `r_B` the offset-corrected differences between each solve's collar Z and the plate
median:

1. If `|Δ| ≤ Δ_agree`, the solves agree. Set `selected = agree` and report A's centre.
2. Else if `|r_A| + r_margin ≤ |r_B|`, A's assumption held and the centre moved in Z. Set
   `selected = A` and flag `z_drift`. Δ says nothing about tilt here.
3. Else if `|r_B| + r_margin ≤ |r_A|`, A's assumption failed. Set `selected = B`, report B's θ as
   the tilt estimate, and flag `tilted`.
4. Otherwise set `selected = ambiguous`, report A, and inflate σ to `|Δ|`.

Starting values: `Δ_agree = 20 mm`, `r_margin = 150 mm` (≈ 3σ of the rest residual).

### 5.3 Measured on 50kg3

| Window | Frames | median `|Δ|` | `r_A` | `r_B` | B's θ | agree / A / B / ambiguous |
|---|---|---|---|---|---|---|
| Rest < 140 | 140 | 1 mm | 0 | 0 | 0.00° | **140** / 0 / 0 / 0 |
| Pull 150–200 | 51 | 18 mm | −108 | −90 | −0.7° | 28 / 0 / 0 / 23 |
| Turnover 205–255 | 51 | 37 mm | **+304** | −18 | −1.5° | 10 / 0 / **37** / 4 |
| 260–285 | 26 | 26 mm | −1 | −99 | −0.9° | 4 / 0 / 0 / 22 |
| Overhead hold 290–370 | 81 | 50 mm | −91 | +85 | +2.0° | 4 / 3 / 8 / **66** |
| After drop 380+ | 33 | 333 mm | −153 | **+517** | −14.5° | 0 / **22** / 2 / 9 |
| **All** | 413 | | | | | 207 / 27 / 54 / 125 |

- **Rest:** every frame agrees, as it must.
- **Turnover:** B is selected. A's yaw is unmeasurable there, and the plate confirms A's depth is
  ~300 mm off.
- **After the drop:** A is selected. The bar has moved ~0.8 m toward the camera, which falsifies
  B's held Z.
- **Overhead hold:** mostly ambiguous. A and B miss the plate by similar amounts in opposite
  directions. Draft 4's single-pixel arbiter called this A, which was overconfident. Either a
  ~2° tilt or ~90 mm of Z drift would explain the hold equally well, and plate depth at ±50 mm
  cannot separate them.

**Sensitivity.** With `r_margin` set to 100 / 150 / 200 mm, the ambiguous count is 81 / 125 / 142.
With `Δ_agree` set to 10 / 20 / 40 mm, the agree count is 175 / 207 / 298.

### 5.4 What this buys

The arbiter catches and attributes the large failures: the turnover yaw (A wrong) and the drop
(B wrong). Below roughly 2° of tilt, or 100 mm of Z drift, the plate cannot decide, and the stage
says `ambiguous` instead of guessing. On 50kg3 that is 30% of frames, concentrated in the pull and
the overhead hold.

## 6. Tried and rejected: B with per-frame plate depth

The idea was to replace B's held `Z_rest` with the plate depth on each frame. That gives a fifth
*measured* constraint (4 pixels at σ 2 px plus collar Z at σ 60 mm), with tilt and yaw both free.

| Window | B-plate θ | B-plate ψ | A's ψ |
|---|---|---|---|
| Rest < 140 (truth: θ = 0, static) | 0.00° ± 1.12 | 38.5° ± 2.1 | 38.6° ± 0.4 |
| Turnover 205–255 | +1.5° | **88°** | unmeasurable |
| Overhead hold 290–370 | **+13.7°** | **90°** | 36–43° |

It was rejected for three reasons:

- **At rest it is worse than A in every output.** The plate's ±50 mm drift goes straight into
  tilt, yaw and height.
- **In motion the video contradicts it.** A 90° yaw puts the plate edge-on, but frame 330 shows
  the plate face-on, as an ellipse. With yaw and tilt both free, a single collar depth cannot pin
  them down: the solve absorbs ~100 mm of depth error by rotating the bar ~50° and tilting it 14°.
  Its overhead height of 2030 mm (A: 1869 mm) comes from that impossible pose, so it is not support
  for B-plate.
- **A constraint cannot also be the arbiter.** B-plate fits plate depth by construction, which
  would leave plate depth able to test A only.

## 7. From "below camera" to "above floor"

These are unchanged, and apply to the selected centre `Y`:

- **Primary, rest-plate anchor:** `height = 225 + (Y_rest − Y)`, assuming 450 mm bumpers.
- **Check, floor-plane anchor:** `height = floor_offset − Y`. It is 31 mm lower on 50kg3 at rest.
  Report both and their gap.

## 8. Output

`bar_pose.json` per capture:

- **Per frame:**
  - `centre_height_mm` and `centre_height_sigma_mm` (from the selected solve);
  - `centre_cam_mm`, `yaw_deg`, `yaw_sigma_deg`, `axis_cam`;
  - from both solves: `a: {centre_cam_mm, yaw_deg}` and `b: {centre_cam_mm, yaw_deg, tilt_deg}`;
  - `delta_mm` (the tilt proxy);
  - `plate: {depth_mm, n_px}` and `plate_residual_mm: {a, b}`;
  - `selected` (`agree`, `A`, `B` or `ambiguous`);
  - `flags`: `yaw_interpolated` (with flank frames), `yaw_check_disagrees`, `tilted`, `z_drift`,
    `collar_only`, `axis_along_ray`, `plate_small` (pixel count below a floor), `missing`.
- **Provenance:** `L`, `C`, landmark σ, `ψ_max`, rest window, `Z_rest` and its spread, the per-solve
  plate rest offsets, the rest residual σ, the plate grow parameters, `Δ_agree`, `r_margin`, both
  height anchors, and the checkpoint behind `predictions.json`.
- **Height is not smoothed.** Velocity and acceleration choose their own filter.

On collar-only frames A and B cannot run. Carry Z and ψ from the nearest solved frame, place the
centre from the collar, and flag `collar_only`. The plate depth is still recorded.

## 9. Open validation items

1. **Calibrate `Δ_agree`, `r_margin`, `ψ_max` and the plate pixel floor** across all seven captures.
   Report the `selected` mix per capture and per lift phase.
2. **Plate depth drift.** Is the ±50 mm slow drift at rest temporal filtering in the phone's depth,
   or the blob boundary breathing? A depth-temporal model could shrink `r_margin` and with it the
   ambiguous share.
3. **Occlusion.** The athlete's hand and arm sit on the plate during the lift (231 pixels at frame
   180, against ~480 typically). Check whether partial blobs bias the median, not just thin it.
4. **Ground truth for height.** The overhead hold reads 1869 mm (A) against a rough 2.0–2.1 m
   estimate. A tape-measured reference capture is needed.
5. **Ground truth for tilt.** A capture with one end of the bar on a 25 mm plate at rest would check
   B's θ and the arbiter's `tilted` branch together.
6. **Ground truth for yaw**, including whether linear interpolation or B's yaw is closer through
   the turnover.
7. **Floor vs gravity disagree by 10°** (S3, warn-only). This defines "level" for both solves.
8. **Plate diameter and bar type must be declared.** The plate radius is used in both the height
   anchor and the plate clip.

## 10. Implementation outline

Pure geometry is kept separate from I/O and built test-first:

- **`solve_level(obs, K, L, C)` (A) and `solve_known_depth(obs, K, L, C, z, x0)` (B).** Synthetic
  fixtures assert:
  - a level bar is recovered exactly by both solves;
  - a tilted bar at fixed Z is recovered by B, while A shows draft 3 §5.1's bias;
  - a level bar moved in Z is recovered by A, while B is biased;
  - a bar at camera height, and a bar along a viewing ray, are flagged with no divide.
- **`plate_depth(depth, collar_px, K, plate_radius)`.** A synthetic depth image with a tilted disc,
  a background, a partial occluder and missing pixels. Assert the blob, the median, and the pixel
  count.
- **`interpolate_yaw(frames, yaw, measured_mask)`.** Interior gap, edge gap (held), 0°/360° wrap,
  and back-to-back gaps.
- **`arbitrate(delta, r_a, r_b, delta_agree, r_margin)`.** A table-driven test covering each branch
  of §5.2.
- **A task that reads `predictions.json`, S4's K and depth, and S3's sidecar, and writes
  `bar_pose.json`.** The manifest reports:
  - rest height under both anchors;
  - `Z_rest`, the plate offsets and the rest residual σ;
  - the `selected` mix;
  - interpolated runs;
  - peak centre height.

---

## Feedback (2026-10-19)

- **User:** drop depth, and with it B, the arbiter and the tilt estimate. Why is tilt needed at all?
  Keep the height on every frame, from the level-bar solve (A) alone.
- **Implemented:** in `powerflow-ui` as the Bar Height view (`powerflow-ui/docs/bar-height-view.md`).
  It uses A with the rest-plate anchor, the S3 floor plane as a printed cross-check only, and tilt
  assumed zero (~25 mm of height per degree of real tilt, unmeasured).
- **50kg3:** 412 of 433 frames have a height. Rest is at 225 mm by construction (192 mm by the floor
  plane). Peak is 1908 mm at 5.885 s.
- **Not yet done:** a draft 6 restating the spec as A-only.

---

## Renumbering note (2026-10-20)

Stages were renumbered so the barbell detector runs before pose: **S7 Barbell → S5**, **S5 Pose →
S6**, **S6 Lift → S7**. This draft is frozen, so its text still uses the old numbers. Read every
"S7" above as S5, "S5" as S6, and "S6" as S7.

Its files were renamed `S7-*.md` → `S5-*.md`, and `S5-*.md` → `S6-*.md` for the pose specs. Links
above that use the old names point to the files of the same name with the new prefix.
