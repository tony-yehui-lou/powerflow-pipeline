# S7 Barbell — Vertical Bar Height

Status: **draft** | Scope: GitHub issue [#63](https://github.com/tony-yehui-lou/PowerFlow/issues/63) (Barbell HVA) under epic [#64](https://github.com/tony-yehui-lou/PowerFlow/issues/64) | Updated: 2026-10-19

Height is the first of H/V/A. Velocity and acceleration are derivatives of it and are out of scope
here, beyond the requirement that the height series is smooth enough to differentiate.

**In:** per-frame cap and collar boxes from
[`S7-barbell-landmark-detection.md`](S7-barbell-landmark-detection.md) (`predictions.json`), S4's
rewritten `camera_matrix.csv`, S3's `retilt_sidecar.json`, and the declared bar type.
**Out:** per frame, the bar's height above the floor in millimetres, the method that produced it,
and a quality flag.

All numbers below come from a read-only prototype on `10 September/50kg3`, the held-out test
capture: 433 frames, 413 with both landmarks.

---

## 1. Why this is possible without depth

Two facts reduce the geometry to a closed form.

- **The camera is level.** S3 rotates every frame so the image plane is vertical. In S4's camera
  frame (CV convention: X right, Y down, Z forward), image `v` is aligned with gravity, and a
  point's height below the camera is `Y = (v − cy) · Z / f`.
- **The bar is a rigid rod of known length that stays near horizontal.** The usual visible pair is
  the cap on one end and the collar on the other, **1755 mm** apart on a men's IWF bar (1660 mm
  women's; see the landmark spec §2).

Landmark centres are taken as box centres. Both lie on the bar axis.

## 2. Method A — level bar (closed form)

If the bar is horizontal, both ends share one `Y`. Then `Z_i = f · Y / (v_i − cy)`, and

```
P_i = Y · ( (u_i − cx)/(v_i − cy),  1,  f/(v_i − cy) )
|P_cap − P_collar| = L   ⇒   |Y| = L / |d|,   d = the difference of the bracketed vectors
```

`sign(Y) = sign(v − cy)`. That is one formula per frame and needs no state.

Measured on 50kg3:

| | |
|---|---|
| Rest, frames 0–140 | Y = 1150 mm below camera, ±3 mm |
| Overhead hold, frames ~280–360 | Y ≈ −500 mm (500 mm above camera) |
| Implied travel | ~1650 mm |

**It fails where the bar crosses camera height.** As `v_cap` and `v_collar` both approach `cy`,
`d` blows up and `Z` loses its conditioning. On 50kg3, frames ~205–255 show the cap depth jumping
from 1.9 m to 3.0 m, and frame 210 puts the collar 1.4 m nearer the camera than the cap, which is
impossible. In this rig the bar crosses the horizon mid-pull, the exact phase that matters.

## 3. Method B — fixed yaw (recommended)

Horizontal position comes from `u` alone, which stays well conditioned through the horizon:

1. **Estimate yaw ψ once**, from rest frames solved with Method A: `ψ = atan2(ΔX, ΔZ)` between the
   two ends. On 50kg3 that gives **ψ = 38.7°, spread 0.42°** over 100 frames.
2. **Per frame**, with `a = (sin ψ, cos ψ)` and `x_i = (u_i − cx)/f`, the bar lies in the X–Z plane
   with `P_collar = P_cap + L·a`, so

   ```
   Z_cap    = L · (a_x − x_collar · a_z) / (x_collar − x_cap)
   Z_collar = Z_cap + L · a_z
   ```
3. Take `Y` as the mean of `(v_i − cy) · Z_i / f` over both ends.

Measured on 50kg3:

- Through the horizon crossing, B keeps the cap at Z = 1.92–1.98 m where A drifts to 3.0 m.
- Elsewhere, A and B agree to within about 5–40 mm.
- B drifts where the lifter turns the bar. During the overhead hold B's Z falls steadily
  (1.86 → 1.48 m) while A's stays near 1.7 m, which suggests the yaw changed after the catch.

**Recommendation:** B as the primary estimate. Use A as a per-frame consistency check wherever
`|v − cy|` is large for both ends, and flag frames where they disagree by more than a threshold
(to be calibrated, §6).

B degenerates when the bar axis points along a viewing ray (`x_collar ≈ x_cap`). At 38.7° yaw
this does not happen, but a pure side-on camera would hit it, and the stage must reject or flag
it rather than divide.

## 4. From "below camera" to "above floor"

Two independent anchors exist. Both should be computed and the stage should report their
disagreement.

| Anchor | Height above floor | 50kg3 rest value |
|---|---|---|
| **Rest-plate** — bar centre at rest = plate radius, 225 mm for a 450 mm bumper | `225 + (Y_rest − Y)` | 225 mm (by construction) |
| **Floor plane** — S3's `floor_offset_m` is the camera height | `floor_offset − Y` | 1344 − 1150 = **194 mm** |

The two anchors differ by 31 mm. **Primary: rest-plate.** It cancels any constant offset in the
landmark-to-axis assumption and in the floor fit, and velocity and acceleration need only the
displacement anyway. It requires a rest window, which every capture has because S1 Cut starts
before the lift. It assumes full-size bumpers, which should be declared rather than assumed (§6).

## 5. What depth is and is not good for

Depth was sampled at landmark centres on the 188×232 depth map:

- **At the cap: unusable.** The cap is a ~30 px tip, about four depth pixels. The window sees
  floor or wall behind it, giving 2.2 m at rest (geometry 1.9 m) and 7–11 m during the lift.
- **At the collar: plausible but coarse.** Against the plate face it reads 3.1–3.3 m where the
  geometry gives 3.3 m, with ±150 mm of noise.

So depth is a **sanity bound only**. It is also the fallback when the cap is missing: the 20
cap-missed frames on 50kg3 are post-lift rest. There, `Z_collar` comes from the last solved frame,
or from collar depth, and `Y` from the collar alone. Those frames are flagged `collar_only`.

## 6. Open validation items

1. **No ground truth yet.** The overhead hold comes out at ~1875 mm above the floor (rest anchor).
   For a 180 cm athlete with a snatch grip, a rough anthropometric estimate is ~2.0–2.1 m. That
   could be a real ~10% scale error, or the estimate could be wrong. The plan is one capture with a
   measured reference: a tape-measured bar height on blocks or a rack, or a known-height marker on
   the platform. Until then, report heights but do not claim accuracy.
2. **Floor vs gravity disagree by 10°.** S3 reports `gravity_agreement_deg = 10.0` on 50kg3, in
   warn mode. If "vertical" is off by θ, horizontal bar travel leaks into height at `sin θ`, which
   is ~2–3 cm for typical 10–20 cm of horizontal drift, and vertical travel shrinks by `1 − cos θ`
   (~1.5%). Find out which of the two is wrong before trusting sub-3 cm numbers.
3. **Rest-anchor vs floor-anchor gap (31 mm).** This may be item 2, a landmark centre sitting off
   the axis, or a floor-fit offset. Measure it across all seven captures before choosing a
   tolerance.
4. **The horizontal-bar assumption.** A tilted bar in a press-out or a missed lift breaks Method A
   and biases B. Test the size of the effect on a synthetic fixture with a known tilt.
5. **Yaw drift.** B fixes yaw from rest. The 50kg3 overhead-hold divergence suggests yaw changes
   after the catch. Options: re-estimate yaw on any window where A is well conditioned, or solve
   yaw per frame with a smoothness prior. Decide after seeing all seven captures.
6. **Plate size and bar type must be declared.** 225 mm assumes 450 mm bumpers, and 1755 mm assumes
   a men's bar. The bar type already reaches the annotation document. Plate diameter has no field
   yet. The preprocess metadata task also drops `barbell_type` before S4 (landmark spec §7).
7. **A vs B disagreement threshold.** Calibrate it from the 413 good frames on 50kg3 plus the
   training captures, then encode it as the flag rule.
8. **Smoothing before differentiation.** Raw centre jitter is ~2 px (≈6 mm). The filter choice
   (Savitzky–Golay vs. a constant-acceleration Kalman) belongs to the velocity spec, but this stage
   must output unsmoothed heights plus per-frame quality so that choice stays open.

## 7. Implementation outline

Pure geometry is kept separate from I/O, per the pipeline conventions, and built test-first:

- `bar_height_level(u, v, K, L)` (Method A) and `bar_height_fixed_yaw(u, v, K, L, ψ)` (Method B):
  pure functions, verified on synthetic fixtures. Project a known 3D bar through a known K, then
  assert that the recovered `Y` and `Z` are exact, including a bar exactly at camera height and a
  bar along a viewing ray.
- A task that reads `predictions.json`, the S4 camera matrix and the S3 sidecar, estimates ψ from
  the rest window, and writes a `bar_height.json` sidecar per capture. Each frame carries
  `height_mm`, `y_below_camera_mm`, `method`, `flags`, and the A–B residual. Provenance records
  L, ψ, the rest window and both anchors.
- The manifest reports per capture: rest height under both anchors, ψ and its spread, the count of
  flagged frames, and peak height.

---

## Feedback (2026-10-19) — incorporated in [`S7-bar-height-draft2.md`](S7-bar-height-draft2.md)

- **User:** measure the bar's angle on every frame instead of fixing yaw from the rest window. It
  should be more accurate per frame, and the angle is needed later to rotate the human model. Keep
  this spec to the barbell.
- **Prototype follow-up on 50kg3:**
  - Method A already solves yaw per frame: four observations, four unknowns.
  - Its yaw is precise where both landmarks sit well off camera height. Under 2 px noise σψ ≈ 0.35°,
    and the rest std is 0.41°.
  - Near camera height, yaw becomes unobservable: σψ reaches 3–27°, and values run to 55–142°.
  - §2's claim that A "fails" was about Z and yaw, not height. Y stays at σ ≈ 3–6 mm on every frame,
    the horizon crossing included.
  - Bar tilt (one end higher) cannot be recovered from one cap and one collar. Adding collar depth
    as a fifth observation produced 2–4° of tilt and a 100 mm height shift at rest, where the truth
    is 0°, so depth is too noisy to supply it.

---

## Renumbering note (2026-10-20)

Stages were renumbered so the barbell detector runs before pose: **S7 Barbell → S5**, **S5 Pose →
S6**, **S6 Lift → S7**. This draft is frozen, so its text still uses the old numbers. Read every
"S7" above as S5, "S5" as S6, and "S6" as S7.

Its files were renamed `S7-*.md` → `S5-*.md`, and `S5-*.md` → `S6-*.md` for the pose specs. Links
above that use the old names point to the files of the same name with the new prefix.
