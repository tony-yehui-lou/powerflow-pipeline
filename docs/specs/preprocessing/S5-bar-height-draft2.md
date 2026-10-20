# S7 Barbell — Bar Pose per Frame: Height and Yaw

Status: **draft 2** | Supersedes: [`S7-bar-height.md`](S7-bar-height.md) (draft 1; feedback at its
bottom) | Scope: GitHub issue [#63](https://github.com/tony-yehui-lou/PowerFlow/issues/63) (Barbell HVA) under epic [#64](https://github.com/tony-yehui-lou/PowerFlow/issues/64) | Updated: 2026-10-19

**Change from draft 1:** the bar's horizontal angle (**yaw**) is now measured on every frame instead
of being fixed from the rest window. It is an output in its own right, consumed later to rotate the
human model, which is out of scope here. Height and yaw come from one per-frame solve. Each carries
its own uncertainty, and the two degrade in different places.

**In:** per-frame cap and collar boxes (`predictions.json`, see
[`S7-barbell-landmark-detection.md`](S7-barbell-landmark-detection.md)), S4's rewritten
`camera_matrix.csv`, S3's `retilt_sidecar.json`, and the declared bar type.
**Out:** per frame, the bar centre's height above the floor (mm), the bar's yaw (deg), a 1σ value
for each, and quality flags.

All numbers are from a read-only prototype on `10 September/50kg3`, the held-out test capture:
433 frames, 413 with both landmarks.

---

## 1. Geometry

S3 makes the camera level, so in S4's camera frame (X right, Y down, Z forward) image `v` is
aligned with gravity. A bar point at depth `Z` sits `Y = (v − cy)·Z/f` below the camera.

The bar is a rigid rod. The usual visible pair is a cap and the **opposite** collar,
`L = 1755 mm` apart on a men's IWF bar (1660 mm women's). The bar centre lies `1100 mm` from the
cap along the axis (`1005 mm` women's).

A bar pose has **five degrees of freedom**: centre (3), yaw ψ about the vertical, and tilt θ
(one end higher). Two landmarks give **four numbers** (`u, v` of each). So exactly one degree of
freedom must be supplied. This spec supplies **θ = 0** (§4).

## 2. Per-frame solve

With θ = 0 both ends share one `Y`. Each end then has `Z_i = f·Y/(v_i − cy)`, and

```
P_i = Y · ( (u_i − cx)/(v_i − cy),  1,  f/(v_i − cy) )
|P_cap − P_collar| = L   ⇒   |Y| = L / |d|,   sign(Y) = sign(v − cy)
ψ = atan2(X_collar − X_cap,  Z_collar − Z_cap)
```

That is closed form, stateless, and needs one division per frame. It returns the centre `(X, Y, Z)`
and ψ together.

**Yaw convention.** ψ is measured in the horizontal plane from the camera's optical axis (+Z)
towards +X, along the direction cap → collar. ψ = 90° means the bar is fronto-parallel. The stage
also emits the unit bar direction vector in the level camera frame, so a consumer never has to
re-derive it from an angle and a convention.

## 3. Uncertainty, and where each output degrades

Each frame's 1σ comes from propagating landmark noise through the solve: the Jacobian, or Monte
Carlo with σ = 2 px per coordinate (the detector's median sharp-box centre error). Measured on
50kg3:

| Phase | Frames | σ height | σ yaw | Yaw value |
|---|---|---|---|---|
| Rest | 0–140 | 5–6 mm | 0.31–0.36° | 38.6° mean, 0.41° std |
| Pull, bar below camera | 150–200 | 3–6 mm | 0.35–1.1° | 36.7–38.5° |
| **Bar near camera height** | **~205–255** | **3–4 mm** | **3–27°** | 55–142°, not credible |
| Overhead hold | 270–370 | 3–4 mm | 0.5–1.0° | 36–43° |
| Drop | 380–420 | 2–5 mm | 0.14–1.1° | 35–40° |

- **Height is well determined on every frame**, the horizon crossing included. Draft 1's statement
  that this solve "fails" at camera height was wrong for height: there `v − cy` is small, so `Y`
  is insensitive to an error in `Z`.
- **Yaw is not.** When both landmarks are near `v = cy`, the information about yaw comes from a few
  pixels of `v` difference, and any small real tilt is absorbed into yaw. In this rig that window
  covers the turnover and catch, about 50 frames (0.8 s).

**Rule:** a frame's yaw is *measured* when `σψ ≤ ψ_max`. Above that threshold it is *bridged*:
an uncertainty-weighted smoother fills it from the measured frames on either side (a
constant-angular-velocity Kalman/RTS smoother, or a Gaussian kernel weighted by `1/σψ²` in the
prototype). The output carries the inflated σ and a `yaw_bridged` flag. The height on bridged
frames is recomputed with the bridged ψ, using the fixed-yaw form below. That moves height by at
most ~10 mm on 50kg3, because height barely depends on ψ there.

```
x_i = (u_i − cx)/f,  a = (sin ψ, cos ψ)
Z_cap = L·(a_x − x_collar·a_z)/(x_collar − x_cap),  Z_collar = Z_cap + L·a_z
Y = mean_i (v_i − cy)·Z_i/f
```

The fixed-yaw form degenerates when the bar axis lies along a viewing ray (`x_collar ≈ x_cap`).
Such frames are flagged and their height is taken from the per-frame solve alone.

## 4. Tilt is assumed, not measured

θ = 0 is an assumption, and this spec states its cost rather than hiding it.

- **It cannot come from the landmarks.** One cap and one collar give four numbers for five unknowns.
- **It cannot come from depth either.** Adding collar depth as a fifth observation (σ = 150 mm) was
  tried. At rest, where the true tilt is 0°, the fit reported 2–4° of tilt and moved the centre
  height by up to 120 mm. Collar depth noise (±150 mm over a 1755 mm bar) is worth roughly ±5° of
  tilt, which is worse than the assumption.
- **Effect on outputs.** With both ends well below or above the camera, a real tilt θ biases
  height by roughly `(L/2 − 655)·sin θ` at the centre: ~4 mm per degree. Near camera height, tilt
  aliases into yaw instead, which is part of why yaw is bridged there.
- **What would measure it:** the far cap or a second landmark on the far end, a second camera, or a
  plate-ellipse fit (the plate face's ellipse encodes the axis direction). All are out of scope for
  this draft.

## 5. From "below camera" to "above floor"

These are unchanged from draft 1 §4:

- **Primary, rest-plate anchor:** `height = 225 + (Y_rest − Y)`. It assumes 450 mm bumpers.
- **Check, floor-plane anchor:** `height = floor_offset − Y`, using S3's `floor_offset_m`.
- On 50kg3 the rest-plate anchor gives 225 mm at rest by construction, and the floor-plane anchor
  gives 194 mm. The stage reports both and their gap.

## 6. Missing landmarks

On frames with the collar only (the 20 cap-missed frames on 50kg3, all after the lift), carry
`Z_collar` from the nearest solved frame, take `Y` from the collar alone, carry ψ forward, and flag
`collar_only` with inflated σ. Collar depth is a sanity bound only. At the cap, depth is unusable:
it reads the background behind the thin sleeve end.

## 7. Output

`bar_pose.json` per capture:

- **Per frame:** `height_mm`, `height_sigma_mm`, `y_below_camera_mm`, `centre_cam_mm` (X, Y, Z),
  `yaw_deg`, `yaw_sigma_deg`, `axis_cam` (unit vector, cap → collar), and `flags`
  (`yaw_bridged`, `collar_only`, `axis_along_ray`, `missing`).
- **Provenance:** `L`, cap-to-centre distance, landmark σ, `ψ_max`, smoother parameters, rest
  window, both anchors and their gap, and the checkpoint behind `predictions.json`.
- **Not smoothed in height.** Velocity and acceleration choose their own filter, and per-frame σ is
  there so they can weight by it.

## 8. Open validation items

1. **Ground truth for height.** The overhead hold reads ~1875 mm above the floor, against a rough
   2.0–2.1 m anthropometric estimate. A capture with a tape-measured bar height is needed before
   any accuracy claim.
2. **Ground truth for yaw.** Is the yaw drift on 50kg3 real or noise? It runs 37.8–39.2° at rest
   and 36–43° overhead. A capture with the bar set at a known angle (e.g. along a taped floor line,
   with the line's angle measured from the depth map) would answer it.
3. **The `ψ_max` threshold, and how many frames get bridged.** Calibrate both across all seven
   captures. If the bridged window routinely covers the whole turnover, per-frame yaw through the
   catch is not available from this camera placement, and the human-rotation consumer must know
   that.
4. **Floor vs gravity disagree by 10°** (S3 `gravity_agreement_deg`, warn-only). "Level" defines
   both height and yaw, so an error here leaks horizontal motion into height and biases yaw. Resolve
   it before trusting sub-3 cm or sub-degree numbers.
5. **The 31 mm gap between the two anchors.** Measure it across captures.
6. **Tilt bias.** Check §4's ~4 mm/deg estimate on a synthetic fixture with known tilt.
7. **Plate diameter and bar type must be declared.** Plate diameter has no metadata field, and
   `barbell_type` is dropped before S4.

## 9. Implementation outline

Pure geometry is kept separate from I/O and built test-first:

- **`solve_bar_level(obs, K, L) -> (centre, yaw)`, `bar_height_fixed_yaw(obs, K, L, yaw)`, and
  `propagate_sigma(...)`.** Synthetic fixtures project a known 3D bar through a known K. The tests
  assert exact recovery, a bar exactly at camera height (height finite, yaw flagged), and a bar
  along a viewing ray (flagged, no divide).
- **`bridge_yaw(yaw, sigma, frames, yaw_max)`.** A fixture with a known yaw ramp and an injected
  unobservable window must recover the ramp within its stated σ.
- **A task that reads `predictions.json`, S4's K and S3's sidecar, and writes `bar_pose.json`.**
  The manifest reports: rest height under both anchors, the yaw mean and spread at rest, the number
  and span of bridged frames, the number of `collar_only` frames, and peak height.

---

## Feedback (2026-10-19) — incorporated in [`S7-bar-height-draft3.md`](S7-bar-height-draft3.md)

- **User:**
  1. Where yaw cannot be calculated, fill it by linear interpolation between the two measured
     frames flanking the gap. No smoother.
  2. Don't lead with a tilt assumption. Cap and collar only *find* the bar; what is tracked is the
     bar's centre (its COM), where tilt should matter much less.
- **Simulation follow-up** (a synthetic bar at 50kg3's geometry, tilted about its centre, then
  solved with the level-bar solve):
  - With these two landmarks, tilt does **not** reduce to a lever arm about the centre. The solve
    reads the extra vertical image offset between the ends as perspective, a depth difference, and
    that corrupts the depth and hence the scale.
  - Per 1° of real tilt, the centre's height is off by **24–28 mm** in every phase tested: rest,
    pull, overhead, and near camera height. That is *more* than the cap end's own 19 mm geometric
    drop. Yaw is off by 1.6–10° per degree.
  - §4's "~4 mm per degree at the centre" estimate is therefore wrong by a factor of ~6.

---

## Renumbering note (2026-10-20)

Stages were renumbered so the barbell detector runs before pose: **S7 Barbell → S5**, **S5 Pose →
S6**, **S6 Lift → S7**. This draft is frozen, so its text still uses the old numbers. Read every
"S7" above as S5, "S5" as S6, and "S6" as S7.

Its files were renamed `S7-*.md` → `S5-*.md`, and `S5-*.md` → `S6-*.md` for the pose specs. Links
above that use the old names point to the files of the same name with the new prefix.
