# S9 Bar HVA — Barbell Velocity and Acceleration by Box-Constrained Tikhonov Regularisation

Status: **draft 2** | Supersedes: [`S9-hva-tikhonov.md`](S9-hva-tikhonov.md) (feedback at its
bottom) | Scope: GitHub issue [#63](https://github.com/tony-yehui-lou/PowerFlow/issues/63)
(Barbell HVA); joints ([#66](https://github.com/tony-yehui-lou/PowerFlow/issues/66)) follow in a
later draft | Updated: 2026-10-24

**Changes from draft 1:**

1. **The barbell first, and only.** This draft specifies the bar centre's vertical height,
   velocity and acceleration. Joints come later, reusing the same `tikhonov.py` unchanged (§13).
2. **The bar height is solved here, not read from S8.** S9 runs `bar_geometry.solve_track` and
   `floor_anchor` on S5's detections itself. It needs `--barbell`, not `--pose` or
   `--kinematics`.
3. **One series per capture.** Draft 1's open question 1 (one α per series or one per capture)
   does not arise for the bar.
4. **Every parameter is listed in one place: §9.** Each one is marked as decided by you,
   calibrated on the captures, or fixed by the theory, with the default proposed and how it is
   checked.

Carried over from draft 1:
- the acceleration-only model with Δt-weighted norms;
- snapping frames onto a uniform grid;
- the energy estimator for δ;
- the constrained discrepancy principle and its four outcomes;
- the banded ADMM with an active-set polish;
- nodal derivatives;
- the a posteriori certificates.

**Method source:** the user's paper on numerical differentiation under bounds on the first and
second derivatives (the "paper"); numbers are the paper's. Equations (22)–(26), Theorem 3.4,
Theorem 3.13 and §5 were not reviewed, and the one place this matters is marked (§8.3).

**In:**
- S5's `predictions.json`, the cap and collar boxes per frame;
- S4's output record, which carries the intrinsics and the image size, and S4's `rgb.mp4` PTS,
  for the frame times;
- S1's `cut_sidecar.json`, for the epoch of frame 0;
- S3's `retilt_sidecar.json`, for the floor-plane fallback and cross-check;
- `lift.barbell_type` from the capture's raw `metadata.yaml`.

**Out:** `bar_hva.json` per capture, at `../data/s9_hva_output/<capture path>/`. Per frame:
- the bar centre's height above the floor, purely calculated;
- its regularised height;
- its vertical velocity (m/s) and vertical acceleration (m/s²). Both are positive upward and
  kinematic: gravity is not added.

The document also carries flags, the selected parameter, the noise estimate and error bounds.

The stage number S9 is a proposal.

---

## 1. Why the bar first

- **It is the simplest series.** It is one rigid point, always visible except where S5 misses it,
  and one series per capture.
- **It is already validated as a height.** The level-bar solve was validated in powerflow-ui's
  Bar Height view: 98–100% of frames have a height on six of the seven captures.
- **It has known physics** to check the bounds against (§9.2).
- **It has a reference trajectory with the boundary conditions the paper wants.** The bar starts
  at rest on the floor, and a lift ends in a still overhead hold. Both ends then have
  `h″ = h‴ = 0`, which is Prop. 4.4's condition for the better ν = 1 rate (§8.2).
- **No pipeline stage publishes the bar's height yet.** The geometry exists in `bar_geometry.py`,
  but today only S8 uses it, to place the hands; the Bar Height view computes its own in a build
  script. `bar_hva.json` becomes the pipeline's bar-height document, and the view can later read
  it instead (Open question 4).

---

## 2. The bar height series

### 2.1 Height

Per frame `i`:
- `bars = solve_track(caps, collars, K, BAR_TYPES[barbell_type])`, the same call S8 makes;
- `anchor = floor_anchor(bars, floor_offset_mm=S3's)`, so the bar and, later, the joints share
  one floor.

Then

```text
heightM[i] = (anchor.y_floor_mm − bars[i].centre_mm[1]) / 1000     // null where bars[i] is None
```

`heightM` is **purely calculated**: no smoothing and no adjustment, the rule S8's feedback set.
It must equal the Bar Height view's height to 0.1 mm (§10.3).

### 2.2 Which frames are observations

`solve_track` labels every pose with how it was placed, in `BarPose.source`. Each label is one
class of observation:

| `source` | How the frame was placed | Error character | Used as an observation |
|---|---|---|---|
| `level` | both landmarks, yaw measured | landmark noise, roughly white | yes |
| `yaw-interpolated` | both landmarks, yaw interpolated between measured flanks | landmark noise plus a smooth yaw error | yes |
| `collar-only` | the collar alone, its depth interpolated across a cap gap of ≤ 30 frames | smooth, correlated: median 1–5 mm, p95 11–27 mm (Bar Height view) | **Decision P3** (§9.1); proposed yes, flagged |
| none | no collar, or a cap gap > 30 frames, or a gap open at either end | — | no: an empty node |

### 2.3 Frame times

- `tMs[i]` comes from S4's `rgb.mp4` PTS, and `startEpochMs` from S1's `cut_sidecar.json`. That
  is how S6 builds `frames`. The two helpers in `tasks/pose.py` (`_frame_offsets_ms` and
  `_capture_start_epoch_ms`) move to `common/frame_clock.py` so both stages share them.
- The number of PTS must equal `record.n_frames`, or the capture is not solved:
  `frame_count_mismatch`.

---

## 3. The model

### 3.1 Continuous

On a lift window `[0, T]`, the stage finds the height `h` that minimises

```text
‖h − z‖²_{L²} + α·μ_a·‖h″‖²_{L²}       subject to |h′| ≤ v_max, |h″| ≤ a_max
```

This is the paper's (3) with `λ_a = α·μ_a` and `λ_j = 0`. The paper's discrete-to-continuum
results (§3.6), boundary analysis (Prop. 4.4, Lemma 4.6) and rates (§4.1) are stated for this
acceleration-only model.

### 3.2 Discrete

Let `z ∈ ℝⁿ` be one segment's heights on the grid nodes (§4). Let `M = diag(m_k)`, with
`m_k = 1` where a node holds an observation (§2.2) and 0 otherwise. With `V = D/Δt` and
`A = D⁽²⁾/Δt²` (paper (6)):

```text
ĥ = argmin_{h ∈ K_h}  F(h) := ‖M^{1/2}(h − z)‖²_Δt + α·μ_a·‖A h‖²_Δt,      ‖u‖²_Δt = Δt·uᵀu
K_h = { h : ‖V h‖_∞ ≤ v_max,  ‖A h‖_∞ ≤ a_max }
```

This is the paper's (47) with `λ_j = 0`, restricted to the observed nodes. On a segment with no
empty node, `M = I` and the paper's §4 applies verbatim.

The weights are physical (Δt-weighted), never algebraic. Then α means the same at any frame rate
(Remark 3.14), and the constants of the stability results do not depend on the mesh.

**Empty nodes** break Assumption 2.3 (`w_min > 0`). The problem stays strictly convex once a
segment has two observed nodes: `R = μ_a AᵀA` vanishes only on affine sequences (Lemma 4.1), so
`M + αR ≻ 0`, and the minimiser is unique. An empty node is filled by the bending-energy
interpolant. The mesh-uniform certificates assume `M = I`, so they are `null` on a segment with
empty nodes (§8.3).

### 3.3 What α means: an equivalent cutoff

In a long segment's interior, `(I + αR)⁻¹` is a zero-phase filter. The eigenvalue of
`μ_a AᵀA` at angular frequency ω is `μ_a(2 sin(ωΔt/2)/Δt)⁴ ≈ μ_a ω⁴`, so the gain is

```text
g(f) ≈ 1 / (1 + (f / f_c)⁴),       f_c = 1 / (2π (α μ_a)^{1/4})
```

This is the magnitude response of a 2nd-order Butterworth run forwards and backwards. Checked at
60 fps with `f_c = 6 Hz`: a 3 Hz sinusoid passes at 0.942, against 0.941 predicted. Every
capture reports `cutoffHz` beside α. The boundary layer is about `ℓ = (α μ_a)^{1/4}` seconds wide.

---

## 4. Grid and segments

The paper needs a uniform grid. The captures are variable frame rate, with dropped frames.

1. **Δt** is the median of `diff(tMs)`, in seconds.
2. **Snap:** frame `i` goes to node `k_i = round((tMs[i] − tMs[0]) / (1000·Δt))`. Values are
   never resampled.
   - A dropped frame leaves an empty node.
   - Two frames on one node: the capture is not solved, `grid_collision`.
   - `snapErrorMs = max|tMs[i] − 1000·k_i·Δt|` is reported. The capture is warned above P10.
3. **Why snap.** At 2 m/s, a 0.5 ms timing error is a 1 mm height error. That is below the
   landmark noise, and it goes into δ. Interpolating onto a grid would instead correlate
   neighbouring noise samples, and that breaks §5's estimator.
4. **Segments:**
   - a run of empty nodes longer than P8 splits the series;
   - shorter runs stay inside the segment as empty nodes, flagged `bridged`;
   - a segment with fewer than P9 observed nodes is not solved (`too_short`).

   On six of the seven captures the bar is one segment covering 98–100% of frames. 45kg1, with
   139 of 367 frames, is the test of the gap rules.

---

## 5. Noise level δ

On every run of four consecutive observed nodes, take `(D⁽³⁾z)_k`. Its variance under
independent noise is `20σ²`. The signal adds about `Δt³h‴`: 0.46 mm at 60 fps and 100 m/s³,
which is negligible.

```text
σ̂     = sqrt( mean((D⁽³⁾z)²) / 20 )                     // energy estimator: drives δ
σ̂_rob = 1.4826 · median|D⁽³⁾z| / √20                     // robust: reported, and checked
δ     = κ · σ̂ · sqrt(n_obs · Δt)                         // κ = P5, the correlation inflation
```

- **The energy estimator drives δ.** It is unbiased for the total noise energy, isolated glitches
  included, and that energy is what the discrepancy principle compares against. The robust
  estimator ignores glitches, so `τδ` would fall below the residual they force.
- **`σ̂ / σ̂_rob > P11`** flags the capture `outliers`: the noise is heavy-tailed.
- **Correlated error is invisible to `D⁽³⁾`.** Smooth error includes collar-only depth
  interpolation and slow detector drift. The regulariser cannot remove it either, so it does not
  inflate the residual. It does make σ̂ understate the error the certificates must cover. κ (P5)
  exists for that, and is calibrated on the rest window (§9.2).

---

## 6. Choosing α: the constrained discrepancy principle

### 6.1 Rule

The stage picks `α_K` with

```text
r_K(α_K) := ‖M^{1/2}(ĥ^{α_K} − z)‖_Δt = τ δ           (paper (70), Δt-norm; τ = P4)
```

`r_K` is continuous and nondecreasing in α (Lemma 4.11, Theorem 4.12). The rule never uses the
smoothness index ν (Theorem 4.17).

### 6.2 Limits and outcomes

The two ends of the residual path:
- `r_lo = r_K(α_min)`, with `α_min` at the top of P12's cutoff bracket. This stands in for
  `r_K(0⁺) = dist(z, K_h)`.
- `r_hi = dist(z, K_h ∩ ker R)`: the best feasible straight line. Its slope is clipped to
  `±v_max` and its intercept refitted, in closed form (Theorem 4.12 (73)).

| `status` | Condition | Published |
|---|---|---|
| `box_inactive` | the unconstrained Morozov fit `h_U^{α_M}` is feasible | `h_U^{α_M}`, since `α_K = α_M` (Theorem 4.12, last claim; Prop. 3.3) |
| `morozov` | `r_lo < τδ < r_hi`, with the box active | `ĥ^{α_K}` (§6.3) |
| `data_outside_bounds` | `τδ ≤ r_lo` | velocity and acceleration `null`, warning. Even the least-regularised feasible fit leaves more residual than the noise explains: the bounds are too tight, δ is too small, or the track has a defect. |
| `noise_dominates` | `τδ ≥ r_hi` | the feasible line, flagged. For the bar this means a static clip or a broken δ. |

### 6.3 Search

1. **Unconstrained Morozov.** Bracket `r(α)` in `log α` over P12 and bisect to `α_M`. Each
   evaluation is one banded solve. If `h_U^{α_M} ∈ K_h`, stop: `box_inactive`. With the bounds
   calibrated as in §9.2, this is the expected case on clean captures.
2. **Constrained.** Otherwise bisect `r_K` in `log α`, warm-starting each QP from the previous
   α's solution and multipliers.
3. **Stop** when `|r_K − τδ| ≤ P13·τδ`, or when the bracket's ratio is below `1 + P13`. Report the
   smallest α that meets the target. By Lemma 4.11 every root gives the same `ĥ`.
4. **Cap.** At most P14 evaluations, or the capture is flagged `alpha_unconverged`. A monotone
   continuous `r_K` cannot hit this, so the flag catches solver failure.

---

## 7. The constrained solve

### 7.1 Banded ADMM, no new dependency

Write `P = M + αR` and `q = Mz` (that is, `F/(2Δt)`). The constraints use the scaled operators,
which give the same polyhedron with better conditioning:

```text
G = [ D ; D⁽²⁾ ],   u = [ v_max·Δt·1 ; a_max·Δt²·1 ],   l = −u

repeat:
  (P + σI + ρ GᵀG) x̃ = σ x − q + Gᵀ(ρ w − y)     // pentadiagonal SPD: cholesky_banded per (α, ρ)
  w̃ = G x̃
  x ← x̃
  w ← clip(w̃ + y/ρ, l, u)
  y ← y + ρ (w̃ − w)
until ‖G x − w‖_∞ ≤ ε_pri  and  ‖P x − q + Gᵀ y‖_∞ ≤ ε_dual          // P15
```

- `scipy.linalg.cholesky_banded` / `cho_solve_banded` make each iteration O(n).
- ρ adapts as in OSQP; the matrix is refactored only when ρ changes by more than 5×.
- **Polish.** Take the active set `𝒜 = {i : y_i ≠ 0}` and solve the KKT system
  `[P G_𝒜ᵀ; G_𝒜 0][h; λ] = [q; b_𝒜]` exactly. Accept it if the multiplier signs are right and the
  inactive rows are strictly feasible; that is Prop. 3.9's local affine structure. Otherwise keep
  the iterate and flag `unpolished`.
- **Feasibility is a hard gate.** Published values satisfy `‖Vĥ‖_∞ ≤ v_max(1 + P16)` and the same
  for `a`, or the capture fails with `solver_infeasible`.
- **Cost:** about 430 nodes and at most P14 evaluations. Well under a second per capture.
- **Alternative:** `osqp` or `qpsolvers`, which needs the user's approval (`CLAUDE.md`). It buys
  nothing at this size and band structure.

### 7.2 From nodes back to frames

Frames sit one-to-one on observed nodes, so no values are resampled.

| Output at frame `i` (node `k`) | Value |
|---|---|
| `heightRegM` | `ĥ_k`. `null` on `bridged` nodes: a height is never stated where none was observed. |
| `velocityMps` | `(ĥ_{k+1} − ĥ_{k−1}) / (2Δt)`, one-sided at segment ends. This is the mean of two staggered `v`, so it obeys `v_max`. |
| `accelMps2` | `(Aĥ)_{k−1}`, the **nodal** second difference centred at `t_k`; the nearest at the ends. Not `π_n A_Δt ĥ`, whose last cell is zero (Lemma 4.6); this is Corollary 4.8's quantity. |

Flags per frame:
- `bridged`;
- `edge`: within `⌈ℓ/Δt⌉` frames of a segment end;
- `v_active` / `a_active`;
- `collar_only`, copied from §2.2.

---

## 8. Diagnostics and certificates

### 8.1 Per segment

`alpha`, `cutoffHz`, `sigmaM`, `sigmaRobustM`, `kappa`, `deltaM`, `tau`, `residualM`, `rLowM`,
`rHighM`, `status`, `activeV`, `activeA`, `solverIterations`, `polished`,
`bendingEnergy = ‖Aĥ‖_Δt`, and:

- **`dAlpha = ‖h_U^{α_K} − ĥ^{α_K}‖_{M_α}`:** the observable part of what the box removed
  (Corollary 3.7). It is 0 on `box_inactive`.

### 8.2 Endpoint check: which rate applies

Prop. 4.4: acceleration converges at the ν = 1 rate only if `h″ = h‴ = 0` at both ends. Otherwise
it falls to ν = ½, which gives no rate for the acceleration. The bar starts at rest. Whether it
*ends* at rest depends on where the operator ended the lift window.

`endpointAccelMps2` = `|â|` at the first and last node. A value above P17 warns that the window
was cut mid-motion. This is the bar-specific reason for the operator rule in §12: **end the lift
window in the overhead hold, before the drop.**

### 8.3 Certificates

These hold if the truth is feasible and `‖ε‖_Δt ≤ δ`. They are published as RMS values, the
Δt-norm divided by `√T_seg`. They are `null` on segments with empty nodes (§3.2).

```text
heightErrorBoundRmsM   = (1 + τ) δ / √T_seg                                      (Theorem 4.14, (74))
B_obs = [ (r_U(α) + δ)² + α (E_U(α) + √(μ_a T) a_max)² ]^{1/2}
          r_U(α) = ‖h_U^α − z‖_Δt,   E_U(α) = ‖R^{1/2} h_U^α‖_Δt
accelErrorBoundRmsMps2 = [ (B_obs² − dAlpha²) / (α μ_a) ]^{1/2} / √T_seg          (Corollary 3.8, (35))
```

- **Bending energy (75):** the selected fit's bending energy never exceeds the truth's, so
  `bendingEnergy` is a lower bound on how much the true trajectory bends.
- **Not reviewed:** the paper defines `‖·‖_{M_α}` in (24)–(26). This spec takes
  `‖x‖²_{M_α} = ‖x‖²_Δt + α‖R^{1/2}x‖²_Δt`, read off Theorem 4.7's proof. **The acceleration
  certificate does not ship until this is confirmed** (decision P18).

---

## 9. Parameters to fix

Every number the stage uses is listed here, grouped by **who fixes it**.

### 9.1 Decide now (yours)

| # | Parameter | Config / symbol | Proposed | The choice | What it changes |
|---|---|---|---|---|---|
| **P1** | Velocity bound | `bar_hva_v_max_mps`, `v_max` | **4.0 m/s** | Peak bar velocity in a snatch is about 2 m/s, so 4 m/s is a 2× margin. The drop is excluded by the lift window (§12); free fall from 2 m would need 6.3 m/s. | Theorem 4.14 needs the true bar inside the bounds. Too tight clips real motion and breaks the certificates; too loose and the box never helps. §9.2 checks it on data. |
| **P2** | Acceleration bound | `bar_hva_a_max_mps2`, `a_max` | **50 m/s²** | Above the bar's acceleration through the second pull and the catch, without measured ground truth. | Same as P1. It also enters the acceleration certificate directly, through `√(μ_a T)·a_max` in `B_obs`, so a loose `a_max` loosens that bound. |
| **P3** | Collar-only frames | `bar_hva_collar_only` | **observe** | `observe`: weight 1, flagged `collar_only`. `mask`: empty nodes, filled by the bending interpolant, with no certificates on that segment. | Their error is smooth (1–5 mm median), which the regulariser can neither see nor remove. `mask` throws away height evidence; `observe` keeps the certificates. Applies mainly to 45kg1. |
| **P4** | Discrepancy factor | `bar_hva_tau`, `τ` | **1.1** | The theory needs `τ ≥ 1`. `‖ε‖_Δt` fluctuates by about `1/√(2·n_obs)`, which is 3.5% at 430 frames, so 1.1 is a 3σ margin. | A larger τ smooths more: peaks are blunted more, and noise is suppressed more. |
| **P6** | Jerk weight | `μ_j` | **0** | Acceleration-only; the theory in §3.6 and §4.1 covers only this case. `μ_j > 0` would penalise jerk too. | Deferred until acceleration curves show staircase artefacts. |
| **P18** | Ship the acceleration certificate? | — | **after (24)–(26) are checked** | Publish `accelErrorBoundRmsMps2` now, as provisional, or hold it as `null`. | Whether the UI can show an error band on acceleration. |

### 9.2 Calibrate on the seven `10 September` captures (50kg3 held out)

| # | Parameter | Config / symbol | Starting value | Calibration procedure | Accept when |
|---|---|---|---|---|---|
| **P1, P2** | Bounds, checked | — | 4.0 m/s, 50 m/s² | Run the unconstrained Morozov fit (bounds off) on the six training captures. Take the max `|v|` and `|a|` over non-`edge`, non-`bridged` frames. Set each bound to ≥ 1.5× that max, rounded up. | Every training capture is `box_inactive`, and 50kg3 is too. Any capture that isn't is read by hand. |
| **P5** | Correlation inflation | `bar_hva_kappa`, `κ` | 1.0 | In the rest window (the first 60 frames, bar on the floor), compute the std of `heightM` about its mean, divided by σ̂. Take κ as that ratio's median across captures, floored at 1. | The ratio is stable across captures (spread < 20%). If it is not, the noise model is wrong, and κ is not the fix. |
| **P7** | Noise estimator order | — | `D⁽³⁾` | Compare σ̂ from `D⁽³⁾` and from `D⁽⁴⁾` (variance 70σ²) on every capture. | They agree within 10%. If they don't, signal is leaking into `D⁽³⁾`, so switch to `D⁽⁴⁾`. |
| **P11** | Outlier ratio | — | 1.5 | Histogram `σ̂/σ̂_rob` across captures, then plant 1–5 synthetic glitches of 30–100 mm. | Clean captures fall below it and planted glitches above it. |
| **P17** | Endpoint warning | — | `0.1·a_max` | Read `endpointAccelMps2` on every capture against its video: is the start and end really still? | It flags exactly the windows cut mid-motion. |
| — | **Resulting `cutoffHz`** | — | — | Report per capture, against the 4–10 Hz bar-velocity literature and Winter's 6 Hz. | All seven in a plausible band, and similar across captures of the same lift. A capture far outside is read by hand. Nothing is tuned to force this. |

### 9.3 Fixed by the theory or by convention (no decision)

| # | Parameter | Value | Why it is not a choice |
|---|---|---|---|
| — | Physical weighting | Δt-norms (42) | Remark 3.14 and §4.2's opening: algebraic weights make α depend on the mesh. |
| — | `μ_a` | 1 s⁴ | It only fixes the unit of α. Only `αμ_a` appears. |
| — | `W` | `M`: 1 observed, 0 empty | §4 of the paper is for `W = I`. Per-sample weights (`W ≠ I`) are draft 3's question, not v1's. |
| — | δ estimator | energy, not robust | §5: the discrepancy principle compares against total noise energy. |
| — | Acceleration output | nodal (Corollary 4.8) | It avoids Lemma 4.6's zeroed last cell. |
| — | Grid Δt | median `diff(tMs)` | It needs no choice, and it maps one frame to one node. |
| P8 | Max bridged gap | 6 frames (100 ms) | Carried over from S8's earlier smoother. It only decides where a series splits, and §4 shows the bar rarely has such gaps. *Revisit only if 45kg1 says so.* |
| P9 | Min segment | 30 observed frames (0.5 s) | Below this, the discrepancy principle has too few samples for δ (`1/√(2n)` > 13%). |
| P10 | Snap warning | `Δt/4` | Beyond it, the frame clock is not the uniform grid this method assumes. |
| P12 | α bracket | `f_c ∈ [0.1, 30] Hz` | It spans from almost a straight line to the Nyquist range at 60 fps. Hitting either end is reported. |
| P13 | Bisection tolerance | 10⁻³ | Far below δ's own uncertainty (3.5%). |
| P14 | Max α evaluations | 60 | Bisection over 2.5 decades at 10⁻³ needs about 25. |
| P15 | ADMM tolerances | `ε_abs = 10⁻⁷ m`, `ε_rel = 10⁻⁶`, ≤ 4000 iterations | The polish step brings the answer to machine precision anyway; these only find the active set. |
| P16 | Feasibility tolerance | 10⁻⁹ (relative) | Floating point only. |
| — | Floor anchor | `floor_anchor` (60-frame rest window, 225 mm plate radius) | Shared with S8, so bar and joints use one floor. |
| — | Detection score | `barbell_min_score` = 0.5 | Already S5's and S8's (false caps scored 0.31–0.33). |

---

## 10. Validation

### 10.1 Synthetic (unit tests, test-first)

The reference is a **bar-like profile** built from C⁴ pieces with zero `h′`, `h″` and `h‴` at both
ends:
- 1 s at rest at 225 mm;
- a pull to about 1.1 m, peaking at about 1.9 m/s;
- a 0.15 m catch dip;
- recovery to about 1.9 m;
- a 1 s overhead hold.

It is sampled with 16/17 ms jitter and Gaussian noise of σ ∈ {1, 2, 4, 8} mm.

- **Operators:**
  - `‖D‖, ‖D⁽²⁾‖, ‖D⁽³⁾‖ ≤ 2, 4, 8` (Lemma 2.2);
  - `ker R = span{1, t}` (Lemma 4.1);
  - the banded `P` equals the dense formula;
  - `h_U = (I + αR)⁻¹z`;
  - the interior gain matches `1/(1+(f/f_c)⁴)` within 1% at 1, 3, 6 and 10 Hz.
- **Mesh uniformity:** at fixed α, sampling at 30, 60 and 120 fps gives `ĥ` and `â` that agree at
  shared times to O(Δt²) (Remark 3.14).
- **Prop. 3.3:** with the box inactive, the QP returns `h_U` with no iterations.
- **KKT (21):** on fixtures with active bounds, every residual is below 10⁻⁸. The solution matches
  a dense SLSQP reference (n = 40, test-only) to 10⁻⁷.
- **Theorem 3.10:** random pairs `z, z′` respect (37) and (39).
- **Lemma 3.15:** the sampled bar profile is in `K_h` on every grid.
- **Lemma 4.11:** `r_K` is nondecreasing and `‖R^{1/2}ĥ‖` nonincreasing, on a 200-point α grid.
- **Theorem 4.12:** `r_K` tends to the closed-form `r_hi`, with one fixture that needs a clipped
  slope.
- **Each `status`** of §6.2 has a fixture.
- **Theorem 4.14 / 4.17:** with `‖ε‖_Δt = δ` exactly:
  - (74) and (75) hold;
  - acceleration error falls at least as `δ^{1/2}` over the four σ.
  - The same profile cut mid-pull shows the slower rate and fires P17's warning.
- **Noise:**
  - σ̂ is unbiased within 5%;
  - three 50 mm glitches fire `outliers`, yet the capture still reaches
    `box_inactive`/`morozov`, not `data_outside_bounds`.
- **Grid:**
  - jittered timestamps snap with `snapErrorMs ≤ 1`;
  - a dropped frame is `bridged`, with no `heightRegM`;
  - a 10-frame gap splits the series;
  - a duplicate PTS gives `grid_collision`.
- **Peaks:** on the profile at σ = 4 mm, the peak-velocity error is below 0.05 m/s and the
  peak-time error is below 1 frame. These numbers are reported in the test output, so the
  blunting is measured, not assumed.

### 10.2 Integration

- The flow under `prefect_test_harness`, with a synthetic `predictions.json`, writes
  `bar_hva.json`.
- A dry run plans the write without solving.
- With `--barbell` missing, `--bar-hva` is refused at config time.

### 10.3 Real data (the seven `10 September` captures, 50kg3 held out)

| Check | Expectation |
|---|---|
| `heightM` against the Bar Height view's height | equal within 0.1 mm: the same method |
| Rest window: V and A | `|V|` < 0.05 m/s and `|A|` < 1 m/s², median over the window |
| `∫V dt` against `ΔheightRegM` | below 5 mm drift |
| Peak velocity in the pull | 1.5–2.5 m/s on every snatch |
| `status` | `box_inactive` on all clean captures (it is the P1/P2 acceptance) |
| `cutoffHz` | §9.2's last row |
| Later, against the joints | near-wrist V against bar V, once joints get HVA |

---

## 11. Code layout

Pure maths is kept apart from I/O (`CLAUDE.md`).

- **`preprocess/tikhonov.py` (new, pure, numpy/scipy):** bar-agnostic, so the joints reuse it as
  is.
  - `snap_to_grid`, `segments`, `estimate_noise`;
  - `unconstrained`, `solve_box` (ADMM and polish, returning KKT residuals);
  - `feasible_line_distance`, `discrepancy`, `nodal_derivatives`, `certificates`.

  Constants P8–P16 are module constants, documented where they are defined.
- **`preprocess/bar_hva.py` (new, pure):** `build(Inputs) -> Built`. It turns the bar poses,
  floor anchor and frame clock into `bar_hva.json`, its manifest values and its warnings.
- **`preprocess/tasks/bar_hva.py` (new `@task(retries=1)`):**
  - reads the inputs of the header, reusing `kinematics._bar_track` (moved to `barbell_input.py`)
    and `kinematics._floor_offset_mm` (moved alongside it);
  - writes `bar_hva.json` by staging, then publishing.

  A capture it cannot solve is never rejected: S0–S4 have already published. The reason is a
  manifest warning.
- **`common/frame_clock.py` (new):** `_frame_offsets_ms` and `_capture_start_epoch_ms`, moved
  from `tasks/pose.py`, which then imports them.
- **`flow.py`:** one guarded step after crop, `if config.bar_hva_root is not None`. It does not
  depend on S6–S8.
- **`config.py` / `cli.py`:**
  - `--bar-hva <root>` requires `--barbell`;
  - the flow raises if `bar_hva_root` is set without `barbell_root`;
  - fields for P1–P5: `bar_hva_v_max_mps`, `bar_hva_a_max_mps2`, `bar_hva_collar_only`,
    `bar_hva_tau`, `bar_hva_kappa`.

## 12. Output contract

`bar_hva.json`, written once per capture (replaced only with `--overwrite`):

```jsonc
{
  "schemaVersion": "1.0.0",
  "captureId": "<id>",
  "method": "docs/specs/preprocessing/S9-hva-tikhonov-draft2.md",
  "frames": { "startEpochMs": 0, "tMs": [ /* from S4's PTS */ ] },
  "grid": { "dtS": 0.0167, "nodes": 433, "snapErrorMs": 0.6, "droppedFrames": [] },
  "settings": { "vMaxMps": 4.0, "aMaxMps2": 50.0, "collarOnly": "observe", "tau": 1.1,
                "kappa": 1.0, "muA": 1.0, "muJ": 0.0 },
  "bar": { "type": "male IWF", "capToCentreM": 1.1, "capToCollarM": 1.755 },
  "floor": { "anchor": "rest-plate | floor-plane | none", "yFloorM": 0.0, "restFrames": 60,
             "s3FloorPlaneYM": 0.0 },
  "com": {
    "heightM":     [ "float | null" ],     // purely calculated, §2.1
    "heightRegM":  [ "float | null" ],     // regularised, §7.2
    "velocityMps": [ "float | null" ],
    "accelMps2":   [ "float | null" ],
    "source":      [ "level | yaw-interpolated | collar-only | null" ],
    "flags":       [ [ "bridged | edge | v_active | a_active | collar_only" ] ],
    "segments": [ {
      "startFrame": 0, "endFrame": 432,
      "status": "box_inactive | morozov | data_outside_bounds | noise_dominates | too_short | grid_collision | solver_infeasible | alpha_unconverged",
      "alpha": 0.0, "cutoffHz": 0.0,
      "sigmaM": 0.0, "sigmaRobustM": 0.0, "deltaM": 0.0,
      "residualM": 0.0, "rLowM": 0.0, "rHighM": 0.0,
      "activeV": 0, "activeA": 0, "dAlpha": 0.0, "bendingEnergy": 0.0,
      "polished": true, "solverIterations": 0, "endpointAccelMps2": [0.0, 0.0],
      "heightErrorBoundRmsM": "float | null",
      "accelErrorBoundRmsMps2": "float | null"     // P18
    } ]
  }
}
```

- **Names follow powerflow-ui's `datatypespec.md`:** `Barbell.com` holds `heightM`, `velocityMps`
  and `accelMps2`, so a Graph `series` of `"bar.com.velocityMps"` resolves directly.
- **Manifest `derived`:** `bar_hva_status`, `bar_hva_cutoff_hz`, `bar_hva_sigma_mm`,
  `bar_hva_peak_velocity_mps` (and its time), `bar_hva_active_frames`, `bar_hva_snap_error_ms`,
  and the floor anchor's method.
- **Operator rule** (README `metadata.yaml` section): start the lift window with the bar at rest,
  which the floor anchor already needs. End it in the overhead hold, **before the drop**. A drop
  breaks P1 and P2 and is not lift data.

## 13. Later: joints

Draft 3 applies the same `tikhonov.py` to each joint's `heightM` from S8's `joint_hva.json`. What
it adds:
- per-series or shared α (draft 1's open question 1);
- per-class bounds (feet against the bar);
- per-sample weights from `heightSigmaM` (`W ≠ I`);
- the wrist-against-bar velocity check.

None of these changes the bar's output.

## 14. Risks

- **Correlated error under-covers the certificates.** Collar-only interpolation and detector
  drift are invisible to `D⁽³⁾`. κ (P5) is the mitigation, measured on the rest window.
- **Tilt is assumed zero** in the height itself: about 25 mm per degree (S5 draft 5). This error
  is smooth and lift-phase dependent. Regularisation cannot see it, and it biases V and A where
  the tilt changes (the turnover).
- **A bound the truth breaks** clips a real peak and voids Theorem 4.14. `a_active` shows where.
  A peak at an active bound is a lower bound, and the UI should say so.
- **Unreviewed pages** gate one output field (P18).

## Open questions

1. **P1, P2, P3, P4 and P18** (§9.1): confirm or change them.
2. **The catch:** is the bar's deceleration at the catch a sharp event that `a_max` = 50 m/s²
   might clip? §9.2's calibration answers this for these seven captures only.
3. **45kg1** (bar on 139 of 367 frames, no cap at rest): solved as several segments on the S3
   floor, or excluded from calibration? Proposed: solved, and excluded from §9.2's statistics.
4. **The Bar Height view** should read `bar_hva.json` once this lands, instead of its own build
   script. That is a separate powerflow-ui change.

---

## Feedback (2026-10-24)

- **User:** P1 `v_max` = **5.0 m/s** and P2 `a_max` = **25 m/s²**. Every other §9.1 parameter
  as proposed: P3 `observe`, P4 τ = 1.1, P6 μ_j = 0, P18 hold the acceleration certificate.
- **Renumbered S9 → S10** (see the note at the bottom of draft 1). Its home is the `analyze`
  pipeline, beside S8 and S9:
  - `analyze/tikhonov.py`, `analyze/bar_hva.py`, `analyze/tasks/bar_hva.py`;
  - `powerflow analyze --barbell … --bar-hva <root>`, stage `s10` in `--stages`;
  - `AnalyzeConfig.bar_hva` holds P1–P5; `powerflow all` writes `s10_hva_output`.
  - It reuses S8's `bar_track` and S3 floor reader, and `common/capture_timing.py` for the
    frame clock, instead of §11's moves.
  - It needs `--barbell` and neither S8 nor S9. When it runs, the analyze manifest sits in its
    root, the last stage run.
- **Implemented**, with two deviations from the text, each measured on the synthetic lift:
  1. **Grid spacing (§4.1).** Rounded 60 fps PTS run 17, 17, 16 ms, so the median interval
     (17 ms) puts a 4 s clip about 80 ms adrift and collides nodes. Δt is the **mean of the
     consecutive intervals** (those ≤ 1.5× the median), refined by least squares against the
     node numbers.
  2. **Endpoint check (§8.2, P17).** The fitted acceleration obeys the bending energy's natural
     boundary condition, `ĥ″ = 0` at both ends, so it cannot see a mid-motion cut: on a lift cut
     while accelerating at 6 m/s², the fit's endpoint read 0.47 m/s². P17 is measured on the
     **data**: a local quadratic least-squares fit to the observations in the first and last
     0.25 s, published as `endpointAccelMps2` with `endpointAccelSeMps2`. It warns above both
     `0.1·a_max` and 3 standard errors. On that lift it read 3.9 m/s², the window mean of the
     true acceleration.
- **Also measured:**
  - **§10.1's 1% gain check holds only well below Nyquist.** At 6 and 10 Hz on a 60 fps grid,
    the gain is the discrete symbol's, `1/(1 + α μ_a (2 sin(ωΔt/2)/Δt)⁴)`; the test checks that
    exactly, and the continuous form at 1 and 3 Hz.
  - **A slope beyond `v_max` everywhere pins every fit to the feasible line.** When the
    least-squares slope exceeds `v_max`, the clipped line saturates every velocity row, so it is
    the solution at every α.
  - **The KKT polish needs iterative refinement at large α,** where `M + αR` is badly
    conditioned. Without it, active rows came back 10⁻⁸ past their bound, and the polish never
    settled.
  - **A bar placed on no frame now warns.** The synthetic S0–S4 fixture's camera is scaled
    1/30, so 2 px of landmark noise leaves no yaw measurable, and S5's bar is placed on no frame.
- **Not yet run on real captures:** §9.2's calibration and §10.3's checks need the
  `10 September` data.
