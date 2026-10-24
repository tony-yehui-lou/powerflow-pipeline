# S9 HVA — Velocity and Acceleration by Box-Constrained Tikhonov Regularisation

Status: **draft 1** | Fills: [`S8-joint-hva-draft3.md`](S8-joint-hva-draft3.md) §5 (placeholder)
| Scope: GitHub issues [#66](https://github.com/tony-yehui-lou/PowerFlow/issues/66) (Joint HVA)
and [#63](https://github.com/tony-yehui-lou/PowerFlow/issues/63) (Barbell HVA) | Updated: 2026-10-24

**Method source:** the user's paper on numerical differentiation of noisy samples under bounds on
the first and second derivatives (the "paper" below). Equation and theorem numbers are the paper's.
Sections §2–§4.2 were reviewed; (22)–(26), Theorem 3.4, Theorem 3.13 and §5 were not, and the one
place this spec depends on them is marked (§6.3).

**In:**
- S8's `joint_hva.json`: `frames` (`startEpochMs`, `tMs[]`), per joint `heightM[]`, `source[]`
  and `flags[]`, and `floor.yFloorM`;
- S5's bar track (`bar_geometry.solve_track`, as S8 reads it), placed on S8's floor.

**Out:** `hva.json` per capture. For every joint and the bar centre, per frame:
- the regularised height;
- the vertical velocity in m/s and the vertical acceleration in m/s², both positive upward and
  kinematic (gravity is not added);
- flags.

Each series also gets its selected parameter, noise estimate, residual diagnostics and
a posteriori error bounds. S8's `heightM[]` is **not touched**. It stays the purely calculated
height, as draft 3's feedback requires.

The stage number S9 is a proposal.

---

## 1. Why a separate stage

- **Write once.** S8's feedback fixed `joint_hva.json` as raw heights with no regularised value.
  A separate stage writes its own document to its own root, `../data/s9_hva_output/`, and reads
  S8's output. Neither document is rewritten.
- **One smoother for bar and joints.** Bar and joint curves are compared against each other, so
  they must be differentiated by the same method under the same parameter rule. Running both
  through one stage makes that true by construction. S8's open question (draft 1, "should the
  bar's HVA move here") is answered: yes, here.
- **One logical step per `@task`** (`CLAUDE.md`). Regularisation is not part of solving the
  skeleton.

---

## 2. The problem solved per series

A *series* is one scalar height track: one joint's `heightM`, or the bar centre's height. Every
series is solved independently, with the same rules.

### 2.1 Continuous model

On a lift window `[0, T]`, find the height `h` that minimises

```text
‖h − z‖²_{L²} + α·μ_a·‖h″‖²_{L²}        subject to |h′| ≤ v_max, |h″| ≤ a_max,
```

This is the paper's (3) with `λ_a = α·μ_a` and `λ_j = 0`.

- **`μ_j = 0` in v1.** The paper's discrete-to-continuum results (§3.6), boundary analysis
  (Prop. 4.4, Lemma 4.6) and rates (§4.1) are all stated for the acceleration-only model. The
  jerk term is a later option (Open question 4).
- **`μ_a = 1 s⁴`.** It only fixes the unit of `α`, so `α` is in s⁴. See §2.4 for what a value
  means.
- **Vertical only.** The bounds act on the height. Horizontal bar path is out of scope (Open
  question 6).

### 2.2 Grid

The paper assumes a uniform grid. Captures are variable frame rate (16/17 ms jitter at ~60 fps),
with dropped frames. The stage therefore **snaps frames to nodes**; it never resamples values:

1. `Δt` is the median of `diff(tMs)` over the capture, in seconds.
2. Frame `i` goes to node `k_i = round((tMs[i] − tMs[0]) / (1000·Δt))`.
3. A dropped frame leaves an empty node.
4. Two frames on one node is a defect: the series is not solved, with reason `grid_collision`.
5. `snapErrorMs = max |tMs[i] − 1000·k_i·Δt|` is reported. The capture is warned if it exceeds
   `Δt/4`.

**Why snap rather than interpolate.** At 2 m/s, a 0.5 ms timing error is a 1 mm height error. That
is below the keypoint noise and is absorbed into δ (§3). Interpolating onto a grid would instead
correlate neighbouring noise samples, and that breaks the noise estimator (§3.1).

### 2.3 Discrete problem

Let `z ∈ ℝⁿ` be the heights on the nodes of one *segment* (§2.5). Let `M = diag(m_k)` be the
observation mask: `m_k = 1` when node `k` has a height, 0 otherwise. With
`V = D/Δt` and `A = D⁽²⁾/Δt²` (paper (6)), the stage solves

```text
ĥ = argmin_{h ∈ K_h}  F(h) := ‖M^{1/2}(h − z)‖²_Δt + α·μ_a·‖A h‖²_Δt
K_h = { h : ‖V h‖_∞ ≤ v_max, ‖A h‖_∞ ≤ a_max }
```

Here `‖u‖²_Δt = Δt·uᵀu` (paper (42)). This is the paper's (47) with `λ_j = 0`, restricted to the
observed nodes. On a segment with no empty node, `M = I` and the paper's §4 applies verbatim.

The weights are in physical (Δt-weighted) form, never in the algebraic form of (11). Then:
- `α` has the same meaning at 30, 60 and 120 fps (Remark 3.14);
- the stability constants of §3.4 do not depend on the mesh.

The paper's §4.2 opening explains why the `Δt⁴`-absorbing scaling is not used.

**Empty nodes** (`m_k = 0`) break Assumption 2.3 (`w_min > 0`). The problem stays strictly
convex: `R` vanishes only on affine sequences (Lemma 4.1), so `M + αR ≻ 0` once a segment has two
observed nodes. The minimiser is still unique, and an empty node is filled by the bending-energy
interpolant. The paper's mesh-uniform certificates (§6.3) assume `M = I`. A segment with empty
nodes therefore publishes them as `null`.

### 2.4 What α means: an equivalent cutoff

In the interior of a long segment, the unconstrained estimator `(I + αR)⁻¹z` is a zero-phase
filter. For `R = μ_a AᵀA`, the eigenvalue at angular frequency `ω` is
`μ_a(2 sin(ωΔt/2)/Δt)⁴ ≈ μ_a ω⁴`, so the gain is

```text
g(f) ≈ 1 / (1 + (f / f_c)⁴),      f_c = 1 / (2π (α μ_a)^{1/4}).
```

This is the magnitude response of a 2nd-order Butterworth low-pass run forwards and backwards,
which is the filter the original S8 draft recommended at 6 Hz. Checked numerically at 60 fps with
`f_c = 6 Hz`: a 3 Hz sinusoid comes through at 0.942, against 0.941 predicted. Every series
reports its `cutoffHz` alongside `α`, so the parameter can be read in the biomechanics
literature's units. The boundary layer, where this picture fails, is about
`ℓ = (α μ_a)^{1/4}` seconds wide (§5.3).

### 2.5 Segments

The heights are split into segments before solving:
- A run of empty nodes longer than `hva_max_gap_frames` (default 6, i.e. 100 ms) splits the
  series.
- Shorter runs stay inside a segment as empty nodes, flagged `bridged`.
- A segment with fewer than `hva_min_segment_frames` observed nodes (default 30, i.e. 0.5 s) is
  not solved: its frames get `null` and the reason `too_short`.

---

## 3. Noise level δ

The discrepancy principle needs δ, a bound on `‖ε‖_Δt` for the segment's data error.

### 3.1 Estimator

On every run of four consecutive observed nodes, take the third difference `(D⁽³⁾z)_k`. For
independent noise of variance σ², its variance is `(1+9+9+1)σ² = 20σ²`. The signal's
contribution is about `Δt³h‴`. At 60 fps with a jerk of 100 m/s³, that is 0.46 mm, against 3 mm
noise, so it is negligible. The estimates are:

```text
σ̂     = sqrt( mean((D⁽³⁾z)²) / 20 )                       // energy estimator: drives δ
σ̂_rob = 1.4826 · median|D⁽³⁾z| / √20                       // robust: reported, and checked
δ     = σ̂ · sqrt(n_obs · Δt)                              // E‖ε‖_Δt for white noise
```

**δ uses the energy estimator, not the robust one.** The discrepancy principle compares the
residual with the *total* noise energy, and an isolated keypoint glitch is part of that energy.
The mean-square estimator is unbiased for it. The robust one ignores the glitch, so it would
report `τδ` below the residual the glitch forces (§4.2 case 1). When `σ̂ / σ̂_rob > 1.5` the series
is flagged `outliers`: its noise is heavy-tailed, and the white-noise reading of δ is optimistic
for most frames.

### 3.2 τ

The default is `τ = 1.1`. `‖ε‖_Δt` fluctuates about its mean with relative standard deviation
about `1/√(2·n_obs)`. That is 3.5% at 430 frames, so τ = 1.1 covers it with margin. The paper
needs `τ ≥ 1` (Theorem 4.14).

### 3.3 Correlated noise: the main risk to δ

Keypoint errors from a detector run on near-identical consecutive frames may be correlated in
time. `D⁽³⁾` cancels slowly varying error, so σ̂ would then underestimate the true error, and the
fit would under-smooth. §7.3's rest-window check measures this directly. If it is confirmed, δ
gets an inflation factor calibrated on rest windows (Open question 2).

---

## 4. Parameter choice: the constrained discrepancy principle

### 4.1 Rule

The stage picks `α_K` such that

```text
r_K(α_K) := ‖M^{1/2}(ĥ^{α_K} − z)‖_Δt = τ δ         (paper (70), Δt-norm)
```

`r_K` is continuous and nondecreasing in α (Lemma 4.11, Theorem 4.12). The rule never uses the
smoothness index ν, which enters only the analysis (Theorem 4.17).

### 4.2 Limits and the four outcomes

The two ends of the residual path:
- `r_lo = r_K(α_min)`, with `α_min` at `f_c = 30 Hz`. This stands in for `r_K(0⁺) =
  dist(z, K_h)`.
- `r_hi = dist(z, K_h ∩ ker R)`: the best **feasible straight line**. On a line `a ≡ 0`, so the
  only constraint left is `|slope| ≤ v_max`. Its distance is computed in closed form: a weighted
  least-squares line, with its slope clipped to `±v_max` and the intercept refitted (Theorem 4.12
  (73)).

The outcome is recorded in each segment's `status`:

| `status` | Condition | What is published |
|---|---|---|
| `morozov` | `r_lo < τδ < r_hi` | `ĥ^{α_K}`, with `α_K` found by §4.3 |
| `box_inactive` | as `morozov`, and the unconstrained Morozov fit is already feasible | `h_U^{α_M}`: `α_K = α_M` (Theorem 4.12, last claim; Prop. 3.3) |
| `data_outside_bounds` | `τδ ≤ r_lo` | `null` velocity and acceleration, warning. Even the least-regularised feasible fit leaves more residual than the noise explains: the bounds are too tight, δ is too small, or the track has a defect. |
| `noise_dominates` | `τδ ≥ r_hi` | the feasible line, flagged. The estimated noise swamps every non-affine motion: a static joint, or an overestimated δ. |

### 4.3 Search

1. **Unconstrained Morozov first.** `r(α) = ‖M^{1/2}(h_U^α − z)‖_Δt` needs one banded solve per α.
   Bracket it in `log α` over `f_c ∈ [0.1, 30] Hz` and bisect to `α_M`. If `h_U^{α_M} ∈ K_h`,
   stop: status `box_inactive`. With physically generous bounds (§5.1) this should be the usual
   case.
2. **Otherwise, bisect `r_K` in `log α`** on the same bracket. Each evaluation is one QP solve,
   warm-started from the previous α's solution and multipliers.
3. **Stop** when `|r_K − τδ| ≤ 10⁻³·τδ`, or when the bracket's ratio is below 1.001. Report the
   smallest α that meets the target. By Lemma 4.11 every root gives the same `ĥ`, so this is the
   root interval's left end.
4. **Cap.** At most 60 evaluations, or the segment is flagged `alpha_unconverged`. That cannot
   happen for a monotone continuous `r_K`; the flag exists to catch solver failure.

---

## 5. The constrained solve

### 5.1 Bounds

| Config | Default | Why |
|---|---|---|
| `hva_v_max_mps` | 6.0 | Free fall from 2 m reaches 6.3 m/s. The cut window should end before the drop (§8), and the bound must still hold for the truth (Theorem 4.14 needs `h† ∈ K`). |
| `hva_a_max_mps2` | 80.0 | Above bar and joint accelerations through the pull and the catch. A foot strike can exceed it; §7.3 calibrates it. |

The bounds are a **safety net, not the smoother**. The discrepancy principle sets the smoothing.
The bounds stop a glitch or a short gap from producing a physically impossible derivative. They
are shared by every series in v1 (Open question 3).

### 5.2 Solver: banded ADMM, no new dependency

SciPy has no box-QP solver that exploits the band structure. The stage implements an OSQP-style
ADMM in numpy/scipy, in `tikhonov.py`. Write `P = M + αR` and `q = M z` (that is, `F` divided by
`2Δt`). The constraints use the **scaled** difference operators, which define the same polyhedron
with better conditioning:

```text
G = [ D ; D⁽²⁾ ]  ((2n−3) × n),     l = −u,   u = [ v_max·Δt·1 ; a_max·Δt²·1 ]

repeat:
  (P + σI + ρ GᵀG) x̃ = σ x − q + Gᵀ(ρ w − y)     // pentadiagonal SPD: cholesky_banded once per (α, ρ)
  w̃ = G x̃
  x ← x̃
  w ← clip(w̃ + y/ρ, l, u)
  y ← y + ρ (w̃ − w)
until ‖G x − w‖_∞ ≤ ε_pri  and  ‖P x − q + Gᵀ y‖_∞ ≤ ε_dual
```

- `scipy.linalg.cholesky_banded` / `cho_solve_banded` make each iteration O(n).
- ρ is adapted as in OSQP: refactor only when ρ changes by more than 5×.
- **Polish step.** Take the active set `𝒜 = {i : y_i ≠ 0}` and solve the equality-constrained KKT
  system `[P G_𝒜ᵀ; G_𝒜 0][h; λ] = [q; b_𝒜]` exactly (sparse). Accept it when the multipliers have
  the right signs and every inactive row is strictly feasible: this is Prop. 3.9's local affine
  structure. Otherwise keep the ADMM iterate and flag `unpolished`.
- **Published values are feasible.** `‖Vĥ‖_∞ ≤ v_max(1 + 10⁻⁹)` and the same for `a`, or the
  segment fails with `solver_infeasible`. A bound is never published as violated.

The alternative is adding `osqp` or `qpsolvers`. That needs the user's approval (`CLAUDE.md`), and
the problem is small and banded enough that it buys nothing. At about 430 nodes, roughly 17 series
and at most 60 α evaluations, the stage costs well under a second per capture.

### 5.3 From nodes back to frames

Frames map one-to-one onto observed nodes (§2.2), so no values are resampled.

| Output | At frame `i` on node `k` | Notes |
|---|---|---|
| `heightM` | `ĥ_k` | `null` on `bridged` nodes. A height is never stated where none was observed; S8's rule, carried over. |
| `velocityMps` | `(ĥ_{k+1} − ĥ_{k−1}) / (2Δt)`, one-sided at the segment ends | The mean of two staggered `v` values, so it obeys `v_max` too. |
| `accelMps2` | `(A ĥ)_{k−1}`, the **nodal** second difference centred at `t_k`; at the ends, the nearest | Nodal, not the piecewise reconstruction `π_n A_Δt ĥ`, whose last cell is zeroed (Lemma 4.6). This is Corollary 4.8's quantity. |

Flags per frame:
- `bridged`;
- `edge`: within `⌈ℓ/Δt⌉` frames of a segment end, `ℓ = (α_K μ_a)^{1/4}`;
- `v_active` / `a_active`: a bound is active here;
- `outliers` (series-level, repeated on every frame);
- `null` with a reason for the segment outcomes of §4.2.

---

## 6. Diagnostics and certificates

### 6.1 Per segment

`alpha`, `cutoffHz`, `sigmaM`, `sigmaRobustM`, `deltaM`, `tau`, `residualM` (= `r_K(α_K)`),
`rLowM`, `rHighM`, `status`, `solverIterations`, `polished`, `activeV` / `activeA` (counts), and:

- **Gain decomposition (Corollary 3.7).**
  - `d_α = ‖h_U^{α_K} − ĥ^{α_K}‖_{M_α}`: the observable part of what the box removed from the
    unconstrained estimate. It is zero on `box_inactive`.
  - `λ_𝒜`: the active multipliers.
- **Endpoint curvature:** `|â|` at the first and last node, against `a_max`. Prop. 4.4: the
  better (ν = 1) rate needs `h″ = h‴ = 0` at both ends. A lift whose window starts with the bar at
  rest and ends in a still overhead hold comes close. A window cut mid-motion falls to ν = ½, with
  no rate for the acceleration. Warn when either `|â|` exceeds `0.1·a_max`.

### 6.2 Certificates that need no source condition

These follow from Theorem 4.14, provided the truth is feasible and `‖ε‖_Δt ≤ δ`. They are
published as RMS values, by dividing the Δt-norm by `√T_seg`:

```text
heightErrorBoundRmsM  = (1 + τ) δ / √T_seg            (74)
```

(75) says the selected fit's bending energy `‖R^{1/2}ĥ‖_Δt` never exceeds the truth's. The
truth's energy is unknown, so the stage only reports the fit's own value, as `bendingEnergy`. It
is a lower bound on how much the true trajectory bends.

### 6.3 Acceleration certificate (Corollary 3.8)

```text
B_obs = [ (r_U(α) + δ)² + α (E_U(α) + √(μ_a T) a_max)² ]^{1/2}
r_U(α) = ‖h_U^α − z‖_Δt,      E_U(α) = ‖R^{1/2} h_U^α‖_Δt
accelErrorBoundRmsMps2 = [ (B_obs² − d_α²) / (α μ_a) ]^{1/2} / √T_seg      (35)
```

This bound uses only the data, the noise bound and `a_max`. It holds only under the two
assumptions above, so the document labels it a *conditional* certificate.

- **Not reviewed:** `d_α` and `‖·‖_{M_α}` are defined in (24)–(26). This spec takes
  `‖x‖²_{M_α} = ‖x‖²_Δt + α‖R^{1/2}x‖²_Δt`, read off Theorem 4.7's proof, and must be checked
  against (24) before this field ships.
- **Masked segments:** published as `null` on any segment with empty nodes (§2.3).

---

## 7. Validation

Synthetic tests follow `CLAUDE.md`: geometry is verified numerically, test-first, and no transform
is accepted because it ran.

### 7.1 Operators and the unconstrained filter

- `‖D‖, ‖D⁽²⁾‖, ‖D⁽³⁾‖ ≤ 2, 4, 8` (Lemma 2.2), by power iteration on n = 3…200.
- `ker R = span{1, t}` (Lemma 4.1).
- The banded `P` equals the dense `M + α μ_a AᵀA`.
- On `M = I`, `h_U` equals `(I + αR)⁻¹z`.
- The interior gain matches `1/(1+(f/f_c)⁴)` within 1% for f ∈ {1, 3, 6, 10} Hz.
- **Mesh uniformity (Remark 3.14):** at a fixed α, the same continuous signal sampled at 30, 60 and
  120 fps gives `ĥ` and `â` that agree at shared times to O(Δt²).

### 7.2 The constrained solve and α

- **Prop. 3.3:** when `h_U ∈ K_h`, the QP returns `h_U` exactly, with no iterations.
- **KKT (21):** on fixtures with active bounds, stationarity, complementarity and sign residuals
  are all below 10⁻⁸. The solution matches a dense reference solve (`scipy.optimize.minimize`,
  SLSQP, n = 40; test-only) to 10⁻⁷.
- **Theorem 3.10:** for random pairs `z, z′`, the observed `‖ĥ − ĥ′‖₂` and `‖â − â′‖₂` respect (37)
  and (39).
- **Lemma 3.15:** sampled feasible trajectories (compact bumps, `|h′| ≤ v_max`, `|h″| ≤ a_max`) are
  in `K_h` on every grid.
- **Lemma 4.11:** `r_K(α)` is nondecreasing on a 200-point log grid; `‖R^{1/2}ĥ^α‖` is
  nonincreasing.
- **Theorem 4.12:** `r_K(α)` tends to the closed-form `r_hi` as α grows. A fixture whose line fit
  needs a clipped slope checks the clipped branch.
- **Each `status` branch of §4.2** has one fixture.
- **Theorem 4.14:** on a ν = 1 reference made of compactly supported bumps, which satisfies (55),
  with Gaussian noise scaled so `‖ε‖_Δt = δ` exactly:
  - (74) and (75) hold;
  - over δ ∈ {1, 2, 4, 8} mm, the acceleration error falls at least as fast as `δ^{1/2}`
    (Theorem 4.17 at ν = 1).

  A reference that violates (55), cut mid-motion, shows the slower rate and fires §6.1's endpoint
  warning.
- **Noise estimator:**
  - σ̂ is unbiased within 5% on white noise over a bar-like profile;
  - `outliers` fires on a fixture with three 50 mm spikes and stays quiet without them;
  - the spiked series still reaches `morozov`, not `data_outside_bounds` (§3.1's reason for the
    energy estimator).
- **Grid:**
  - jittered 16/17 ms timestamps snap with `snapErrorMs ≤ 1`;
  - a dropped frame becomes a `bridged` node with no published height;
  - a 10-frame gap splits the segment;
  - a duplicate timestamp yields `grid_collision`.
- **Integration:**
  - the flow under `prefect_test_harness` writes `hva.json` next to S8's tree;
  - a dry run plans the write without solving;
  - S8's `joint_hva.json` is byte-identical before and after.

### 7.3 Real data (the seven `10 September` captures, 50kg3 held out)

| Check | Expectation |
|---|---|
| **Rest window** (first 60 frames, static): V and A per series | V ≈ 0, A ≈ 0. The detrended std of `heightM` against σ̂ measures noise correlation (§3.3). |
| **Near wrist vs bar centre**, hands on bar: V(t) | Peak V within 0.1 m/s; peak time within 1–2 frames. The two detectors are independent. |
| **Integration:** `∫V dt` against `Δĥ` per segment | Below 5 mm drift. |
| **`cutoffHz` per series** | Report the spread per capture against Winter's 6 Hz. A spread of more than 2× between joints of one capture is a finding (Open question 1). |
| **`status` and active-bound counts** | Mostly `box_inactive`. Every `data_outside_bounds` is read by hand, and the bounds are calibrated from them. |
| **Endpoint curvature** | Which captures meet the ν = 1 condition (§6.1). |

---

## 8. Code layout

Pure maths is kept apart from I/O, as `CLAUDE.md` requires:

- **`preprocess/tikhonov.py` (new, pure, numpy/scipy only):**
  - `snap_to_grid(t_ms) -> Grid` (§2.2);
  - `segments(grid, heights, max_gap, min_frames)` (§2.5);
  - `estimate_noise(z, mask) -> Noise` (§3.1);
  - `unconstrained(z, mask, dt, alpha)`;
  - `solve_box(z, mask, dt, alpha, v_max, a_max, warm=None) -> BoxSolution` (§5.2, with the
    polish step and KKT residuals);
  - `feasible_line_distance(z, mask, dt, v_max)` (§4.2);
  - `discrepancy(z, mask, dt, v_max, a_max, tau) -> Fit` (§4.3; returns the status, α, ĥ and
    multipliers);
  - `nodal_derivatives(h, dt)` (§5.3);
  - `certificates(fit, ...)` (§6).

  Thresholds are module constants, documented where they are defined.
- **`preprocess/hva.py` (new, pure):** `build(Inputs) -> Built`. It assembles `hva.json`, the
  manifest `derived` values and the warnings from S8's document and the bar heights. Same shape
  as `joint_hva.build`.
- **`preprocess/tasks/hva.py` (new `@task(retries=1)`):**
  - reads `joint_hva.json` and the bar track, reusing `kinematics._bar_track`, which moves to
    `barbell_input.py` so both tasks share it;
  - places the bar on `floor.yFloorM`, so bar and joints share one floor;
  - writes `hva.json` through the staging-then-publish helpers.

  A capture it cannot solve is never rejected: S0–S8 have already published. The reason goes in a
  manifest warning.
- **`flow.py`:** one more guarded step after kinematics: `if config.hva_root is not None`.
- **`config.py` / `cli.py`:**
  - `--hva <root>`, which requires `--kinematics`. The flow raises if `hva_root` is set without
    `kinematics_root`, as it does for the existing pairs.
  - New config fields: `hva_tau`, `hva_v_max_mps`, `hva_a_max_mps2`, `hva_max_gap_frames`,
    `hva_min_segment_frames`. `μ_a`, the α bracket and the solver tolerances are module constants.

**Operator note** (README): the lift window in `metadata.yaml` should **end before the bar is
dropped**. A drop breaks `v_max` and `a_max` and is not lift data.

## 9. Output contract

`hva.json`, written once per capture (replaced only with `--overwrite`):

```jsonc
{
  "schemaVersion": "1.0.0",
  "captureId": "<id>",
  "method": "docs/specs/preprocessing/S9-hva-tikhonov.md",
  "frames": { "startEpochMs": 0, "tMs": [ /* S8's, verbatim */ ] },
  "grid": { "dtS": 0.0167, "nodes": 433, "snapErrorMs": 0.6, "droppedFrames": [] },
  "settings": { "tau": 1.1, "muA": 1.0, "muJ": 0.0, "vMaxMps": 6.0, "aMaxMps2": 80.0,
                "maxGapFrames": 6, "minSegmentFrames": 30 },
  "floor": { "yFloorM": 0.0, "anchor": "<S8's>" },
  "series": {
    "<jointId> | bar": {
      "heightM":     [ "float | null" ],    // regularised; S8's raw heightM stays in joint_hva.json
      "velocityMps": [ "float | null" ],
      "accelMps2":   [ "float | null" ],
      "flags":       [ [ "bridged | edge | v_active | a_active | outliers" ] ],
      "segments": [ {
        "startFrame": 0, "endFrame": 432,
        "status": "morozov | box_inactive | data_outside_bounds | noise_dominates | too_short | grid_collision | solver_infeasible",
        "alpha": 0.0, "cutoffHz": 0.0,
        "sigmaM": 0.0, "sigmaRobustM": 0.0, "deltaM": 0.0,
        "residualM": 0.0, "rLowM": 0.0, "rHighM": 0.0,
        "activeV": 0, "activeA": 0, "dAlpha": 0.0, "bendingEnergy": 0.0, "polished": true, "solverIterations": 0,
        "endpointAccelMps2": [0.0, 0.0],
        "heightErrorBoundRmsM": "float | null",
        "accelErrorBoundRmsMps2": "float | null"   // conditional certificate, §6.3
      } ]
    }
  }
}
```

Field names follow powerflow-ui's `datatypespec.md` (`heightM`, `velocityMps`, `accelMps2`), so
the Graph entries (`"bar.com.velocityMps"`) can point at them.

Manifest `derived`:
- `hva_status_counts`;
- the median and range of `cutoffHz`;
- the series flagged `outliers`;
- the active-bound frame counts;
- `snapErrorMs`;
- the endpoint warnings.

## 10. Risks

- **Correlated keypoint noise** under-smooths (§3.3). Measured by §7.3's rest window.
- **Bounds that the truth breaks.** A foot strike or a catch above `a_max` makes `h† ∉ K`. Theorem
  4.14's certificates then do not hold, and the box clips a real peak. The `a_active` flags show
  where; §7.3 calibrates the bounds. Peak acceleration at an active bound is a lower bound, and the
  UI should label it so.
- **Per-series α.** Joints regularised differently are blunted differently. The filter is
  zero-phase, so peak *times* do not shift, but peak *values* are not equally biased.
- **The paper's unreviewed pages** (§6.3's `M_α` norm) gate one output field.
- **Masked segments** fall outside the paper's §4 assumptions. They get no certificates (§2.3).

## Open questions

1. **One α per series, or one per capture?** The discrepancy principle is per series, and that is
   what the theory certifies. A shared α (say the median of the joints' α_K) makes curves blunted
   alike. Recommended: per series in v1, and decide on §7.3's `cutoffHz` spread.
2. **δ inflation for correlated noise** (§3.3). Needed if the rest-window std exceeds σ̂ by more
   than about 20%.
3. **Per-class bounds.** Should there be separate `v_max` and `a_max` for the bar, the upper body
   and the feet? §7.3 decides.
4. **The jerk term** `μ_j > 0`. The paper covers it only formally (§4.1 preamble). Deferred until
   acceleration curves show staircase artefacts.
5. **Per-sample weights** from S8's `heightSigmaM` or `source`. This is `W ≠ I`: §3 of the paper
   still holds, §4 does not. It answers S8 draft 3's open question 5 with "not in v1".
6. **Horizontal bar path.** The same machinery per axis, with an L∞ bound per axis, not on speed.
   Deferred to bar-height draft 6.

---

## Feedback (2026-10-24)

- **User:** do the **barbell first**. Rewrite the spec for the bar alone, with a clear section
  listing the parameters that have to be fixed.
- **Incorporated in:** [`S9-hva-tikhonov-draft2.md`](S9-hva-tikhonov-draft2.md).
  - The bar's height is no longer read from S8. S9 solves the bar track itself from S5's
    detections, so it needs `--barbell` and does not need `--pose` or `--kinematics`.
  - With one series per capture, open question 1 (one α per series or per capture) goes away for
    now.
  - The parameters are collected in one section, split three ways: decide now, calibrate on the
    captures, and fixed by the theory.
  - Joints move to a later draft, which reuses `tikhonov.py` unchanged.
