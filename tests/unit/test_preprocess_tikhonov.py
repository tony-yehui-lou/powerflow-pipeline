"""The box-constrained Tikhonov differentiator, checked against the paper's results numerically.

`S9-hva-tikhonov-draft2.md` §10.1. Every fixture is built from a known trajectory with known
derivatives, so each assertion is against the truth it was made from, never against "it ran". The
reference profile is the spec's bar-like lift: rest, pull, catch dip, recovery, overhead hold,
built from septic smoothsteps so `h'`, `h''` and `h'''` vanish at both ends (Prop. 4.4, nu = 1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import minimize

from powerflow_pipeline.data.preprocess import tikhonov as tk

Array = NDArray[np.float64]

# (start s, duration s, rise m): rest 1 s, pull, catch dip, recovery, hold 1 s.
PIECES = ((1.0, 1.0, 0.85), (2.0, 0.3, -0.15), (2.3, 1.0, 0.8))
REST_M = 0.225
DURATION_S = 4.3
BOUNDS = tk.Bounds(v_max=5.0, a_max=25.0)


def _smoothstep(x: Array, order: int) -> Array:
    """The septic smoothstep `35x^4 - 84x^5 + 70x^6 - 20x^7` (or a derivative), 0 off [0, 1]."""

    coefficients = np.polynomial.Polynomial([0, 0, 0, 0, 35, -84, 70, -20])
    out = coefficients.deriv(order)(np.clip(x, 0.0, 1.0)) if order else coefficients(x)
    if order:
        out = np.where((x > 0) & (x < 1), out, 0.0)
    else:
        out = np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, out))
    return np.asarray(out, dtype=np.float64)


def profile(t: Array, order: int = 0) -> Array:
    """The reference lift's height (order 0), velocity (1) or acceleration (2) at `t` seconds."""

    out = np.full_like(t, REST_M if order == 0 else 0.0)
    for start, duration, rise in PIECES:
        out = out + rise * _smoothstep((t - start) / duration, order) / duration**order
    return out


def bang_bang(t: Array, accel: float) -> Array:
    """Rest, then `+accel` for 0.5 s and `-accel` for 0.5 s, then rest: a feasible trajectory
    that saturates `a_max = accel`, so noise pushes the unconstrained fit out of the box."""

    rise = np.where(t < 1.5, accel * (t - 1.0) ** 2 / 2, 0.0)
    fall = accel * (0.125 + 0.5 * (t - 1.5) - (t - 1.5) ** 2 / 2)
    out = np.where(t < 1.0, 0.0, np.where(t < 1.5, rise, np.where(t < 2.0, fall, accel / 4)))
    return np.asarray(REST_M + out, dtype=np.float64)


def grid(fps: float, duration: float = DURATION_S) -> tuple[Array, float]:
    dt = 1.0 / fps
    n = round(duration * fps) + 1
    return np.arange(n) * dt, dt


def observed(n: int) -> NDArray[np.bool_]:
    return np.ones(n, dtype=bool)


def noisy(sigma: float, seed: int, fps: float = 60.0) -> tuple[Array, Array, float]:
    t, dt = grid(fps)
    truth = profile(t)
    rng = np.random.default_rng(seed)
    return truth + rng.normal(0.0, sigma, t.size), truth, dt


def norm_dt(x: Array, dt: float) -> float:
    return math.sqrt(dt) * float(np.linalg.norm(x))


# --- operators -------------------------------------------------------------------------------


@pytest.mark.parametrize("m", [3, 4, 10, 57, 200])
def test_difference_operator_norms_are_bounded_as_lemma_2_2(m: int) -> None:
    for order, bound in ((1, 2.0), (2, 4.0), (3, 8.0)):
        if m <= order:
            continue
        d = tk.difference(m, order).toarray()
        assert np.linalg.norm(d, 2) <= bound + 1e-12


def test_the_kernel_of_r_is_the_affine_sequences_lemma_4_1() -> None:
    t, dt = grid(60.0, 1.0)
    r = tk.regulariser(t.size, dt).toarray()

    assert np.allclose(r @ np.ones(t.size), 0.0, atol=1e-6)
    assert np.allclose(r @ t, 0.0, atol=1e-6)
    assert np.linalg.matrix_rank(r) == t.size - 2


def test_alpha_and_cutoff_are_inverse() -> None:
    for f in (0.1, 1.0, 6.0, 30.0):
        assert tk.cutoff_for_alpha(tk.alpha_for_cutoff(f)) == pytest.approx(f)


# --- the unconstrained filter ----------------------------------------------------------------


def test_unconstrained_solves_the_masked_normal_equations() -> None:
    z, _, dt = noisy(0.003, seed=1)
    mask = observed(z.size)
    mask[[5, 40, 41, 42, 200]] = False
    z[~mask] = np.nan
    alpha = tk.alpha_for_cutoff(6.0)

    got = tk.unconstrained(z, mask, dt, alpha)

    m = np.diag(mask.astype(float))
    dense = np.linalg.solve(m + alpha * tk.regulariser(z.size, dt).toarray(), m @ np.nan_to_num(z))
    assert np.allclose(got, dense, atol=1e-10)


@pytest.mark.parametrize("f", [1.0, 3.0, 6.0, 10.0])
def test_interior_gain_is_a_fourth_order_cutoff(f: float) -> None:
    """§3.3: away from the ends the filter's gain is `1 / (1 + alpha mu_a s(w)^4)`, with the
    discrete symbol `s(w) = 2 sin(w dt / 2) / dt`; for `f` well below Nyquist that is
    `1 / (1 + (f/f_c)^4)`."""

    t, dt = grid(60.0, 40.0)
    omega = 2 * math.pi * f
    z = np.sin(omega * t)
    alpha = tk.alpha_for_cutoff(6.0)
    h = tk.unconstrained(z, observed(t.size), dt, alpha)

    middle = slice(t.size // 3, 2 * t.size // 3)
    basis = np.stack([np.sin(omega * t[middle]), np.cos(omega * t[middle])], axis=1)
    gain = float(np.hypot(*np.linalg.lstsq(basis, h[middle], rcond=None)[0]))
    symbol = 2 * math.sin(omega * dt / 2) / dt
    assert gain == pytest.approx(1 / (1 + alpha * tk.MU_A * symbol**4), rel=1e-6)
    if f <= 3.0:
        assert gain == pytest.approx(1 / (1 + (f / 6.0) ** 4), rel=0.01)


def test_a_fixed_alpha_means_the_same_filter_at_every_frame_rate() -> None:
    """Remark 3.14: with physical weights the discrete solutions converge under refinement."""

    alpha = tk.alpha_for_cutoff(6.0)
    solutions = {}
    for fps in (30.0, 60.0, 120.0):
        t, dt = grid(fps)
        h = tk.unconstrained(profile(t), observed(t.size), dt, alpha)
        _, a = tk.nodal_derivatives(h, dt)
        solutions[fps] = (t, h, a)

    def gap(coarse: float, fine: float) -> tuple[float, float]:
        _, hc, ac = solutions[coarse]
        _, hf, af = solutions[fine]
        step = round(fine / coarse)
        inner = slice(2, -2)  # nodal accelerations are one-sided at the ends
        return (
            float(np.max(np.abs(hc - hf[::step]))),
            float(np.max(np.abs(ac[inner] - af[::step][inner]))),
        )

    h_coarse, a_coarse = gap(30.0, 60.0)
    h_fine, a_fine = gap(60.0, 120.0)
    assert h_fine < h_coarse / 3  # O(dt^2): 4x per halving
    assert a_fine < a_coarse / 3


def test_nodal_derivatives_are_exact_on_a_quadratic() -> None:
    t, dt = grid(60.0, 1.0)
    h = 0.5 + 1.2 * t - 4.9 * t**2

    v, a = tk.nodal_derivatives(h, dt)

    assert np.allclose(v[1:-1], 1.2 - 9.8 * t[1:-1])
    assert np.allclose(a, -9.8)


# --- the constrained solve -------------------------------------------------------------------


def test_an_inactive_box_returns_the_unconstrained_estimate_prop_3_3() -> None:
    z, _, dt = noisy(0.003, seed=2)
    alpha = tk.alpha_for_cutoff(6.0)

    solution = tk.solve_box(z, observed(z.size), dt, alpha, BOUNDS)

    assert solution.iterations == 0
    assert np.array_equal(solution.h, tk.unconstrained(z, observed(z.size), dt, alpha))
    assert not np.any(solution.multipliers)


def _kkt(z: Array, dt: float, alpha: float, bounds: tk.Bounds, solution: tk.BoxSolution) -> None:
    """Stationarity, sign, complementarity and feasibility of (21), in the scaled rows."""

    n = z.size
    p = np.eye(n) + alpha * tk.regulariser(n, dt).toarray()
    g = tk.constraint_matrix(n).toarray()
    u = tk.constraint_bounds(n, dt, bounds)
    gh, y = g @ solution.h, solution.multipliers
    assert np.max(np.abs(p @ solution.h - z + g.T @ y)) < 1e-8
    assert np.all(np.abs(gh) <= u * (1 + 1e-9))
    upper, lower = y > 0, y < 0
    assert np.allclose(gh[upper], u[upper], rtol=1e-9)
    assert np.allclose(gh[lower], -u[lower], rtol=1e-9)


def test_an_active_box_solves_the_kkt_system() -> None:
    z, _, dt = noisy(0.003, seed=3)
    alpha = tk.alpha_for_cutoff(10.0)
    tight = tk.Bounds(v_max=1.2, a_max=4.0)

    solution = tk.solve_box(z, observed(z.size), dt, alpha, tight)

    assert solution.polished
    assert np.count_nonzero(solution.multipliers) > 0
    _kkt(z, dt, alpha, tight, solution)
    assert tk.is_feasible(solution.h, dt, tight)


def test_the_box_solution_matches_a_dense_reference_solver() -> None:
    rng = np.random.default_rng(4)
    dt = 1 / 60
    n = 40
    z = np.cumsum(rng.normal(0, 0.01, n))
    alpha = tk.alpha_for_cutoff(8.0)
    tight = tk.Bounds(v_max=0.6, a_max=15.0)
    p = np.eye(n) + alpha * tk.regulariser(n, dt).toarray()
    g = tk.constraint_matrix(n).toarray()
    u = tk.constraint_bounds(n, dt, tight)

    reference = minimize(
        lambda h: 0.5 * h @ p @ h - z @ h,
        np.zeros(n),
        jac=lambda h: p @ h - z,
        constraints=[
            {"type": "ineq", "fun": lambda h: u - g @ h, "jac": lambda h: -g},
            {"type": "ineq", "fun": lambda h: u + g @ h, "jac": lambda h: g},
        ],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    solution = tk.solve_box(z, observed(n), dt, alpha, tight)

    assert reference.success
    assert np.max(np.abs(solution.h - reference.x)) < 1e-7


def test_masked_nodes_are_filled_and_the_solution_stays_feasible() -> None:
    z, truth, dt = noisy(0.003, seed=5)
    mask = observed(z.size)
    mask[100:105] = False
    z[~mask] = np.nan
    tight = tk.Bounds(v_max=1.2, a_max=4.0)
    near_truth = tk.Bounds(v_max=2.0, a_max=13.0)

    solution = tk.solve_box(z, mask, dt, tk.alpha_for_cutoff(10.0), tight)
    filled = tk.solve_box(z, mask, dt, tk.alpha_for_cutoff(10.0), near_truth)

    assert np.all(np.isfinite(solution.h))
    assert tk.is_feasible(solution.h, dt, tight)
    assert np.max(np.abs(filled.h[100:105] - truth[100:105])) < 0.01


def test_lipschitz_stability_theorem_3_10() -> None:
    """(37) and (39) with W = I: the algebraic acceleration weight is `alpha * mu_a`."""

    dt = 1 / 60
    alpha = tk.alpha_for_cutoff(8.0)
    tight = tk.Bounds(v_max=1.0, a_max=5.0)
    t, _ = grid(60.0, 2.0)
    rng = np.random.default_rng(6)
    for _ in range(5):
        z = profile(t + 1.0) + rng.normal(0, 0.01, t.size)
        z2 = z + rng.normal(0, 0.02, t.size)
        h = tk.solve_box(z, observed(t.size), dt, alpha, tight).h
        h2 = tk.solve_box(z2, observed(t.size), dt, alpha, tight).h
        _, a = tk.nodal_derivatives(h, dt)
        _, a2 = tk.nodal_derivatives(h2, dt)
        gap = float(np.linalg.norm(z - z2))
        assert np.linalg.norm(h - h2) <= gap * (1 + 1e-6)
        inner = slice(1, -1)  # the nodal ends repeat their neighbours
        assert np.linalg.norm(a[inner] - a2[inner]) <= gap / math.sqrt(alpha * tk.MU_A) * (1 + 1e-6)


@pytest.mark.parametrize("fps", [30.0, 60.0, 120.0])
def test_samples_of_a_feasible_trajectory_are_feasible_lemma_3_15(fps: float) -> None:
    t, dt = grid(fps)
    fine = np.linspace(0, DURATION_S, 200_001)
    exact = tk.Bounds(
        v_max=float(np.max(np.abs(profile(fine, 1)))),
        a_max=float(np.max(np.abs(profile(fine, 2)))),
    )

    assert tk.is_feasible(profile(t), dt, exact, rtol=1e-9)


# --- the discrepancy principle ---------------------------------------------------------------


def test_the_constrained_residual_is_monotone_lemma_4_11() -> None:
    z, _, dt = noisy(0.004, seed=7)
    tight = tk.Bounds(v_max=1.2, a_max=4.0)
    mask = observed(z.size)

    residuals, energies = [], []
    warm = None
    for f in np.geomspace(30.0, 0.1, 40):
        warm = tk.solve_box(z, mask, dt, tk.alpha_for_cutoff(float(f)), tight, warm=warm)
        residuals.append(tk.residual(warm.h, z, mask, dt))
        energies.append(tk.bending_energy(warm.h, dt))

    assert np.all(np.diff(residuals) >= -1e-9)
    assert np.all(np.diff(energies) <= 1e-9)


def test_the_feasible_line_clips_its_slope_to_the_velocity_bound() -> None:
    t, dt = grid(60.0, 1.0)
    z = 0.3 + 8.0 * t
    mask = observed(t.size)

    line = tk.feasible_line(z, mask, dt, v_max=5.0)

    assert np.allclose(np.diff(line) / dt, 5.0)
    assert np.mean(line - z) == pytest.approx(0.0, abs=1e-12)  # intercept refitted


def test_the_constrained_residual_tends_to_the_feasible_line_theorem_4_12() -> None:
    z, _, dt = noisy(0.003, seed=8)
    mask = observed(z.size)
    tight = tk.Bounds(v_max=1.2, a_max=4.0)
    line = tk.residual(tk.feasible_line(z, mask, dt, tight.v_max), z, mask, dt)

    gaps = [
        line - tk.residual(tk.solve_box(z, mask, dt, tk.alpha_for_cutoff(f), tight).h, z, mask, dt)
        for f in (3.0, 1.0, 0.3, 0.1)
    ]

    assert all(gap >= -1e-9 for gap in gaps)  # r_K never passes its limit (73)
    assert np.all(np.diff(gaps) < 0)
    assert gaps[-1] < gaps[0] / 10


def test_a_slope_beyond_the_bound_everywhere_pins_every_fit_to_the_feasible_line() -> None:
    t, dt = grid(60.0, 1.0)
    rng = np.random.default_rng(8)
    z = 0.3 + 8.0 * t + 0.05 * np.sin(2 * math.pi * t) + rng.normal(0, 0.002, t.size)
    mask = observed(t.size)
    line = tk.feasible_line(z, mask, dt, BOUNDS.v_max)

    for f in (3.0, 0.3, 0.1):
        solution = tk.solve_box(z, mask, dt, tk.alpha_for_cutoff(f), BOUNDS)
        assert solution.polished
        assert np.allclose(solution.h, line, atol=1e-9)


def _fit(z: Array, dt: float, target: float, bounds: tk.Bounds = BOUNDS) -> tk.Fit:
    return tk.discrepancy(z, observed(z.size), dt, bounds, target=target)


def test_generous_bounds_end_on_the_unconstrained_morozov_parameter() -> None:
    z, _, dt = noisy(0.003, seed=9)
    target = 1.1 * 0.003 * math.sqrt(z.size * dt)

    fit = _fit(z, dt, target)

    assert fit.status == "box_inactive"
    assert fit.alpha is not None and fit.h is not None
    assert fit.residual == pytest.approx(target, rel=2e-3)
    assert 2.0 < tk.cutoff_for_alpha(fit.alpha) < 20.0


def test_an_active_box_ends_on_the_constrained_morozov_parameter() -> None:
    t, dt = grid(60.0, 3.0)
    z = bang_bang(t, 4.0) + np.random.default_rng(10).normal(0, 0.003, t.size)
    target = 1.1 * 0.003 * math.sqrt(z.size * dt)
    tight = tk.Bounds(v_max=5.0, a_max=4.0)

    fit = _fit(z, dt, target, tight)

    assert fit.status == "morozov"
    assert fit.h is not None and fit.multipliers is not None
    assert fit.residual == pytest.approx(target, rel=2e-3)
    assert tk.is_feasible(fit.h, dt, tight)
    assert np.count_nonzero(fit.multipliers) > 0


def test_a_target_below_every_feasible_residual_is_data_outside_bounds() -> None:
    z, _, dt = noisy(0.003, seed=11)
    tight = tk.Bounds(v_max=0.5, a_max=1.0)

    fit = _fit(z, dt, target=1e-6, bounds=tight)

    assert fit.status == "data_outside_bounds"
    assert fit.h is None
    assert fit.r_lo is not None and fit.r_lo > 1e-6


def test_a_target_above_the_feasible_line_is_noise_dominates() -> None:
    z, _, dt = noisy(0.003, seed=12)
    line = tk.feasible_line(z, observed(z.size), dt, BOUNDS.v_max)

    fit = _fit(z, dt, target=10.0)

    assert fit.status == "noise_dominates"
    assert fit.h is not None
    assert np.allclose(fit.h, line)


def _exact_noise(truth: Array, delta: float, dt: float, seed: int) -> Array:
    """White noise rescaled so `||eps||_dt` is exactly `delta`."""

    eps = np.random.default_rng(seed).normal(0.0, 1.0, truth.size)
    return truth + eps * (delta / norm_dt(eps, dt))


def test_the_a_posteriori_bounds_of_theorem_4_14_hold() -> None:
    t, dt = grid(60.0)
    truth = profile(t)
    tau = 1.1
    for seed, sigma in enumerate((0.001, 0.003, 0.008)):
        delta = sigma * math.sqrt(t.size * dt)
        z = _exact_noise(truth, delta, dt, seed)

        fit = _fit(z, dt, target=tau * delta)

        assert fit.h is not None
        assert norm_dt(fit.h - truth, dt) <= (1 + tau) * delta  # (74)
        assert tk.bending_energy(fit.h, dt) <= tk.bending_energy(truth, dt) + 1e-12  # (75)


def test_acceleration_error_falls_with_the_noise_theorem_4_17() -> None:
    """At nu = 1 the bound falls like δ^(1/2); the observed log-log slope must be at least that."""

    t, dt = grid(60.0)
    truth = profile(t)
    true_a = profile(t, 2)
    sigmas = (0.001, 0.002, 0.004, 0.008)
    errors = []
    for sigma in sigmas:
        delta = sigma * math.sqrt(t.size * dt)
        runs = []
        for seed in range(5):
            fit = _fit(_exact_noise(truth, delta, dt, 100 + seed), dt, target=1.1 * delta)
            assert fit.h is not None
            _, a = tk.nodal_derivatives(fit.h, dt)
            runs.append(norm_dt(a - true_a, dt))
        errors.append(float(np.mean(runs)))

    slope = np.polyfit(np.log(sigmas), np.log(errors), 1)[0]
    assert np.all(np.diff(errors) > 0)
    assert slope >= 0.5


def test_peak_velocity_is_recovered_within_the_spec_tolerance() -> None:
    """§10.1: at 4 mm the peak velocity error is under 0.05 m/s and its time within a frame."""

    t, dt = grid(60.0)
    truth = profile(t)
    true_v = profile(t, 1)
    delta = 0.004 * math.sqrt(t.size * dt)
    for seed in range(5):
        fit = _fit(_exact_noise(truth, delta, dt, 200 + seed), dt, target=1.1 * delta)
        assert fit.h is not None
        v, _ = tk.nodal_derivatives(fit.h, dt)
        assert abs(float(np.max(v)) - float(np.max(true_v))) < 0.05
        assert abs(int(np.argmax(v)) - int(np.argmax(true_v))) <= 1


# --- noise -----------------------------------------------------------------------------------


def test_the_noise_estimates_are_unbiased_on_white_noise() -> None:
    estimates = []
    for seed in range(20):
        z, _, _ = noisy(0.003, seed=300 + seed)
        noise = tk.estimate_noise(z, observed(z.size))
        estimates.append((noise.sigma, noise.sigma_robust))
    sigma, robust = np.mean(estimates, axis=0)

    assert sigma == pytest.approx(0.003, rel=0.05)
    assert robust == pytest.approx(0.003, rel=0.05)


def test_the_noise_estimate_only_uses_four_consecutive_observations() -> None:
    z, _, _ = noisy(0.003, seed=13)
    mask = observed(z.size)
    mask[::4] = False
    z[~mask] = 1e3  # would wreck the estimate if a masked node were read

    noise = tk.estimate_noise(z, mask)

    assert noise.samples == 0
    assert math.isnan(noise.sigma)


def test_isolated_glitches_raise_the_energy_estimate_but_not_the_robust_one() -> None:
    z, _, dt = noisy(0.002, seed=14)
    z[[60, 130, 200]] += 0.05

    noise = tk.estimate_noise(z, observed(z.size))

    assert noise.sigma / noise.sigma_robust > 1.5
    target = 1.1 * noise.sigma * math.sqrt(z.size * dt)
    assert _fit(z, dt, target).status in ("box_inactive", "morozov")


# --- grid and segments -----------------------------------------------------------------------


def test_jittered_frame_times_snap_to_one_node_each() -> None:
    t_ms = np.round(np.arange(300) * 1000 / 60).astype(int).tolist()  # 16/17 ms jitter

    snapped = tk.snap_to_grid(t_ms)

    assert snapped.dt_s == pytest.approx(1 / 60, rel=1e-4)
    assert snapped.nodes == tuple(range(300))
    assert snapped.snap_error_ms <= 1.0
    assert snapped.dropped == ()


def test_a_dropped_frame_leaves_an_empty_node() -> None:
    t_ms = [round(i * 1000 / 60) for i in range(120) if i != 50]

    snapped = tk.snap_to_grid(t_ms)

    assert snapped.n_nodes == 120
    assert snapped.dropped == (50,)
    assert snapped.nodes[50] == 51


def test_two_frames_on_one_node_is_a_grid_collision() -> None:
    t_ms = [round(i * 1000 / 60) for i in range(60)]
    t_ms.insert(30, t_ms[30] + 2)

    with pytest.raises(tk.GridError, match="collision"):
        tk.snap_to_grid(t_ms)


def test_a_single_frame_has_no_grid() -> None:
    with pytest.raises(tk.GridError):
        tk.snap_to_grid([0])


def test_long_gaps_split_segments_and_short_ones_are_bridged() -> None:
    present = [False] * 3 + [True] * 40 + [False] * 4 + [True] * 40 + [False] * 10 + [True] * 20
    present += [False] * 2

    found = tk.segments(present, max_gap=6)

    assert [(s.start, s.stop, s.observed) for s in found] == [(3, 87, 80), (97, 117, 20)]
