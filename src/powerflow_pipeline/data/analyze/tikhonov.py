"""Box-constrained Tikhonov differentiation of one height series: pure maths, no I/O.

`docs/specs/preprocessing/S10-hva-tikhonov-draft2.md` §3-§8, after the user's paper on numerical
differentiation under bounds on the first and second derivatives (numbers below are the paper's).
On a uniform grid of spacing `dt`, with an observation mask `M`, this minimises

    F(h) = ||M^(1/2) (h - z)||^2_dt + alpha * mu_a * ||A h||^2_dt     (47), lambda_j = 0
    over K_h = { |V h| <= v_max, |A h| <= a_max }                      (10)

with `V = D / dt`, `A = D2 / dt^2` and `||u||^2_dt = dt * u'u` (42), and picks `alpha` by the
constrained discrepancy principle (70). Units are metres and seconds throughout. Bar-agnostic:
nothing here knows what the series is the height of.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.sparse.linalg import splu

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]

# Only `alpha * MU_A` ever appears, so this fixes the unit of alpha (s^4) and nothing else.
MU_A: Final = 1.0

# P12: the alpha search, as equivalent cutoffs. 30 Hz is the 60 fps Nyquist limit, 0.1 Hz is all
# but a straight line over a lift; hitting either end is reported, never silent.
CUTOFF_MAX_HZ: Final = 30.0
CUTOFF_MIN_HZ: Final = 0.1
# P13, P14: bisection stops within this fraction of the target, far inside delta's own ~3.5%
# uncertainty; 2.5 decades of alpha at this tolerance take about 25 evaluations.
BISECTION_TOL: Final = 1e-3
MAX_EVALUATIONS: Final = 60

# P15: ADMM only has to find the active set; the polish step then solves the KKT system exactly.
RHO_INIT: Final = 0.1
SIGMA: Final = 1e-6
RELAXATION: Final = 1.6
EPS_ABS: Final = 1e-7
EPS_REL: Final = 1e-6
MAX_ITERATIONS: Final = 4000
CHECK_EVERY: Final = 10
ADAPT_EVERY: Final = 50
MAX_POLISH_ROUNDS: Final = 25
REFINEMENT_STEPS: Final = 2
# P16: how far past a bound a published value may sit. Polished solutions sit within 1e-9; an
# unpolished ADMM iterate is held to this, or the series is not solved.
FEASIBILITY_RTOL: Final = 1e-6

# Frames whose interval is above this multiple of the median are not consecutive: a drop.
DROP_INTERVAL_FACTOR: Final = 1.5

# Var(D3 eps) = (1 + 9 + 9 + 1) sigma^2 for independent noise; MAD to sigma for a Gaussian.
D3_VARIANCE: Final = 20.0
MAD_TO_SIGMA: Final = 1.4826

Status = Literal["box_inactive", "morozov", "data_outside_bounds", "noise_dominates"]


class GridError(ValueError):
    """The frame times cannot be placed one per node on a uniform grid."""


class SolverError(RuntimeError):
    """The box QP returned a point outside the box."""


@dataclass(frozen=True)
class Bounds:
    v_max: float  # m/s
    a_max: float  # m/s^2


@dataclass(frozen=True)
class Grid:
    dt_s: float
    nodes: tuple[int, ...]  # the node of each frame, strictly increasing
    n_nodes: int
    snap_error_ms: float

    @property
    def dropped(self) -> tuple[int, ...]:
        """Nodes no frame landed on."""

        taken = set(self.nodes)
        return tuple(k for k in range(self.n_nodes) if k not in taken)


@dataclass(frozen=True)
class Segment:
    start: int  # first node, observed
    stop: int  # one past the last node, observed
    observed: int


@dataclass(frozen=True)
class Noise:
    sigma: float  # energy estimator: drives delta
    sigma_robust: float  # MAD estimator: the outlier check
    samples: int  # third differences used


@dataclass(frozen=True)
class BoxSolution:
    h: Array
    multipliers: Array  # y in `P h - q + G'y = 0`, one per row of `constraint_matrix`
    iterations: int  # 0 when the unconstrained estimate was already feasible
    polished: bool


@dataclass(frozen=True)
class Fit:
    status: Status
    target: float
    r_hi: float
    alpha: float | None = None
    h: Array | None = None
    h_unconstrained: Array | None = None
    multipliers: Array | None = None
    residual: float | None = None
    r_lo: float | None = None
    evaluations: int = 0
    iterations: int = 0
    polished: bool = True
    at_bracket: bool = False


# --- grid ------------------------------------------------------------------------------------


def snap_to_grid(t_ms: Sequence[int]) -> Grid:
    """Place each frame on the node nearest its time; values are never resampled (§4).

    `dt` is the mean of the consecutive intervals, refined by least squares against the node
    numbers. Not their median: rounded 60 fps PTS run 17, 17, 16 ms, so the median would put a
    4 s capture 80 ms adrift.
    """

    if len(t_ms) < 2:
        raise GridError("a grid needs at least two frames")
    t = np.asarray(t_ms, dtype=np.float64) - float(t_ms[0])
    intervals = np.diff(t)
    if np.any(intervals <= 0):
        raise GridError("grid collision: frame times are not strictly increasing")
    consecutive = intervals[intervals <= DROP_INTERVAL_FACTOR * float(np.median(intervals))]
    dt_ms = float(np.mean(consecutive))
    for _ in range(2):
        nodes = np.rint(t / dt_ms).astype(np.int64)
        if nodes[-1] > 0:
            dt_ms = float(np.dot(t, nodes) / np.dot(nodes, nodes))
    nodes = np.rint(t / dt_ms).astype(np.int64)
    repeated = np.flatnonzero(np.diff(nodes) == 0)
    if repeated.size:
        i = int(repeated[0])
        raise GridError(f"grid collision: frames {i} and {i + 1} fall on node {nodes[i]}")
    return Grid(
        dt_s=dt_ms / 1000.0,
        nodes=tuple(int(k) for k in nodes),
        n_nodes=int(nodes[-1]) + 1,
        snap_error_ms=float(np.max(np.abs(t - nodes * dt_ms))),
    )


def segments(present: Sequence[bool], *, max_gap: int) -> list[Segment]:
    """Runs of observed nodes, split wherever more than `max_gap` nodes in a row are empty."""

    observed = [i for i, p in enumerate(present) if p]
    found: list[Segment] = []
    if not observed:
        return found
    start = previous = observed[0]
    count = 1
    for i in observed[1:]:
        if i - previous - 1 > max_gap:
            found.append(Segment(start=start, stop=previous + 1, observed=count))
            start, count = i, 0
        previous = i
        count += 1
    found.append(Segment(start=start, stop=previous + 1, observed=count))
    return found


# --- operators -------------------------------------------------------------------------------


def difference(m: int, order: int) -> sparse.csr_matrix:
    """The `order`-th forward difference, (m - order) x m (paper §2.2)."""

    d = sparse.identity(m, format="csr")
    for k in range(order):
        size = m - k
        step = sparse.diags([-1.0, 1.0], [0, 1], shape=(size - 1, size), format="csr")
        d = step @ d
    return sparse.csr_matrix(d)


def regulariser(n: int, dt: float) -> sparse.csr_matrix:
    """`R = mu_a A'A`, so that `alpha ||R^(1/2) h||^2 = alpha mu_a ||A h||^2` (52)."""

    d2 = difference(n, 2)
    return sparse.csr_matrix((MU_A / dt**4) * (d2.T @ d2))


def constraint_matrix(n: int) -> sparse.csr_matrix:
    """`G = [D; D2]`: the box rows, scaled so velocity and acceleration rows are alike (§7.1)."""

    return sparse.csr_matrix(sparse.vstack([difference(n, 1), difference(n, 2)]))


def constraint_bounds(n: int, dt: float, bounds: Bounds) -> Array:
    """`|G h| <= u` is exactly `K_h` in the scaled rows."""

    return np.concatenate(
        [np.full(n - 1, bounds.v_max * dt), np.full(n - 2, bounds.a_max * dt * dt)]
    )


def alpha_for_cutoff(f_hz: float) -> float:
    """The alpha whose interior gain is `1 / (1 + (f / f_hz)^4)` (§3.3)."""

    return float((2.0 * math.pi * f_hz) ** -4 / MU_A)


def cutoff_for_alpha(alpha: float) -> float:
    return float(1.0 / (2.0 * math.pi * (alpha * MU_A) ** 0.25))


def _banded(matrix: sparse.spmatrix) -> Array:
    """A symmetric pentadiagonal matrix in `cholesky_banded`'s upper form."""

    n = matrix.shape[0]
    ab = np.zeros((3, n))
    ab[2] = matrix.diagonal(0)
    ab[1, 1:] = matrix.diagonal(1)
    ab[0, 2:] = matrix.diagonal(2)
    return ab


def _factor(matrix: sparse.spmatrix) -> Array:
    return np.asarray(cholesky_banded(_banded(matrix), lower=False), dtype=np.float64)


def _solve(factor: Array, rhs: Array) -> Array:
    return np.asarray(cho_solve_banded((factor, False), rhs), dtype=np.float64)


def _system(z: Array, mask: Mask, dt: float, alpha: float) -> tuple[sparse.csr_matrix, Array]:
    """`P = M + alpha R` and `q = M z`: `F / (2 dt)` as `1/2 h'P h - q'h` plus a constant."""

    p = sparse.diags(mask.astype(np.float64)) + alpha * regulariser(z.size, dt)
    return sparse.csr_matrix(p), np.where(mask, z, 0.0)


# --- norms -----------------------------------------------------------------------------------


def residual(h: Array, z: Array, mask: Mask, dt: float) -> float:
    """`||M^(1/2)(h - z)||_dt`, the discrepancy (70)."""

    return math.sqrt(dt) * float(np.linalg.norm((h - z)[mask]))


def bending_energy(h: Array, dt: float) -> float:
    """`||A h||_dt`, so `||R^(1/2) h||_dt = sqrt(mu_a) * bending_energy`."""

    return math.sqrt(dt) * float(np.linalg.norm(np.diff(h, 2))) / dt**2


def energy_norm(x: Array, dt: float, alpha: float) -> float:
    """`||x||_{M_alpha}`, taken as `||x||^2_dt + alpha ||R^(1/2) x||^2_dt` (spec §8.3)."""

    return math.sqrt(dt * float(x @ x) + alpha * MU_A * bending_energy(x, dt) ** 2)


def is_feasible(h: Array, dt: float, bounds: Bounds, rtol: float = FEASIBILITY_RTOL) -> bool:
    u = constraint_bounds(h.size, dt, bounds)
    return bool(np.all(np.abs(constraint_matrix(h.size) @ h) <= u * (1 + rtol)))


def nodal_derivatives(h: Array, dt: float) -> tuple[Array, Array]:
    """Velocity and acceleration at every node (§7.2).

    Velocity is the central difference, one-sided at the ends: the mean of two staggered `V h`,
    so it obeys `v_max` too. Acceleration is the nodal second difference centred on each node
    (Corollary 4.8), repeated from the neighbour at the two ends.
    """

    v = np.gradient(h, dt)
    a = np.empty_like(h)
    a[1:-1] = np.diff(h, 2) / dt**2
    a[0], a[-1] = a[1], a[-2]
    return np.asarray(v, dtype=np.float64), a


def end_acceleration(z: Array, mask: Mask, dt: float, window: int) -> tuple[Array, Array]:
    """The data's own acceleration at each end, `[start, end]`, with its standard errors.

    A local quadratic least-squares fit to the observations in the first and last `window`
    nodes. Not the regularised estimate: that obeys the bending energy's natural boundary
    condition, `h'' = 0` at both ends, so it reads near zero however the window was cut.
    """

    values, errors = np.full(2, np.nan), np.full(2, np.nan)
    for side, span in enumerate((slice(0, window), slice(z.size - window, z.size))):
        t = (np.arange(z.size) * dt)[span][mask[span]]
        if t.size < 4:
            continue
        x = t - t.mean()
        design = np.stack([np.ones_like(x), x, x**2], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, z[span][mask[span]], rcond=None)
        fitted = design @ coefficients
        scale = float(np.sum((z[span][mask[span]] - fitted) ** 2)) / (t.size - 3)
        covariance = scale * np.linalg.inv(design.T @ design)
        values[side] = 2 * coefficients[2]
        errors[side] = 2 * math.sqrt(float(covariance[2, 2]))
    return values, errors


# --- noise -----------------------------------------------------------------------------------


def estimate_noise(z: Array, mask: Mask) -> Noise:
    """sigma from the third differences over every four consecutive observed nodes (§5)."""

    ok = mask[:-3] & mask[1:-2] & mask[2:-1] & mask[3:]
    d3 = np.diff(np.where(mask, z, 0.0), 3)[ok]
    if not d3.size:
        return Noise(sigma=math.nan, sigma_robust=math.nan, samples=0)
    return Noise(
        sigma=math.sqrt(float(np.mean(d3**2)) / D3_VARIANCE),
        sigma_robust=MAD_TO_SIGMA * float(np.median(np.abs(d3))) / math.sqrt(D3_VARIANCE),
        samples=int(d3.size),
    )


# --- solves ----------------------------------------------------------------------------------


def unconstrained(z: Array, mask: Mask, dt: float, alpha: float) -> Array:
    """`h_U = (M + alpha R)^(-1) M z`; on `M = I` the spectral filter (18)."""

    p, q = _system(z, mask, dt, alpha)
    return _solve(_factor(p), q)


def feasible_line(z: Array, mask: Mask, dt: float, v_max: float) -> Array:
    """The closest feasible straight line, `K_h ∩ ker R` (Theorem 4.12 (73)).

    A line has no acceleration, so only its slope is bounded. With the intercept eliminated the
    residual is a convex quadratic in the slope, so clipping the free slope is optimal.
    """

    t = np.arange(z.size) * dt
    tt, zz = t[mask], z[mask]
    spread = float(np.sum((tt - tt.mean()) ** 2))
    slope = 0.0 if spread == 0 else float(np.sum((tt - tt.mean()) * (zz - zz.mean())) / spread)
    slope = min(max(slope, -v_max), v_max)
    return np.asarray(float(np.mean(zz - slope * tt)) + slope * t, dtype=np.float64)


def _polish(
    p: sparse.csr_matrix, q: Array, g: sparse.csr_matrix, u: Array, w: Array, y: Array
) -> tuple[Array, Array] | None:
    """Solve the KKT system on ADMM's active set exactly, correcting the set until it is
    consistent (Prop. 3.9). None when it does not settle or the system is singular."""

    n = q.size
    sign = np.zeros(u.size)
    sign[u - w < y] = 1.0
    sign[w + u < -y] = -1.0
    for _ in range(MAX_POLISH_ROUNDS):
        rows = np.flatnonzero(sign)
        g_a = g[rows]
        kkt = sparse.bmat([[p, g_a.T], [g_a, None]], format="csc")
        rhs = np.concatenate([q, sign[rows] * u[rows]])
        try:
            lu = splu(kkt)
        except RuntimeError:  # singular: the active rows are dependent
            return None
        solution = lu.solve(rhs)
        for _ in range(REFINEMENT_STEPS):  # at large alpha the KKT matrix is ill-conditioned
            solution = solution + lu.solve(rhs - kkt @ solution)
        if not np.all(np.isfinite(solution)):
            return None
        h, lam = solution[:n], solution[n:]
        gh = g @ h
        wrong = rows[sign[rows] * lam < 0]
        inactive = sign == 0
        violated = np.flatnonzero(inactive & (np.abs(gh) > u * (1 + 1e-9)))
        if not wrong.size and not violated.size:
            multipliers = np.zeros(u.size)
            multipliers[rows] = lam
            return h, multipliers
        sign[wrong] = 0.0
        sign[violated] = np.sign(gh[violated])
    return None


def solve_box(
    z: Array,
    mask: Mask,
    dt: float,
    alpha: float,
    bounds: Bounds,
    warm: BoxSolution | None = None,
) -> BoxSolution:
    """The minimiser of `F` over `K_h` (Theorem 3.1), by OSQP-style ADMM and a KKT polish.

    The unconstrained estimate is returned as is when it is already feasible (Prop. 3.3). Each
    ADMM step is one banded solve, O(n). Raises `SolverError` if the answer leaves the box.
    """

    n = z.size
    p, q = _system(z, mask, dt, alpha)
    h_u = _solve(_factor(p), q)
    g = constraint_matrix(n)
    u = constraint_bounds(n, dt, bounds)
    if np.all(np.abs(g @ h_u) <= u):
        return BoxSolution(h=h_u, multipliers=np.zeros(u.size), iterations=0, polished=True)

    x = warm.h.copy() if warm is not None else h_u
    y = warm.multipliers.copy() if warm is not None else np.zeros(u.size)
    w = np.clip(g @ x, -u, u)
    gtg = sparse.csr_matrix(g.T @ g)
    rho = RHO_INIT
    factor = _factor(p + SIGMA * sparse.identity(n) + rho * gtg)
    eps_abs = EPS_ABS
    for iteration in range(1, MAX_ITERATIONS + 1):
        x_tilde = _solve(factor, SIGMA * x + q + g.T @ (rho * w - y))
        z_relaxed = RELAXATION * (g @ x_tilde) + (1 - RELAXATION) * w
        x = RELAXATION * x_tilde + (1 - RELAXATION) * x
        w_next = np.clip(z_relaxed + y / rho, -u, u)
        y = y + rho * (z_relaxed - w_next)
        w = w_next
        if iteration % CHECK_EVERY:
            continue
        gx, px, gty = g @ x, p @ x, g.T @ y
        primal = float(np.max(np.abs(gx - w)))
        dual = float(np.max(np.abs(px - q + gty)))
        primal_scale = max(float(np.max(np.abs(gx))), float(np.max(np.abs(w))))
        dual_scale = max(float(np.max(np.abs(px))), float(np.max(np.abs(gty))), float(np.max(q)))
        if primal <= eps_abs + EPS_REL * primal_scale and dual <= eps_abs + EPS_REL * dual_scale:
            polished = _polish(p, q, g, u, w, y)
            if polished is not None:
                return BoxSolution(polished[0], polished[1], iteration, polished=True)
            eps_abs /= 100  # a wrong active set: tighten and keep going
        if iteration % ADAPT_EVERY == 0 and primal > 0 and dual > 0:
            proposed = rho * math.sqrt(
                (primal / max(primal_scale, 1e-30)) / (dual / max(dual_scale, 1e-30))
            )
            if not 0.2 <= proposed / rho <= 5.0:
                rho = proposed
                factor = _factor(p + SIGMA * sparse.identity(n) + rho * gtg)

    if not is_feasible(x, dt, bounds):
        raise SolverError(f"ADMM ended {np.max(np.abs(g @ x) / u) - 1:.2e} outside the box")
    return BoxSolution(h=x, multipliers=y, iterations=MAX_ITERATIONS, polished=False)


# --- the discrepancy principle ---------------------------------------------------------------


def _bisect[T](
    evaluate: Callable[[float], tuple[float, T]], lo: float, hi: float, target: float
) -> tuple[float, float, T, bool]:
    """The smallest alpha in `[lo, hi]` whose residual meets `target`, bisecting in log alpha.

    `evaluate(alpha)` returns the residual, nondecreasing in alpha (Lemmas 4.9, 4.11), and its
    payload. `lo`'s residual is below the target. Returns `alpha, residual, payload, at_bracket`;
    `at_bracket` says even `hi` fell short, and `hi` is returned.
    """

    r_hi, best = evaluate(hi)
    if r_hi < target:
        return hi, r_hi, best, True
    alpha, r_best = hi, r_hi
    for _ in range(MAX_EVALUATIONS):
        if hi / lo <= 1 + BISECTION_TOL:
            break
        mid = math.sqrt(lo * hi)
        r_mid, payload = evaluate(mid)
        if r_mid >= target or abs(r_mid - target) <= BISECTION_TOL * target:
            hi, alpha, r_best, best = mid, mid, r_mid, payload
            if abs(r_mid - target) <= BISECTION_TOL * target:
                break
        else:
            lo = mid
    return alpha, r_best, best, False


def discrepancy(z: Array, mask: Mask, dt: float, bounds: Bounds, *, target: float) -> Fit:
    """Choose alpha with `||M^(1/2)(h_alpha - z)||_dt = target` (§6), and solve at it.

    `target` is `tau * delta`. In order:
    1. `target >= dist(z, K_h ∩ ker R)`: no alpha reaches it; the feasible line (`noise_dominates`).
    2. The unconstrained Morozov parameter, by bisection on banded solves; if its estimate is
       feasible it is also the constrained one (Theorem 4.12, last claim): `box_inactive`.
    3. Otherwise `target <= r_K(alpha_min)` is `data_outside_bounds`, and anything else is found
       by bisecting `r_K` in log alpha (Lemma 4.11), warm-starting each QP: `morozov`.
    """

    line = feasible_line(z, mask, dt, bounds.v_max)
    r_hi = residual(line, z, mask, dt)
    if target >= r_hi:
        return Fit(status="noise_dominates", target=target, r_hi=r_hi, h=line, residual=r_hi)

    a_lo, a_hi = alpha_for_cutoff(CUTOFF_MAX_HZ), alpha_for_cutoff(CUTOFF_MIN_HZ)
    evaluations = 0
    iterations = 0

    def r_u(alpha: float) -> tuple[float, Array]:
        nonlocal evaluations
        evaluations += 1
        h = unconstrained(z, mask, dt, alpha)
        return residual(h, z, mask, dt), h

    r_u_lo, _ = r_u(a_lo)
    if r_u_lo < target:
        alpha_m, r_m, h_m, at_bracket = _bisect(r_u, a_lo, a_hi, target)
        if is_feasible(h_m, dt, bounds, rtol=0.0):
            return Fit(
                status="box_inactive",
                target=target,
                r_hi=r_hi,
                alpha=alpha_m,
                h=h_m,
                h_unconstrained=h_m,
                multipliers=np.zeros(2 * z.size - 3),
                residual=r_m,
                evaluations=evaluations,
                at_bracket=at_bracket,
            )

    warm: BoxSolution | None = None

    def r_k(alpha: float) -> tuple[float, BoxSolution]:
        nonlocal evaluations, iterations, warm
        evaluations += 1
        warm = solve_box(z, mask, dt, alpha, bounds, warm=warm)
        iterations += warm.iterations
        return residual(warm.h, z, mask, dt), warm

    r_lo, _ = r_k(a_lo)
    if target <= r_lo:
        return Fit(
            status="data_outside_bounds",
            target=target,
            r_hi=r_hi,
            r_lo=r_lo,
            evaluations=evaluations,
            iterations=iterations,
        )
    alpha_k, r_best, best, at_bracket = _bisect(r_k, a_lo, a_hi, target)
    return Fit(
        status="morozov",
        target=target,
        r_hi=r_hi,
        alpha=alpha_k,
        h=best.h,
        h_unconstrained=unconstrained(z, mask, dt, alpha_k),
        multipliers=best.multipliers,
        residual=r_best,
        r_lo=r_lo,
        evaluations=evaluations,
        iterations=iterations,
        polished=best.polished,
        at_bracket=at_bracket,
    )
