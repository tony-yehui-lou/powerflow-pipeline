"""S8 · 3D joints from 2D keypoints, the lifter's bone lengths, the bar and the floor.

`docs/specs/preprocessing/S8-joint-hva-draft3.md` §3. A pixel fixes a joint's direction from the
camera but not its range. The body supplies the range: bones of known length chain the joints
together, and the chain is anchored at both ends -- the hands hold the bar (S5), the grounded feet
stand on the floor. Each frame is one small least-squares problem over every joint at once:

- reprojection onto S6's pixels (robust, so one misplaced keypoint is outvoted by the bones);
- bone lengths, each with a sigma (`athlete_input`);
- wrists on the bar at the measured half grip, while the hands hold it;
- grounded ankles at the ankle height above the floor.

Nothing else: no symmetry prior, no pull towards a start, no smoothing across frames. A joint
S6 did not see (`unseen`) is not solved and gets no position -- its height would be a guess.

Units are millimetres in S4's levelled camera frame (X right, Y down, Z forward), as in
`bar_geometry`. Heights are `y_floor - Y`.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares

from powerflow_pipeline.data.common.models import JointId
from powerflow_pipeline.data.preprocess import athlete_input as ai
from powerflow_pipeline.data.preprocess.bar_geometry import Pixel, Point, normalised
from powerflow_pipeline.data.preprocess.body_plane import (
    Side,
    lateral_offsets,
    place_on_floor,
    place_on_plane,
)
from powerflow_pipeline.data.preprocess.models import Intrinsics

Vector = NDArray[np.float64]
Anchor = Literal["bar", "floor"]
JointSource = Literal["observed", "weak", "unseen"]
NoFitReason = Literal["no_anchor", "too_few_joints", "no_fit"]

# The clavicles are interpolated in S6 and carry no information of their own: derived afterwards.
SOLVED_JOINTS: Final[tuple[JointId, ...]] = (
    "leftAnkle",
    "rightAnkle",
    "leftKnee",
    "rightKnee",
    "leftHip",
    "rightHip",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "head",
)
INDEX: Final[dict[JointId, int]] = {joint: i for i, joint in enumerate(SOLVED_JOINTS)}
ANKLES: Final[tuple[JointId, JointId]] = ("leftAnkle", "rightAnkle")
ALL_ANCHORS: Final[frozenset[Anchor]] = frozenset({"bar", "floor"})

# Confidence at or above which a joint is `observed`; between MIN_CONF and this it is `weak`;
# below MIN_CONF it is `unseen`: not solved, and no position is published for it.
OBSERVED_CONF: Final = 0.7
MIN_CONF: Final = 0.3
# Keypoint noise at full confidence; a joint's pixel sigma is this over its confidence.
SIGMA_PX: Final = 3.0
# Draft 3 §3.2's sigmas, in millimetres.
WRIST_PLANE_SIGMA_MM: Final = 20.0
BAR_AXIS_SLACK_MM: Final = 60.0  # a wrist sits off the bar; beyond this it is not holding it
# Fewer observed joints than this and the frame is not solved.
MIN_OBSERVED_JOINTS: Final = 6
# The fit's RMS normalised residual above this is no fit at all.
MAX_RMS: Final = 5.0
# Two starts whose solutions differ by more than this at any joint found different minima.
BRANCH_MM: Final = 50.0
# ... and two fits this close in RMS normalised residual fit equally well.
BRANCH_RMS_TIE: Final = 0.2
# A grounded ankle further than this many sigmas from the floor after the solve is a conflict.
GROUND_CONFLICT_SIGMAS: Final = 3.0
# Grounded, in 2D: the ankle's pixel moves less than this per frame over +-GROUND_WINDOW frames.
GROUND_PX: Final = 4.0
GROUND_WINDOW: Final = 3
# Shoulder and hip widths for the start when the body has neither (mm).
DEFAULT_SHOULDER_WIDTH_MM: Final = 380.0
DEFAULT_HIP_WIDTH_MM: Final = 180.0


@dataclass(frozen=True)
class BarHold:
    """The bar on a frame the hands hold it: where it is, and where the hands are on it."""

    centre_mm: Point
    axis: Point  # unit, cap -> collar
    half_grip_mm: float
    near_side: Side  # the side nearer the cap, at negative offsets along `axis`


@dataclass(frozen=True)
class FrameObs:
    """Everything one frame contributes: S6's pixels, the bar if held, which ankles are down."""

    pixels: Mapping[JointId, Pixel | None]
    confidence: Mapping[JointId, float]
    hold: BarHold | None
    grounded: frozenset[JointId]


@dataclass(frozen=True)
class Body:
    """Per-capture constants: each bone's (length, sigma) in mm, and the floor's Y."""

    lengths: Mapping[str, tuple[float, float]]
    y_floor_mm: float


@dataclass(frozen=True)
class FrameFit:
    points: dict[JointId, Point]
    height_sigma_mm: dict[JointId, float]
    source: dict[JointId, JointSource]
    anchors: frozenset[Anchor]
    rms: float
    branch_ambiguous: bool = False
    ground_conflict: frozenset[JointId] = frozenset()
    # The full solver state, unseen joints included, only to start the next frame's solve.
    solution: tuple[float, ...] = ()

    def vector(self) -> Vector:
        if self.solution:
            return np.array(self.solution, dtype=float)
        return np.array([self.points[j] for j in SOLVED_JOINTS], dtype=float).ravel()


@dataclass(frozen=True)
class NoFit:
    reason: NoFitReason


@dataclass(frozen=True)
class CaptureSolve:
    fits: list[FrameFit | NoFit]
    # Per frame: the shoulder midpoint reached from the feet against from the hands (mm); None
    # unless the frame had both anchors and both one-anchor solves succeeded.
    closure_mm: list[float | None]


def joint_source(obs: FrameObs, joint: JointId) -> JointSource:
    conf = obs.confidence.get(joint, 0.0)
    if obs.pixels.get(joint) is None or conf < MIN_CONF:
        return "unseen"
    return "observed" if conf >= OBSERVED_CONF else "weak"


def grounded_flags(pixels: Sequence[Pixel | None], confidence: Sequence[float]) -> list[bool]:
    """Per frame, whether this ankle is on the floor -- decided in 2D, before any solve.

    Grounded when its pixel moves less than GROUND_PX per frame on every step within
    +-GROUND_WINDOW frames (at least GROUND_WINDOW such steps seen), at confidence >= MIN_CONF.
    That drops the heel rise of the second pull, the jump, and a split.
    """

    n = len(pixels)
    steps: list[float | None] = [None] * n  # step i: from frame i - 1 to frame i
    for i in range(1, n):
        a, b = pixels[i - 1], pixels[i]
        if a is not None and b is not None:
            steps[i] = math.dist(a, b)
    flags = []
    for i in range(n):
        if pixels[i] is None or confidence[i] < MIN_CONF:
            flags.append(False)
            continue
        near = [s for s in steps[max(1, i - GROUND_WINDOW + 1) : i + GROUND_WINDOW + 1]]
        seen = [s for s in near if s is not None]
        flags.append(len(seen) >= GROUND_WINDOW and all(s < GROUND_PX for s in seen))
    return flags


def body_from_lengths(lengths: Mapping[str, ai.Length], *, y_floor_mm: float) -> Body:
    """The solve's constants from `athlete_input`'s lengths; a bone with none is left out."""

    return Body(
        lengths={
            bone_id: (length.used_mm, length.sigma_mm)
            for bone_id, length in lengths.items()
            if length.used_mm is not None
        },
        y_floor_mm=y_floor_mm,
    )


def _endpoint(end: ai.Endpoint) -> tuple[int, int]:
    """A bone end as two indices whose midpoint it is (the same index twice for a joint)."""

    if isinstance(end, tuple):
        return INDEX[end[0]], INDEX[end[1]]
    return INDEX[end], INDEX[end]


def _partner(joint: JointId) -> JointId | None:
    if joint.startswith("left"):
        return f"right{joint[4:]}"  # type: ignore[return-value]
    if joint.startswith("right"):
        return f"left{joint[5:]}"  # type: ignore[return-value]
    return None


def initial_guess(obs: FrameObs, body: Body, k: Intrinsics) -> Vector | None:
    """A closed-form start for the solve, or None when nothing anchors the frame.

    With the bar: draft 1's body-plane solve, each joint on its plane beside the bar's
    mid-plane. With only grounded ankles: the ankles on the floor, every other joint on its ray
    at the ankles' depth. A joint left unplaced takes its partner mirrored across the mid-plane,
    or the centroid of what was placed.
    """

    points: dict[JointId, Point | None] = dict.fromkeys(SOLVED_JOINTS)

    def usable(joint: JointId) -> Pixel | None:
        return obs.pixels.get(joint) if joint_source(obs, joint) != "unseen" else None

    offsets: dict[JointId, float] | None = None
    hold = obs.hold
    if hold is not None:
        offsets = lateral_offsets(
            near_side=hold.near_side,
            half_grip_mm=hold.half_grip_mm,
            shoulder_width_mm=body.lengths.get("shoulderWidth", (DEFAULT_SHOULDER_WIDTH_MM,))[0],
            hip_width_mm=body.lengths.get("hipWidth", (DEFAULT_HIP_WIDTH_MM,))[0],
        )
        for joint in SOLVED_JOINTS:
            px = usable(joint)
            if px is not None:
                points[joint] = place_on_plane(px, k, hold.centre_mm, hold.axis, offsets[joint])
    elif obs.grounded and ai.ANKLE_HEIGHT in body.lengths:
        y_ankle = body.y_floor_mm - body.lengths[ai.ANKLE_HEIGHT][0]
        for ankle in obs.grounded:
            px = usable(ankle)
            if px is not None:
                points[ankle] = place_on_floor(px, k, y_ankle)
        depths = [p[2] for p in points.values() if p is not None]
        if not depths:
            return None
        z = statistics.fmean(depths)
        for joint in SOLVED_JOINTS:
            px = usable(joint)
            if points[joint] is None and px is not None:
                x, y = normalised(px, k)
                points[joint] = (x * z, y * z, z)
    else:
        return None

    placed = [p for p in points.values() if p is not None]
    if not placed:
        return None
    centroid = tuple(statistics.fmean(c) for c in zip(*placed, strict=True))
    for joint in SOLVED_JOINTS:
        if points[joint] is not None:
            continue
        partner = _partner(joint)
        mirror = points.get(partner) if partner is not None else None
        if mirror is not None and offsets is not None and hold is not None and partner is not None:
            shift = offsets[joint] - offsets[partner]
            points[joint] = tuple(m + shift * a for m, a in zip(mirror, hold.axis, strict=True))  # type: ignore[assignment]
        else:
            points[joint] = centroid  # type: ignore[assignment]
    return np.array([points[j] for j in SOLVED_JOINTS], dtype=float).ravel()


def _cauchy(r: Vector) -> Vector:
    """Each component rescaled so its square is Cauchy's rho, `ln(1 + r^2)`: quadratic near 0,
    and an influence that falls away far out, so a keypoint on something else is all but ignored.

    Draft 3 named soft-L1. Its influence stays constant however far out a keypoint is, and on the
    synthetic lift a knee pixel 210 px off still dragged the ankle 85 mm; Cauchy does not.
    """

    return np.sign(r) * np.sqrt(np.log1p(r * r))


class _Problem:
    """The residual vector of one frame, with every index and constant precomputed."""

    def __init__(
        self,
        obs: FrameObs,
        body: Body,
        k: Intrinsics,
        *,
        hold: BarHold | None,
        grounded: Sequence[JointId],
    ) -> None:
        self.k = k
        self.y_floor = body.y_floor_mm
        seen = {j for j in SOLVED_JOINTS if joint_source(obs, j) != "unseen"}
        pix = [(INDEX[j], obs.pixels[j], obs.confidence[j]) for j in SOLVED_JOINTS if j in seen]
        self.pix_idx = np.array([i for i, _, _ in pix], dtype=int)
        self.pix = np.array([p for _, p, _ in pix], dtype=float).reshape(-1, 2)
        self.pix_sigma = np.array([SIGMA_PX / c for _, _, c in pix], dtype=float)

        # A bone ending at an unseen joint constrains nothing that is solved.
        bones = [
            (bone, body.lengths[bone.bone_id])
            for bone in ai.BONES
            if bone.b is not None
            and bone.bone_id in body.lengths
            and set(_bone_joints(bone)) <= seen
        ]
        ends = [(*_endpoint(bone.a), *_endpoint(bone.b)) for bone, _ in bones]  # type: ignore[arg-type]
        self.bone_ends = np.array(ends, dtype=int).reshape(-1, 4)
        self.bone_len = np.array([length for _, (length, _) in bones], dtype=float)
        self.bone_sigma = np.array([sigma for _, (_, sigma) in bones], dtype=float)

        self.hold = hold
        if hold is not None:
            self.centre = np.array(hold.centre_mm, dtype=float)
            self.axis = np.array(hold.axis, dtype=float)
            near = f"{hold.near_side}Wrist"
            wrists = [w for w in ("leftWrist", "rightWrist") if w in seen]
            self.wrist_idx = np.array([INDEX[w] for w in wrists], dtype=int)
            self.wrist_target = np.array(
                [-hold.half_grip_mm if w == near else hold.half_grip_mm for w in wrists],
                dtype=float,
            )

        self.ground_idx = np.array([INDEX[a] for a in grounded], dtype=int)
        if grounded:
            self.ankle_height, self.ankle_sigma = body.lengths[ai.ANKLE_HEIGHT]

    def residuals(self, x: Vector) -> Vector:
        p = x.reshape(-1, 3)
        parts: list[Vector] = []

        q = p[self.pix_idx]
        z = np.maximum(q[:, 2], 1.0)
        uv = np.stack([self.k.fx * q[:, 0] / z + self.k.cx, self.k.fy * q[:, 1] / z + self.k.cy])
        parts.append(_cauchy(((uv.T - self.pix) / self.pix_sigma[:, None]).ravel()))

        e = self.bone_ends
        a = (p[e[:, 0]] + p[e[:, 1]]) / 2.0
        b = (p[e[:, 2]] + p[e[:, 3]]) / 2.0
        parts.append((np.linalg.norm(a - b, axis=1) - self.bone_len) / self.bone_sigma)

        if self.hold is not None:
            rel = p[self.wrist_idx] - self.centre
            along = rel @ self.axis
            parts.append((along - self.wrist_target) / WRIST_PLANE_SIGMA_MM)
            off_axis = np.linalg.norm(rel - np.outer(along, self.axis), axis=1)
            parts.append(np.maximum(0.0, off_axis - BAR_AXIS_SLACK_MM) / WRIST_PLANE_SIGMA_MM)

        if self.ground_idx.size:
            heights = self.y_floor - p[self.ground_idx, 1]
            parts.append((heights - self.ankle_height) / self.ankle_sigma)

        return np.concatenate(parts)

    def jacobian(self, x: Vector) -> Vector:
        """`residuals`' Jacobian, analytically: a finite-difference one costs 40 residual
        evaluations, and a capture runs a thousand-odd solves (two starts plus two closure
        solves per frame)."""

        p = x.reshape(-1, 3)
        n = x.size
        blocks: list[Vector] = []

        q = p[self.pix_idx]
        z = np.maximum(q[:, 2], 1.0)
        live = q[:, 2] >= 1.0
        fx, fy = self.k.fx, self.k.fy
        u = fx * q[:, 0] / z + self.k.cx
        v = fy * q[:, 1] / z + self.k.cy
        r = ((np.stack([u, v], axis=1) - self.pix) / self.pix_sigma[:, None]).ravel()
        log = np.log1p(r * r)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(np.abs(r) < 1e-8, 1.0, np.abs(r) / ((1 + r * r) * np.sqrt(log)))
        block = np.zeros((2 * len(self.pix_idx), n))
        for row, (i, sigma) in enumerate(zip(self.pix_idx, self.pix_sigma, strict=True)):
            c = 3 * int(i)
            block[2 * row, c] = fx / z[row]
            block[2 * row + 1, c + 1] = fy / z[row]
            if live[row]:
                block[2 * row, c + 2] = -fx * q[row, 0] / z[row] ** 2
                block[2 * row + 1, c + 2] = -fy * q[row, 1] / z[row] ** 2
            block[2 * row : 2 * row + 2] /= sigma
        blocks.append(block * scale[:, None])

        e = self.bone_ends
        a = (p[e[:, 0]] + p[e[:, 1]]) / 2.0
        b = (p[e[:, 2]] + p[e[:, 3]]) / 2.0
        diff = a - b
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-9)
        grad = diff / dist[:, None] / self.bone_sigma[:, None] / 2.0
        block = np.zeros((len(e), n))
        for row in range(len(e)):
            for col, sign in ((0, 1.0), (1, 1.0), (2, -1.0), (3, -1.0)):
                c = 3 * int(e[row, col])
                block[row, c : c + 3] += sign * grad[row]
        blocks.append(block)

        if self.hold is not None:
            axis = self.axis
            block = np.zeros((len(self.wrist_idx), n))
            hinge = np.zeros((len(self.wrist_idx), n))
            rel = p[self.wrist_idx] - self.centre
            along = rel @ axis
            perp = rel - np.outer(along, axis)
            off = np.linalg.norm(perp, axis=1)
            for row, i in enumerate(self.wrist_idx):
                c = 3 * int(i)
                block[row, c : c + 3] = axis / WRIST_PLANE_SIGMA_MM
                if off[row] > BAR_AXIS_SLACK_MM:
                    hinge[row, c : c + 3] = perp[row] / off[row] / WRIST_PLANE_SIGMA_MM
            blocks.extend([block, hinge])

        if self.ground_idx.size:
            block = np.zeros((len(self.ground_idx), n))
            for row, i in enumerate(self.ground_idx):
                block[row, 3 * int(i) + 1] = -1.0 / self.ankle_sigma
            blocks.append(block)

        return np.vstack(blocks)


def solve_frame(
    obs: FrameObs,
    body: Body,
    k: Intrinsics,
    *,
    starts: Sequence[Vector],
    anchors: frozenset[Anchor] = ALL_ANCHORS,
    previous: Vector | None = None,
) -> FrameFit | NoFit:
    """One frame's skeleton, from each start in turn; the lowest-cost acceptable fit wins.

    Two acceptable fits that disagree by more than BRANCH_MM flag `branch_ambiguous`. Where
    they also fit equally well (RMS within BRANCH_RMS_TIE) the frame alone cannot choose -- a
    floor-only frame has exact mirror solutions -- and the one nearer `previous`, the last
    frame's solution, wins: continuity decides what geometry cannot.

    `anchors` restricts which anchors may be used, for the one-anchor solves of the closure
    test.
    """

    hold = obs.hold if "bar" in anchors else None
    grounded = (
        [a for a in ANKLES if a in obs.grounded]
        if "floor" in anchors and ai.ANKLE_HEIGHT in body.lengths
        else []
    )
    used_anchors: list[Anchor] = []
    if hold is not None:
        used_anchors.append("bar")
    if grounded:
        used_anchors.append("floor")
    used = frozenset(used_anchors)
    if not used:
        return NoFit("no_anchor")
    observed = [j for j in SOLVED_JOINTS if joint_source(obs, j) == "observed"]
    if len(observed) < MIN_OBSERVED_JOINTS:
        return NoFit("too_few_joints")
    if not starts:
        return NoFit("no_fit")

    # Only the joints that were seen are compared: an unseen joint's coordinates never move.
    seen = np.array([joint_source(obs, j) != "unseen" for j in SOLVED_JOINTS])
    fits: list[tuple[float, Vector, FrameFit]] = []
    for start in starts:
        problem = _Problem(obs, body, k, hold=hold, grounded=grounded)
        result = least_squares(problem.residuals, start, jac=problem.jacobian, method="trf")
        m, n = result.fun.size, result.x.size
        rms = math.sqrt(2.0 * result.cost / m)
        if not result.success or rms > MAX_RMS:
            continue
        fits.append((float(result.cost), result.x, _frame_fit(obs, body, result, used, rms, m, n)))
    if not fits:
        return NoFit("no_fit")

    fits.sort(key=lambda item: item[0])
    best_rms = fits[0][2].rms
    tied = [item for item in fits if item[2].rms - best_rms <= BRANCH_RMS_TIE]
    if previous is not None and len(tied) > 1:
        last = previous.reshape(-1, 3)[seen]
        tied.sort(
            key=lambda item: float(
                np.sum(np.linalg.norm(item[1].reshape(-1, 3)[seen] - last, axis=1))
            )
        )
    best_x = tied[0][1].reshape(-1, 3)[seen]
    ambiguous = any(
        float(np.max(np.linalg.norm(x.reshape(-1, 3)[seen] - best_x, axis=1))) > BRANCH_MM
        for _, x, _ in fits
    )
    return replace(tied[0][2], branch_ambiguous=ambiguous)


def _frame_fit(
    obs: FrameObs,
    body: Body,
    result: OptimizeResult,
    anchors: frozenset[Anchor],
    rms: float,
    m: int,
    n: int,
) -> FrameFit:
    x = np.asarray(result.x, dtype=np.float64).reshape(-1, 3)
    jac = np.asarray(result.jac, dtype=np.float64)
    scale = max(1.0, 2.0 * float(result.cost) / (m - n)) if m > n else 1.0
    cov = np.linalg.pinv(jac.T @ jac) * scale
    source = {j: joint_source(obs, j) for j in SOLVED_JOINTS}
    seen = [j for j in SOLVED_JOINTS if source[j] != "unseen"]
    points = {
        j: (float(x[INDEX[j], 0]), float(x[INDEX[j], 1]), float(x[INDEX[j], 2])) for j in seen
    }
    sigma = {j: math.sqrt(max(0.0, float(cov[3 * INDEX[j] + 1, 3 * INDEX[j] + 1]))) for j in seen}

    conflict: set[JointId] = set()
    if "floor" in anchors:
        height, ankle_sigma = body.lengths[ai.ANKLE_HEIGHT]
        for ankle in ANKLES:
            if ankle in obs.grounded:
                off = abs(body.y_floor_mm - points[ankle][1] - height)
                if off > GROUND_CONFLICT_SIGMAS * ankle_sigma:
                    conflict.add(ankle)
    return FrameFit(
        points=points,
        height_sigma_mm=sigma,
        source=source,
        anchors=anchors,
        rms=rms,
        ground_conflict=frozenset(conflict),
        solution=tuple(float(v) for v in x.ravel()),
    )


def _shoulder_mid(fit: FrameFit) -> Point | None:
    a, b = fit.points.get("leftShoulder"), fit.points.get("rightShoulder")
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def solve_capture(frames: Sequence[FrameObs], body: Body, k: Intrinsics) -> CaptureSolve:
    """Every frame's fit, each started from its own closed-form guess and its predecessor's
    solution; then, where both anchors held, the two one-anchor solves of the closure test."""

    fits: list[FrameFit | NoFit] = []
    previous: Vector | None = None
    for obs in frames:
        guess = initial_guess(obs, body, k)
        starts = [s for s in (guess, previous) if s is not None]
        fit = solve_frame(obs, body, k, starts=starts, previous=previous)
        fits.append(fit)
        if isinstance(fit, FrameFit):
            previous = fit.vector()

    closure: list[float | None] = [None] * len(frames)
    for i, (obs, fit) in enumerate(zip(frames, fits, strict=True)):
        if not isinstance(fit, FrameFit) or fit.anchors != ALL_ANCHORS:
            continue
        # From the frame's own solution: each one-anchor solve then moves only as far as its
        # anchor pulls it. From the cold guess the legs' branch is free and can flip.
        start = fit.vector()
        by_bar = solve_frame(obs, body, k, starts=[start], anchors=frozenset({"bar"}))
        by_floor = solve_frame(obs, body, k, starts=[start], anchors=frozenset({"floor"}))
        if isinstance(by_bar, FrameFit) and isinstance(by_floor, FrameFit):
            a, b = _shoulder_mid(by_bar), _shoulder_mid(by_floor)
            if a is not None and b is not None:
                closure[i] = math.dist(a, b)
    return CaptureSolve(fits=fits, closure_mm=closure)


def _bone_joints(bone: ai.BoneSpec) -> list[JointId]:
    assert bone.b is not None
    return [j for end in (bone.a, bone.b) for j in (end if isinstance(end, tuple) else (end,))]
