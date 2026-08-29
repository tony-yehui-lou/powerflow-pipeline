# S4 Crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the S4 Crop stage specified in `docs/specs/preprocessing/6-cropping.md`:
for each camera, measure residual translation from odometry, read S3's valid-content region,
intersect a motion guard band with it, and crop RGB/depth/confidence to one fixed rectangle —
publishing to a new `s4_crop_output` root and establishing the per-camera form of invariant
**I5**.

**Architecture:** A new pure-maths module, `data/preprocess/crop.py` (no I/O, no Prefect — the
boundary `retilt.py` already holds), carries the reference-frame displacement, the guard-band
formula, the valid/motion intersection, the even-extent fix, and the intrinsics rewrite. A new
task module, `data/preprocess/tasks/crop.py`, wires that maths to I/O and follows
`retilt_camera`'s staging/publish shape exactly. The flow calls `crop_camera` after
`retilt_camera` inside the existing per-camera loop, binding the record `retilt_camera`
currently discards.

**Tech Stack:** Python 3.12, PyAV, OpenCV, numpy, scipy (`Rotation`), pandas, Prefect 3,
pytest, Ruff, mypy.

## Global Constraints

- No new dependencies — `torch`, `opencv-contrib-python`, `av`, `pandas`, `scipy` are already
  in `pyproject.toml`.
- Do not run the real `powerflow preprocess` pipeline until Task 7, and only then as directed.
- One rectangle per camera, never per frame.
- This is a pure slice: no resampling, no interpolation, anywhere. Depth stays `uint16`;
  confidence stays `⊆ {0, 1, 2}`, exactly the input's values in the cropped region.
- RGB output width and height must both be even — `libx264` + `yuv420p` refuses an odd
  dimension (verified directly: encoding 1420×1673 raises `ExternalError` from
  `avcodec_open2`; 1420×1672 succeeds). Three of the four real cameras produce an odd height
  before this fix.
- Default tunables must let `tests/integration/test_preprocess_flow.py::test_the_cli_runs_the_flow`
  pass with **no** tunable overrides, the same constraint the existing `build_capture` comment
  already records for `retilt_min_floor_points`.
- Preserve unrelated pre-existing workspace changes.
- TDD per `CLAUDE.md`: failing test first, then minimum implementation, for every task that
  touches `src/`.
- Coverage must stay ≥ 90% (current baseline: 197 tests, 96.17%). `mypy` is `strict = true`
  over both `src` and `tests` — every new test needs full type annotations.

## Measured calibration (real cameras, `../data/s3_retilt_output/11 July/`)

Computed from each camera's `retilt_sidecar.json`, `odometry.csv`, and `camera_matrix.csv`,
using every retained frame's odometry row (not S3's 32-frame sample) and a depth sample every
20th frame:

| camera | max‖Δp‖ | Z_guard (q=0.05) | (m_x, m_y) | `crop_bounds_px` (pre-shrink) | `bound_source` | RGB out (pre-shrink) |
|---|---|---|---|---|---|---|
| 30kg_Set1/Front | 1.14 mm | 1.40 m | 2, 1 px | `[10, 238, 1430, 1911]` | L/R/B=motion, T=valid | 1420×1673 |
| 30kg_Set1/Side | 13.25 mm | 1.88 m | 10, 4 px | `[18, 268, 1422, 1908]` | L/R/B=motion, T=valid | 1404×1640 |
| 50kg_Set3/Front | 1.73 mm | 1.40 m | 2, 2 px | `[10, 241, 1430, 1910]` | L/R/B=motion, T=valid | 1420×1669 |
| 50kg_Set3/Side | 16.55 mm | 1.87 m | 13, 3 px | `[21, 246, 1419, 1909]` | L/R/B=motion, T=valid | 1398×1663 |

`valid_bounds_px` for all four is `[0, {238,268,241,246}, 1440, 1920]` (S3's top-only border).
After `shrink_to_even` (Task 3), only Front/Front's odd heights (1673, 1669) change, by one row
each — widths are already even on all four.

Three findings this plan corrects that `6-cropping.md` does not anticipate:

1. **§3's "1–5 mm" translation figure is an S3 sampling artifact.** S3 only samples 32 frames;
   across every retained frame, the Side cameras drift 13.25 mm and 16.55 mm. Default
   `crop_max_residual_translation_m = 0.025` (see Task 1) — well above the real spread, still
   stricter than S3's own `0.05`.
2. **`retilt.translation_span_m` is not reusable for §1.** It is max *pairwise* distance over
   world-frame positions (O(N²), and it discards the signed per-axis components §3's margin
   formula needs). Task 3 writes a dedicated `reference_displacements` helper.
3. **Odd output height crashes the encoder** — see Global Constraints above; fixed by
   `shrink_to_even` (Task 3) and exercised by every test that checks output dimensions.

## Decision on I5's scope (resolved before this plan was written)

`6-cropping.md`'s open question — whether invariant I5 is per-camera or requires a
cross-camera reconciliation step — was raised with the project owner. **Decision: per-camera,
exactly as `6-cropping.md` §3 is written.** Consequence, recorded deliberately: the four real
cameras keep four different output sizes (1420×1672, 1404×1640, 1420×1668, 1398×1662 after
even-shrink), so `1-ingestion_orient.md`'s literal invariant table wording ("every image has
identical pixel dimensions") is not satisfied by S4 alone. Task 8 updates the spec's open
question to record this rather than leaving the invariant table silently wrong.

---

### Task 1: Config, CLI, and stage-root wiring

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/config.py`
- Modify: `src/powerflow_pipeline/data/cli.py`
- Modify: `prefect.yaml`
- Modify: `tests/unit/test_cli.py`
- Modify: `tests/unit/test_preprocess_ingest.py` (shared `make_config`)
- Modify: `tests/integration/test_preprocess_flow.py` (its own `make_config`)

**Interfaces:**
- Produces: `PreprocessConfig.crop_root`, in `stage_roots`; six `crop_*` tunables; a `--crop`
  CLI flag

- [ ] **Step 1: Write the failing CLI test**

Add to `tests/unit/test_cli.py`, mirroring the existing `--retilt` assertion:

```python
def test_preprocess_command_accepts_crop_root(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    result = runner.invoke(
        app,
        [
            "preprocess",
            "--input", str(raw),
            "--records", str(tmp_path / "s0"),
            "--cut", str(tmp_path / "s1"),
            "--retilt", str(tmp_path / "s3"),
            "--crop", str(tmp_path / "s4"),
            "--output", str(tmp_path / "s2"),
        ],
    )

    assert result.exit_code == 0, result.output
```

Run `uv run pytest --no-cov tests/unit/test_cli.py -q` and confirm it fails: `--crop` is not a
recognized option yet, and (separately) `test_preprocess_command_accepts_retilt_root` /
`test_the_cli_runs_the_flow` will start failing once Step 3 makes `--crop` required — expected,
fixed in Step 5.

- [ ] **Step 2: Add `crop_root` and the tunables to `PreprocessConfig`**

```python
crop_root: Path
crop_max_residual_translation_m: float = Field(default=0.025, gt=0)
crop_depth_guard_quantile: float = Field(default=0.05, gt=0, lt=1)
crop_safety_px: int = Field(default=8, ge=0)
crop_max_crop_fraction: float = Field(default=0.25, gt=0, lt=1)
crop_depth_sample_stride: int = Field(default=10, ge=1)
crop_max_sampled_depth_frames: int = Field(default=32, ge=1)
```

The last two are implementation tunables the design spec's §Configuration does not list (it
names four); they control how densely `Z_guard_m`'s depth sample is taken, mirroring S3's
`retilt_sample_stride`/`retilt_max_sampled_frames` pattern.

Add `self.crop_root` to the `stage_roots` tuple, after `retilt_root` and before `output_root`:

```python
return (self.record_root, self.cut_root, self.retilt_root, self.crop_root, self.output_root)
```

(`output_root` is S2 Orient's tree and stays last in the tuple — field declaration order does
not need to match stage order, but `stage_roots`'s order is what `write_session_metadata`
iterates, and it should read newest-to-not-yet-created stages last... actually check: S2's
`output_root` already sits after `retilt_root` in the existing tuple despite S2 running before
S3. Preserve that existing ordering exactly; only insert `crop_root` between `retilt_root` and
`output_root`.)

- [ ] **Step 3: Add `--crop` to the CLI and thread it into `PreprocessConfig`**

In `cli.py`, add a required `typer.Option` immediately after `retilt_root`, following the exact
pattern:

```python
    crop_root: Annotated[
        Path, typer.Option("--crop", help="Where S4 publishes the cropped streams.")
    ],
```

Add `crop_root=crop_root,` to the `PreprocessConfig(...)` construction, and update the command
docstring to mention S4 ("...retilt it level with the floor (S3), and crop it to a stable
common region (S4).").

- [ ] **Step 4: Add `crop_root` to `prefect.yaml`**

```yaml
    parameters:
      config:
        raw_root: ../data/raw
        record_root: ../data/s0_ingest_output
        cut_root: ../data/s1_cut_output
        retilt_root: ../data/s3_retilt_output
        crop_root: ../data/s4_crop_output
        output_root: ../data/s2_orient_output
        rotation: cw
```

Update the deployment `description` to mention S4.

- [ ] **Step 5: Fix the two now-broken call sites and verify GREEN**

`--crop` is required, so `test_preprocess_command_accepts_retilt_root` (`tests/unit/test_cli.py`)
and `test_the_cli_runs_the_flow` (`tests/integration/test_preprocess_flow.py`) now fail with a
missing-option error. Add `"--crop", str(<a new tmp path>),` to each invocation's argument list.

Update the two config builders so every other unit/integration test keeps working:

```python
# tests/unit/test_preprocess_ingest.py::make_config
def make_config(tmp_path: Path, **overrides: Any) -> PreprocessConfig:
    return PreprocessConfig(
        raw_root=tmp_path / "raw",
        record_root=tmp_path / "s0",
        cut_root=tmp_path / "s1",
        retilt_root=tmp_path / "s3",
        crop_root=tmp_path / "s4",
        output_root=tmp_path / "s2",
        **overrides,
    )
```

```python
# tests/integration/test_preprocess_flow.py::make_config
def make_config(tmp_path: Path, raw: Path, **overrides: Any) -> PreprocessConfig:
    overrides.setdefault("retilt_min_floor_points", 10)
    return PreprocessConfig(
        raw_root=raw,
        record_root=tmp_path / "s0_ingest_output",
        cut_root=tmp_path / "s1_cut_output",
        retilt_root=tmp_path / "s3_retilt_output",
        crop_root=tmp_path / "s4_crop_output",
        output_root=tmp_path / "s2_orient_output",
        **overrides,
    )
```

```bash
uv run pytest --no-cov tests/unit/test_cli.py -q
uv run pytest --no-cov tests/unit -q
uv run mypy src
```

Expected: all pass. (`tests/integration` is not run yet — nothing calls `crop_camera` until
Task 6, so its config-builder change is inert for now; it is fixed here so the file stays
importable.)

- [ ] **Step 6: Commit**

```bash
git add src/powerflow_pipeline/data/preprocess/config.py src/powerflow_pipeline/data/cli.py \
  prefect.yaml tests/unit/test_cli.py tests/unit/test_preprocess_ingest.py \
  tests/integration/test_preprocess_flow.py
git commit -m "feat(preprocess): add S4 crop config, CLI flag, and stage root"
```

---

### Task 2: Promote `_quat_to_matrix` to public

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/retilt.py`

**Interfaces:**
- Changes: `_quat_to_matrix(qx, qy, qz, qw) -> np.ndarray` becomes `quat_to_matrix(qx, qy, qz,
  qw) -> np.ndarray` — Task 3 imports this from `retilt.py` for `reference_displacements`

Rename-only; no behavior change. Its own task so a reviewer can approve it independently of
Task 3's new maths.

- [ ] **Step 1: Rename the function and its one call site**

```python
def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Standard unit-quaternion -> rotation-matrix conversion, `(x, y, z, w)` order."""

    matrix: np.ndarray = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return matrix
```

At `retilt.py:313` (inside `gravity_down_camera`):

```python
    r_wc = quat_to_matrix(qx, qy, qz, qw)  # world -> ARKit camera-local
```

- [ ] **Step 2: Run the full retilt suite to confirm nothing else referenced the private name**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_retilt_math.py tests/unit/test_preprocess_retilt.py -q
uv run mypy src
uv run ruff check src/powerflow_pipeline/data/preprocess/retilt.py
```

Expected: all green — this is a pure rename, so nothing should change behaviorally.

- [ ] **Step 3: Commit**

```bash
git add src/powerflow_pipeline/data/preprocess/retilt.py
git commit -m "refactor(preprocess): make quat_to_matrix public for reuse by S4 crop"
```

---

### Task 3: Pure maths — `data/preprocess/crop.py`

**Files:**
- New: `src/powerflow_pipeline/data/preprocess/crop.py`
- New: `tests/unit/test_preprocess_crop_math.py`

**Interfaces:**
- Consumes: `CropBounds`, `ScanRejected` (`common`); `Intrinsics` (`preprocess/models.py`);
  `quat_to_matrix` (`preprocess/retilt.py`, Task 2)
- Produces: every function in the table below — no Prefect `@task`, no filesystem writes

| Function | Contract |
|---|---|
| `read_valid_bounds(sidecar: dict[str, Any]) -> CropBounds` | §2; parses the sidecar's flat 4-list; `ScanRejected` if absent, not a 4-list, non-integer, or degenerate |
| `reference_displacements(positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray` | §1; `Δp_i = R_0ᵀ(p_i − p_0)`, shape `(N,3)`; `ScanRejected` on any non-finite value |
| `max_translation_m(displacements: np.ndarray) -> float` | `max_i ‖Δp_i‖₂` |
| `guard_depth_m(depth_m: np.ndarray, quantile: float) -> float` | lower quantile of positive, confidence-nonzero depth samples, metres; `ScanRejected` on an empty sample |
| `motion_bounds_px(displacements, k, valid, z_guard_m, safety_px, size) -> tuple[int,int,int,int]` | §3 verbatim; returns raw `(left, top, right, bottom)`, possibly too large for `size` |
| `intersect_crop(valid, motion, size) -> tuple[CropBounds, dict[str,str]]` | §3's intersection plus `bound_source` (`"valid"`/`"motion"` per edge); `ScanRejected` on non-positive extent |
| `shrink_to_even(bounds: CropBounds) -> CropBounds` | trims at most one row/column off the bottom/right so both extents are even |
| `depth_bounds(crop, rgb_size, depth_size) -> CropBounds` | §3's inward-rounded `ceil`/`floor` rescale; `ScanRejected` on non-positive extent |
| `crop_intrinsics(k: Intrinsics, bounds: CropBounds) -> Intrinsics` | §4: `fx/fy` unchanged, `cx -= x0`, `cy -= y0` |
| `crop_fractions(bounds: CropBounds, size: tuple[int,int]) -> tuple[float,float]` | `(1 − width_frac, 1 − height_frac)` removed, for §6's `max_crop_fraction` gate |

- [ ] **Step 1: Write failing numeric tests, function by function**

```python
"""Pure maths for S4 Crop: residual translation, guard band, intersection, even-extent fix.

No I/O, no Prefect -- numeric behavior is checked against hand-worked values and the four
real cameras' measured crop rectangles (never accepted merely because it ran, per CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.models import CropBounds
from powerflow_pipeline.data.preprocess.crop import (
    crop_fractions,
    crop_intrinsics,
    depth_bounds,
    guard_depth_m,
    intersect_crop,
    max_translation_m,
    motion_bounds_px,
    read_valid_bounds,
    reference_displacements,
    shrink_to_even,
)
from powerflow_pipeline.data.preprocess.models import Intrinsics


# --- read_valid_bounds ------------------------------------------------------------------


def test_read_valid_bounds_parses_the_flat_list() -> None:
    sidecar = {"valid_bounds_px": [0, 238, 1440, 1920]}
    bounds = read_valid_bounds(sidecar)
    assert (bounds.x0, bounds.y0, bounds.x1, bounds.y1) == (0, 238, 1440, 1920)


def test_read_valid_bounds_rejects_missing_key() -> None:
    with pytest.raises(ScanRejected, match="valid_bounds_px"):
        read_valid_bounds({})


def test_read_valid_bounds_rejects_wrong_length() -> None:
    with pytest.raises(ScanRejected, match="valid_bounds_px"):
        read_valid_bounds({"valid_bounds_px": [0, 238, 1440]})


def test_read_valid_bounds_rejects_degenerate() -> None:
    with pytest.raises(ScanRejected, match="degenerate"):
        read_valid_bounds({"valid_bounds_px": [100, 238, 50, 1920]})  # x0 > x1


# --- reference_displacements -------------------------------------------------------------


def test_reference_displacements_zero_motion_is_zero() -> None:
    positions = np.zeros((5, 3))
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (5, 1))  # identity
    displacements = reference_displacements(positions, quaternions)
    assert displacements == pytest.approx(np.zeros((5, 3)))


def test_reference_displacements_expresses_world_motion_in_frame_zero() -> None:
    # Frame 0's camera-to-world rotation is 90 deg about Y: world X <- camera Z.
    from scipy.spatial.transform import Rotation

    r0 = Rotation.from_euler("y", 90, degrees=True)
    q0 = r0.as_quat()  # (x, y, z, w)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # world moves +1 in X
    quaternions = np.array([q0, q0])

    displacements = reference_displacements(positions, quaternions)

    # R_0^T maps world +X to camera +Z for this rotation.
    assert displacements[1] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


def test_reference_displacements_rejects_non_finite_pose() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]])
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    with pytest.raises(ScanRejected, match="finite"):
        reference_displacements(positions, quaternions)


# --- max_translation_m / guard_depth_m ----------------------------------------------------


def test_max_translation_m_is_the_largest_norm() -> None:
    displacements = np.array([[0.001, 0.0, 0.0], [0.0, 0.0, 0.02], [0.0, 0.0, 0.0]])
    assert max_translation_m(displacements) == pytest.approx(0.02)


def test_guard_depth_m_is_the_lower_quantile() -> None:
    samples = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    z = guard_depth_m(samples, quantile=0.0)
    assert z == pytest.approx(1.0)


def test_guard_depth_m_rejects_empty_sample() -> None:
    with pytest.raises(ScanRejected, match="depth"):
        guard_depth_m(np.array([]), quantile=0.05)


# --- motion_bounds_px: hand-worked against 30kg_Set1/Front (real capture) -----------------


def test_motion_bounds_px_matches_real_camera_30kg_front() -> None:
    # camera_matrix.csv: fx=1343.002, cx=724.818, cy=968.286; valid_bounds_px [0,238,1440,1920]
    k = Intrinsics(fx=1343.002, fy=1343.002, cx=724.818, cy=968.286, frame="portrait")
    valid = CropBounds(x0=0, y0=238, x1=1440, y1=1920)
    # A single displacement reproducing the measured max: dX=0.75mm, dY=0.80mm, dZ=0.93mm.
    displacements = np.array([[0.00075, 0.00080, 0.00093]])

    left, top, right, bottom = motion_bounds_px(
        displacements, k, valid, z_guard_m=1.40, safety_px=8, size=(1440, 1920)
    )

    assert (left, right) == (10, 10)  # ceil(~2.0) + 8
    assert (top, bottom) == (9, 9)  # ceil(~1.0) + 8


def test_motion_bounds_px_zero_displacement_is_exactly_safety_px() -> None:
    k = Intrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, frame="portrait")
    valid = CropBounds(x0=0, y0=0, x1=1000, y1=1000)
    displacements = np.zeros((3, 3))

    left, top, right, bottom = motion_bounds_px(
        displacements, k, valid, z_guard_m=1.5, safety_px=8, size=(1000, 1000)
    )

    assert (left, top, right, bottom) == (8, 8, 8, 8)


# --- intersect_crop: real-camera reproduction ---------------------------------------------


def test_intersect_crop_matches_real_camera_30kg_front() -> None:
    valid = CropBounds(x0=0, y0=238, x1=1440, y1=1920)
    bounds, source = intersect_crop(valid, (10, 9, 10, 9), size=(1440, 1920))

    assert (bounds.x0, bounds.y0, bounds.x1, bounds.y1) == (10, 238, 1430, 1911)
    assert source == {"left": "motion", "top": "valid", "right": "motion", "bottom": "motion"}


def test_intersect_crop_rejects_empty_overlap() -> None:
    valid = CropBounds(x0=0, y0=0, x1=100, y1=100)
    with pytest.raises(ScanRejected, match="overlap"):
        intersect_crop(valid, (60, 0, 60, 0), size=(100, 100))  # left+right >= width


# --- shrink_to_even: the encoder-parity fix -------------------------------------------------


def test_shrink_to_even_trims_an_odd_height() -> None:
    bounds = CropBounds(x0=10, y0=238, x1=1430, y1=1911)  # 1420 x 1673
    shrunk = shrink_to_even(bounds)
    assert (shrunk.width, shrunk.height) == (1420, 1672)
    assert (shrunk.x0, shrunk.y0, shrunk.x1) == (10, 238, 1430)  # only y1 moved


def test_shrink_to_even_leaves_already_even_bounds_alone() -> None:
    bounds = CropBounds(x0=18, y0=268, x1=1422, y1=1908)  # 1404 x 1640, both even
    assert shrink_to_even(bounds) == bounds


# --- depth_bounds: real-camera rescale ------------------------------------------------------


def test_depth_bounds_matches_real_camera_30kg_front() -> None:
    crop = CropBounds(x0=10, y0=238, x1=1430, y1=1910)  # post-even-shrink
    bounds = depth_bounds(crop, rgb_size=(1440, 1920), depth_size=(192, 256))
    assert (bounds.x0, bounds.y0, bounds.x1, bounds.y1) == (2, 32, 190, 254)


def test_depth_bounds_rejects_degenerate_result() -> None:
    crop = CropBounds(x0=0, y0=0, x1=2, y1=1920)  # sub-one-depth-pixel wide
    with pytest.raises(ScanRejected, match="degenerate"):
        depth_bounds(crop, rgb_size=(1440, 1920), depth_size=(192, 256))


# --- crop_intrinsics --------------------------------------------------------------------


def test_crop_intrinsics_shifts_the_principal_point_only() -> None:
    k = Intrinsics(fx=1343.002, fy=1343.002, cx=724.818, cy=968.286, frame="portrait")
    bounds = CropBounds(x0=10, y0=238, x1=1430, y1=1910)

    cropped = crop_intrinsics(k, bounds)

    assert cropped.fx == k.fx
    assert cropped.fy == k.fy
    assert cropped.cx == pytest.approx(714.818)
    assert cropped.cy == pytest.approx(730.286)
    assert cropped.frame == "portrait"


# --- crop_fractions ----------------------------------------------------------------------


def test_crop_fractions_matches_real_camera_30kg_front() -> None:
    bounds = CropBounds(x0=10, y0=238, x1=1430, y1=1910)
    frac_w, frac_h = crop_fractions(bounds, size=(1440, 1920))
    assert frac_w == pytest.approx(1.0 - 1420 / 1440)
    assert frac_h == pytest.approx(1.0 - 1672 / 1920)
```

Run `uv run pytest --no-cov tests/unit/test_preprocess_crop_math.py -q` and confirm RED — the
module does not exist.

- [ ] **Step 2: Implement `crop.py`**

```python
"""Pure maths for S4 Crop: residual translation, guard band, intersection (6-cropping.md)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import ValidationError

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.models import CropBounds
from powerflow_pipeline.data.preprocess.models import Intrinsics
from powerflow_pipeline.data.preprocess.retilt import quat_to_matrix


def read_valid_bounds(sidecar: dict[str, Any]) -> CropBounds:
    """Parse S3's `valid_bounds_px` flat 4-list, in rectified RGB pixel coordinates (§2)."""

    raw = sidecar.get("valid_bounds_px")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ScanRejected(f"retilt_sidecar.json valid_bounds_px missing or malformed: {raw!r}")
    try:
        x0, y0, x1, y1 = (int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ScanRejected(f"valid_bounds_px entries are not integers: {raw!r}") from exc
    try:
        return CropBounds(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValidationError as exc:
        raise ScanRejected(f"valid_bounds_px degenerate: {raw!r}") from exc


def reference_displacements(positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    """`Δp_i = R_0^T (p_i - p_0)`, expressed in the frame-0 camera frame (§1)."""

    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(quaternions)):
        raise ScanRejected("odometry pose is not finite")
    r0 = quat_to_matrix(*quaternions[0])  # camera-0 -> world
    return (positions - positions[0]) @ r0


def max_translation_m(displacements: np.ndarray) -> float:
    """`max_i ||Δp_i||_2` (§1)."""

    return float(np.linalg.norm(displacements, axis=1).max())


def guard_depth_m(depth_m: np.ndarray, quantile: float) -> float:
    """Lower `quantile` of positive, confidence-nonzero depth samples, in metres (§3)."""

    if depth_m.size == 0:
        raise ScanRejected("no valid depth samples available to derive Z_guard_m")
    return float(np.quantile(depth_m, quantile))


def motion_bounds_px(
    displacements: np.ndarray,
    k: Intrinsics,
    valid: CropBounds,
    z_guard_m: float,
    safety_px: int,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """§3's conservative pixel guard band: `(left, top, right, bottom)`, possibly larger than
    `size` allows -- callers intersect against §2's valid region before using them."""

    r_x = max(k.cx - valid.x0, valid.x1 - 1 - k.cx)
    r_y = max(k.cy - valid.y0, valid.y1 - 1 - k.cy)
    dx, dy, dz = displacements[:, 0], displacements[:, 1], displacements[:, 2]
    m_x = k.fx * np.abs(dx) / z_guard_m + r_x * np.abs(dz) / z_guard_m
    m_y = k.fy * np.abs(dy) / z_guard_m + r_y * np.abs(dz) / z_guard_m
    left = right = int(math.ceil(float(m_x.max()))) + safety_px
    top = bottom = int(math.ceil(float(m_y.max()))) + safety_px
    return left, top, right, bottom


def intersect_crop(
    valid: CropBounds, motion: tuple[int, int, int, int], size: tuple[int, int]
) -> tuple[CropBounds, dict[str, str]]:
    """§3's final rectangle: the intersection of the valid-content region and the motion guard
    band, plus which rectangle determined each edge (a tie resolves to `"valid"`)."""

    left, top, right, bottom = motion
    width, height = size
    x0, x1 = max(valid.x0, left), min(valid.x1, width - right)
    y0, y1 = max(valid.y0, top), min(valid.y1, height - bottom)
    try:
        bounds = CropBounds(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValidationError as exc:
        raise ScanRejected(
            f"motion guard band and valid-content region do not overlap: "
            f"valid={valid}, motion={motion}"
        ) from exc
    source = {
        "left": "valid" if x0 == valid.x0 else "motion",
        "top": "valid" if y0 == valid.y0 else "motion",
        "right": "valid" if x1 == valid.x1 else "motion",
        "bottom": "valid" if y1 == valid.y1 else "motion",
    }
    return bounds, source


def shrink_to_even(bounds: CropBounds) -> CropBounds:
    """Trim at most one row/column off the bottom/right so both extents are even -- libx264 +
    yuv420p refuses an odd width or height."""

    x1 = bounds.x1 - (bounds.width % 2)
    y1 = bounds.y1 - (bounds.height % 2)
    return CropBounds(x0=bounds.x0, y0=bounds.y0, x1=x1, y1=y1)


def depth_bounds(
    crop: CropBounds, rgb_size: tuple[int, int], depth_size: tuple[int, int]
) -> CropBounds:
    """§3's inward-rounded rescale from the RGB crop to depth/confidence resolution."""

    rgb_width, rgb_height = rgb_size
    depth_width, depth_height = depth_size
    x0 = math.ceil(crop.x0 * depth_width / rgb_width)
    x1 = math.floor(crop.x1 * depth_width / rgb_width)
    y0 = math.ceil(crop.y0 * depth_height / rgb_height)
    y1 = math.floor(crop.y1 * depth_height / rgb_height)
    try:
        return CropBounds(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValidationError as exc:
        raise ScanRejected(f"depth crop rectangle degenerate: {[x0, y0, x1, y1]}") from exc


def crop_intrinsics(k: Intrinsics, bounds: CropBounds) -> Intrinsics:
    """§4: a pure origin shift, no resampling -- `fx`/`fy` unchanged."""

    return Intrinsics(fx=k.fx, fy=k.fy, cx=k.cx - bounds.x0, cy=k.cy - bounds.y0, frame=k.frame)


def crop_fractions(bounds: CropBounds, size: tuple[int, int]) -> tuple[float, float]:
    """Fraction of width and of height removed, for §6's `max_crop_fraction` gate."""

    width, height = size
    return 1.0 - bounds.width / width, 1.0 - bounds.height / height
```

- [ ] **Step 3: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_crop_math.py -q
uv run mypy src
uv run ruff check src/powerflow_pipeline/data/preprocess/crop.py
```

- [ ] **Step 4: Commit**

```bash
git add src/powerflow_pipeline/data/preprocess/crop.py tests/unit/test_preprocess_crop_math.py
git commit -m "feat(preprocess): pure maths for S4 crop (translation, guard band, intersection)"
```

---

### Task 4: Fixture extensions — `odometry_translation` and `rgb_size`

**Files:**
- Modify: `tests/conftest.py`

**Interfaces:**
- Extends: `MakeCamera.__call__(..., odometry_translation: Callable[[int], tuple[float,float,float]] | None = None, rgb_size: tuple[int,int] = RGB_SIZE)`

Both default to today's exact behavior, so every existing test stays byte-identical.

- [ ] **Step 1: Add the `odometry_translation` knob**

`make_camera`'s odometry-row loop currently hard-codes `x, y, z = 0.0, 0.0, 0.0` on every row —
no fixture can exercise crop's §1 or §3 without real per-frame displacement. Add a parameter:

```python
        odometry_translation: Callable[[int], tuple[float, float, float]] | None = None,
```

and change the row-building loop:

```python
        if "odometry" not in omit:
            rows = [
                "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy,"
                " distortion_center_x, distortion_center_y"
            ]
            for i in range(odometry_rows):
                fx = ODOMETRY_FX[i % len(ODOMETRY_FX)]
                x, y, z = odometry_translation(i) if odometry_translation else (0.0, 0.0, 0.0)
                rows.append(
                    f"{uptime_base + i / odometry_hz:.6f}, {i:06d}, {x}, {y}, {z}, 0.0, 0.0, 0.0,"
                    f" 1.0, {fx}, {fx}, {ODOMETRY_CX}, {ODOMETRY_CY}, , "
                )
            (camera_dir / "odometry.csv").write_text("\n".join(rows) + "\n")
```

- [ ] **Step 2: Add the `rgb_size` knob**

`_write_rgb` currently always renders at the module constant `RGB_SIZE`. Give `make_camera` a
`rgb_size` parameter (default `RGB_SIZE`) and pass it through:

```python
        rgb_size: tuple[int, int] = RGB_SIZE,
```

```python
        if "rgb" not in omit:
            _write_rgb(camera_dir / "rgb.mp4", rgb_frames, rgb_size, creation_time)
```

`_render_floor_plane`'s own `rgb_size` parameter already exists and is currently always called
with the module constant too (`_render_floor_plane(depth_size, RGB_SIZE, ...)` inside
`make_camera`); change that call site to pass the new `rgb_size` parameter through so
`render_floor_plane=True` and a custom `rgb_size` compose correctly:

```python
            floor_depth, floor_confidence = _render_floor_plane(
                depth_size,
                rgb_size,
                avg_fx,
                ODOMETRY_CX,
                ODOMETRY_CY,
                floor_tilt_deg,
                floor_roll_deg,
                floor_distance_m,
            )
```

- [ ] **Step 3: Update the `MakeCamera` Protocol and verify no existing test moved**

Add both new keywords to the `MakeCamera.__call__` Protocol signature (same defaults). Then:

```bash
uv run pytest --no-cov tests/unit tests/integration -q
uv run mypy tests
```

Expected: identical pass/fail counts to before this task — both knobs are opt-in and default
to prior behavior.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add odometry_translation and rgb_size knobs to make_camera"
```

---

### Task 5: `data/preprocess/tasks/crop.py`

**Files:**
- New: `src/powerflow_pipeline/data/preprocess/tasks/crop.py`
- New: `tests/unit/test_preprocess_crop.py`

**Interfaces:**
- Produces: `crop_camera(record: CameraRecord, config: PreprocessConfig) -> tuple[CameraRecord, StepResult]`
  — no `raw_root`: every input lives in S3's output tree already.

`crop_camera` follows `retilt_camera`'s shape end to end: `log_task_paths` → plan → early
return on `dry_run` → `create_staging_root` → write → `_validate_exit` → `publish_staging` →
`cleanup` in a `finally`. Reuse `frame_paths`/`read_frame`/`CONFIDENCE_VALUES`
(`tasks/ingest.py`), `OUTPUT_TIME_BASE`/`write_camera_matrix`/`preflight` (`tasks/orient.py`),
`create_staging_root`/`publish_staging`/`cleanup`/`write_json` (`common/filesystem.py`).

- [ ] **Step 1: Write the RED happy-path test, building on Task 4's fixture**

```python
"""S4 establishes per-camera I5: one stable, common-region rectangle across every frame."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.tasks.crop import crop_camera
from powerflow_pipeline.data.preprocess.tasks.cut import cut_camera, resolve_cut_interval
from powerflow_pipeline.data.preprocess.tasks.discover import discover_sessions
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from powerflow_pipeline.data.preprocess.tasks.orient import orient_camera
from powerflow_pipeline.data.preprocess.tasks.retilt import retilt_camera
from tests.conftest import MakeCamera, MakeSessionMetadata
from tests.unit.test_preprocess_ingest import make_config as _base_make_config


def make_config(tmp_path: Path, **overrides: object) -> PreprocessConfig:
    return _base_make_config(tmp_path, **overrides)


def _build_retilt_record(
    raw_root: Path, config: PreprocessConfig, camera: str = "Side"
) -> CameraRecord:
    """Run S0 -> S1 -> S2 -> S3 on the one synthetic camera named `camera`."""

    (camera_dir,) = [c for c in discover_sessions.fn(raw_root) if c.camera == camera]
    ingested = ingest_camera.fn(camera_dir, config)
    interval = resolve_cut_interval.fn(raw_root, camera_dir.date, camera_dir.session, ingested)
    cut_record, _ = cut_camera.fn(ingested, interval, config)
    orient_record, _ = orient_camera.fn(cut_record, config)
    retilt_record, _ = retilt_camera.fn(orient_record, raw_root, config)
    return retilt_record


def _make_crop_session(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_session_metadata: MakeSessionMetadata,
    *,
    odometry_translation: object = None,
    **camera_overrides: object,
) -> Path:
    raw = tmp_path / "raw"
    make_camera(
        raw,
        camera="Side",
        rgb_frames=5,
        depth_frames=6,
        render_floor_plane=True,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
        odometry_translation=odometry_translation,  # type: ignore[arg-type]
        **camera_overrides,
    )
    make_session_metadata(
        raw, lift_start_ms=16, lift_end_ms=80, floor_regions="full_frame"
    )
    return raw


# --- the happy path ------------------------------------------------------------------


def test_crop_camera_produces_the_expected_rectangle(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    crop_record, step = crop_camera.fn(retilt_record, config)

    assert crop_record.rgb_width % 2 == 0
    assert crop_record.rgb_height % 2 == 0
    assert crop_record.rgb_width < retilt_record.rgb_width  # S3's border was trimmed
    assert step.derived["crop_bounds_px"][2] - step.derived["crop_bounds_px"][0] == (
        crop_record.rgb_width
    )
```

Run `uv run pytest --no-cov tests/unit/test_preprocess_crop.py -q` and confirm RED — the module
does not exist.

- [ ] **Step 2: Implement `crop.py`'s helpers and `_validate_exit`**

```python
"""S4 · Common-region crop -> establishes per-camera I5.

Measures residual camera translation from odometry, reads S3's valid-content region, and
crops RGB/depth/confidence to their intersection -- a pure slice, never a reprojection
(see docs/specs/preprocessing/6-cropping.md).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from prefect import task

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.filesystem import (
    cleanup,
    create_staging_root,
    publish_staging,
    write_json,
)
from powerflow_pipeline.data.common.models import CropBounds, FileOp, StepResult
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.crop import (
    crop_fractions,
    crop_intrinsics,
    depth_bounds,
    guard_depth_m,
    intersect_crop,
    max_translation_m,
    motion_bounds_px,
    read_valid_bounds,
    reference_displacements,
    shrink_to_even,
)
from powerflow_pipeline.data.preprocess.models import CameraRecord, Intrinsics
from powerflow_pipeline.data.preprocess.tasks.ingest import (
    CONFIDENCE_VALUES,
    frame_paths,
    read_frame,
)
from powerflow_pipeline.data.preprocess.tasks.orient import (
    OUTPUT_TIME_BASE,
    preflight,
    write_camera_matrix,
)

PASSTHROUGH = ("imu.csv",)


def crop_rgb(
    source: Path, destination: Path, bounds: CropBounds, size: tuple[int, int], crf: int
) -> None:
    """Slice every RGB frame to `bounds`; PTS/time-base carried through unchanged -- a crop
    never touches frame count or timing, only pixel extent (§4)."""

    width, height = size
    with av.open(str(source)) as inp, av.open(str(destination), "w") as out:
        in_stream = inp.streams.video[0]
        in_stream.thread_type = "AUTO"

        out_stream = out.add_stream(
            "libx264", rate=in_stream.average_rate, options={"crf": str(crf)}
        )
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"
        out_stream.time_base = OUTPUT_TIME_BASE
        out_stream.codec_context.time_base = OUTPUT_TIME_BASE
        if (creation_time := inp.metadata.get("creation_time")) is not None:
            out.metadata["creation_time"] = creation_time

        for frame in inp.decode(in_stream):
            array = frame.to_ndarray(format="bgr24")
            cropped = array[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
            out_frame = av.VideoFrame.from_ndarray(cropped.astype(np.uint8), format="bgr24")
            out_frame.pts = frame.pts
            out_frame.time_base = frame.time_base
            for packet in out_stream.encode(out_frame):
                out.mux(packet)
        for packet in out_stream.encode():
            out.mux(packet)


def crop_png_stream(paths: list[Path], destination: Path, bounds: CropBounds) -> None:
    """Slice a PNG frame stream to `bounds`. No interpolation, no value change (§4)."""

    destination.mkdir(parents=True)
    for path in paths:
        frame = read_frame(path)
        cropped = frame[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
        cv2.imwrite(str(destination / path.name), cropped)


def _validate_exit(
    staging: Path,
    record: CameraRecord,
    rgb_size: tuple[int, int],
    depth_size: tuple[int, int],
    depth_paths: list[Path],
    confidence_paths: list[Path],
    crop_bounds: CropBounds,
    valid_bounds: CropBounds,
    cropped_intrinsics: Intrinsics,
) -> None:
    """Mirror S3 Retilt's `_validate_exit`: prove per-camera I5 on the staged result."""

    with av.open(str(staging / "rgb.mp4")) as container:
        stream = container.streams.video[0]
        out_size = (stream.codec_context.width, stream.codec_context.height)
        n_rgb = sum(1 for _ in container.decode(stream))
    if out_size != rgb_size:
        raise ScanRejected(f"rgb size changed during crop: {out_size} vs expected {rgb_size}")
    if out_size[0] % 2 or out_size[1] % 2:
        raise ScanRejected(f"rgb output dimensions are not both even: {out_size}")
    if n_rgb != record.n_frames:
        raise ScanRejected(f"rgb frame count changed during crop: {n_rgb} vs {record.n_frames}")

    for name, source_paths in (("depth", depth_paths), ("confidence", confidence_paths)):
        emitted = sorted((staging / name).glob("*.png"))
        if [path.name for path in emitted] != [path.name for path in source_paths]:
            raise ScanRejected(f"{name} frame filenames changed during crop")

    depth = read_frame(staging / "depth" / depth_paths[0].name)
    if depth.dtype != np.uint16:
        raise ScanRejected(f"depth dtype changed during crop: {depth.dtype}")
    if (depth.shape[1], depth.shape[0]) != depth_size:
        raise ScanRejected(f"depth size changed during crop: {depth.shape[::-1]} vs {depth_size}")

    confidence = read_frame(staging / "confidence" / confidence_paths[0].name)
    outside = sorted(set(np.unique(confidence).tolist()) - CONFIDENCE_VALUES)
    if outside:
        raise ScanRejected(f"confidence values invented during crop: {outside}")

    if not (
        valid_bounds.x0 <= crop_bounds.x0
        and crop_bounds.x1 <= valid_bounds.x1
        and valid_bounds.y0 <= crop_bounds.y0
        and crop_bounds.y1 <= valid_bounds.y1
    ):
        raise ScanRejected("crop_bounds_px is not inside valid_bounds_px")

    check = preflight(cropped_intrinsics, rgb_size[0], rgb_size[1])
    if not check["principal_point_inside_image"]:
        raise ScanRejected("cropped principal point falls outside the cropped image")
```

- [ ] **Step 3: Implement `crop_camera` itself**

```python
@task(retries=1)
def crop_camera(
    record: CameraRecord, config: PreprocessConfig
) -> tuple[CameraRecord, StepResult]:
    """Measure residual translation, intersect it with S3's valid-content region, and crop
    one camera's streams to the result whole -- establishing per-camera I5.

    `record` is S3's output record; unlike S3 Retilt, no `raw_root` is needed -- every input
    S4 reads lives in S3's own output tree.
    """

    source = record.source
    destination = config.crop_root / record.relative
    log_task_paths(source, destination)

    sidecar = json.loads((source / "retilt_sidecar.json").read_text())
    valid = read_valid_bounds(sidecar)

    file_ops = [
        FileOp(op="write", src=source / "rgb.mp4", dst=destination / "rgb.mp4"),
        FileOp(op="write", src=source / "depth", dst=destination / "depth"),
        FileOp(op="write", src=source / "confidence", dst=destination / "confidence"),
        FileOp(op="write", src=source / "camera_matrix.csv", dst=destination / "camera_matrix.csv"),
        FileOp(op="copy", src=source / "odometry.csv", dst=destination / "odometry.csv"),
        FileOp(op="copy", src=source / "imu.csv", dst=destination / "imu.csv"),
        FileOp(op="write", src=source, dst=destination / "crop_sidecar.json"),
        FileOp(op="publish", src=source, dst=destination),
    ]
    crop_record = record.model_copy(update={"source": destination})

    if config.dry_run:
        return crop_record, StepResult(file_ops=file_ops)

    odometry = pd.read_csv(source / "odometry.csv", skipinitialspace=True).iloc[: record.n_frames]
    displacements = reference_displacements(
        odometry[["x", "y", "z"]].to_numpy(dtype=float),
        odometry[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float),
    )
    max_t = max_translation_m(displacements)
    if max_t > config.crop_max_residual_translation_m:
        raise ScanRejected(
            f"camera translated {max_t:.4f} m across the retained frames, exceeding "
            f"crop_max_residual_translation_m={config.crop_max_residual_translation_m}"
        )

    depth_paths = frame_paths(source / "depth")
    confidence_paths = frame_paths(source / "confidence")
    sample = list(range(0, len(depth_paths), config.crop_depth_sample_stride))[
        : config.crop_max_sampled_depth_frames
    ]
    depth_samples: list[np.ndarray] = []
    for index in sample:
        depth_frame = read_frame(depth_paths[index])
        confidence_frame = read_frame(confidence_paths[index])
        mask = (depth_frame > 0) & (confidence_frame > 0)
        depth_samples.append(depth_frame[mask].astype(np.float64) / 1000.0)
    z_guard_m = guard_depth_m(np.concatenate(depth_samples), config.crop_depth_guard_quantile)

    rgb_size = (record.rgb_width, record.rgb_height)
    depth_size = (record.depth_width, record.depth_height)
    motion = motion_bounds_px(
        displacements, record.intrinsics, valid, z_guard_m, config.crop_safety_px, rgb_size
    )
    intersected, bound_source = intersect_crop(valid, motion, rgb_size)
    crop_bounds = shrink_to_even(intersected)
    even_shrink = (intersected.width - crop_bounds.width, intersected.height - crop_bounds.height)

    frac_w, frac_h = crop_fractions(crop_bounds, rgb_size)
    if frac_w > config.crop_max_crop_fraction or frac_h > config.crop_max_crop_fraction:
        raise ScanRejected(
            f"crop removes {frac_w:.3f}/{frac_h:.3f} of width/height, exceeding "
            f"crop_max_crop_fraction={config.crop_max_crop_fraction}"
        )

    depth_crop = depth_bounds(crop_bounds, rgb_size, depth_size)
    cropped_intrinsics = crop_intrinsics(record.intrinsics, crop_bounds)
    crop_output_size = (crop_bounds.width, crop_bounds.height)
    depth_output_size = (depth_crop.width, depth_crop.height)

    sidecar_out = {
        "reference_frame": 0,
        "max_translation_m": max_t,
        "crop_max_residual_translation_m": config.crop_max_residual_translation_m,
        "z_guard_m": z_guard_m,
        "crop_depth_guard_quantile": config.crop_depth_guard_quantile,
        "crop_safety_px": config.crop_safety_px,
        "valid_bounds_px": [valid.x0, valid.y0, valid.x1, valid.y1],
        "motion_bounds_px": list(motion),
        "crop_bounds_px": [crop_bounds.x0, crop_bounds.y0, crop_bounds.x1, crop_bounds.y1],
        "bound_source": bound_source,
        "even_shrink_px": list(even_shrink),
        "depth_crop_bounds_px": [depth_crop.x0, depth_crop.y0, depth_crop.x1, depth_crop.y1],
        "confidence_crop_bounds_px": [
            depth_crop.x0, depth_crop.y0, depth_crop.x1, depth_crop.y1,
        ],
        "rgb_input_size": list(rgb_size),
        "rgb_output_size": list(crop_output_size),
        "depth_input_size": list(depth_size),
        "depth_output_size": list(depth_output_size),
        "crop_fraction_wh": [frac_w, frac_h],
        "camera_translation_corrected": False,
        "pixels_shifted": False,
        "k_rewritten": True,
    }

    result = StepResult(
        derived={
            "max_translation_m": max_t,
            "z_guard_m": z_guard_m,
            "crop_bounds_px": sidecar_out["crop_bounds_px"],
            "rgb_output_size": list(crop_output_size),
        },
        file_ops=file_ops,
    )

    staging = create_staging_root(destination)
    try:
        crop_rgb(source / "rgb.mp4", staging / "rgb.mp4", crop_bounds, crop_output_size, config.rgb_crf)
        crop_png_stream(depth_paths, staging / "depth", depth_crop)
        crop_png_stream(confidence_paths, staging / "confidence", depth_crop)

        write_camera_matrix(staging / "camera_matrix.csv", cropped_intrinsics)
        shutil.copy2(source / "odometry.csv", staging / "odometry.csv")
        for name in PASSTHROUGH:
            shutil.copy2(source / name, staging / name)
        write_json(staging / "crop_sidecar.json", sidecar_out)

        _validate_exit(
            staging, record, crop_output_size, depth_output_size, depth_paths, confidence_paths,
            crop_bounds, valid, cropped_intrinsics,
        )

        if config.overwrite and destination.exists():
            shutil.rmtree(destination)
        publish_staging(staging, destination)
    finally:
        cleanup(staging)

    crop_record = crop_record.model_copy(
        update={
            "rgb_width": crop_output_size[0],
            "rgb_height": crop_output_size[1],
            "depth_width": depth_output_size[0],
            "depth_height": depth_output_size[1],
            "intrinsics": cropped_intrinsics,
        }
    )
    return crop_record, result
```

- [ ] **Step 4: Write the §6 rejection tests**

One worked in full; the rest follow the same `_make_crop_session` + `pytest.raises(ScanRejected,
match=...)` pattern already established in `test_preprocess_retilt.py`:

```python
def test_translation_beyond_tolerance_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(
        tmp_path, make_camera, make_session_metadata,
        odometry_translation=lambda i: (0.05 * i, 0.0, 0.0),  # drifts well past any tolerance
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    with pytest.raises(ScanRejected, match="translated"):
        crop_camera.fn(retilt_record, config)
```

| Test | Construction | `match=` |
|---|---|---|
| `test_missing_valid_bounds_is_rejected` | after building `retilt_record`, rewrite `retilt_sidecar.json` at `retilt_record.source` to drop `valid_bounds_px` before calling `crop_camera.fn` | `"valid_bounds_px"` |
| `test_degenerate_valid_bounds_is_rejected` | same, but set `valid_bounds_px` to `[100, 0, 50, 100]` (`x0 > x1`) | `"degenerate"` |
| `test_translation_beyond_tolerance_is_rejected` | shown above | `"translated"` |
| `test_no_valid_depth_is_rejected` | after building `retilt_record`, overwrite every file under `retilt_record.source / "confidence"` with all-zero PNGs before calling `crop_camera.fn` | `"depth"` |
| `test_empty_intersection_is_rejected` | pass `config = make_config(tmp_path, retilt_min_floor_points=10, crop_safety_px=100_000)` — an absurd safety margin forces the motion band past the whole frame | `"overlap"` |
| `test_excessive_crop_fraction_is_rejected` | `config = make_config(tmp_path, retilt_min_floor_points=10, crop_max_crop_fraction=0.001)` | `"crop_max_crop_fraction"` |

Also write the "no pixels invented" test:

```python
def test_cropped_depth_matches_the_input_slice_exactly(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    crop_record, step = crop_camera.fn(retilt_record, config)

    x0, y0, x1, y1 = step.derived["crop_bounds_px"]
    input_frame = read_frame(sorted((retilt_record.source / "depth").glob("*.png"))[0])
    from powerflow_pipeline.data.preprocess.crop import depth_bounds
    from powerflow_pipeline.data.common.models import CropBounds as _CB

    dbounds = depth_bounds(
        _CB(x0=x0, y0=y0, x1=x1, y1=y1),
        (retilt_record.rgb_width, retilt_record.rgb_height),
        (retilt_record.depth_width, retilt_record.depth_height),
    )
    expected_slice = input_frame[dbounds.y0 : dbounds.y1, dbounds.x0 : dbounds.x1]
    output_frame = read_frame(sorted((crop_record.source / "depth").glob("*.png"))[0])
    assert np.array_equal(output_frame, expected_slice)
```

(Add `from powerflow_pipeline.data.preprocess.tasks.ingest import read_frame` and `import
numpy as np` to the test module's imports.)

- [ ] **Step 5: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_crop.py -q
uv run mypy src tests
uv run ruff check src/powerflow_pipeline/data/preprocess/tasks/crop.py tests/unit/test_preprocess_crop.py
```

- [ ] **Step 6: Commit**

```bash
git add src/powerflow_pipeline/data/preprocess/tasks/crop.py tests/unit/test_preprocess_crop.py
git commit -m "feat(preprocess): implement S4 crop_camera task"
```

---

### Task 6: Flow wiring

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/flow.py`
- Modify: `tests/integration/test_preprocess_flow.py`

**Interfaces:**
- Changes: the per-camera loop calls `crop_camera` after `retilt_camera`, binding the record
  `retilt_camera` currently discards; manifest `steps` gains `"crop"`

- [ ] **Step 1: Write the failing integration assertion**

Extend `build_capture` in `tests/integration/test_preprocess_flow.py` to pass a large-enough
`rgb_size` (Task 4) so the default crop tunables pass without overrides — the real motivation
being that the default 64×48 landscape frame (48×64 portrait) would lose 33% of width to an
8 px guard band on each side, tripping `crop_max_crop_fraction=0.25`:

```python
    floor = {
        "render_floor_plane": True,
        "floor_tilt_deg": 8.0,
        "floor_roll_deg": -2.0,
        "depth_size": (64, 48),
        "rgb_size": (256, 192),  # portrait 192x256 after S1 -- large enough that S4's
                                  # 8px default guard band stays well under max_crop_fraction
    }
```

Add the manifest/output assertion:

```python
    assert all("crop" in scan.steps for scan in manifest.scans)
    assert (config.crop_root / "9 July" / "cnj_45kg_Set1" / "Front" / "rgb.mp4").is_file()
```

Run `uv run pytest --no-cov tests/integration/test_preprocess_flow.py -q` and confirm it fails
— `"crop"` is not yet in `steps` and `s4_crop_output` does not exist.

- [ ] **Step 2: Wire the call**

```python
                cut_record, cut_step = cut_camera(record, interval, config)
                orient_record, orient_step = orient_camera(cut_record, config)
                # A camera rejected here has already published S2 output -- the same
                # shape as a camera rejected at S2 having already published S1 output.
                retilt_record, retilt_step = retilt_camera(orient_record, config.raw_root, config)
                _, crop_step = crop_camera(retilt_record, config)
```

```python
                    steps=["ingest", "cut", "orient", "retilt", "crop"],
                    derived={
                        **cut_step.derived,
                        **orient_step.derived,
                        **retilt_step.derived,
                        **crop_step.derived,
                    },
                    warnings=(
                        cut_step.warnings + orient_step.warnings
                        + retilt_step.warnings + crop_step.warnings
                    ),
                    file_ops=(
                        cut_step.file_ops + orient_step.file_ops
                        + retilt_step.file_ops + crop_step.file_ops
                    ),
```

Add the import: `from powerflow_pipeline.data.preprocess.tasks.crop import crop_camera`.

Note for implementers: `retilt_step` was previously assigned via `_, retilt_step =
retilt_camera(...)` — the record is now bound and consumed one line later.

- [ ] **Step 3: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/integration -q
uv run mypy src
```

- [ ] **Step 4: Commit**

```bash
git add src/powerflow_pipeline/data/preprocess/flow.py tests/integration/test_preprocess_flow.py
git commit -m "feat(preprocess): wire S4 crop into the preprocess flow"
```

---

### Task 7: Calibration on real data

**Files:**
- None modified — data-collection only.

**Interfaces:**
- Consumes: a real `powerflow preprocess` run over `data/raw/11 July`
- Produces: per-camera `crop_bounds_px`/`bound_source`/output-size numbers recorded in this
  plan's own notes, checked against the "Measured calibration" table above

- [ ] **Step 1: Dry-run first**

```bash
uv run powerflow preprocess --input ../data/raw --records ../data/s0_ingest_output \
  --cut ../data/s1_cut_output --retilt ../data/s3_retilt_output \
  --crop ../data/s4_crop_output --output ../data/s2_orient_output --dry-run
```

Expected: 4 cameras planned, 0 rejected, `crop` present in each `steps` list.

- [ ] **Step 2: Real run**

```bash
uv run powerflow preprocess --input ../data/raw --records ../data/s0_ingest_output \
  --cut ../data/s1_cut_output --retilt ../data/s3_retilt_output \
  --crop ../data/s4_crop_output --output ../data/s2_orient_output
```

- [ ] **Step 3: Compare against the measured table**

```bash
for f in ../data/s4_crop_output/*/*/*/crop_sidecar.json; do
  jq '{camera: input_filename, max_translation_m, z_guard_m, crop_bounds_px, bound_source, rgb_output_size}' "$f"
done
```

Expected: `crop_bounds_px` and `bound_source` match this plan's "Measured calibration" table
(pre-shrink numbers there; `rgb_output_size` reflects the even-shrink applied to the two Front
cameras). A mismatch means either the odometry-row alignment or the motion-margin formula has a
bug — debug against these four sidecars before proceeding, do not adjust the expected table to
match an unexplained result.

- [ ] **Step 4: Record the result in this file**

Append a "Results" subsection here (mirroring `2026-08-26-retilt-implementation.md`'s Task 9)
with the actual per-camera numbers and the date of the run.

---

### Task 8: README and the honest I5 note

**Files:**
- Modify: `powerflow-pipeline/README.md`
- Modify: `docs/specs/preprocessing/6-cropping.md`

**Interfaces:**
- None — documentation only.

- [ ] **Step 1: Add S4 to the README's stage list, flag table, and outputs table**

Follow the existing S0–S3 bullet style and the "Verifying a Retilt run" section's `jq` pattern:

```bash
jq '{max_translation_m, z_guard_m, crop_bounds_px, bound_source}' \
  ../data/s4_crop_output/11\ July/30kg_Set1/Front/crop_sidecar.json
```

- [ ] **Step 2: Update `6-cropping.md`'s open question**

Replace the "Still open" paragraph with the decision actually taken:

> **Resolved (2026-08-27):** implemented per-camera, exactly as this document's §3 specifies.
> The four real cameras therefore keep four different output sizes (see the implementation
> plan's measured-calibration table). `1-ingestion_orient.md`'s literal I5 wording ("every
> image has identical pixel dimensions") is *not* satisfied by S4 alone as a result — that
> would require a separate cross-camera reconciliation stage, applied to the two cameras'
> `crop_bounds_px`, which does not exist in any spec. This was a deliberate scope decision, not
> an oversight.

Also add a short note that `RetiltResult.valid_bounds_px` is typed `CropBounds` in
`preprocess/models.py`, but the sidecar actually written to disk stores a flat 4-list — the
model is never serialized directly, and S4's `read_valid_bounds` parses the list.

- [ ] **Step 3: Read through once**

No automated check — confirm `--crop` matches the flag name from Task 1, and the `jq` paths
match Task 7's real output.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/specs/preprocessing/6-cropping.md
git commit -m "docs: document S4 crop and resolve the I5 per-camera scope question"
```

---

### Task 9: Full-repository verification

**Files:**
- Verify only: all modified source, test, and documentation files

**Interfaces:**
- Consumes: completed Tasks 1–8
- Produces: evidence that formatting, linting, typing, coverage, and the full test suite pass

- [ ] **Step 1: Run the four CI gates in order**

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src
uv run pytest --cov --cov-fail-under=90
```

Expected: all four exit 0; coverage at or above 90% (baseline before this plan: 197 tests,
96.17%).

- [ ] **Step 2: Inspect the final diff**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the files this plan describes differ, in addition to any
pre-existing workspace changes unrelated to this plan. (Note, per the retilt plan's own Task
11: this directory is not a git repository at the top level — `/Users/yeethui/github/PowerFlow`
— so if `git status`/`git diff` are unavailable here too, substitute a trailing-whitespace grep
over the touched documentation and rely on `ruff format --check` for source/test whitespace, as
the retilt plan did.)
