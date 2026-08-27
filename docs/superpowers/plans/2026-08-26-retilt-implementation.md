# S3 Retilt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the S3 Retilt stage specified in `docs/specs/preprocessing/4-retilt.md`
(as corrected by Task 1 below): fit one floor plane per camera from depth, derive the tilt/
roll that levels it, rectify RGB/depth/confidence with the resulting homography, and publish
to a new `s3_retilt_output` root — establishing invariant **I3**.

**Architecture:** A new pure-maths module, `data/preprocess/retilt.py` (no I/O, no Prefect —
the same boundary `geometry.py` and `timeline.py` already hold), carries region parsing,
back-projection, plane fitting, angle derivation, homography, and the gravity check. A new
task module, `data/preprocess/tasks/retilt.py`, wires that maths to I/O and follows
`orient_camera`'s staging/publish shape exactly. `orient_camera` must first start returning a
`CameraRecord` (Task 5) — the shape `cut_camera` already returns — because Retilt needs S2's
output dimensions and rotated `K`, and today only `cut_camera` hands the next stage a record.
The flow calls `retilt_camera` after `orient_camera` inside the existing per-camera loop.

**Tech Stack:** Python 3.12, PyAV, OpenCV, `torch.linalg.lstsq`, pandas, Prefect 3, pytest,
Ruff, mypy.

## Global Constraints

- No new dependencies — `torch`, `opencv-contrib-python`, `av`, `pandas`, `pyyaml` are already
  in `pyproject.toml`.
- Do not run the real `powerflow preprocess` pipeline until Task 9, and only then as directed.
- One plane and one homography **per camera**, never per frame — recompute nothing inside the
  per-frame warp loop that does not depend on the frame.
- Never resample confidence into a value the sensor did not report (`{0,1,2}` only); depth and
  confidence use nearest-neighbour only, matching S1 Orient's rule.
- The gravity-agreement check computes and records `gravity_agreement_deg` and warns past
  tolerance; it must **not** reject, until a follow-up plan promotes it (Task 9).
- Preserve unrelated pre-existing workspace changes.
- TDD per `CLAUDE.md`: failing test first, then minimum implementation, for every task that
  touches `src/`.

---

### Task 1: Correct `docs/specs/preprocessing/4-retilt.md`

**Files:**
- Modify: `docs/specs/preprocessing/4-retilt.md`

**Interfaces:**
- Consumes: real captures at `data/raw/11 July/{30kg_Set1,50kg_Set3}/metadata.yaml`
- Produces: a design spec the rest of this plan can be implemented against without
  contradiction

This task is **already complete** — done directly against the spec ahead of this plan, since
every later task depends on reading correct rules rather than re-deriving them from raw
capture evidence mid-implementation. Verify before proceeding:

- [x] **Step 1: Confirm the three corrections are present**

```bash
grep -n "parenthesised" docs/specs/preprocessing/4-retilt.md
grep -n "legacy and wrong" docs/specs/preprocessing/4-retilt.md
grep -n "y1 >= y0" docs/specs/preprocessing/4-retilt.md
```

Expected: each returns at least one match — §1 states the on-disk value is a parenthesised
string requiring an explicit parser (not a numeric YAML value), marks the `_in_pixels` suffix
as legacy (values are normalized `[0,1]`, not pixels), and §7's degeneracy rule reads
`x0 >= x1` **or** `y1 >= y0` (not `y0 >= y1`, which rejects every real region under §1's
top→bottom convention).

- [x] **Step 2: Confirm the two convention notes are present**

```bash
grep -n "pixel-centre" docs/specs/preprocessing/4-retilt.md
grep -n "warn-only" docs/specs/preprocessing/4-retilt.md
```

Expected: §2 states `K_d`'s principal point is scaled pixel-centre (`(c + 0.5) * s - 0.5`),
and §7 states the gravity check ships warn-only pending calibration (Task 9).

---

### Task 2: Config, CLI, and stage-root wiring

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/config.py`
- Modify: `src/powerflow_pipeline/data/cli.py`
- Modify: `prefect.yaml`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Produces: `PreprocessConfig.retilt_root`, in `stage_roots`; nine `retilt_*` tunables; a
  `--retilt` CLI flag

- [x] **Step 1: Write the failing CLI test**

Add to `tests/unit/test_cli.py`, mirroring the existing `--cut` assertion:

```python
def test_preprocess_command_accepts_retilt_root(...) -> None:
    ...
    result = runner.invoke(app, ["preprocess", "--input", ..., "--records", ...,
                                  "--cut", ..., "--retilt", str(retilt_dir), "--output", ...])
    assert result.exit_code == 0
```

Run `uv run pytest --no-cov tests/unit/test_cli.py -q` and confirm it fails — `--retilt` is
not yet a recognized option.

- [x] **Step 2: Add `retilt_root` and the tunables to `PreprocessConfig`**

```python
retilt_root: Path
retilt_sample_stride: int = Field(default=10, ge=1)
retilt_max_sampled_frames: int = Field(default=32, ge=1)
retilt_max_pooled_points: int = Field(default=200_000, ge=1)
retilt_min_floor_points: int = Field(default=500, ge=1)
retilt_max_plane_rms_m: float = Field(default=0.02, gt=0)
retilt_gravity_tolerance_deg: float = Field(default=5.0, gt=0)
retilt_max_tilt_deg: float = Field(default=45.0, gt=0)
retilt_max_roll_deg: float = Field(default=45.0, gt=0)
retilt_max_translation_m: float = Field(default=0.05, gt=0)
retilt_conf2_area_fraction: float = Field(default=1 / 6, gt=0, le=1)
```

Add `self.retilt_root` to the `stage_roots` tuple, after `cut_root` and before `output_root`
— `write_session_metadata` already iterates `stage_roots`, so a new stage's `metadata.yaml`
copy comes free (this is the documented extension point in `config.py`'s docstring).

- [x] **Step 3: Add `--retilt` to the CLI and thread it into `PreprocessConfig`**

Follow the exact pattern of `--cut` in `cli.py`'s `preprocess` command: a `typer.Option`
named `--retilt`, required, help text "Where S3 publishes the rectified streams."

- [x] **Step 4: Add `retilt_root` to `prefect.yaml`**

```yaml
    parameters:
      config:
        raw_root: ../data/raw
        record_root: ../data/s0_ingest_output
        cut_root: ../data/s1_cut_output
        retilt_root: ../data/s3_retilt_output
        output_root: ../data/s2_orient_output
```

Note the existing `output_root` names S2's tree — do not rename it; `retilt_root` is
additive. Update the deployment `description` to mention S3.

- [x] **Step 5: Run focused tests and verify GREEN**

```bash
uv run pytest --no-cov tests/unit/test_cli.py -q
```

---

### Task 3: Models — `FloorRegion`, `PlaneFit`, `RetiltResult`

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/models.py`
- Test: `tests/unit/test_common_models.py` (or a new `tests/unit/test_preprocess_retilt_models.py`)

**Interfaces:**
- Produces: `FloorRegion.pixel_bounds(width, height) -> tuple[int,int,int,int]`;
  `PlaneFit`; `RetiltResult`

- [x] **Step 1: Write failing validator tests**

```python
def test_floor_region_rejects_x_degenerate() -> None:
    with pytest.raises(ValidationError):
        FloorRegion(x0=0.5, y0=0.2, x1=0.5, y1=0.1)

def test_floor_region_rejects_y_degenerate() -> None:
    # y1 must be strictly less than y0 (top-right above bottom-left) — the CORRECTED rule
    with pytest.raises(ValidationError):
        FloorRegion(x0=0.0, y0=0.5, x1=1.0, y1=0.6)

def test_floor_region_accepts_real_capture_values() -> None:
    # data/raw/11 July/30kg_Set1/metadata.yaml, side region
    region = FloorRegion(x0=0.0, y0=1.0, x1=0.40268, y1=0.64503)
    assert region.pixel_bounds(1440, 1920) == (0, 1239, 582, 1920)
```

Run `uv run pytest --no-cov tests/unit/test_common_models.py -q -k floor_region` and confirm
it fails — `FloorRegion` does not exist yet.

- [x] **Step 2: Implement the models**

```python
class FloorRegion(BaseModel):
    """A normalized floor-selection rectangle, operator-annotated per camera (4-retilt.md §1)."""

    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)  # bottom_left.y -- larger, since y grows downward
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)  # top_right.y -- smaller

    @model_validator(mode="after")
    def validate_extent(self) -> FloorRegion:
        if self.x0 >= self.x1:
            raise ValueError("floor region degenerate on x: x0 >= x1")
        if self.y1 >= self.y0:
            raise ValueError("floor region degenerate on y: y1 >= y0")
        return self

    def pixel_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        """`(col_start, row_start, col_end, row_end)` per §1's conversion."""

        return (
            round(width * self.x0),
            round(height * self.y1),
            round(width * self.x1),
            round(height * self.y0),
        )


class PlaneFit(BaseModel):
    normal: list[float]  # unit vector, camera frame, n_y < 0
    rms_residual_m: float
    n_points: int
    n_frames_sampled: int
    confidence_mode: Literal["conf2_only", "conf1_and_2"]


class RetiltResult(BaseModel):
    tilt_deg: float
    roll_deg: float
    plane: PlaneFit
    gravity_agreement_deg: float | None  # None when odometry orientation is unavailable
    homography_rgb: list[list[float]]
    homography_depth: list[list[float]]
    valid_bounds_px: CropBounds
    translation_span_m: float
```

Import `CropBounds` from `powerflow_pipeline.data.common.models` rather than redefining it —
it already validates positive extent and exposes `.width`/`.height` (`common/models.py:11`).

- [x] **Step 3: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/unit/test_common_models.py -q
uv run mypy src
```

---

### Task 4: Pure maths — `data/preprocess/retilt.py`

**Files:**
- New: `src/powerflow_pipeline/data/preprocess/retilt.py`
- New: `tests/unit/test_preprocess_retilt_math.py`

**Interfaces:**
- Produces: every function in the table below. No Prefect `@task`, no filesystem writes — a
  boundary already established by `geometry.py`/`timeline.py` so these functions stay fast
  and exactly checkable.

| Function | Contract |
|---|---|
| `parse_region_point(raw: str) -> tuple[float, float]` | `"(0, 0.75093)"` → `(0.0, 0.75093)`; `ValueError` on anything else |
| `read_floor_region(metadata: dict, camera: str) -> FloorRegion` | picks `front_`/`side_` prefix from `camera`, raises `ScanRejected` naming the exact missing/malformed field |
| `depth_intrinsics(k: Intrinsics, rgb_size, depth_size) -> Intrinsics` | pixel-centre scaling per the corrected §2 |
| `select_floor_pixels(depth, confidence, bounds) -> tuple[np.ndarray, Literal[...]]` | §1 selection; returns `(row, col)` index arrays plus the mode used |
| `backproject(rows, cols, depth_mm, k_d) -> np.ndarray` | vectorized §2, shape `(N, 3)`, metres |
| `fit_plane(points: np.ndarray) -> PlaneFit` | `torch.linalg.lstsq`, `Y = aX + bZ + c`; normal `(a,-1,b)` normalized, `n_y < 0` |
| `tilt_roll_from_normal(n: np.ndarray) -> tuple[float, float]` | §4 verbatim, degrees |
| `rectifying_rotation(tilt_deg: float, roll_deg: float) -> np.ndarray` | `R_x(tilt) @ R_z(roll)`, 3×3 |
| `homography(k: Intrinsics, r: np.ndarray) -> np.ndarray` | `K @ R @ K⁻¹`, 3×3 |
| `depth_scale_map(k_d: Intrinsics, r: np.ndarray, size: tuple[int,int]) -> np.ndarray` | per-output-pixel `R_row3 · K_d⁻¹ · x̃`, shape `(H, W)` |
| `valid_bounds(h: np.ndarray, size: tuple[int,int]) -> CropBounds` | bbox of the all-ones mask warped by `H` |
| `gravity_down_camera(qx, qy, qz, qw: float) -> np.ndarray` | the three-step basis chain below |
| `gravity_agreement_deg(normal: np.ndarray, g_camera: np.ndarray) -> float` | `degrees(angle(normal, -g_camera))` |
| `translation_span_m(xyz: np.ndarray) -> float` | max pairwise distance across sampled-frame `(x,y,z)` odometry rows |

Gravity chain (write these four lines as inline comments in the implementation, matching the
spec's assumptions section):

```python
def gravity_down_camera(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Odometry quaternion -> 'down' in S2's rotated portrait camera frame.

    g_world -> ARKit camera axes (R_wc^T) -> CV camera axes (flip Y,Z) -> S2's 90 deg CW
    rotation of the image (not the odometry columns). See 4-retilt.md Assumptions.
    """
    r_wc = _quat_to_matrix(qx, qy, qz, qw)         # world -> ARKit camera-local
    g_world = np.array([0.0, -1.0, 0.0])           # ARKit world is gravity-aligned, Y up
    g_arkit = r_wc.T @ g_world
    g_cv = np.diag([1.0, -1.0, -1.0]) @ g_arkit    # ARKit (Y up, Z back) -> CV (Y down, Z fwd)
    g_portrait = _R_Z_90_CW @ g_cv                 # S2 rotates the image, not odometry
    return g_portrait
```

- [x] **Step 1: Write failing numeric tests, function by function**

Key assertions (each its own test):

```python
def test_parse_region_point_real_capture_value() -> None:
    assert parse_region_point("(0.40268, 0.64503)") == pytest.approx((0.40268, 0.64503))

def test_depth_intrinsics_pixel_centre_scaling() -> None:
    k = Intrinsics(fx=1343.0, fy=1343.0, cx=724.8, cy=968.3, frame="portrait")
    k_d = depth_intrinsics(k, rgb_size=(1440, 1920), depth_size=(192, 256))
    s = 192 / 1440
    assert k_d.cx == pytest.approx((724.8 + 0.5) * s - 0.5)

def test_select_floor_pixels_conf2_dominant_uses_conf2_only() -> None:
    # a region where conf-2 covers >= 1/6 of the area -> mode == "conf2_only"
    ...

def test_select_floor_pixels_conf2_sparse_falls_back_to_conf1_and_2() -> None:
    ...

def test_fit_plane_recovers_known_normal() -> None:
    # synthetic points on Y = 0.05*X + 0.02*Z + 1.5 plus small noise
    points = _synthetic_plane_points(a=0.05, b=0.02, c=1.5, n=2000, noise_m=0.001)
    fit = fit_plane(points)
    expected = _unit_normal(0.05, 0.02)
    assert np.dot(fit.normal, expected) > 0.9999
    assert fit.rms_residual_m < 0.01

def test_tilt_roll_recovers_baked_in_angles() -> None:
    n = _normal_from_tilt_roll(tilt_deg=7.0, roll_deg=-3.0)
    tilt, roll = tilt_roll_from_normal(n)
    assert tilt == pytest.approx(7.0, abs=1e-3)
    assert roll == pytest.approx(-3.0, abs=1e-3)

def test_rectifying_rotation_levels_the_normal() -> None:
    n = _normal_from_tilt_roll(tilt_deg=12.0, roll_deg=4.0)
    r = rectifying_rotation(*tilt_roll_from_normal(n))
    assert r @ n == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)

def test_homography_identity_at_zero_rotation() -> None:
    k = Intrinsics(fx=100, fy=100, cx=50, cy=50, frame="portrait")
    assert homography(k, np.eye(3)) == pytest.approx(np.eye(3))

def test_valid_bounds_shrinks_under_a_roll() -> None:
    # matches the deleted test_preprocess_plane.py's assertion by name/intent
    h_identity = homography(k, np.eye(3))
    h_rolled = homography(k, rectifying_rotation(0.0, 15.0))
    assert valid_bounds(h_rolled, size).width < valid_bounds(h_identity, size).width

def test_gravity_down_camera_identity_quaternion() -> None:
    g = gravity_down_camera(0.0, 0.0, 0.0, 1.0)
    # work out the expected vector by hand from the four-line chain and assert against it
    assert g == pytest.approx(_expected_g_identity, abs=1e-9)

def test_depth_scale_map_matches_hand_worked_point() -> None:
    # one specific (u, v, Z_src) -> Z' computed by hand against R_row3 . K_d^-1 . x
    ...
```

`test_valid_bounds_shrinks_under_a_roll` deliberately reuses the name found in
`.pytest_cache`/`.pyc` residue from a deleted `test_preprocess_plane.py` — same assertion,
same intent, now under this plan's ownership.

- [x] **Step 2: Run and verify RED, then implement each function to GREEN**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_retilt_math.py -q
```

Implement in the order listed in the interface table — each function's tests are
independent, but `fit_plane` and `tilt_roll_from_normal` are exercised together by the
recovery tests, so implement both before running those two.

- [x] **Step 3: Full-module run and type check**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_retilt_math.py -q
uv run mypy src
```

---

### Task 5: `orient_camera` returns `(CameraRecord, StepResult)`

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/tasks/orient.py:147-236`
- Modify: `src/powerflow_pipeline/data/preprocess/flow.py:72-96`
- Modify: `tests/unit/test_preprocess_orient.py` (every call site)
- Modify: `tests/integration/test_preprocess_flow.py` if it inspects orient's return directly

**Interfaces:**
- Changes: `orient_camera(record, config) -> StepResult` becomes
  `orient_camera(record, config) -> tuple[CameraRecord, StepResult]`

This is mechanical but must land before Task 6 — Retilt needs a `CameraRecord` describing
S2's *output* geometry (portrait dimensions, rotated `K`), which today only `cut_camera`
returns.

- [x] **Step 1: Update the failing call sites first (RED)**

Update every `orient_camera(...)` call in `tests/unit/test_preprocess_orient.py` to unpack
`orient_record, orient_step = orient_camera.fn(...)`, and add one new assertion:

```python
def test_orient_camera_returns_record_with_rotated_dimensions(...) -> None:
    orient_record, step = orient_camera.fn(cut_record, config)
    assert (orient_record.rgb_width, orient_record.rgb_height) == rgb_size  # already rotated
    assert orient_record.intrinsics == rotated  # the same K orient_camera already computes
    assert orient_record.source == config.output_root / record.relative
```

Run and confirm it fails — the return type does not match yet.

- [x] **Step 2: Implement the record construction**

At the end of `orient_camera`, build the output record via `record.model_copy(update={...})`
— the same pattern `cut_camera` uses (`cut.py:318`) — setting `source=destination`,
`rgb_width, rgb_height = rgb_size`, `depth_width, depth_height = depth_size`,
`intrinsics=rotated`. Return `(orient_record, result)`.

- [x] **Step 3: Update the flow's call site**

`flow.py:76`: `orient_step = orient_camera(cut_record, config)` becomes
`orient_record, orient_step = orient_camera(cut_record, config)`. `orient_record` is not yet
consumed by anything downstream of S2 — Task 7 wires it into `retilt_camera`.

- [x] **Step 4: Run and verify GREEN across all affected suites**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_orient.py tests/integration/test_preprocess_flow.py tests/integration/test_prefect_spine.py -q
```

---

### Task 6: `data/preprocess/tasks/retilt.py`

**Files:**
- New: `src/powerflow_pipeline/data/preprocess/tasks/retilt.py`
- New: `tests/unit/test_preprocess_retilt.py`

**Interfaces:**
- Produces: `retilt_camera(record: CameraRecord, raw_root: Path, config: PreprocessConfig) -> tuple[CameraRecord, StepResult]`
  — `raw_root` is needed because the floor region lives in the *raw* session `metadata.yaml`
  (§1), not anywhere in S2's output tree.

`retilt_camera` follows `orient_camera`'s shape end to end: `log_task_paths` → plan → early
return on `dry_run` → `create_staging_root` → write → `_validate_exit` → `publish_staging` →
`cleanup` in a `finally`. Reuse `frame_paths`/`read_frame` (`ingest.py:85,145`), `write_json`,
`OUTPUT_TIME_BASE` (`orient.py:42`).

- [x] **Step 1: Write the RED happy-path test using the new fixture (see Task 8 first)**

This task depends on Task 8's `make_camera(floor_tilt_deg=..., floor_roll_deg=...)` fixture
extension to have a real plane to fit against. Implement Task 8's fixture extension before
writing this test; the fixture and this task's tests may be developed together in one
RED→GREEN cycle if that is more natural, but the fixture must exist before any retilt test
can assert a numeric result.

```python
def test_retilt_camera_recovers_baked_in_tilt(tmp_path, make_camera, make_session_metadata) -> None:
    raw_root = tmp_path / "raw"
    make_camera(raw_root, floor_tilt_deg=8.0, floor_roll_deg=-2.0, floor_distance_m=1.8)
    make_session_metadata(raw_root, floor_regions="full_frame")
    orient_record = _build_orient_record(...)  # via ingest -> cut -> orient in the test, or a direct constructor

    retilt_record, step = retilt_camera.fn(orient_record, raw_root, make_config(tmp_path))

    assert step.derived["tilt_deg"] == pytest.approx(8.0, abs=0.5)
    assert step.derived["roll_deg"] == pytest.approx(-2.0, abs=0.5)
    assert retilt_record.n_frames == orient_record.n_frames
```

Run and confirm RED — the module does not exist.

- [x] **Step 2: Implement region reading and rejection (§7's region rules)**

```python
metadata = yaml.safe_load((raw_root / record.date / record.session / "metadata.yaml").read_text())
region = read_floor_region(metadata, record.camera)  # raises ScanRejected, per Task 4
```

Test each rejection individually: missing fields, unparseable string, out-of-bounds, and both
degenerate cases (`x0 >= x1`, `y1 >= y0` — the corrected rule).

- [x] **Step 3: Implement translation-span rejection**

Read `odometry.csv`'s `x, y, z` at the sampled-frame indices (Step 4 defines the sample), and
reject when `translation_span_m(...) > config.retilt_max_translation_m`.

- [x] **Step 4: Implement sampling, pooling, and the plane fit**

```python
sample_indices = list(range(0, record.n_frames, config.retilt_sample_stride))[
    : config.retilt_max_sampled_frames
]
```

For each sampled index: read depth/confidence frames, run `select_floor_pixels` against
`region.pixel_bounds(...)`, back-project, and append to the pooled cloud (subsample to
`retilt_max_pooled_points` if the pool exceeds it — deterministic, e.g. every Nth pooled
point). Reject on `< retilt_min_floor_points`. Fit the plane; reject on RMS or on
`|tilt| > retilt_max_tilt_deg` / `|roll| > retilt_max_roll_deg`.

- [x] **Step 5: Implement the gravity check — warn only**

```python
g = gravity_down_camera(*_first_sample_quaternion(...))
agreement = gravity_agreement_deg(fit.normal, g)
if agreement > config.retilt_gravity_tolerance_deg:
    result.warnings.append(f"gravity disagreement {agreement:.2f} deg exceeds tolerance")
```

Never raise `ScanRejected` from this branch. Test both the pass and the warn path; assert no
exception either way.

- [x] **Step 6: Implement the warp**

RGB: decode with PyAV, `cv2.warpPerspective(frame, H_rgb, size, flags=cv2.INTER_LINEAR)` per
frame, re-encode carrying PTS through unchanged (frame count and timing are untouched by
Retilt — only pixel content and depth values change) exactly as `orient.py:rotate_rgb`
carries PTS through its filter graph. Depth/confidence: precompute `cv2.remap` coordinate
maps once from `H_depth⁻¹` (`cv2.INTER_NEAREST`, `cv2.BORDER_CONSTANT` with a sentinel that
`_validate_exit` never sees as valid — e.g. confidence border fills as `0`, matching "never
use confidence-0"). Apply `depth_scale_map` to the nearest-neighbour-sampled depth, clip to
`uint16` range, store.

- [x] **Step 7: Implement `_validate_exit` and publish**

Mirror `orient.py:_validate_exit` (lines 239–265): depth stays `uint16`; confidence stays
`⊆ {0,1,2}`; depth/confidence frame counts and filenames unchanged from input; RGB and depth
output dimensions equal their inputs (no resize). Then stage → validate → publish → cleanup,
matching `orient_camera`'s control flow exactly.

- [x] **Step 8: Write `retilt_sidecar.json` per §6, plus resolved ambiguities**

Include every §6 field. Additionally:

- `region_normalized`, `region_px` — both representations of the used `FloorRegion`, for
  debuggability without re-deriving pixel bounds from the raw fraction.
- `sample_indices`, `translation_span_m`.
- `gravity_check_mode: "warn"`.
- `confidence_mode`: report `"conf2_only"` only if **every** sampled frame individually chose
  `conf2_only`; otherwise `"conf1_and_2"`, alongside a `confidence_mode_per_frame: [...]` list
  of counts — resolving §1's per-frame-selection-vs-per-camera-field ambiguity honestly rather
  than picking the mode of the majority.
- Confirm "confidence `1` and `2`" in §1 step 3 is read as *from the annotated region*, not
  from the largest conf-2 component (which is conf-2 by construction and would make the
  fallback vacuous) — assert this with a dedicated fixture where conf-1 pixels lie outside the
  largest conf-2 component but inside the region, and confirm they are included when the
  conf2-only threshold is not met.

- [x] **Step 9: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_retilt.py -q
uv run mypy src
```

---

### Task 7: Flow wiring

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/flow.py`
- Modify: `tests/integration/test_preprocess_flow.py`

**Interfaces:**
- Changes: the per-camera loop calls `retilt_camera` after `orient_camera`; manifest
  `steps` gains `"retilt"`

- [x] **Step 1: Write the failing integration assertion**

Extend the existing flow integration test to assert `"retilt"` appears in
`manifest.scans[i].steps` and that `s3_retilt_output/<date>/<session>/<camera>/rgb.mp4`
exists after a non-dry-run invocation with a real (small) fixture session.

- [x] **Step 2: Wire the call**

In `flow.py`'s per-camera `try` block:

```python
cut_record, cut_step = cut_camera(record, interval, config)
orient_record, orient_step = orient_camera(cut_record, config)
retilt_record, retilt_step = retilt_camera(orient_record, config.raw_root, config)
```

Extend `steps=["ingest", "cut", "orient", "retilt"]` and merge `retilt_step.derived`,
`.warnings`, `.file_ops` into the `ScanOutcome` the same way `cut_step`/`orient_step` are
merged today (`flow.py:92-94`).

Note for implementers: a camera rejected at S3 has already published S2 output — this
matches the existing shape (a camera rejected at S2 has already published S1 output) and is
intentionally not special-cased.

- [x] **Step 3: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/integration -q
```

---

### Task 8: Fixtures for a real fittable floor

**Files:**
- Modify: `tests/conftest.py`

**Interfaces:**
- Extends: `make_camera(..., floor_tilt_deg=0.0, floor_roll_deg=0.0, floor_distance_m=1.5)`
- Extends: `make_session_metadata(..., floor_regions=None)`

`make_camera` currently writes depth as all-zero plus one marker pixel (`conftest.py:150`) —
there is no plane in it to fit. `CLAUDE.md` requires verifying geometry numerically against
synthetic fixtures, so this stage's tests need a fixture that renders an actual tilted plane.

- [x] **Step 1: Extend `make_camera` to render a real depth plane**

For every depth frame, instead of (or in addition to) the existing marker, fill the frame
with `Z(u,v)` sampled from the plane `Y = a·X + b·Z + c` (derived from `floor_tilt_deg`/
`floor_roll_deg`/`floor_distance_m` via the inverse of `retilt.tilt_roll_from_normal`),
back-projected through `STATIC_K`'s depth-scaled intrinsics, converted to millimetre `uint16`
depth values. Set confidence `2` everywhere inside the annotated region, `0` outside it, so
`select_floor_pixels` has a real, large conf-2 component to find.

- [x] **Step 2: Extend `make_session_metadata` for the `video:` block**

```python
def _make(..., floor_regions: Literal["full_frame"] | dict[str, str] | None = None, ...):
    ...
    if floor_regions == "full_frame":
        lines += [
            "video:",
            '  front_floor_region_bottom_left_in_pixels: "(0, 1)"',
            '  front_floor_region_top_right_in_pixels: "(1, 0)"',
            '  side_floor_region_bottom_left_in_pixels: "(0, 1)"',
            '  side_floor_region_top_right_in_pixels: "(1, 0)"',
        ]
    elif isinstance(floor_regions, dict):
        lines += ["video:"] + [f"  {key}: {value}" for key, value in floor_regions.items()]
```

Support `omit` values that drop individual fields (mirroring the existing lift-window `omit`
knob) so each §7 region-rejection test can construct exactly the malformed input it needs.

- [x] **Step 3: Verify the fixture round-trips through Task 4's maths**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_retilt_math.py -q
uv run pytest --no-cov tests/unit/test_preprocess_retilt.py -q
```

Both suites now exercise real rendered geometry rather than stub data.

---

### Task 9: Calibrate on real data; do not yet promote the gravity check

**Files:**
- None modified — this is a data-collection task, not a code task.

**Interfaces:**
- Consumes: a real `powerflow preprocess` run over `data/raw/11 July`
- Produces: four measured values per camera, recorded in this plan file's own notes (or a
  follow-up doc) as the basis for a later promotion decision

- [x] **Step 1: Dry-run first**

```bash
uv run powerflow preprocess --input ../data/raw --records ../data/s0_ingest_output \
  --cut ../data/s1_cut_output --retilt ../data/s3_retilt_output \
  --output ../data/s2_orient_output --dry-run
```

Expected: 4 cameras planned, 0 rejected, `retilt` present in each `steps` list.

Result: matched — dry-run planned all 4 cameras, 0 rejected, `retilt_camera` present for each.

- [x] **Step 2: Real run**

```bash
uv run powerflow preprocess --input ../data/raw --records ../data/s0_ingest_output \
  --cut ../data/s1_cut_output --retilt ../data/s3_retilt_output \
  --output ../data/s2_orient_output
```

- [x] **Step 3: Record the four calibration numbers per camera**

```bash
for f in ../data/s3_retilt_output/*/*/*/retilt_sidecar.json; do
  jq '{camera: input_filename, tilt_deg, roll_deg, gravity_agreement_deg, plane_rms_residual_m, translation_span_m}' "$f"
done
```

If `gravity_agreement_deg` is small and consistent (well under
`retilt_gravity_tolerance_deg`) across all four cameras, the axis-convention chain in Task 4
is confirmed correct and a follow-up plan can flip §7's gravity rule from warn to reject. If
it is large or inconsistent, the chain has a sign error — debug against these specific
sidecars before promoting anything, and do **not** flip the check while its correctness is
unconfirmed.

**Results (2026-08-26, `../data/raw/11 July`, `--overwrite` over pre-existing 2026-07-16 S1/S2
output; 4 cameras processed, 0 rejected):**

| camera | tilt_deg | roll_deg | gravity_agreement_deg | plane_rms_residual_m | translation_span_m |
|---|---|---|---|---|---|
| 30kg_Set1/Front | -8.013 | -2.177 | 15.124 | 0.00498 | 0.00113 |
| 30kg_Set1/Side  | -8.892 | -1.795 | 14.774 | 0.00814 | 0.00437 |
| 50kg_Set3/Front | -8.197 | -2.493 | 15.280 | 0.00432 | 0.00132 |
| 50kg_Set3/Side  | -8.215 | -1.934 | 19.850 | 0.00379 | 0.00548 |

Plane fits are tight (RMS residual 4–8 mm, well under `retilt_max_plane_rms_m` = 0.02 m) and
cameras were effectively static during sampling (translation span 1–5 mm, well under
`retilt_max_translation_m` = 0.05 m), so the plane-fitting side of the pipeline is trustworthy.

`gravity_agreement_deg` is **large across all four cameras** (14.8°–19.9°), well past
`retilt_gravity_tolerance_deg` = 5.0°. Three of four cluster tightly around 15° (Front/Front/
Side of 30kg and 50kg), with the fourth (50kg_Set3/Side) at 19.9° — consistent enough in
magnitude to suggest a systematic error (most likely a sign or axis-order mistake in the
`gravity_down_camera` chain: world→ARKit→CV→S2-portrait) rather than four independent sensor
miscalibrations, but not so tightly clustered as to rule out a genuine, camera-specific mount
tilt contributing on top of a smaller convention bug. **Conclusion: do not promote the gravity
check from warn to reject.** A follow-up task should debug the `gravity_down_camera` axis
chain against these four sidecars (e.g. by comparing the recovered `plane.normal` and the
computed `g_camera` vector component-by-component for one sample) before any promotion
decision.

---

### Task 10: README

**Files:**
- Modify: `powerflow-pipeline/README.md`

**Interfaces:**
- None — documentation only.

- [x] **Step 1: Add S3 to the stage list, `--retilt` to the flag table, `retilt_sidecar.json`
  to the output table**

Follow the existing S0/S1/S2 bullet style and the "Verifying a Cut run" section's `jq`/`yq`
pattern for a new "Verifying a Retilt run" section, e.g.:

```bash
jq '{tilt_deg, roll_deg, gravity_agreement_deg}' \
  ../data/s3_retilt_output/11\ July/30kg_Set1/Front/retilt_sidecar.json
```

- [x] **Step 2: Confirm the doc renders and cross-references are correct**

No automated check — read it through once against the actual flag names added in Task 2.

Read through: `--retilt` matches the CLI flag added in Task 2; also fixed two stale references
that Task 2's addition otherwise left inconsistent — "all three stage roots" → "all four", and
the run-time/output-size note (now three re-encode passes / four stage outputs, softened to
avoid asserting an unverified precise number for the four-stage timing).

---

### Task 11: Full-repository verification

**Files:**
- Verify only: all modified source, test, and documentation files

**Interfaces:**
- Consumes: completed Tasks 1–10
- Produces: evidence that formatting, linting, typing, coverage, and the full test suite pass

- [x] **Step 1: Run the four CI gates in order**

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src
uv run pytest --cov --cov-fail-under=90
```

Expected: all four exit 0; coverage at or above 90%.

Result: all four green. `ruff check` — all checks passed. `ruff format --check` — 49 files
already formatted. `mypy src` — no issues found in 27 source files. `pytest --cov
--cov-fail-under=90` — 197 passed, total coverage 96.17%.

- [x] **Step 2: Inspect the final diff**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the files this plan describes differ, in addition to any
pre-existing workspace changes unrelated to this plan.

Note: `/Users/yeethui/github/PowerFlow` is not a git repository, so `git diff`/`git status`
were not available. Substituted a direct trailing-whitespace grep over the touched
documentation (`README.md`, this plan file) — none found — and relied on `ruff format --check`
already having verified whitespace/formatting across all 49 tracked source/test files.
