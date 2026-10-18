# S3 Retilt: single-frame fit implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the opt-in single-frame floor fit specified in
`docs/superpowers/specs/2026-10-18-retilt-single-frame-design.md`: a new
`retilt_single_frame` config field and `--retilt-single-frame` CLI flag that, when set, fits
S3's floor plane from frame 0 of the cut clip alone instead of pooling up to
`retilt_max_sampled_frames` frames. Default behaviour (flag unset) must be byte-for-byte
unchanged.

**Architecture:** One new config field (`config.py`), one new CLI flag (`cli.py`), a three-line
change to `_sample_indices` (`tasks/retilt.py`), and one new sidecar/derived field
(`frame_sample_mode`). No new modules, no new dependencies, no change to any pure-maths
function in `retilt.py` — `select_floor_pixels`, `backproject`, and `fit_plane` already work
over however many points `_sample_indices` hands them.

**Tech Stack:** Python 3.12, Prefect 3, pytest, Ruff, mypy — all already in `pyproject.toml`.

## Global Constraints

- No new dependencies.
- Default (`retilt_single_frame=False`, or omitted) must reproduce today's sidecar output
  exactly, field for field, aside from the new `frame_sample_mode: "pooled"` key.
- Do not touch `fit_plane`, `select_floor_pixels`, `backproject`, or any other pure-maths
  function — this is a sampling-policy change, not a fitting-algorithm change.
- Do not run the real `powerflow preprocess` pipeline until the final task, and only then as
  directed.
- TDD per `CLAUDE.md`: failing test first, then minimum implementation, for every task that
  touches `src/`.
- Preserve unrelated pre-existing workspace changes.

---

### Task 1: Config field

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/config.py`

**Interfaces:**
- Produces: `PreprocessConfig.retilt_single_frame: bool = False`

- [ ] **Step 1: Add the field**

Add next to the other `retilt_*` tunables (`config.py:94-109`):

```python
# When true, fit the floor plane from frame 0 of the cut clip alone instead of pooling up
# to `retilt_max_sampled_frames` frames (docs/superpowers/specs/2026-10-18-retilt-single-
# frame-design.md). `retilt_sample_stride`/`retilt_max_sampled_frames` are ignored in this
# mode. Default false: unchanged behaviour for every capture day calibrated against pooled
# sampling (11 July, 22 August).
retilt_single_frame: bool = False
```

No test needed for a bare pydantic default — covered by Task 2's CLI test and Task 3's
`_sample_indices` test.

---

### Task 2: CLI flag

**Files:**
- Modify: `src/powerflow_pipeline/data/cli.py`
- Modify: `tests/unit/test_cli.py`

**Interfaces:**
- Produces: `--retilt-single-frame` boolean option, threaded into `PreprocessConfig`

- [ ] **Step 1: Write the failing test (RED)**

Mirror `test_preprocess_command_accepts_retilt_root`'s shape — build a minimal raw fixture,
invoke the CLI with `--retilt-single-frame --dry-run`, assert `exit_code == 0`. Add a second
test asserting that omitting the flag still exits 0 (the default path), and — if the command
is invoked via the `preprocess_flow` call captured through a monkeypatched `preprocess_flow`
the way other CLI tests do, or by inspecting the `PreprocessConfig` built — assert
`retilt_single_frame` is `True`/`False` respectively.

Run `uv run pytest --no-cov tests/unit/test_cli.py -q -k retilt_single_frame` and confirm it
fails — the option doesn't exist yet.

- [ ] **Step 2: Add the option**

In `cli.py`'s `preprocess` command, alongside `overwrite` (a same-shaped boolean with no `/no-`
negation, matching `--overwrite`'s own style rather than `--rtmpose-subject-depth`'s, since
there's no reason to ever need an explicit `--no-retilt-single-frame`):

```python
retilt_single_frame: Annotated[
    bool,
    typer.Option(
        "--retilt-single-frame",
        help=(
            "Fit S3's floor plane from frame 0 of the cut clip alone, instead of pooling "
            "multiple frames. Use when the lifter or bar is expected to cross the annotated "
            "floor rectangle later in the clip."
        ),
    ),
] = False,
```

Thread it into the `PreprocessConfig(...)` construction: `retilt_single_frame=retilt_single_frame`.

- [ ] **Step 3: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/unit/test_cli.py -q
```

---

### Task 3: `_sample_indices` single-frame path

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/tasks/retilt.py`
- Modify: `tests/unit/test_preprocess_retilt.py`

**Interfaces:**
- Changes: `_sample_indices(n_frames: int, config: PreprocessConfig) -> list[int]` returns
  `[0]` when `config.retilt_single_frame` is true

- [ ] **Step 1: Write the failing unit test (RED)**

Direct test of the function, no fixture needed:

```python
def test_sample_indices_single_frame_ignores_stride_and_max() -> None:
    config = make_config(
        tmp_path, retilt_single_frame=True, retilt_sample_stride=10, retilt_max_sampled_frames=32
    )
    assert _sample_indices(500, config) == [0]

def test_sample_indices_pooled_is_unchanged_by_default() -> None:
    config = make_config(tmp_path, retilt_sample_stride=10, retilt_max_sampled_frames=3)
    assert _sample_indices(100, config) == [0, 10, 20]
```

(`_sample_indices` is module-private; import it the same way this test file already imports
other private helpers from `tasks.retilt`, or re-export nothing new — follow the existing
import style at the top of `tests/unit/test_preprocess_retilt.py`.)

Run `uv run pytest --no-cov tests/unit/test_preprocess_retilt.py -q -k sample_indices` and
confirm it fails for the single-frame case (the pooled case should already pass — it's a
regression guard).

- [ ] **Step 2: Implement**

```python
def _sample_indices(n_frames: int, config: PreprocessConfig) -> list[int]:
    if config.retilt_single_frame:
        if n_frames < 1:
            raise ScanRejected("no frames available to sample for the floor fit")
        return [0]
    indices = list(range(0, n_frames, config.retilt_sample_stride))[
        : config.retilt_max_sampled_frames
    ]
    if not indices:
        raise ScanRejected("no frames available to sample for the floor fit")
    return indices
```

- [ ] **Step 3: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_retilt.py -q -k sample_indices
uv run mypy src
```

---

### Task 4: Sidecar/derived `frame_sample_mode`

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/tasks/retilt.py`
- Modify: `tests/unit/test_preprocess_retilt.py`

**Interfaces:**
- Changes: `retilt_camera`'s written sidecar and returned `StepResult.derived` both gain
  `frame_sample_mode: Literal["pooled", "first"]`

- [ ] **Step 1: Write the failing test (RED)**

Extend (or add alongside) `test_retilt_camera_recovers_baked_in_tilt` — assert
`sidecar["frame_sample_mode"] == "pooled"` for the default config, and, with a second call
using `retilt_single_frame=True`, `== "first"`. Read the sidecar back from
`destination / "retilt_sidecar.json"` the way this test file's existing assertions do (or from
the staged dict directly if the test already captures it before publish — follow whichever
pattern `test_retilt_camera_recovers_baked_in_tilt` uses today).

Run and confirm RED — the key doesn't exist yet.

- [ ] **Step 2: Implement**

In `retilt_camera` (`tasks/retilt.py:221-387`), right after `sample_indices = _sample_indices(...)`:

```python
frame_sample_mode: Literal["pooled", "first"] = (
    "first" if config.retilt_single_frame else "pooled"
)
```

Add `"frame_sample_mode": frame_sample_mode` to the `sidecar` dict (next to
`"sample_indices"`) and to `StepResult(derived={...})` (next to `"plane_rms_residual_m"`).

- [ ] **Step 3: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_retilt.py -q
uv run mypy src
```

---

### Task 5: The motivating regression test — occlusion in a later pooled frame

**Files:**
- Modify: `tests/conftest.py` (only if the existing fixture helpers can't already express a
  per-frame depth/confidence override — check first; `make_camera` already renders a real
  tilted plane per Task 8 of `2026-08-26-retilt-implementation.md`, so a per-index override may
  already be a small, additive extension rather than new machinery)
- Modify: `tests/unit/test_preprocess_retilt.py`

**Interfaces:**
- Consumes: `make_camera`, `make_session_metadata`, `_make_floor_session`,
  `_build_orient_record` (all already used by this test file's existing tests)
- Produces: one fixture session whose frame 0 is clean floor and whose frame `20` (inside the
  default `retilt_sample_stride=10` sampling but outside a single-frame sample) has the
  annotated rectangle overwritten with non-floor depth/confidence

- [ ] **Step 1: Check whether the fixture needs extending**

Read `make_camera`'s current signature in `tests/conftest.py`. If it only supports one tilt/
roll/distance for every frame, add an optional knob — e.g.
`occlude_frame_indices: dict[int, ...] | None` — that overwrites the rendered depth/confidence
for specific frame indices within the rectangle with values that are not the fitted plane
(e.g. depth scaled to a much closer surface, confidence still `2` so it isn't filtered out by
`select_floor_pixels`'s confidence logic — the point is to simulate an occluding body the
pipeline has no way to distinguish from floor, not to simulate a sensor dropout it already
handles).

- [ ] **Step 2: Write the failing test (RED)**

```python
def test_pooled_mode_is_contaminated_by_a_later_occluded_frame(
    tmp_path, make_camera, make_session_metadata
) -> None:
    raw = _make_floor_session(
        tmp_path, make_camera, make_session_metadata,
        floor_tilt_deg=8.0, floor_roll_deg=-2.0, floor_distance_m=1.8,
        occlude_frame_indices={20: ...},  # non-floor surface inside the rectangle
    )
    config = make_config(tmp_path, retilt_min_floor_points=10, retilt_sample_stride=10,
                          retilt_max_sampled_frames=32)
    orient_record = _build_orient_record(raw, config)

    retilt_record, step = retilt_camera.fn(orient_record, config)

    # The contaminated frame measurably pulls the fit away from the baked-in truth —
    # looser tolerance than the clean-fixture test, which is the point being proven.
    assert abs(step.derived["tilt_deg"] - 8.0) > 0.5 or abs(step.derived["roll_deg"] - (-2.0)) > 0.5


def test_single_frame_mode_is_unaffected_by_the_same_occluded_frame(
    tmp_path, make_camera, make_session_metadata
) -> None:
    raw = _make_floor_session(
        tmp_path, make_camera, make_session_metadata,
        floor_tilt_deg=8.0, floor_roll_deg=-2.0, floor_distance_m=1.8,
        occlude_frame_indices={20: ...},
    )
    config = make_config(tmp_path, retilt_min_floor_points=10, retilt_single_frame=True)
    orient_record = _build_orient_record(raw, config)

    retilt_record, step = retilt_camera.fn(orient_record, config)

    assert step.derived["tilt_deg"] == pytest.approx(8.0, abs=0.5)
    assert step.derived["roll_deg"] == pytest.approx(-2.0, abs=0.5)
```

If the first assertion doesn't actually fail reliably (the contamination might be too small to
cross 0.5° depending on rectangle size/point density), tune the occlusion's extent or depth
offset until it does — the test only has value if it demonstrates real contamination, not a
rounding difference. Run both and confirm the first is RED-by-construction before any
implementation (it should already pass, since Task 3/4 didn't touch the maths — this task is
pure test-writing) and the second passes immediately once Task 3 is done.

- [ ] **Step 3: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_retilt.py -q
```

Both tests should pass without further `src/` changes — this task is verifying the design's
central claim, not implementing anything new.

---

### Task 6: Translation-span and gate tests under single-frame mode

**Files:**
- Modify: `tests/unit/test_preprocess_retilt.py`

**Interfaces:**
- None new — exercises existing code paths under the new config value.

- [ ] **Step 1: `translation_span_m` is trivially zero**

Reuse `test_translation_beyond_tolerance_is_rejected`'s fixture and
`_perturb_odometry_x` helper, but with `retilt_single_frame=True` and a tight
`retilt_max_translation_m`. Assert the run **succeeds** (no `ScanRejected`) and the written
sidecar's `translation_span_m == 0.0`, proving the perturbation that rejects pooled mode is
inert under one sample — documented behaviour, not a silent gap.

- [ ] **Step 2: `retilt_min_floor_points` and `retilt_max_plane_rms_m` still apply**

Reuse `test_tilt_beyond_bound_is_rejected`'s fixture shape with `retilt_single_frame=True`;
assert the same `ScanRejected` still fires. Add one case with a rectangle sized so a single
frame's point count falls under `retilt_min_floor_points` (e.g. a tiny rectangle that still
clears it when pooled over many frames isn't the right test here — construct the rectangle or
threshold so frame 0 alone is insufficient) and assert the rejection names
`retilt_min_floor_points`, proving single-frame mode doesn't get a free pass on the gates.

- [ ] **Step 3: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/unit/test_preprocess_retilt.py -q
uv run mypy src
```

---

### Task 7: Flow-level regression guard

**Files:**
- Modify: `tests/integration/test_preprocess_flow.py`

**Interfaces:**
- None new — one additional flow invocation.

- [ ] **Step 1: Write the test**

Run the existing small fixture session twice through `preprocess()` end to end (not
`.fn()`-level): once with `config.retilt_single_frame` left at its default, once with it set
`True`. Assert both runs reach `"retilt"` in `steps` and publish. Assert the default run's
`retilt_sidecar.json` is unchanged from what the suite already expects elsewhere (if an
existing integration test already snapshots or spot-checks specific sidecar fields for this
fixture, add `frame_sample_mode == "pooled"` to that assertion rather than duplicating the
whole check).

- [ ] **Step 2: Run and verify GREEN**

```bash
uv run pytest --no-cov tests/integration -q
```

---

### Task 8: README

**Files:**
- Modify: `powerflow-pipeline/README.md`

**Interfaces:**
- None — documentation only.

- [ ] **Step 1: Add `--retilt-single-frame` to the flag table**

One row, next to the other `--retilt*`-adjacent flags, describing when to use it (lifter/bar
expected to cross the annotated rectangle later in the clip) and that the default is pooled
sampling.

- [ ] **Step 2: Add a short note to "Measuring the floor region"**

One sentence: if the rectangle can't stay clear of the lifter for the whole cut clip,
`--retilt-single-frame` fits from frame 0 alone instead of pooling — cross-reference
`docs/superpowers/specs/2026-10-18-retilt-single-frame-design.md` for why frame 0 specifically.

---

### Task 9: Full-repository verification

**Files:**
- Verify only.

- [ ] **Step 1: Run the four CI gates in order**

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src
uv run pytest --cov --cov-fail-under=90
```

Expected: all four exit 0; coverage at or above 90%.

- [ ] **Step 2: Confirm the default path is untouched**

```bash
git diff --stat -- src/powerflow_pipeline/data/preprocess/tasks/retilt.py
```

Read the diff by hand: every change should be additive (`if config.retilt_single_frame:`
branches and the new sidecar/derived key), with no line inside the existing pooled branch
altered.

---

### Task 10: Calibrate on 10 September (data task, not a code task)

**Files:** None modified.

**Prerequisite:** 10 September's floor region must already be re-measured against its own S2
output — not this plan's job; tracked separately (see the retilt-single-frame spec's "Out of
scope" and the earlier README work on measuring the floor region).

- [ ] **Step 1: Run both modes per the spec's "Verification" section**, compare
  `tilt_deg`/`roll_deg`/`plane_rms_residual_m`/`n_floor_points`/`frame_sample_mode` per
  capture, and record the comparison (in this plan's notes, or a short follow-up note) as the
  basis for deciding whether `--retilt-single-frame` becomes 10 September's standing recipe.
- [ ] **Step 2: Re-run 11 July without the flag** and diff `retilt_sidecar.json` against the
  values recorded in `2026-08-26-retilt-implementation.md` Task 9 — confirm unchanged.
