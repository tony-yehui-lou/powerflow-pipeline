# Odometry-Authoritative Cut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `odometry.csv:timestamp` the authoritative frame clock and apply one timestamp-derived index selection to RGB, depth, confidence, and odometry.

**Architecture:** `cut_camera` will map odometry timestamps to epoch milliseconds, select rows inside the shared closed cut interval, and cap that selection to indices having an RGB counterpart. The resulting paired indices drive all frame-paired outputs; IMU retains its independent timestamp selection.

**Tech Stack:** Python 3.12, PyAV, pandas, Prefect 3, pytest, Ruff, mypy

## Global Constraints

- Do not run the real `powerflow preprocess` pipeline.
- Do not resample or interpolate any stream.
- Keep IMU timestamp-selected because it is not frame-index paired.
- Preserve unrelated pre-existing workspace changes.
- Do not add dependencies.

---

### Task 1: Lock the authoritative-clock behavior with a failing test

**Files:**
- Modify: `tests/unit/test_preprocess_cut.py:155-229`

**Interfaces:**
- Consumes: `cut_camera.fn(record: CameraRecord, interval: CutInterval, config: PreprocessConfig)`
- Produces: regression coverage proving RGB, depth, confidence, and odometry use identical odometry-derived indices

- [ ] **Step 1: Replace the obsolete post-cut tolerance test with the regression test**

Use the existing variable-PTS fixture to assert that the odometry interval `[33,100]` selects
indices `2..6` for all paired streams even though MP4 PTS would select only indices `2..5`:

```python
def test_rgb_uses_odometry_timestamps_instead_of_mp4_pts(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    record = build_record(tmp_path, make_camera, rgb_frames=8, depth_frames=9, imu_rows=20)
    interval = make_interval(start_ms=CREATED_EPOCH_MS + 33, end_ms=CREATED_EPOCH_MS + 100)

    cut_record, step = cut_camera.fn(record, interval, make_config(tmp_path, dry_run=True))

    assert cut_record.counts.rgb == 5
    assert cut_record.counts.depth == 5
    assert cut_record.counts.confidence == 5
    assert cut_record.counts.odometry == 5
    assert cut_record.n_frames == 5
    assert step.derived["frame_timestamp_source"] == "odometry.csv:timestamp"
```

Update the existing happy-path and dry-run expectations from four RGB frames to five, and assert
that the sidecar records `frame_timestamp_source: odometry.csv:timestamp`.

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
uv run pytest --no-cov tests/unit/test_preprocess_cut.py::test_rgb_uses_odometry_timestamps_instead_of_mp4_pts -q
```

Expected: FAIL because current `cut_camera` returns four RGB frames selected from MP4 PTS rather
than five frames selected from odometry indices.

---

### Task 2: Select every paired frame stream from odometry indices

**Files:**
- Modify: `src/powerflow_pipeline/data/preprocess/tasks/cut.py:1-110,238-349`
- Modify: `src/powerflow_pipeline/data/preprocess/timeline.py:1-12`
- Test: `tests/unit/test_preprocess_cut.py`

**Interfaces:**
- Consumes: `record.counts.rgb`, odometry timestamps, `select_closed_indices`
- Produces: one `paired_keep: list[int]` used for RGB, depth, confidence, and odometry; provenance key `frame_timestamp_source`

- [ ] **Step 1: Implement the minimal paired-index selection**

Remove `probe_rgb_pts`. After building `odom_epochs`, first select all odometry rows in the closed
interval so the existing empty-odometry error remains precise. Then restrict them to RGB-backed
indices and use the result everywhere:

```python
depth_keep = select_closed_indices(odom_epochs, start_ms, end_ms)
imu_keep = select_closed_indices(imu_epochs, start_ms, end_ms)

if not depth_keep:
    raise ScanRejected("empty odometry.csv in cut interval")

paired_count = min(record.counts.rgb, len(depth_paths), len(confidence_paths), len(odometry_ts))
paired_keep = [index for index in depth_keep if index < paired_count]
if not paired_keep:
    raise ScanRejected(f"no rgb frames in cut interval for camera {record.camera}")
if not imu_keep:
    raise ScanRejected("empty imu.csv in cut interval")

rgb_keep = paired_keep
depth_keep = paired_keep
n_frames = len(paired_keep)
```

Use `odom_epochs` for both RGB and depth retained epoch ranges. Add
`"frame_timestamp_source": "odometry.csv:timestamp"` to the sidecar and `StepResult.derived`.
Update module documentation to state that RGB/depth/confidence/odometry share the odometry clock;
update `timeline.py` documentation consistently.

- [ ] **Step 2: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest --no-cov tests/unit/test_preprocess_cut.py -q
```

Expected: all Cut tests PASS.

- [ ] **Step 3: Run adjacent unit and integration tests**

Run:

```bash
uv run pytest --no-cov tests/unit/test_preprocess_timeline.py tests/integration/test_preprocess_flow.py tests/integration/test_prefect_spine.py -q
```

Expected: all selected tests PASS.

---

### Task 3: Verify repository quality without running the real pipeline

**Files:**
- Verify only: all modified source, test, and documentation files

**Interfaces:**
- Consumes: completed Tasks 1-2
- Produces: evidence that formatting, linting, typing, and automated tests pass

- [ ] **Step 1: Run formatting, lint, and type checks**

Run:

```bash
uv run ruff format --check
uv run ruff check
uv run mypy src
```

Expected: all commands exit 0.

- [ ] **Step 2: Run the complete automated test suite**

Run:

```bash
uv run pytest --no-cov -q
```

Expected: all tests PASS. Do not invoke `powerflow preprocess`.

- [ ] **Step 3: Inspect the final diff**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files differ in addition to the user's pre-existing
workspace changes.
