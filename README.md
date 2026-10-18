# powerflow-pipeline

Data and training pipelines for PowerFlow. Prefect 3 orchestrates every data pipeline under
`src/powerflow_pipeline/data/`.

The pipeline that exists today is **`preprocess`** — Step 1 of Preprocessing V2:

Two raw directory shapes are handled by one invocation, detected per capture from where the
operator `metadata.yaml` sits — there is no layout flag:

| | two-camera session | single-camera capture |
|---|---|---|
| Raw shape | `<date>/<session>/<camera>/` streams | `<date>/<liftType>/<trial>/` streams, or just `<date>/<trial>/` |
| `metadata.yaml` | at the **session**, shared | **inside** the capture |
| Roles | `front`, `side` from the directory name | `single` |
| Cut interval owner | the `side` camera | the capture itself |

The lift-type folder (`Snch`, `CnJ`) in the single-camera shape is only ever used to *infer*
the lift type for `metadata.py`'s derived fields — discovery itself doesn't care whether it's
there, so `<date>/<trial>/` (no lift-type folder) and `<date>/<liftType>/<trial>/` both work.

Output **mirrors the raw path exactly**, so a stage tree has whatever depth its raw counterpart
has. A *role* is only a lookup key for which `metadata.yaml` fields a capture reads; it says
nothing about where the camera pointed.

- **S0 · Ingest** — validate each camera in a raw capture and emit a normalized `record.json`.
- **S1 · Cut** — derive a shared real-world epoch interval from the interval owner's
  operator-declared lift window, then trim every time-indexed stream in the group to that
  interval. This establishes invariant **I2**: frame `k` of every stream refers to the same
  real-world instant across every camera of the group.
- **S2 · Orient** — rotate RGB, depth, and confidence to portrait, and rotate the camera matrix
  with them. This establishes invariant **I1**: every image is portrait, in the frame its
  intrinsics describe.
- **S3 · Retilt** — fit one floor plane per camera from depth in an operator-annotated region,
  derive the tilt/roll that levels it, and rectify RGB, depth, and confidence with the resulting
  per-camera homography. This establishes invariant **I3**: the floor plane is level in every
  image. Tilt and roll only — **yaw is never corrected**, so the floor-anchored frame's
  horizontal axes are the camera's, not the athlete's.

A camera that fails validation is **rejected with a reason**, never emitted degraded.

---

## Build

Python 3.12, managed by [`uv`](https://docs.astral.sh/uv/). Use `uv run` for everything — never
`pip` or the system Python.

```bash
uv sync                 # create .venv and install runtime + dev dependencies
uv sync --no-dev        # runtime dependencies only
```

There is nothing to compile. `uv sync` is the whole build.

To add a dependency (ask first — see `CLAUDE.md`):

```bash
uv add <package>            # runtime
uv add --dev <package>      # dev only
```

No system `ffmpeg` binary is required. Video work goes through **PyAV**, which bundles the ffmpeg
libraries and ships ffmpeg's own filters and the libx264 encoder.

---

## Test

The four CI gates, in order. Each is runnable on its own:

```bash
uv run ruff check                            # lint
uv run ruff format --check                   # formatting (drop --check to apply)
uv run mypy src                              # types, strict
uv run pytest --cov --cov-fail-under=90      # tests, coverage floor 90%
```

A bare `uv run pytest` already applies coverage and the 90% floor — they are configured in
`pyproject.toml`. A bare `uv run mypy` checks `tests/` as well as `src/`.

### Running a subset

```bash
uv run pytest tests/unit                     # fast: no Prefect backend
uv run pytest tests/integration              # flow + CLI, under a disposable Prefect server
uv run pytest tests/unit/test_preprocess_geometry.py -v
uv run pytest -k "rotation or intrinsics"    # by name
uv run pytest --no-cov -q                    # skip coverage while iterating
uv run pytest -x --lf                        # stop at first failure, rerun last failures
```

### What the tests actually assert

Geometry is verified **numerically against synthetic fixtures**, never by checking that a transform
merely ran. The fixture (`tests/conftest.py::make_camera`) builds a miniature of the real capture —
a landscape clip, `uint16` depth, `{0,2}` confidence, a drifting per-frame `odometry.csv` — carrying
an **asymmetric marker**, so that a wrong rotation direction, a bare transpose, and a flip each
produce a *different* wrong answer.

The fixture's clip is deliberately **variable frame rate**, like the real capture. A uniform clip
hides a real failure: irregular timestamps make libx264's B-frame decode timestamps collide at a
coarse timebase, and the muxer then rejects the packet.

Cut is verified with parametrized rejection cases mirroring the Ingest V1–V11 pattern: one test per
rejection rule from the spec (missing lift window, non-numeric, negative, out-of-order, interval
outside capture, empty stream after cut, timestamp unavailable, depth/confidence mismatch,
frame-count tolerance exceeded). The happy path asserts retained index ranges, reindexed filenames,
byte-for-byte `camera_matrix.csv`, and per-stream epoch ranges in `cut_sidecar.json`.

`tests/integration/` runs the flow under `prefect_test_harness`, which spins up a throwaway local
Prefect database. Nothing is written to a real Prefect server.

---

## Writing `metadata.yaml`

Every capture needs an operator-annotated `metadata.yaml`: **inside** the capture for a
single-camera trial, or **shared at the session** (one level up) for a two-camera session.
Two fields are load-bearing — S1 Cut and S3 Retilt reject the camera without them. Everything
else in the file (`weight_in_kg`, `type`, `result`, `dateTime_epoch`, the whole `athlete:`
block) is **never read by the pipeline**; it's carried through to the audit trail as
informational only (and the stage output's own `athlete:` block is always empty — nothing in
the capture supplies those values).

Single-camera example:

```yaml
lift:
  lift_start_time_side_in_ms: 31100   # REQUIRED (S1): ms into the raw rgb.mp4 where the lift starts
  lift_end_time_side_in_ms: 38330     # REQUIRED (S1): ms into the raw rgb.mp4 where it ends
  weight_in_kg: 50.0                  # informational only, not read
  type: snch                          # informational only, not read
  result: success                     # informational only, not read
athlete:                              # informational only, not read
  name: string
  height_in_cm: 180.0
  ...
video:
  floor_region_bottom_left_in_pixels: (0.20, 0.78)  # REQUIRED (S3): normalized (x, y)
  floor_region_top_right_in_pixels: (0.55, 0.68)    # normalized (x, y)
```

A two-camera session's shared file instead prefixes the lift window and floor region by role,
since one file governs both cameras:

```yaml
lift:
  lift_start_time_side_in_ms: 31100   # always "_side_", even though this is the shared file
  lift_end_time_side_in_ms: 38330
video:
  front_floor_region_bottom_left_in_pixels: (x0, y0)
  front_floor_region_top_right_in_pixels:   (x1, y1)
  side_floor_region_bottom_left_in_pixels:  (x0, y0)
  side_floor_region_top_right_in_pixels:    (x1, y1)
```

### Measuring the lift window

Open the **raw** `rgb.mp4` — the `Side` camera's, for a two-camera session — and read the
player timestamp where the lift starts and ends, in milliseconds from the start of that file.
S1 Cut rejects the group if:

- either field is missing, non-numeric, or negative
- `lift_start_time_side_in_ms >= lift_end_time_side_in_ms`
- the resulting interval falls outside the camera's own recording

A window under 1 second only warns — a real snatch can be that fast.

### Measuring the floor region

Measure on **S2's portrait output** (`data/s2_orient_output/<capture>/rgb.mp4`), not the raw
landscape video — S3 applies the rectangle to S2's rotated depth/confidence frames, and the
raw frame is a different orientation. Coordinates are normalized `[0, 1]`: `x` left→right,
`y` **top→bottom** (`y=0` top of frame), written as parenthesised strings (`(0.20, 0.78)`,
not a YAML list). `bottom_left`'s `y` must be numerically larger than `top_right`'s — S3
rejects a region where that's reversed.

Pick a patch that is:
- flat floor, not a riser, mat edge, or anything else at a different height
- clear of feet, plates, and bystanders during roughly the **first 5 seconds** of the cut clip
  (S3 samples every 10th frame, up to 32 frames)
- close enough to the camera for the LiDAR to return clean depth

A region that looks fine by eye can still fail: on 22 August, the previous default rectangle
sat on a PA speaker cabinet in the foreground rather than the floor, and every capture that
used it failed S3's plane-fit RMS check even though nothing about the YAML itself was wrong.

Workflow for a day whose camera position hasn't been measured yet: run once through S4 (below)
with whatever placeholder region is in the file, open the resulting S2 `rgb.mp4`, measure the
real rectangle, write it into the **raw** `metadata.yaml`, then rerun just that capture with
`--only '<date>/<trial>' --overwrite` — S1 and S2 are cheap to redo.

### Checklist before a run

- [ ] Both `lift_start_time_side_in_ms` and `lift_end_time_side_in_ms` are this capture's own
      values, not copied from another trial (a copy-pasted window is accepted silently — S1 has
      no way to tell it's wrong, it just cuts the wrong few seconds).
- [ ] The floor region was measured on this camera's own S2 output, not inherited from a
      previous day's tripod position.
- [ ] A two-camera session's shared file has both `front_` and `side_` prefixed pairs.

---

## Run the pipeline

There is no flag to stop early: every run does S0 Ingest → S1 Cut → S2 Orient → S3 Retilt →
S4 Crop, and adding `--pose` appends S5 Pose on top. If S3 or S4 rejects a camera, its earlier
stage outputs (S2, S3) are already published and are not rolled back — only the rejected
stage's output is missing.

### Recipe A — through S4 Crop

```bash
cd powerflow-pipeline
uv sync

uv run powerflow preprocess \
  --input   ../data/raw \
  --records ../data/s0_ingest_output \
  --cut     ../data/s1_cut_output \
  --output  ../data/s2_orient_output \
  --retilt  ../data/s3_retilt_output \
  --crop    ../data/s4_crop_output \
  --only    '10 September/*' \
  --dry-run
```

Drop `--dry-run` once it reports what you expect.

### Recipe B — through S5 Pose

Same as Recipe A, plus `--pose`:

```bash
uv run powerflow preprocess \
  --input   ../data/raw \
  --records ../data/s0_ingest_output \
  --cut     ../data/s1_cut_output \
  --output  ../data/s2_orient_output \
  --retilt  ../data/s3_retilt_output \
  --crop    ../data/s4_crop_output \
  --pose    ../data/s5_pose_output \
  --only    '10 September/*' \
  --dry-run
```

This re-runs S0–S4 too — S5 isn't a separate pass over existing S4 output, it's the same flow
with one more stage attached. Optionally choose the 2D detector:

```bash
--pose-detector rtmpose --rtmpose-variant balanced   # default: --pose-detector mediapipe
```

The first run against a given detector/variant downloads its pretrained weights into
`models_cache/` (git-ignored) — expect network access on that first run only. S5 writes one
`pose.json` per surviving camera plus a single `skeleton.human-v2.json` at the pose root's top
level, describing the joint topology every `pose.json` under it shares.

### Flags

| Flag | Meaning |
|---|---|
| `--input` | Raw capture root; both shapes above, mixed freely. |
| `--records` | Where S0 writes `record.json`, `metadata.yaml`, and `manifest.json`. |
| `--cut` | Where S1 publishes the trimmed, time-aligned landscape streams. |
| `--output` | Where S2 publishes the rotated portrait streams. |
| `--retilt` | Where S3 publishes the rectified, floor-levelled streams. |
| `--crop` | Where S4 publishes the cropped streams. |
| `--pose` | Where S5 publishes detected 2D poses. Omit to skip S5 entirely. |
| `--pose-detector` | `mediapipe` (default) or `rtmpose`. |
| `--rtmpose-variant` | `lightweight`, `balanced` (default), or `performance` — only with `--pose-detector rtmpose`. |
| `--only` | Process only captures whose id matches this glob, e.g. `'22 August/Snch/*'`. Keep a two-camera session whole — see below. |
| `--rotation` | `cw` (default) or `ccw`. |
| `--dry-run` | Validate and plan, write nothing. |
| `--overwrite` | Replace cameras already published (default: refuse). |

Start with `--dry-run`: it validates every camera and reports what *would* be written.

`--only` selects **captures**, and a two-camera session derives its cut interval from its
`Side` camera, so a glob that excludes it rejects the group with `session has no Side camera`.
Match the session, not one camera:

```bash
uv run powerflow preprocess ... --only '11 July/30kg_Set1/*'    # the session, both cameras
uv run powerflow preprocess ... --only '22 August/Snch/110kgSnch1'  # a single-camera capture
```

A full run over the four cameras in `data/raw/9 July` takes on the order of **10+ minutes** — it
decodes, cuts, rotates, retilts, crops, and re-encodes ~22k frames four times — and produces
several GB across all four stage outputs. Adding `--pose` adds a detector pass over S4's RGB on
top of that.

There is no `--in-place` mode. It would rewrite the raw capture, and raw data is write-once: each
stage publishes to its own output root instead.

### Where the output goes

| What | Where | Written by |
|---|---|---|
| **Run manifest** (JSON) | `<--records>/manifest.json` | end of the flow |
| **Run manifest** (markdown) | Prefect artifact, key `preprocess-run-manifest` | end of the flow |
| Per-camera validated record | `<--records>/<capture path>/record.json` | S0 |
| Per-group metadata | the raw `metadata.yaml`'s own relative path, in **all five** stage roots | S0 |
| Trimmed landscape streams + provenance | `<--cut>/<capture path>/` incl. `cut_sidecar.json` | S1 |
| Portrait streams + provenance | `<--output>/<capture path>/` incl. `sidecar.json` | S2 |
| Floor-levelled portrait streams + provenance | `<--retilt>/<capture path>/` incl. `retilt_sidecar.json` | S3 |
| Cropped streams + provenance | `<--crop>/<capture path>/` | S4 |
| 2D pose document | `<--pose>/<capture path>/pose.json`, plus `<--pose>/skeleton.human-v2.json` once per run | S5 |

`<capture path>` is the capture's raw-relative path — `11 July/30kg_Set1/Front` or
`22 August/Snch/110kgSnch1` — so every stage tree matches its raw counterpart byte for byte in
path shape.

Every stage output root **through S4** gets its own byte-identical copy of `metadata.yaml`, so a
consumer of one stage never has to reach back into an earlier stage's tree to learn which lift the
pixels came from. S5's `--pose` root deliberately does **not** get one: unlike S0–S4 it doesn't
run for every camera in every run, so copying it there would leave orphaned files with no pose
output beside them whenever a camera is skipped or `--pose` is omitted.

The **manifest is the audit record for one run**: every camera processed, the values derived
(rotation applied, `k_rewritten`, frame counts, cut epoch bounds), any warnings, and **every
rejected camera with its exact reason**. Read it first when a run does not do what you expected.

The JSON copy is the one to grep:

```bash
jq '.rejected_scans' data/s0_ingest_output/manifest.json          # what was rejected, and why
jq '.scans[].derived' data/s0_ingest_output/manifest.json         # rotation, k_rewritten, cut bounds, n_frames
```

### Verifying a Cut run

The shared epoch bounds are recorded in two places — the session `metadata.yaml` and each camera's
`cut_sidecar.json`:

```bash
# The session sees the same cut interval for both cameras.
yq '.lift | {cut_start_epoch_ms, cut_end_epoch_ms}' \
  ../data/s1_cut_output/9\ July/cnj_45kg_Set1/metadata.yaml

# Each camera records its own retained epoch ranges.
jq '{camera: .camera_creation_time, rgb: .retained.rgb, depth: .retained.depth}' \
  ../data/s1_cut_output/9\ July/cnj_45kg_Set1/Side/cut_sidecar.json
jq '{camera: .camera_creation_time, rgb: .retained.rgb, depth: .retained.depth}' \
  ../data/s1_cut_output/9\ July/cnj_45kg_Set1/Front/cut_sidecar.json
```

Depth and confidence are always selected as matched pairs — `cut_sidecar.json` records identical
counts and epoch ranges for both.

### Verifying a Retilt run

Each camera's fitted plane, derived tilt/roll, and gravity cross-check land in
`retilt_sidecar.json`:

```bash
jq '{tilt_deg, roll_deg, gravity_agreement_deg}' \
  ../data/s3_retilt_output/11\ July/30kg_Set1/Front/retilt_sidecar.json
```

For a single-camera capture day, the run also compares every capture's fit against the day's
median tilt, roll and camera height — one tripod shot them all, so a capture that disagrees is
suspect even when its own RMS is tight. That check is **warn-only** and lands in the run
manifest, not the sidecar: the median is not knowable when the sidecar is written.

```bash
jq -r '.scans[] | select(.warnings | length > 0) | "\(.scan_id): \(.warnings | join("; "))"' \
  ../data/s0_ingest_output/manifest.json
```

`gravity_agreement_deg` compares the fitted floor normal against gravity derived from odometry —
it is recorded and **warns** past `retilt_gravity_tolerance_deg`, but never rejects a camera on
its own. `plane_rms_residual_m` and `translation_span_m` are the other two numbers worth checking
first: a run with a loose plane fit or a camera that moved during sampling is a run worth
distrusting even if nothing was rejected.

---

## Layout

```
src/powerflow_pipeline/data/
  cli.py                  # typer entrypoint: builds config, invokes the flow
  common/                 # shared @tasks and helpers: discovery, filesystem, manifest, errors
  preprocess/
    flow.py               # @flow, ordered orchestration only
    config.py             # PreprocessConfig
    models.py             # CameraRecord, Intrinsics, CutInterval, ...
    geometry.py           # pure rotation maths: no I/O, no Prefect
    timeline.py           # pure epoch-time maths: no I/O, no Prefect
    retilt.py             # pure plane-fit/tilt/homography maths: no I/O, no Prefect
    tasks/                # one @task per logical step: discover, ingest, cut, orient, retilt, crop, pose, metadata
prefect.yaml              # deployments
tests/unit/               # fast, no Prefect backend
tests/integration/        # flow + CLI, under prefect_test_harness
```

Data lives in `data/`, write-once: one task writes a file, many read it. Raw capture in `data/raw/`,
each stage's output in `data/<stage>_output/`.
