# powerflow-pipeline

Data and training pipelines for PowerFlow. Prefect 3 orchestrates every data pipeline under
`src/powerflow_pipeline/data/`.

The pipeline that exists today is **`preprocess`** — Step 1 of Preprocessing V2:

- **S0 · Ingest** — validate each camera in a raw capture and emit a normalized `record.json`.
- **S1 · Cut** — derive a shared real-world epoch interval from the Side camera's operator-declared
  lift window, then trim every time-indexed stream in both cameras to that interval. This establishes
  invariant **I2**: frame `k` of every stream refers to the same real-world instant across both
  cameras.
- **S2 · Orient** — rotate RGB, depth, and confidence to portrait, and rotate the camera matrix
  with them. This establishes invariant **I1**: every image is portrait, in the frame its
  intrinsics describe.

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

## Run the pipeline

```bash
uv run powerflow preprocess \
  --input   ../data/raw \
  --records ../data/s0_ingest_output \
  --cut     ../data/s1_cut_output \
  --output  ../data/s2_orient_output
```

| Flag | Meaning |
|---|---|
| `--input` | Raw capture root: `<date>/<session>/<camera>/`. |
| `--records` | Where S0 writes `record.json`, `metadata.yaml`, and `manifest.json`. |
| `--cut` | Where S1 publishes the trimmed, time-aligned landscape streams. |
| `--output` | Where S2 publishes the rotated portrait streams. |
| `--rotation` | `cw` (default) or `ccw`. |
| `--dry-run` | Validate and plan, write nothing. |
| `--overwrite` | Replace cameras already published (default: refuse). |

Start with `--dry-run`: it validates every camera and reports what *would* be written.

A full run over the four cameras in `data/raw/9 July` takes about **10 minutes** — it decodes,
cuts, rotates, and re-encodes ~22k frames twice — and produces about 2.5 GB across all three
stage outputs.

There is no `--in-place` mode. It would rewrite the raw capture, and raw data is write-once: each
stage publishes to its own output root instead.

### Where the output goes

| What | Where | Written by |
|---|---|---|
| **Run manifest** (JSON) | `<--records>/manifest.json` | end of the flow |
| **Run manifest** (markdown) | Prefect artifact, key `preprocess-run-manifest` | end of the flow |
| Per-camera validated record | `<--records>/<date>/<session>/<camera>/record.json` | S0 |
| Per-session metadata | `<date>/<session>/metadata.yaml` in **all three** stage roots | S0 |
| Trimmed landscape streams + provenance | `<--cut>/<date>/<session>/<camera>/` incl. `cut_sidecar.json` | S1 |
| Portrait streams + provenance | `<--output>/<date>/<session>/<camera>/` incl. `sidecar.json` | S2 |

Every stage output root gets its own byte-identical copy of `metadata.yaml`, so a consumer of one
stage never has to reach back into an earlier stage's tree to learn which lift the pixels came from.
A new stage adds its root to `PreprocessConfig.stage_roots` and inherits the copy.

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
    tasks/                # one @task per logical step: discover, ingest, cut, orient, metadata
prefect.yaml              # deployments
tests/unit/               # fast, no Prefect backend
tests/integration/        # flow + CLI, under prefect_test_harness
```

Data lives in `data/`, write-once: one task writes a file, many read it. Raw capture in `data/raw/`,
each stage's output in `data/<stage>_output/`.
