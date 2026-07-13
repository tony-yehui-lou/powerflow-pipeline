# powerflow-pipeline

Data and training pipelines for PowerFlow. Prefect 3 orchestrates every data pipeline under
`src/powerflow_pipeline/data/`.

The pipeline that exists today is **`preprocess`** — Step 1 of Preprocessing V2:

- **S0 · Ingest** — validate each camera in a raw capture and emit a normalized `record.json`.
- **S1 · Orient** — rotate RGB, depth, and confidence to portrait, and rotate the camera matrix
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

`tests/integration/` runs the flow under `prefect_test_harness`, which spins up a throwaway local
Prefect database. Nothing is written to a real Prefect server.

---

## Run the pipeline

```bash
uv run powerflow preprocess \
  --input   data/raw \
  --output  data/s1_orient_output \
  --records data/s0_ingest_output
```

| Flag | Meaning |
|---|---|
| `--input` | Raw capture root: `<date>/<session>/<camera>/`. |
| `--output` | Where S1 publishes the portrait streams. |
| `--records` | Where S0 writes `record.json`, `metadata.yaml`, and `manifest.json`. |
| `--rotation` | `cw` (default) or `ccw`. |
| `--dry-run` | Validate and plan, write nothing. |
| `--overwrite` | Replace cameras already published (default: refuse). |

Start with `--dry-run`: it validates every camera and reports what *would* be written.

A full run over the four cameras in `data/raw/9 July` takes about **7 minutes** — it decodes,
rotates, and re-encodes ~22k frames of 1920×1440 — and produces about 1.4 GB.

There is no `--in-place` mode. It would rewrite the raw capture, and raw data is write-once: each
stage publishes to its own output root instead.

### Where the output goes

| What | Where | Written by |
|---|---|---|
| **Run manifest** (JSON) | `<--records>/manifest.json` | end of the flow |
| **Run manifest** (markdown) | Prefect artifact, key `preprocess-run-manifest` | end of the flow |
| Per-camera validated record | `<--records>/<date>/<session>/<camera>/record.json` | S0 |
| Per-session metadata | `<date>/<session>/metadata.yaml` in **both** `<--records>` and `<--output>` | S0 |
| Portrait streams + provenance | `<--output>/<date>/<session>/<camera>/` incl. `sidecar.json` | S1 |

Every stage output root gets its own byte-identical copy of `metadata.yaml`, so a consumer of one
stage never has to reach back into an earlier stage's tree to learn which lift the pixels came from.
A new stage adds its root to `PreprocessConfig.stage_roots` and inherits the copy.

The **manifest is the audit record for one run**: every camera processed, the values derived
(rotation applied, `k_rewritten`, frame counts), any warnings, and **every rejected camera with its
exact reason**. Read it first when a run does not do what you expected.

Both copies are written **at the end of the flow**, after all cameras. A run still in progress has
`record.json` files on disk but no manifest yet — that is normal, not a failure. `--dry-run` writes
neither.

The markdown copy is the one to browse in the Prefect UI. The JSON copy is the one to grep:

```bash
jq '.rejected_scans' data/s0_ingest_output/manifest.json          # what was rejected, and why
jq '.scans[].derived' data/s0_ingest_output/manifest.json         # rotation, k_rewritten, n_frames
```

---

## Layout

```
src/powerflow_pipeline/data/
  cli.py                  # typer entrypoint: builds config, invokes the flow
  common/                 # shared @tasks and helpers: discovery, filesystem, manifest, errors
  preprocess/
    flow.py               # @flow, ordered orchestration only
    config.py             # PreprocessConfig
    models.py             # CameraRecord, Intrinsics, ...
    geometry.py           # pure rotation maths: no I/O, no Prefect
    tasks/                # one @task per logical step: discover, ingest, orient, metadata
prefect.yaml              # deployments
tests/unit/               # fast, no Prefect backend
tests/integration/        # flow + CLI, under prefect_test_harness
```

Data lives in `data/`, write-once: one task writes a file, many read it. Raw capture in `data/raw/`,
each stage's output in `data/<stage>_output/`.
