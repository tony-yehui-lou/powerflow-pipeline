# S7 Barbell Detection — a Prefect stage, and the stage registry it lands on

Status: **draft** | Scope: issue [#121](https://github.com/tony-yehui-lou/PowerFlow/issues/121) under epic [#59](https://github.com/tony-yehui-lou/PowerFlow/issues/59); consumed by [#63](https://github.com/tony-yehui-lou/PowerFlow/issues/63) | Updated: 2026-10-19

## Purpose

Two things:

- **Make the S7 cap/collar detector a pipeline stage.** It is described in
  [`S7-barbell-landmark-detection.md`](S7-barbell-landmark-detection.md) and today runs as
  `scripts/predict_landmarks.py` against one capture at a time. As a stage it runs inside the
  `preprocess` Prefect flow, for every capture, with its output published, audited in the run
  manifest, and reproducible from a named checkpoint.
- **Make adding the *next* stage, before or after it, a configuration change rather than a flow
  rewrite.**

The second goal shapes the first. S7 is therefore specified as the first stage built on a small
**stage registry**, not as an eighth hand-wired block in `flow.py`.

```text
S0 Ingest → S1 Cut → S2 Orient → S3 Retilt → S4 Crop ─┬→ S5 Pose → S6 Lift
                                                       └→ S7 Barbell ─→ (S8 Bar Pose, later)
          └──────────── core: unchanged ─────────────┘   └──── registry stages ────┘
```

---

## 1. What adding a stage costs today

Adding S5 and S6 touched these places:

| # | Edit | Where |
|---|---|---|
| 1 | A root field (`pose_root`, `lift_root`) | `preprocess/config.py` |
| 2 | Every tunable as a flat, prefixed field (`rtmpose_variant`, `lift_patch_radius`, …) | `preprocess/config.py` |
| 3 | A CLI option for the root, and one per tunable | `data/cli.py` |
| 4 | A `None` check to enable the stage, plus cross-checks (`lift_root` without `pose_root`) | `preprocess/flow.py` |
| 5 | Its own block merging `steps` / `derived` / `warnings` / `file_ops` | `preprocess/flow.py` |
| 6 | A one-off flag to reuse published output instead of rerunning (`skip_detect`) | config + CLI + flow |
| 7 | A deployment parameter. S5 and S6 never got one, so the `preprocess` deployment cannot run them | `prefect.yaml` |

Inserting a stage *between* two others also means re-reading every downstream block to check
what it reads in memory.

The rule this spec adopts already half-exists. S6 reads S5's **published document**, never its
in-memory result, and that is why the two can be rerun independently. This spec makes that rule
universal for registry stages.

## 2. The stage registry

### 2.1 Principle

**A registry stage reads only published trees and writes only its own.** It never receives
another stage's in-memory result. As a consequence:

- order is a list in the config;
- a stage can run against output an *earlier run* published, without rerunning anything upstream;
- inserting a stage changes nothing about its neighbours unless they are reconfigured to read it.

### 2.2 The contract

Every registry stage is one module under `src/powerflow_pipeline/data/preprocess/stages/`,
exporting one `StageSpec`:

```python
# preprocess/stages/base.py
StageId = str  # "s7_barbell"; also the default output dir name: <data_root>/<id>_output

@dataclass(frozen=True)
class StageSpec(Generic[S, R]):
    id: StageId
    title: str                                    # "S7 Barbell detection", for logs and the manifest
    settings: type[S]                             # the stage's own pydantic model (§3.3)
    reads: Callable[[S], tuple[StageId, ...]]     # trees it opens, given its settings (§4)
    setup: Callable[[S, RunContext], R] | None    # once per run: load a model, open a session
    run: Callable[[StageInput, S, R], StepResult] # a Prefect @task, once per capture
    copies_metadata: bool = True                  # write metadata.yaml beside each published capture
```

```python
# preprocess/stages/base.py
@dataclass(frozen=True)
class StageInput:
    """Everything a stage knows about one capture: identity, plus where every tree is."""
    capture: CaptureUnit                  # capture_id, relative, role, group_id, layout
    roots: Mapping[StageId, Path]         # resolved root of every known stage, core or registry
    output: Path                          # <this stage's root>/<capture.relative>
    dry_run: bool
    overwrite: bool

    def tree(self, stage: StageId) -> Path:
        """<root of `stage`>/<capture.relative>; KeyError names the stage if it isn't declared."""
```

`run` returns the existing `StepResult`, so the manifest code doesn't change.

### 2.3 The registry

```python
# preprocess/stages/__init__.py
REGISTRY: dict[StageId, StageSpec] = {
    spec.id: spec
    for spec in (
        s7_barbell.SPEC,
        # s5_pose.SPEC, s6_lift.SPEC  ← phase 2 (§8)
    )
}
```

**Adding a stage is three steps:** write `stages/<id>.py`, add one line here, and name it in a run
config. `flow.py`, `config.py`, `cli.py` and `prefect.yaml` are not edited.

The core stages S0–S4 are **not** registry stages, and they stay the existing hand-wired chain:
- Cut needs a group-level interval owner.
- S3's day-level plane check needs every capture fitted first.

They are still *known ids* (`s0_ingest` … `s4_crop`), so a registry stage can `read` their trees,
and they resolve roots exactly as today.

## 3. Configuration

### 3.1 One run file

A run is described by one YAML file. The CLI and `prefect.yaml` both accept it, and any value in it
can be overridden on the command line (§3.4).

```yaml
# runs/preprocess.yaml
data_root: ../data          # every default root is <data_root>/<stage id>_output
raw_root: ../data/raw
only: "10 September/*"      # the existing capture glob
dry_run: false
overwrite: false

core:
  run: true                 # false: don't execute S0–S4; read their published trees (§5.2)
  rotation: cw
  # … every existing S0–S4 tunable, unchanged in name and default

stages:                     # registry stages, run in this order, after the core
  - id: s7_barbell
    checkpoint: ../data/models/s7_barbell/frcnn-v2-2026-10-19.pt
```

`stages` is the whole switchboard:
- **A stage runs iff it is listed.** There is no `enabled` flag to forget.
- **Order is list order.**
- **Settings not given take the stage's defaults.**

### 3.2 Roots by convention

The pipeline already follows `../data/{task}_output`. The registry makes it the default:

| Stage | Default root |
|---|---|
| `s4_crop` | `<data_root>/s4_crop_output` |
| `s7_barbell` | `<data_root>/s7_barbell_output` |
| any new `sN_name` | `<data_root>/sN_name_output` |

A stage entry may set `root:` to override this. Core roots keep their existing fields (`crop_root`
etc.), which default by the same convention when unset. So a run file names **no paths at all**
beyond `data_root` and a checkpoint.

### 3.3 Per-stage settings are the stage's own model

Each stage owns a pydantic model. The registry validates the stage's entry against it, so a typo
is an error that names the stage and the key. Nothing is flattened into `PreprocessConfig`, and
there is no `rtmpose_`-style prefixing.

```python
class StageEntry(BaseModel):          # what `stages:` holds; extra keys go to the stage's model
    model_config = ConfigDict(extra="allow")
    id: StageId
    root: Path | None = None
```

### 3.4 CLI

A few generic flags replace per-stage flags:

```bash
powerflow preprocess --config runs/preprocess.yaml
powerflow preprocess --config runs/preprocess.yaml --set s7_barbell.score_threshold=0.5
powerflow preprocess --config runs/preprocess.yaml --no-core --only "10 September/45kg1"
powerflow preprocess --config runs/preprocess.yaml --stages s7_barbell
```

| Flag | Effect |
|---|---|
| `--config PATH` | the run file |
| `--set ID.KEY=VALUE` | override one stage setting; repeatable; the value is parsed as YAML |
| `--set core.KEY=VALUE` | override one core setting |
| `--stages ID[,ID…]` | replace the `stages:` list's ids, keeping their settings from the file |
| `--no-core` | `core.run = false` |
| `--only`, `--dry-run`, `--overwrite` | as today |

Today's flags (`--input`, `--crop`, `--pose`, `--lift`, `--skip-detect`, …) keep working through
phase 2 as aliases that write into the same config, so no existing recipe breaks.

`prefect.yaml` gets one parameter, the parsed run file, instead of a growing list of roots.

### 3.5 Worked examples

**Run S7 alone over captures that are already cropped:**

```yaml
core: {run: false}
stages:
  - id: s7_barbell
    checkpoint: ../data/models/s7_barbell/frcnn-v2-2026-10-19.pt
```

**Insert a hypothetical stage *before* S7**, e.g. `s4b_deblur`, which publishes sharpened RGB.
Add it to the list and point S7 at it. Nothing else changes:

```yaml
stages:
  - id: s4b_deblur
  - id: s7_barbell
    input: s4b_deblur         # S7 now reads s4b_deblur's rgb.mp4 instead of s4_crop's
    checkpoint: …
```

**Add a stage *after* S7**, e.g. `s8_bar_pose`: the Bar Height geometry moving out of powerflow-ui
into the pipeline (§9). It declares `reads = ("s7_barbell", "s4_crop", "s3_retilt")`, and is
listed after S7:

```yaml
stages:
  - id: s7_barbell
    checkpoint: …
  - id: s8_bar_pose
    min_score: 0.5
```

**Rerun only S8 against S7's published output:**

```bash
powerflow preprocess --config runs/preprocess.yaml --no-core --stages s8_bar_pose
```

## 4. Validation at startup

Before anything runs, the flow resolves the plan and refuses a bad one with a message that says
what to change. For each listed stage, in order:

1. **The id is known.**
   > `stages[1]: unknown stage 's7_barbel'; known: s7_barbell, …`
2. **Its settings validate** against the stage's model.
   > `s7_barbell.score_threshold: must be in [0, 1], got 3`
3. **Every id in `reads(settings)` is satisfiable:**
   - it is a core stage with `core.run: true`; or
   - it is a registry stage listed *earlier* in this run; or
   - its root exists on disk.

   Otherwise:
   > `s7_barbell reads s4b_deblur, which is neither listed before it nor published at ../data/s4b_deblur_output`
4. **No stage is listed twice, and no two stages share a root.**

`setup` is called only after the whole plan validates. A typo in stage 3 never costs the minutes
it takes to load stage 2's model.

## 5. Execution

### 5.1 Flow shape

```python
@flow(name="preprocess")
def preprocess(config: PreprocessConfig) -> RunManifest:
    plan = resolve_plan(config, REGISTRY)            # §4; raises before any work
    captures = discover(config)                      # as today
    if config.core.run:
        run_core(captures, config, manifest)         # today's S0–S4 loop, moved verbatim
    resources = {s.id: s.spec.setup(s.settings, ctx) for s in plan.stages if s.spec.setup}
    for capture in captures_surviving_core(captures, manifest, config):
        for stage in plan.stages:                    # list order
            result = stage.spec.run(stage.input_for(capture), stage.settings, resources.get(stage.id))
            manifest.record(capture, stage.id, result)   # one generic merge, replacing §1 row 5
    return emit_manifest(manifest, …)
```

`manifest.record` appends the stage id to the capture's `steps`, merges `derived` under a
`<stage id>.` key prefix, and concatenates `warnings` and `file_ops`. The prefix stops two stages
that both report `n_frames` from overwriting each other.

### 5.2 Without the core

When `core.run` is false, the captures come from discovery exactly as today. A capture whose input
tree is missing is skipped with a manifest warning, never a rejection: it may simply not have been
processed yet. The rest of the run is unchanged.

### 5.3 A failure is per capture, per stage

- **`ScanRejected` from a registry stage** rejects that capture *for the remaining stages* and is
  recorded as today.
- **Any other exception** fails the task, and Prefect's retries apply as each stage's `@task`
  declares.
- **Captures are independent:** one capture's failure never stops another's.

## 6. S7 Barbell detection

### 6.1 What it does

For one capture:
1. Read the input tree's `rgb.mp4`.
2. Run the detector on every frame.
3. Keep the highest-scoring **cap** and **collar** per frame — the rule already used, since the
   scene holds at most one of each that matters.
4. Publish one document.

It does no geometry. Pairing, scale and height belong to a later stage (§9).

### 6.2 Reads

`reads = (settings.input, "s1_cut")`:
- **`<input>/<relative>/rgb.mp4`**, default `s4_crop`. Its image space is the one the boxes are
  drawn in, and is recorded as `stage` in the output.
- **`s1_cut/<relative>/cut_sidecar.json`**, for the capture's epoch anchor. This is exactly how S5
  obtains it (`tasks/pose.py: _capture_start_epoch_ms`); the two helpers move to `common/` and both
  stages use them.

### 6.3 The model boundary

S5's pattern is reused: the stage calls a protocol, and the torchvision model is one implementation.

```python
# preprocess/barbell_model.py
@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in input-tree pixels
    score: float

@runtime_checkable
class BarbellDetector(Protocol):
    def detect(self, rgb_path: Path, n_frames: int) -> dict[str, list[Detection | None]]:
        """{"cap": [...], "collar": [...]}, each n_frames long; None where nothing passed."""
```

This keeps every test free of torch. A stub detector exercises the stage's plumbing, publishing
and manifest.

`TorchvisionBarbellDetector` is the implementation:

- **Model registry moves into `src/`.** `MODELS` and `build_model` move from
  `scripts/train_landmark_detector.py` to `preprocess/barbell_model.py`. The training script
  imports them from there, so training and inference can never build different models.
- **Explicit checkpoint, verified.** It loads exactly the checkpoint the settings name and records
  its sha256. There is no "latest under `runs/`" lookup, which is what the script does today, and
  is not reproducible.
- **Native resolution.** It builds the model at the input's own size (`min_size`/`max_size` =
  frame short/long side), as the script does. The detector was trained that way, and a resize
  shifts boxes.
- **Streaming decode.** Frames are decoded with PyAV and inferred in batches of `batch_size`. The
  script loads every frame first, about 3 GB for one 433-frame capture.
- **Device.** `device: auto` picks `mps`, then `cuda`, then `cpu`. MPS needs
  `PYTORCH_ENABLE_MPS_FALLBACK=1` for one torchvision op; `setup` sets it before importing torch
  and records that it did.
- **Loaded once per run**, in `setup`, and shared across captures.

### 6.4 Output

Published at `<s7_barbell root>/<relative>/`:

| File | When | What |
|---|---|---|
| `barbell.json` | always | the document below |
| `metadata.yaml` | always | copied, as every stage does (`copies_metadata`) |
| `overlay.mp4` | `render_overlay: true` | boxes drawn on every frame, for review; off by default (it re-encodes the video) |

`barbell.json` is camelCase on the wire, like `PoseDocument`, and is a pydantic model in
`common/models.py`:

```json
{
  "schemaVersion": "1.0.0",
  "captureId": "10 September/50kg3",
  "role": "single",
  "stage": "s4_crop",
  "pixelSpace": {"width": 1416, "height": 1744},
  "frames": {"count": 433, "fps": 59.98, "startEpochMs": 1789030457100, "tMs": [0, 17, 33, …]},
  "classes": ["cap", "collar"],
  "landmarks": {
    "cap":    [{"box": [42.9, 1583.5, 68.7, 1623.0], "score": 0.995}, …, null, …],
    "collar": [{"box": [763.0, 1256.3, 788.1, 1287.6], "score": 0.928}, …]
  },
  "detector": {
    "architecture": "fasterrcnn_resnet50_fpn_v2",
    "checkpoint": "../data/models/s7_barbell/frcnn-v2-2026-10-19.pt",
    "checkpointSha256": "…",
    "trainingEpoch": 6,
    "scoreThreshold": 0.3,
    "rule": "highest-scoring detection per class",
    "device": "mps",
    "versions": {"torch": "…", "torchvision": "…"}
  }
}
```

Differences from today's `predictions.json`, each deliberate:

- **Per-frame arrays, `frames.count` long, with `null` for no detection.** Not a dict keyed by
  stringified frame index.
- **`frames.tMs` comes from the video's own PTS, and `startEpochMs` from S1.** Not from a nominal
  fps, which [`body-model-viewer`](https://github.com/tony-yehui-lou/PowerFlow/issues/120) showed
  can be wrong.
- **The image space (`stage`, `pixelSpace`) is explicit.** A box means nothing without it, and an
  inserted stage (§3.5) can change it.
- **Provenance names the exact checkpoint by hash.**

The low recording threshold (0.3) is intentional: the stage records candidates, and consumers gate.
Bar Height drops below 0.5. A higher value would destroy evidence that a consumer cannot recover.

### 6.5 Settings

| Key | Default | Meaning |
|---|---|---|
| `checkpoint` | **required** | path to the `.pt`; the stage refuses to plan without it |
| `input` | `s4_crop` | stage whose `rgb.mp4` is read |
| `architecture` | from the checkpoint | override only for a checkpoint that doesn't record one |
| `device` | `auto` | `auto`, `mps`, `cuda`, `cpu` |
| `score_threshold` | `0.3` | detections below are not recorded |
| `frame_stride` | `1` | run on every Nth frame; skipped frames are `null` (for quick looks only) |
| `batch_size` | `4` | frames per forward pass |
| `render_overlay` | `false` | also publish `overlay.mp4` |
| `warn_cap_coverage` | `0.5` | warn when fewer than this fraction of frames have a cap |
| `warn_collar_coverage` | `0.9` | likewise for the collar |
| `root` | `<data_root>/s7_barbell_output` | override the output root |

### 6.6 Publish, overwrite, dry run

These follow the existing rules in `common/filesystem.py`:
- **Publish:** write into a staging directory, validate the document against its model, then
  `publish_staging`.
- **Existing output:** with `overwrite`, it is replaced. Without it, the capture is **skipped with
  a manifest note**, not an error, so a rerun over a day only fills in what is missing.
- **Dry run:** plan the `file_ops` and write nothing. `setup` is not called, so a dry run never
  loads the model.

### 6.7 Manifest

All of these are under the `s7_barbell.` prefix:
- **`derived`:** `frames`, `frames_with_cap`, `frames_with_collar`, `frames_with_both`, median cap
  and collar score, seconds per frame, `checkpoint_sha256`, `device`.
- **`warnings`:** coverage below the `warn_*` thresholds, and frames where every detection fell
  under the threshold.
- **Never a rejection for low coverage.** 45kg1 shows a cap on only a quarter of its frames, and is
  still valid S7 output.

### 6.8 Retries and cost

- **Retries:** the `@task` declares `retries=0`. Inference is deterministic, so a retry only
  repeats a failure.
- **Cost:** about 0.43 s/frame on MPS, about 3 minutes per 433-frame capture. The manifest's
  seconds per frame keeps that measured.

## 7. Where the checkpoint lives

Today the only checkpoint sits in a training run directory
(`data/annotations/barbell/v3/runs/<run>/best.pt`), next to the labels it was trained from. A
pipeline stage should name a **model**, not a training run. Proposed:

```text
../data/models/s7_barbell/
  frcnn-v2-2026-10-19.pt        # copied once from the chosen run; never written again
  frcnn-v2-2026-10-19.json      # architecture, classes, training run, split, val metrics (§5 of the landmark spec)
```

Promotion is a deliberate copy by hand or by a script, matching the data rule "write once by one
task". The stage then reads `.json` beside the `.pt` for `architecture` and records both in
provenance.

## 8. Phase 2 — S5 and S6 onto the registry

This is not required for S7, but it is the reason the registry is worth having.

- **S5 Pose** becomes `stages/s5_pose.py`.
  - Its settings model absorbs `pose_detector`, `rtmpose_variant`, `rtmpose_det_frequency`,
    `rtmpose_clavicle_shoulder_weight` and `rtmpose_subject_depth`, without the prefixes.
  - `setup` builds the detector; that code moves from `cli.py: _build_detector`.
  - `ensure_skeleton_published` becomes part of `setup`.
- **S6 Lift** becomes `stages/s6_lift.py`, with `reads = ("s5_pose", "s4_crop")`.
- **`skip_detect` disappears.** Lifting an earlier run's detections is `stages: [s6_lift]`, with
  `s5_pose` satisfied from disk by §4 rule 3.
- **Old flags stay as aliases** (`--pose`, `--lift`, `--skip-detect`), mapped onto `stages:`, for
  one release.

## 9. Downstream

- **`scripts/predict_landmarks.py`** becomes a thin wrapper that runs the stage on one capture with
  `render_overlay: true`, or is retired. Its output location moves from `annotations/barbell/v3/`
  (a labelled corpus, which inference should not write into) to `s7_barbell_output/`.
- **powerflow-ui's `build_bar_height.py`** reads `s7_barbell_output/<relative>/barbell.json`
  instead of `predictions.json`.
  - The change is small: per-frame arrays instead of a dict, and `frames.tMs` instead of `i/fps`.
  - It also gets real timing for free.
  - That is a UI-repo change, made when this lands.
- **S8 Bar Pose (future).** The level-bar geometry now in `build_bar_height.py` becomes a registry
  stage with `reads = ("s7_barbell", "s4_crop", "s3_retilt")`, listed after S7. This spec is the
  registry it would be added to, and §3.5 is its config.

## 10. Testing

TDD, as the repo requires; the coverage gate stays at 90%. None of these tests import torch.

| Test | Asserts |
|---|---|
| plan resolution | unknown id, bad setting, unsatisfiable `reads`, duplicate id, shared root: each raises naming the stage and key |
| ordering | stages run in list order; a stage listed before its dependency is rejected unless the dependency is on disk |
| config layering | file < `--set` < legacy alias; `--stages` keeps file settings; roots default to `<data_root>/<id>_output` |
| `--no-core` | S0–S4 tasks are not called; captures with a missing input tree are skipped with a warning |
| S7 stage, stub detector | document shape, arrays `frames.count` long, `null` placement, `tMs` from PTS, `stage`/`pixelSpace` from the input tree |
| S7 publish | staging then publish; existing output skipped without `overwrite`, replaced with it; dry run writes nothing and never calls `setup` |
| S7 manifest | derived keys prefixed; coverage warnings at the thresholds; low coverage never rejects |
| top-1 rule | given several detections per class, the highest-scoring one above threshold is kept |
| `barbell_model.build_model` | every registered architecture builds with the replaced head (moved from the script's existing check) |
| one real-model smoke test | marked `slow`, skipped by default; one short clip through the real checkpoint |

## 11. Implementation order

1. **`StageSpec`, `StageInput`, `REGISTRY`, plan resolution, `StageEntry` parsing.** Tests first.
   No stages registered yet.
2. **Move `run_core` out of `preprocess()` verbatim, and add `core.run`.** Today's tests must pass
   unchanged.
3. **Add the run file and `--config` / `--set` / `--stages` / `--no-core`,** with the legacy flags
   as aliases.
4. **Move `MODELS` / `build_model` to `preprocess/barbell_model.py`;** the training script imports
   them. Add `BarbellDetector`.
5. **`stages/s7_barbell.py`** against a stub detector, then `TorchvisionBarbellDetector`.
6. **Move the epoch and PTS helpers from `tasks/pose.py` to `common/`.**
7. **Run on `10 September`; compare with today's `predictions.json`** on 50kg3. Boxes should match
   to rounding, given the same checkpoint.
8. Phase 2 (§8), separately.

## 12. Decisions needed

1. **Add `torchvision` as a runtime dependency.** `torch` already is; `predict_landmarks.py` gets
   torchvision today through `uv run --with torchvision --with pillow`. Pillow is not needed by
   the stage, since frames come from PyAV. The repo rule is to ask before adding a dependency.
2. **The checkpoint home.** `../data/models/s7_barbell/` as in §7, or keep pointing into
   `annotations/.../runs/`.
3. **The run file's home.** `runs/*.yaml` in the pipeline repo (versioned, reviewable recipes) is
   proposed.
4. **Whether phase 2 follows immediately.** Until it does, the flow has two ways to add a stage.

## 13. Out of scope

- **Retraining, or changing the detector's classes.**
- **Bar geometry** (pairing, scale, height): that is S8, specified separately.
- **Moving S0–S4 onto the registry.** Their group-level and day-level coupling needs its own design.
- **The preprocess metadata task dropping `barbell_type`** (landmark spec §7). S7 doesn't need it,
  but S8 will.
- **Concurrency across captures.** One model per run, captures sequential, as S5 runs today.
