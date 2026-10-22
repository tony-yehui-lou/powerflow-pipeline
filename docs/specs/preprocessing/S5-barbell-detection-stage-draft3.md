# S5 Barbell Detection — a Prefect stage (draft 3)

Status: **draft 3 — decisions approved, ready to implement** | Supersedes: [draft 2](S5-barbell-detection-stage-draft2.md), [draft 1](S5-barbell-detection-stage.md) | Scope: issue #121 under epic #59 | Updated: 2026-10-22

## Purpose

Run the S5 cap/collar detector inside the `preprocess` Prefect flow, for every capture, between
S4 Crop and S6 Pose. Its output is published under `../data/s5_barbell_output/`, audited in the
run manifest, and reproducible from a named checkpoint. S6's wrist correction and S8 Kinematics
then read that output instead of a hand-made `predictions.json`.

```text
S0 Ingest → S1 Cut → S2 Orient → S3 Retilt → S4 Crop → S5 Barbell → S6 Pose → S7 Lift → S8 Kinematics
                                                       └── new ──┘   reads S5   (n/a)    reads S5
```

Draft 1 bundled S5 with a stage-registry refactor. This draft separates them: S5 lands first, in
the same hand-wired style as S6–S8. The registry becomes its own spec (§11).

---

## 0. Why S5 isn't in the flow today

S5 exists only outside the flow:

- **Training:** `scripts/train_landmark_detector.py`.
- **Inference:** `scripts/predict_landmarks.py`, run by hand on one capture at a time.
  - It needs `uv run --with torchvision --with pillow`.
  - It writes `predictions.json` and `predictions.mp4` into the labelled corpus at
    `data/annotations/barbell/v3/<capture>/`.

The flow only ever *reads* those files, through `--barbell` (`barbell_input.py`). Six things kept
it that way:

1. **The detector came after the pipeline.**
   - S5 started as a labelling and training experiment (annotation guide, SAM2 label tools,
     model recommendation).
   - Inference was a script for watching boxes on a video, not a stage.
   - The S5/S6/S7 renumbering on 2026-10-20 put barbell detection before pose *in name*, but
     `flow.py` was never changed. Its docstring still lists "S4 crop, S6 pose".
2. **Draft 1 made S5 wait on a refactor.**
   - It specified S5 as the first stage of a new stage registry, with run files, `--set`,
     `--no-core`, and moving S0–S4 into `run_core`.
   - That is a large change with no stage that needs it urgently, so nothing landed, S5
     included.
3. **Draft 1's ordering assumption broke.**
   - Draft 1 said "S5 and S6 do not read each other", and put registry stages *after* the
     hand-wired loop.
   - Since then, S6 corrects wrists against the bar (`wrist_refine.py`, `S6-wrist-refinement.md`)
     and S8 anchors the hands to it. Both run inside the hand-wired loop.
   - So a registry-run S5 would run after the stages that consume it. A fresh run would correct
     wrists against stale or missing detections.
4. **`torchvision` is not a runtime dependency.**
   - `torch` is, but the detector's model builder (`fasterrcnn_resnet50_fpn_v2`) is in
     torchvision.
   - The repo rule is to ask before adding a dependency. Nobody asked.
5. **There is no pipeline-owned checkpoint.**
   - The only weights are a training run's `best.pt` under
     `data/annotations/barbell/v3/runs/fasterrcnn_resnet50_fpn_v2-20261019-003709/`.
   - The script picks "latest by mtime", which a reproducible stage can't do.
6. **The deployment can't build a model.**
   - `preprocess()` takes its S6 detector as a Python object that `cli.py` builds.
   - A Prefect deployment can only pass serialisable parameters, so `prefect.yaml` runs S0–S4
     and nothing after. S6, S7 and S8 are CLI-only for the same reason.
   - S5, done the same way, would be CLI-only too.

The result today: 33 captures are cropped, and 7 have detections (all `10 September`). Those 7
are the only captures S6 and S8 can use the bar on.

## 1. Approach

S5 is a hand-wired task, `tasks/barbell.py`, called in the per-capture loop right after
`crop_camera` and before `detect_pose_camera`. It follows the patterns S6–S8 already use:

- an optional root;
- a detector behind a protocol;
- publish-then-read: later stages read S5's published `barbell.json`, never its in-memory
  result.

Ordering then follows from position in the loop, with no registry, and a single run can go from
raw to S8 with the bar in place.

The flat config naming (`barbell_*`) is the debt draft 1 §1 describes. It is accepted here
knowingly; the registry spec pays it off for S5–S8 together.

## 2. Configuration

New and changed `PreprocessConfig` fields:

| Field | CLI | Default | Meaning |
|---|---|---|---|
| `barbell_root` | `--barbell` | `None` | **Changed.** S5's tree: written when S5 runs, read by S6/S8 either way. It holds `barbell.json`, no longer `predictions.json`. |
| `barbell_checkpoint` | `--barbell-checkpoint` | `None` | **New.** Set ⇒ S5 runs. Unset ⇒ S5 doesn't run, and `--barbell` is read as published (today's behaviour). |
| `barbell_device` | `--barbell-device` | `auto` | `auto` (mps → cuda → cpu), `mps`, `cuda`, `cpu` |
| `barbell_record_threshold` | — | `0.3` | Detections below this are not *recorded*. |
| `barbell_min_score` | — | `0.5` | Unchanged. Consumers (S6, S8) ignore detections below this. |
| `barbell_batch_size` | — | `4` | Frames per forward pass. |
| `barbell_render_overlay` | `--barbell-overlay` | `false` | Also publish `overlay.mp4`. |
| `barbell_warn_cap_coverage` | — | `0.5` | Warn below this fraction of frames with a cap. |
| `barbell_warn_collar_coverage` | — | `0.9` | Likewise for the collar. |

Two thresholds, on purpose:

- **S5 records at 0.3.** It keeps candidates, because a consumer can't recover evidence that was
  never written.
- **Consumers gate at 0.5,** as today: real detections score ≥ 0.95, and false caps on a
  bystander scored 0.31–0.33.

**Validation at flow start.** Each of these raises before any work:

- `barbell_checkpoint` without `barbell_root`;
- a checkpoint path that doesn't exist;
- a checkpoint without its `.json` sidecar (§6).

Whether S5 runs is decided by `barbell_checkpoint`, not by a `skip_*` flag:

- reusing published detections means passing `--barbell` alone;
- running S5 means adding `--barbell-checkpoint`.

## 3. The model boundary

This is S6's pattern again: the task calls a protocol, so every test except one smoke test runs
without torch.

```python
# preprocess/barbell_model.py
@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in the input video's pixels
    score: float

@runtime_checkable
class BarbellDetector(Protocol):
    provenance: Mapping[str, Any]            # architecture, checkpoint, sha256, device, versions
    def detect(self, rgb_path: Path, n_frames: int) -> dict[str, list[Detection | None]]:
        """{"cap": [...], "collar": [...]}, each n_frames long; top-1 per class, None below threshold."""
```

`TorchvisionBarbellDetector` carries the following from draft 1 §6.3 unchanged:

- **One model builder.** `MODELS` and `build_model` move from the training script into
  `preprocess/barbell_model.py`, and the script imports them.
- **The named checkpoint only,** with its sha256 recorded. There is no "latest" lookup.
- **Native-resolution model build,** because a resize shifts boxes.
- **PyAV streaming decode in batches,** not about 3 GB of frames held in memory.
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`,** set before torch is imported.
- **Loaded once per run.**

### 3.1 Built by the flow, so the deployment can run it

`preprocess()` gains `barbell_detector: BarbellDetector | None = None`:

- When it is `None` and `barbell_checkpoint` is set, the flow builds one with
  `build_barbell_detector(config)`. This happens after validation, and never in a dry run.
- Tests inject a stub.
- `cli.py` passes nothing and lets the flow build it.

This is the change that lets `prefect.yaml` run S5 (§8). S6's detector stays CLI-built for now.
Moving it the same way is a follow-up (§11).

## 4. The task

`tasks/barbell.py: detect_barbell_camera(record, config, detector) -> (record, StepResult)`, a
Prefect `@task` with `retries=0`. Inference is deterministic, so a retry only repeats a failure.

For one camera:

1. **Inputs:** `record.source / "rgb.mp4"` (S4's output), and the epoch anchor from S1's
   `cut_sidecar.json`.
   - S6 reads the anchor the same way. `_capture_start_epoch_ms` and `_frame_offsets_ms` move
     from `tasks/pose.py` to `common/`, and both tasks use them.
2. **Detect:** `detector.detect(...)`.
3. **Build `BarbellDocument`** (§5) and validate it.
4. **Stage and publish:** write `barbell.json`, `metadata.yaml`, and optionally `overlay.mp4`
   into staging, then call `publish_staging`.
5. **Return** `derived` and `warnings` (§7).

**Existing output:**

- **Without `overwrite`:** the capture is not re-detected. A manifest note says so, and S6/S8
  read what is there. A rerun over a day only fills gaps, which at about 3 min per capture
  matters.
- **With `overwrite`:** it is replaced.

**Dry run:** plan the `file_ops`, write nothing, and never build the detector.

**Low coverage never rejects.** 45kg1 has a cap on about a quarter of its frames and is valid
output. A decode failure raises (the task fails); it is not turned into a `ScanRejected`.

## 5. Output: `barbell.json`

The schema is draft 1 §6.4 unchanged, as a pydantic `BarbellDocument` in `common/models.py`. It
is camelCase on the wire, like `PoseDocument`:

- **`frames.count`-long arrays per class,** with `null` where nothing was detected.
- **`frames.tMs` from the video's PTS,** and `startEpochMs` from S1.
- **`stage: "s4_crop"` and `pixelSpace`,** both explicit.
- **`detector`** carries the checkpoint path and sha256, the architecture, the training epoch,
  `recordThreshold`, the top-1 rule, the device, and library versions.

Layout:

```text
../data/s5_barbell_output/<capture path>/
  barbell.json
  metadata.yaml        # written by the task; S5 is optional, so it is not in `stage_roots`
  overlay.mp4          # only with barbell_render_overlay
```

## 6. Checkpoint home

```text
../data/models/s5_barbell/
  frcnn-v2-2026-10-19.pt     # copied once from the chosen training run; never rewritten
  frcnn-v2-2026-10-19.json   # architecture, classes, training run id, split, val metrics
```

Promoting a model is a deliberate copy, matching the data rule "write once by one task". The
first promotion is
`annotations/barbell/v3/runs/fasterrcnn_resnet50_fpn_v2-20261019-003709/best.pt`. A small
`scripts/promote_barbell_model.py` writes both files; the `.json` is required, not optional.

## 7. Consumers and the manifest

### 7.1 `barbell_input.py`

This module was written to be the single swap point.

- **`DETECTIONS_FILENAME`** becomes `barbell.json`.
- **`read_detections`** parses a `BarbellDocument`:
  - It refuses a `pixelSpace` that differs from the capture's frame size. This replaces today's
    `frame_size` check, which is `[h, w]` in `predictions.json`.
  - It applies `min_score` and returns box centres exactly as now.
  - So `wrist_refine` and `bar_geometry` don't change.
- **`predictions.json` is no longer read.** The 7 captures that have one are re-detected by S5
  (§10 step 7) instead of converted.
- **S6's `wrist_refinement.json`** records `barbell.json`'s path and the checkpoint sha256 in
  place of the `predictions.json` path.

### 7.2 Manifest

S5's `derived` keys are prefixed `barbell_`. The flow merges every stage's `derived` into one
flat dict, so an unprefixed key like `n_frames` would overwrite S6's. The keys:

- `barbell_frames_with_cap`, `barbell_frames_with_collar`, `barbell_frames_with_both`;
- `barbell_median_cap_score`, `barbell_median_collar_score`;
- `barbell_s_per_frame`;
- `barbell_checkpoint_sha256`, `barbell_device`;
- `barbell_reused`, true when published output was kept.

`steps` gains `"barbell"`. **Warnings** cover coverage below the `warn_*` thresholds.

### 7.3 Flow wiring

The new block in `flow.py`, after `crop_camera`:

```python
if barbell_detector is not None:
    _, barbell_step = detect_barbell_camera(crop_record, config, barbell_detector)
    steps.append("barbell")
    # merge derived / warnings / file_ops, as for pose, lift and kinematics
```

The module docstring and the `preprocess()` docstring are updated to list S5.

## 8. Prefect deployment

`prefect.yaml`'s `preprocess` deployment gains:

```yaml
        barbell_root: ../data/s5_barbell_output
        barbell_checkpoint: ../data/models/s5_barbell/frcnn-v2-2026-10-19.pt
```

and its description mentions S5. Because the flow builds the detector itself (§3.1), this is the
first model stage the deployment can actually run.

## 9. Dependency

Add `torchvision` to `[project].dependencies`, pinned to the release paired with the locked
`torch`. Pillow is not needed: frames come from PyAV, and the overlay is drawn with OpenCV.

## 10. Implementation order (TDD throughout; coverage stays ≥ 90%)

1. **Dependency and checkpoint.**
   - Add `torchvision` (decision 1).
   - Add `scripts/promote_barbell_model.py`, and promote the current run.
2. **`barbell_model.py`.**
   - Move `MODELS`/`build_model` in, and make the training script import them.
   - Add the `Detection` and `BarbellDetector` protocol, then `TorchvisionBarbellDetector`.
3. **`BarbellDocument`** in `common/models.py`, with round-trip tests.
4. **Move the epoch and PTS helpers** from `tasks/pose.py` to `common/`, keeping S6's tests green.
5. **`tasks/barbell.py`** against a stub detector:
   - shape and `null` placement;
   - publish, reuse and overwrite;
   - dry run;
   - prefixed `derived` keys;
   - coverage warnings.
6. **Config and CLI:**
   - the new fields;
   - startup validation;
   - `build_barbell_detector`;
   - the flow block;
   - the integration test (stub detector) that S6 reads S5's output from the same run.
7. **`barbell_input.py` → `barbell.json`.**
   - Rerun S5 on `10 September`.
   - Compare boxes with the existing `predictions.json` on all 7 captures. They should match to
     rounding with the same checkpoint.
   - Then rerun S6/S8 there and diff `wrist_refinement.json`.
8. **`prefect.yaml`, then docs.**
   - Update `prefect.yaml`.
   - README Recipe C: `--barbell ../data/s5_barbell_output`, plus the new S5 recipe.
   - `S6-wrist-refinement.md` §config: path and filename.
9. **Retire the script.** `scripts/predict_landmarks.py` becomes a thin wrapper over the task with
   the overlay on, and stops writing into `annotations/`.
10. **One `slow`-marked smoke test** through the real checkpoint on a short clip, skipped by
    default.

**Then, separately:** run S5 over the 26 cropped captures that have no detections (about 1.5 h on
MPS at 0.43 s/frame).

## 11. Out of scope (follow-ups)

- **The stage registry** (draft 1 §2–§5, §8): run files, `--set`, `--no-core`, and moving S5–S8
  onto it together.
  - Its own spec, written with this ordering lesson: registry stages that *core* stages consume
    can't run after the core loop.
- **Building S6's detector inside the flow,** so the deployment can run S6–S8. Same mechanism as
  §3.1.
- **powerflow-ui `build_bar_height.py`** reading `s5_barbell_output/.../barbell.json` instead of
  `predictions.json`. A UI-repo change, made after step 7.
- **S8 Bar Pose / moving bar geometry out of the UI.**
- **Retraining, or new classes.**
- **Concurrency across captures.**

## 12. Decisions (approved 2026-10-22)

1. **`torchvision` becomes a runtime dependency,** pinned to the release paired with the locked
   `torch` (§9).
2. **Checkpoints live in `../data/models/s5_barbell/`,** each `.pt` with its required `.json`
   sidecar (§6).
3. **`predictions.json` support is dropped.** No converter. The 7 captures that have one are
   re-detected by S5 (§10 step 7), which doubles as the parity check against the script.
4. **Flat `barbell_*` config fields are accepted for now.** The stage registry is a follow-up
   spec (§11).

## Changes from draft 1

| Draft 1 | Draft 2 | Why |
|---|---|---|
| S5 is the first registry stage, run after the S0–S4 loop | S5 is hand-wired between S4 and S6 | S6 and S8 now consume S5; a post-loop S5 would run after its consumers |
| Registry, run files, `--set`, `--no-core` in scope | Deferred to their own spec | They aren't needed to ship S5, and they blocked it |
| `scoreThreshold` 0.3 in provenance | `barbell_record_threshold` (0.3) vs `barbell_min_score` (0.5) named apart | Draft 1 left the consumer gate implicit; it already exists in config |
| Detector built in `setup` (registry) | Built by the flow from config when not injected | Lets the Prefect deployment run it, which draft 1 didn't address for any model stage |
| `derived` prefixed by the registry's `manifest.record` | `barbell_` key prefix by hand | No registry yet; the collision with `n_frames` is real |
| `frame_stride` setting | Dropped | Quick looks are the script's job; every published document is full-rate |
| Consumers named: `predict_landmarks.py`, UI | Adds `barbell_input.py`, S6's sidecar, S8, README Recipe C | Those consumers appeared after draft 1 |

## Changes from draft 2

- **§12:** all four decisions approved as proposed; it now records them as settled.
- **Status line:** ready to implement.
- **No design changes.**
