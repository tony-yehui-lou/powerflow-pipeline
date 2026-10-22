# Pipeline phase split: preprocess, detect, analyze

Date: 2026-10-22
Status: **draft, not implemented**
Closes: the "home" open question in `S8-joint-hva-draft3.md`, `S8-joint-angles-draft3.md` and
`S9-phase-intervals-draft7.md` (Open question 8). Their next drafts take the answer from here.

## Context

One Prefect flow, `preprocess` (`data/preprocess/flow.py`), runs all ten stages: S0 Ingest,
S1 Cut, S2 Orient, S3 Retilt, S4 Crop, S5 Barbell, S6 Pose, S7 Lift, S8 Kinematics and S9 Phase.
Both S8 and S9 asked whether they belong there. They don't, and neither do S5–S7. The stages
are three kinds of work:

- **S0–S4 are fixed video geometry.** There are no learned models, and they are mostly
  settled.
- **S5–S7 are detection.** They run learned models (the S5 cap/collar detector, RTMPose,
  MediaPipe), are slow on CPU, and are tied to a named checkpoint. They change when a model
  changes.
- **S8–S9 are quantitative analysis.** These are maths on published detections and the
  lifter's measurements. They read no video and run no vision model. S9's v2 may load a small
  trained sequence model over per-frame features. They are cheap and change with every rule
  revision (S9 is on draft 8).

Today, re-running S9 after a threshold change means invoking the same flow that owns ingest and
RTMPose. That flow also carries cross-stage guards that only make sense inside one flow
("`phase_root` set but `kinematics_root` unset").

## Decision

Split `preprocess` into three pipelines, one per phase. Each follows the pipeline layout in
`CLAUDE.md`, with its own `flow.py`, `config.py` and `tasks/`:

| Pipeline | Stages | Reads | Publishes |
|---|---|---|---|
| `preprocess` | S0 Ingest, S1 Cut, S2 Orient, S3 Retilt, S4 Crop | `data/raw` | `s0`–`s4_*_output` |
| `detect` | S5 Barbell, S6 Pose (wrist refinement included), S7 Lift (optional, **off by default**) | S0 record, S4 video, depth and sidecar | `s5_barbell_output`, `s6_pose_output`, `s7_lift_output` |
| `analyze` | S8 Kinematics (joint HVA, joint angles), S9 Phase | S0 record and `metadata.yaml`, S3/S4 sidecars, S5, S6 | `s8_kinematics_output`, `s9_phase_output` |

The pipelines talk to each other only through published output in `../data`, which is
already the rule (write once, read many). No pipeline imports another pipeline's tasks.

### Stage numbers stay global

S-numbers are not restarted per pipeline. Spec filenames, output folder names (`s5_barbell_output`
…), GitHub issues and the UI all use them, and the 2026-10-20 renumbering already cost one
cleanup. The phase is a property of the stage, not a new numbering.

### S7 Lift: parked in `detect`, off by default

S7 back-projects S6's pixel joints through S4's LiDAR depth. Nothing downstream reads it: S8
solves 3D joints from bone lengths, the bar and the floor, and reads no depth. There is no
`s7_lift_output` in `data/` today. It is kept as an optional `detect` stage because it is still
the only depth-based 3D joint source, and it may serve as a cross-check for S8.

- It runs only when `--lift <root>` is given. The `detect` default skips it.
- It reads S6's published document, so it can run on an earlier detection without re-running
  RTMPose (today's `skip_detect` behaviour, kept).
- Its classification is "detect" by placement, not by nature: it is a measurement on a
  detection. If it is ever revived as an S8 input, it moves to `analyze` (Open question 2).

## 1. Pipeline boundaries

### 1.1 `preprocess` (S0–S4)

Unchanged in behaviour. It keeps the per-group loop: S1 Cut needs the group's interval owner,
and S3 runs its day-level plane-consistency check after every capture of a day is fitted. Its
output contract to the next phase is, per camera:

- S0's `record.json` and `metadata.yaml` (intrinsics, the `athlete:` and `lift:` blocks);
- S3's floor offset sidecar;
- S4's cropped video, `depth/`, `confidence/` and `crop_sidecar.json` (`px_per_mm`,
  `crop_bounds`, provenance).

A camera rejected anywhere in S0–S4 publishes no S4 output, so it is never discovered
downstream.

### 1.2 `detect` (S5–S7)

Per camera, independent. There is no group loop: once S4 has published, nothing in S5–S7
is cross-camera.

- **S5 Barbell** runs by default and needs `--barbell-checkpoint` (a promoted model with its
  card). The checkpoint is checked before any work, and the detector is never built in a dry
  run, as now.
- **S6 Pose** runs by default. When S5's output is available, wrist refinement corrects both
  wrists against the bar. Without it, wrists are left as detected (as now).
- **S7 Lift** runs only with `--lift`.
- `--only s5,s6` style selection lets you re-run one model without the other.

### 1.3 `analyze` (S8–S9)

Per camera, independent. No video is read and no vision model is loaded.

- **S8 Kinematics** reads S6's pose, S5's bar, S3's floor and the `athlete:` block.
- **S9 Phase** reads S8's `joint_hva.json`, S5's bar, S6's grip (wrist refinement) and
  `lift.type`.
- S9 can run alone against an existing S8 tree (`--only s9`). This is the main reason for the
  split: re-running S9 after a rule change takes seconds and loads no detector.

### 1.4 What replaces the in-flow guards

Today's flow refuses inconsistent root combinations up front. After the split, the same
intent becomes an **input check**: a stage whose upstream output is missing for a camera
rejects that camera with a reason in the manifest ("no S6 pose published for <capture>"). A
missing upstream *root* (the whole tree) is still refused before any work.

v1 checks existence only. Because stages now run at different times, a stale S8 tree can sit
under a fresh S6 run. Detecting that means recording upstream provenance, which is a schema
change and out of scope here (Open question 3).

## 2. Shared code

| Module | Today | Moves to | Why |
|---|---|---|---|
| `preprocess/models.py` (`CameraRecord`, `CaptureUnit`, …) | preprocess | `common/` | imported by tasks in all three phases |
| `preprocess/bar_geometry.py` | preprocess | `common/` | used by S6 wrist refinement (detect) and S8, S9 (analyze) |
| `preprocess/athlete_input.py` | preprocess | `analyze/` | only S8 reads `athlete:` |
| `geometry.py`, `timeline.py`, `retilt.py`, `crop.py` | preprocess | stay | S0–S4 only |
| `pose_geometry.py` | preprocess | `detect/` | used by RTMPose and S7 |

### 2.1 `detect/`

`barbell_model`, `barbell_input`, `pose_model`, `rtmpose_assets`, `rtmpose_detector`,
`mediapipe_assets`, `mediapipe_pose_detector`, `halpe26_mapping`, `pose_landmark_mapping`,
`pose_geometry`, `wrist_refine`, `pose_lift`; tasks `barbell`, `pose`, `lift`.

### 2.2 `analyze/`

`athlete_input`, `body_plane`, `skeleton_solve`, `joint_hva`, `joint_angles`, `phase_events`,
`phase_eval`; tasks `kinematics`, `phase`.

### 2.3 Rehydrating a camera from published output

Today S5–S9 receive the `crop_record` that S4 just produced, in memory. Split pipelines have
to rebuild it from disk. A new `common/camera_records.py` does this. For each camera,
it reads S0's `record.json` and S4's `crop_sidecar.json` into the same `CameraRecord` that
S4 returns today, and discovers cameras by walking the S4 tree. It is the one new piece of
logic in this change. It is tested by round-tripping: run S4 on a fixture, rehydrate, and
compare with S4's returned record field by field.

## 3. Configuration and CLI

Each pipeline gets its own pydantic config. The roots each one names:

- `PreprocessConfig`: `raw_root`, `record_root`, `cut_root`, `retilt_root`, `crop_root`,
  `output_root`.
- `DetectConfig`: reads `record_root`, `crop_root`; writes `barbell_root`, `pose_root`,
  `lift_root | None`; plus `barbell_checkpoint`, `barbell_device`, `skip_detect`.
- `AnalyzeConfig`: reads `record_root`, `retilt_root`, `crop_root`, `barbell_root`,
  `pose_root`; writes `kinematics_root`, `phase_root`; plus the `phase` threshold block that
  lives in `PreprocessConfig` today.

The `powerflow` CLI gains one command per pipeline, plus a convenience chain:

```text
powerflow preprocess --input … --records … --cut … --retilt … --crop … --output …
powerflow detect     --records … --crop … --barbell … --barbell-checkpoint … --pose … [--lift …] [--only s5,s6,s7]
powerflow analyze    --records … --retilt … --crop … --barbell … --pose … --kinematics … --phase … [--only s8,s9]
powerflow all        (runs the three in order; each reads what the previous one published)
```

Each command keeps the shared contract: `--output` or `--in-place`, `--dry-run`, capture
selection, staging before publish, and one manifest per run. `prefect.yaml` gets one
deployment per pipeline.

## 4. Documentation

- `docs/specs/preprocessing/` splits into `docs/specs/preprocess/`, `docs/specs/detect/` and
  `docs/specs/analyze/`, by stage. The files are moved with `git mv`; no draft's content is edited.
  Relative links between specs in the same phase keep working. Cross-phase links (S9 → S8 is
  same-phase; S8 → S5/S6 is cross-phase) are fixed in each spec's next draft, not in place.
- The "home" open question is closed by appending feedback to the bottom of
  `S8-joint-hva-draft3.md`, `S8-joint-angles-draft3.md` and `S9-phase-intervals-draft7.md`,
  then writing S8 draft 4 (both specs) and S9 draft 8. Those drafts update their code-layout
  sections (`preprocess/…` → `analyze/…`).
- The module docstring of each `flow.py` names its stages, replacing today's ten-stage
  list.

## 5. Code changes (for the implementation plan, not done here)

1. Move `models.py` and `bar_geometry.py` to `common/`. Fix imports. No behaviour change.
2. Add `common/camera_records.py` (§2.3), test first.
3. Create `detect/` with its flow, config and tasks. Move the S5–S7 modules. S7 is gated on
   `lift_root`.
4. Create `analyze/` with its flow, config and tasks. Move the S8–S9 modules.
5. Cut `preprocess/flow.py` back to S0–S4, and remove the downstream guards and roots from
   `PreprocessConfig`.
6. Split the CLI into commands. Add `all`. Add the Prefect deployments.
7. Rename tests to `test_detect_*` / `test_analyze_*` to match their modules. Coverage stays
   above 90%.

Each step leaves the suite green. Steps 1 and 2 can land on their own.

## Out of scope

- Any change to a stage's maths, thresholds or output schema.
- Moving output folders: `s5`–`s9_*_output` keep their names and locations.
- Re-parenting GitHub issues. Phase (#94/#95/#96) sits under epic #77 "Detection" while S9
  now lives in `analyze` (Open question 4).

## Test plan

- **Rehydration round-trip** (§2.3): for every fixture capture, the rehydrated `CameraRecord`
  equals S4's in-memory record.
- **Equivalence:** on the fixture set, running `preprocess` → `detect` → `analyze` publishes
  byte-identical `barbell.json`, `pose.json`, `joint_hva.json` and `phase.json` to the
  current single flow. Run on the branch before the old flow is cut down.
- **Isolation:** `analyze --only s9` against an existing S8 tree runs with no detector loaded (the
  test asserts no detector is constructed).
- **Missing upstream:** a camera with no S6 output is rejected by `analyze` with a reason, and
  the rest of the run publishes.
- **S7 default off:** `detect` without `--lift` writes nothing to any lift root, and the manifest
  lists S7 as skipped.

## Verification

Before deleting the old stages from `preprocess/flow.py`, run both paths on the 10 September
captures and diff every published document. Every difference must be explained or fixed.

## Open questions

1. **Where does the bar's 3D track live?** Today S9's task solves it from S5's detections
   ("reads S5's detections (and solves the track)"), and S8 uses the bar as an anchor
   separately. It is a measurement, so it belongs in `analyze`. Should it become its own
   step there, published once and read by both S8 and S9? If it is numbered, it needs a new
   number (S7 is taken by Lift).
2. **S7's future.** Kept in `detect` and off by default. If it becomes an S8 input or
   cross-check, it moves to `analyze`. If it stays unused for a set time, it is retired.
3. **Stale upstream:** should each stage record its upstream documents' `schemaVersion` and
   source hash, and then reject a camera whose upstream changed since its own last run, or
   only warn in the manifest?
4. **GitHub epics:** re-parent Phase (#94/#95/#96) from "Detection" (#77) to an analysis epic,
   or leave the epics as feature groupings independent of the pipeline split?
5. **Naming:** `analyze` vs `analysis`, `detect` vs `detection`. This spec uses verbs, to
   match `preprocess`.
