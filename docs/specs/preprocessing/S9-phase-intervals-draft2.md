# S9 Phase — Snatch Phase Intervals from the Bar Track and Joints

Status: **draft 2** | Supersedes: [`S9-phase-intervals.md`](S9-phase-intervals.md) (feedback at its bottom) | Scope: GitHub issue [#95](https://github.com/tony-yehui-lou/PowerFlow/issues/95) (Phase Time Interval Detection) under [#94](https://github.com/tony-yehui-lou/PowerFlow/issues/94) (Phase), epic [#77](https://github.com/tony-yehui-lou/PowerFlow/issues/77) (Detection) | Sibling: [#96](https://github.com/tony-yehui-lou/PowerFlow/issues/96) Phase Issue Detection | Updated: 2026-10-22

The stage number **S9** is a proposal: the next free number after S8 Kinematics. In the
architecture this is the **Phase Analyzer** with its **Phase Rules**. Whether it lives in
`preprocess` or starts an analysis pipeline is S8's open question too; this spec assumes
`preprocess` for now.

**Changes from draft 1:**

1. **10 September only.** The corpus is its 7 single-camera snatch captures. 11 July is out of
   scope.
2. **One camera per rep.** Draft 1's multi-camera rules are gone: shared labels across a group,
   front captures reusing Side events, the `front_only` flag and the Front/Side consistency check.
3. **Cross-validation is leave-one-capture-out**, since each capture is one rep on one camera.

Carried over: the five events and their rules, the label contract, v1 rules first and v2 behind an
adoption gate, the `phase.json` output.

## Purpose

Split each snatch rep into four phases, plus the frames before and after the lift:

| Phase | From | To |
|---|---|---|
| `setup` | clip start | E0 liftoff |
| `first_pull` | E0 liftoff | E1 bar passes knee |
| `second_pull` | E1 bar passes knee | E2 full extension |
| `turnover` | E2 full extension | E3 catch |
| `recover` | E3 catch | E4 stand |
| `finish` | E4 stand | clip end |

The stage finds **five events**, E0–E4. The phases are the intervals between them. It does not
classify frames one at a time. #96 reads these intervals to decide where to look for faults.

Two detectors share one output contract:

- **v1, rules (§3).** Built first. Mostly reads the bar track, with knee height for E1.
- **v2, learned (§4).** Trained on the labels in §2. It replaces v1 for an event only when it beats
  v1 on held-out reps (§4.4).

**In:**
- S5's bar centre height per frame (`bar_pose.json`). Until that file exists, recompute it in the
  task with `bar_geometry.solve_track` and `floor_anchor`, the same way S8 does;
- S8's `joint_hva.json`: knee, hip, shoulder and wrist heights with σ and flags;
- the capture's `metadata.yaml`: `lift_type_in_source_file`, `lift.result`, `layout`, `cameras[].role`;
- for evaluation and training only: the phase labels (§2).

**Out:** one `phase.json` per capture (§5): event frames and times, phase intervals, per-event
uncertainty and source, and flags.

**Out of scope:**
- every session but 10 September: the 11 July captures, and the 22 August clean & jerk (`CnJ`);
- fault detection (#96);
- the labeling UI itself. This spec fixes the label file contract (§2.2). The UI gets its own spec
  in `powerflow-ui/docs/`.

---

## 1. The events

Each event is one instant you can see on video and measure from the signals. All times are
`startEpochMs + tMs[i]`, never `i / fps`, because `frames.fps` is nominal and dropped frames leave
gaps in `tMs`.

Signals (all in the floor frame from S8 §4, heights in mm):
- `h_bar(t)` is the bar centre height and `v_bar(t)` its vertical velocity. Velocity comes from a
  zero-phase low-pass filter (default cutoff 8 Hz, see Open questions) on the height, then a central
  difference over the real `tMs` spacing.
- `h_knee(t)`: the mean of left and right knee `heightM`, weighted by `1/σ²`. Frames flagged
  `uncertain` or with σ above `KNEE_SIGMA_MAX` are left out.
- `h_hip(t)` and `h_shoulder(t)`: the same rule, for hips and shoulders.

| Event | What it is | v1 definition (§3 has the thresholds) |
|---|---|---|
| **E0 Liftoff** | the plates leave the floor | the first frame where `h_bar` is above the rest height by `LIFT_MM` and keeps rising for `LIFT_HOLD` frames; then step back to the last frame where `v_bar ≤ V_REST` |
| **E1 Bar passes knee** | the end of the first pull | the first frame after E0 where `h_bar ≥ h_knee`. Fallback when the knee is unavailable: the local minimum of `v_bar` between E0 and E2 (the scoop dip) |
| **E2 Full extension** | the end of the drive, where the lifter starts pulling under | the frame of peak upward `v_bar` between E1 and the bar's peak height |
| **E3 Catch** | the lifter is under the bar and the bar stops descending | the first frame after the bar's peak height where `v_bar` rises through 0 (the bar's lowest point after its peak), with `h_bar` above `h_shoulder` |
| **E4 Stand** | standing, with the bar overhead | the first frame after E3 where `|v_bar| < V_STILL` for `STILL_HOLD` frames, with `h_hip` within `STAND_FRAC` of its maximum after E3 |

**Missed lift.** The rep counts as missed if, after E3, `h_bar` drops below `h_shoulder` before E4.
Then E4 is `absent`, the `recover` phase runs to the drop frame, and the `missed` flag is set. A
non-null `lift.result` overrides this guess and is recorded as the source.

**Why E3 is the bottom, not first contact.** In a squat snatch the lifter meets the bar and the two
ride down together. The bar's lowest point is easy to measure and repeatable. The moment of first
contact is neither. For a power snatch, the bottom and first contact are close together. This is a
default, and the Open questions revisit it.

---

## 2. Labels

### 2.1 What gets labeled

The five event times per rep, nothing per frame. Five ordered events can't produce phases out of
order, they take seconds to label instead of minutes, and error is measured in milliseconds per
event (§7).

Labels are in **epoch ms**, so they stay valid if a capture is re-cut or its frames are renumbered.

**Corpus at draft 2:** the 7 captures from 10 September: 30kg1, 30kg2, 40kg1, 45kg1, 50kg1, 50kg2
and 50kg3. Each is `layout: single_camera`, one rep at about 60 fps (50kg2 reports 59.71), with S8 `joint_hva.json` on disk.

That is 7 reps, all one athlete. Enough to measure v1 and to tune its thresholds. It is not enough
to train v2 (§4.4).

### 2.2 File contract

`data/annotations/phase/v1/<captureId>/phases.json`, one per labeled capture, written once per pass:

```json
{
  "schemaVersion": "1.0.0",
  "captureId": "10 September/50kg3",
  "lift": "snatch",
  "labeler": "<initials>",
  "pass": 1,
  "labeledAt": "2026-10-22T10:00:00Z",
  "events": {
    "liftoff":       {"epochMs": 1789029617780, "certainty": "sure"},
    "barPassesKnee": {"epochMs": 1789029618200, "certainty": "unsure"},
    "fullExtension": {"epochMs": 1789029618400, "certainty": "sure"},
    "catch":         {"epochMs": 1789029618750, "certainty": "sure"},
    "stand":         {"epochMs": null, "certainty": "absent"}
  },
  "result": "made",
  "notes": ""
}
```

- `certainty` is `sure`, `unsure` or `absent`. An `absent` event has `epochMs: null`.
- `result` is `made`, `missed` or `unknown`.
- Each pass writes its own file, `phases.pass<N>.json`. A pass is never edited after it is written.
  `phases.json` points at the adjudicated set: the pass to use, or a merge recorded with both
  sources.

### 2.3 Protocol

1. Label from `rgb.mp4` at full frame rate with `h_bar` drawn under the video. The labeler judges
   by eye. The curve is there to help find the frame, not to decide it.
2. Use the definitions in §1, written out in the labeling guide with one reference frame per event
   (a guide in the same style as `S5-barbell-annotation-guide.md`).
3. **Two passes, at least three days apart**, or a second labeler. The gap between passes per event
   is the **human floor**: the best accuracy any detector can be held to (§7.2).
4. Label before looking at v1's output, so the labels aren't biased toward the rules.

---

## 3. v1 — Rules

### 3.1 Starting thresholds

| Name | Value | Meaning |
|---|---|---|
| `LIFT_MM` | 20 mm | rise above the rest height that counts as off the floor |
| `LIFT_HOLD` | 83 ms (5 frames at 60 Hz) | how long the rise must continue |
| `V_REST` | 50 mm/s | speed below which the bar counts as still at rest |
| `V_STILL` | 50 mm/s | speed below which the bar counts as still overhead |
| `STILL_HOLD` | 500 ms | how long it must stay still for E4 |
| `STAND_FRAC` | 0.05 | how close to its post-catch maximum the hip height must be |
| `KNEE_SIGMA_MAX` | 50 mm | knee σ above which the E1 rule falls back |

Holds are durations, not frame counts. The task converts them to frames with each capture's real
`tMs` cadence. All values are config, recorded in the provenance (§5), and tuned against the labels
(§7.3).

### 3.2 Order and search windows

Search for each event only inside the window its neighbours leave:
- E0, from the end of the rest window;
- E2, the bar's peak and E3, all after E0;
- E1, between E0 and E2;
- E4, after E3.

If an event isn't found, it is `absent` and every phase that depends on it is null. Nothing is
guessed. Durations are checked against starting plausible ranges, which the label statistics will
replace:

| Phase | Plausible duration |
|---|---|
| first pull | 0.30–1.20 s |
| second pull | 0.08–0.40 s |
| turnover | 0.15–0.60 s |

A duration outside its range sets the `implausible_duration` flag. It never moves an event.

### 3.3 Bar track gaps

S5's draft 5 measurements show the bar's yaw can't be measured through the turnover. There, B is
selected, and some frames are collar-only or `missing`.

- Gaps up to `MAX_GAP` (100 ms) are bridged for `v_bar` only, by linear interpolation of `h_bar`,
  and flagged `bar_gap_bridged`.
- An event that falls inside a bridged gap gets its σ widened to half the gap.
- A longer gap at an event makes that event `absent`.

### 3.4 Uncertainty

Each event reports `sigmaMs`: half the width of the window where the rule's condition is within
its tolerance (for peaks, where `v_bar` is within 5% of its peak), floored at half a frame. E1 from
the fallback rule also adds the knee/scoop disagreement measured in §7.2.

---

## 4. v2 — Learned detector

Built only after v1 has been measured against the labels and there are enough labels (§4.4).

### 4.1 Features per frame

All features are relative to the rep, so different loads and athletes compare:
- `h_bar`, `v_bar` and the bar's vertical acceleration, with heights divided by the standing hip
  height;
- knee, hip, shoulder and wrist heights over the same scale, with σ as a separate input channel;
- knee and hip angles from S8 positions where both bones are `measured` or `fitted`;
- frame validity masks: bar `missing`, joint `uncertain`.

IMU and odometry describe the **phone**, not the lifter, so they are left out.

### 4.2 Candidates

1. **Left-to-right HMM** with six states in the phase order of the Purpose table and Gaussian
   emissions over the features. Viterbi decoding gives the boundaries directly. It needs very
   little data, and its order constraint matches the problem.
2. **Event-heatmap temporal network.** A small 1D dilated conv net outputs one Gaussian heatmap per
   event, decoded under the same order constraint. It needs more data than the HMM.

The HMM goes first. The network is tried only once the corpus reaches the size in §4.4.

### 4.3 Training code

`src/powerflow_pipeline/training/phase/`, next to the existing `training/` packages. Each training
run writes its corpus manifest (captures, label passes used), its features configuration and its
leave-one-capture-out results. Models that pass §4.4 are registered with the Model Hub, and S9
loads them through it, never from a training output folder.

### 4.4 Adoption gate

v2 replaces v1 **per event** only when, under leave-one-capture-out cross-validation:

- the median absolute error is lower than v1's;
- the 90th-percentile error is not worse; and
- the corpus has at least **30 labeled reps** from at least **2 athletes**.

Until then, S9 runs v1, and v2 is reported in the training manifest only.

---

## 5. Output

`data/s9_phase_output/<captureId>/phase.json`:

```json
{
  "schemaVersion": "1.0.0",
  "captureId": "10 September/50kg3",
  "lift": "snatch",
  "method": "docs/specs/preprocessing/S9-phase-intervals.md",
  "detector": {"liftoff": "rules-v1", "barPassesKnee": "rules-v1-fallback", "...": "..."},
  "frames": {"count": 433, "startEpochMs": 0, "tMs": [0, 17, "..."]},
  "events": {
    "liftoff": {"frame": 141, "tMs": 2350, "epochMs": 0, "sigmaMs": 17, "status": "found"},
    "stand":   {"frame": null, "tMs": null, "epochMs": null, "sigmaMs": null, "status": "absent"}
  },
  "phases": [
    {"phase": "setup", "startFrame": 0, "endFrame": 141},
    {"phase": "first_pull", "startFrame": 141, "endFrame": 178}
  ],
  "result": {"value": "made", "source": "rules"},
  "flags": ["bar_gap_bridged"],
  "provenance": {"thresholds": {}, "filterCutoffHz": 8, "inputs": {"barPose": "", "jointHva": ""}}
}
```

- `endFrame` is exclusive. Phases never overlap, and together they cover every frame from 0 to
  `count`. A phase bounded by an `absent` event is left out, and its frames belong to no phase.
- `status` is `found`, `fallback` or `absent`.
- `result.source` is `metadata` or `rules`.
- `flags` can be: `missed`, `bar_gap_bridged`, `knee_unavailable`, `implausible_duration`.
- A capture whose `lift_type_in_source_file` is not a snatch is rejected in the manifest, not
  guessed.
- A capture from outside 10 September is rejected in the manifest as out of scope for draft 2.

The run manifest (a Prefect markdown artifact, like every stage) reports per capture: each event
with its status and σ, the phase durations, flags, rejected captures, and the thresholds used.

---

## 6. Code layout

- `data/preprocess/phase_events.py` holds the pure functions: signal prep (filter, velocity,
  gap bridging), each event rule, interval assembly, the plausibility checks. It has no I/O.
- `data/preprocess/tasks/phase.py` holds the `@task`: it reads `bar_pose.json` (or recomputes the
  track), `joint_hva.json` and the metadata, and writes `phase.json` through staging like every
  stage.
- `data/preprocess/phase_eval.py` loads labels and scores `phase.json` against them (§7). The
  pipeline never imports it. It is called from a CLI subcommand.
- Thresholds go in `config.py` under a `phase` block.

---

## 7. Validation

### 7.1 Synthetic (unit tests)

Build `h_bar` from piecewise curves with known event times: rest, a rise accelerating to a peak
velocity, the drop under, the catch dip, the rise, then still. Check that:
- each rule recovers its event to within one frame;
- the event order holds;
- adding noise at the measured S5 rest-residual σ moves each event by no more than its `sigmaMs`;
- a gap across E3 either widens σ (short gap) or makes E3 `absent` (long gap);
- the missed-lift curve sets `missed` and leaves E4 `absent`;
- a dropped frame (a gap in `tMs`) leaves the event *times* unchanged.

### 7.2 Against the labels: the primary real-data test

Per event, over the corpus:
- the median and 90th-percentile absolute error, in ms;
- the share of events `found` when the label is `sure`.

Each result is reported next to the **human floor** (§2.3). Starting targets, to be re-set once the
human floor is measured:

| Event | Median abs. error | 90th percentile |
|---|---|---|
| E0, E2, E3 | ≤ 33 ms (2 frames at 60 Hz) | ≤ 67 ms |
| E1, E4 | ≤ 67 ms | ≤ 133 ms |

### 7.3 Tuning without overfitting

Thresholds are tuned with leave-one-capture-out cross-validation, the same as §4.4. The spec records
the chosen values with the cross-validated error, never the error on the reps they were tuned on.

### 7.4 Consistency

- **Hand phases in S5.** The windows S5's draft 5 marked by eye on 50kg3 (Pull 150–200, Turnover
  205–255) should broadly agree. That is a sanity check, not ground truth.

---

## 8. Risks

- **Seven reps, one athlete, one session, one camera position.** Thresholds tuned here may not carry over to another lifter, another
  camera height or another bar speed. That is why §4.4 asks for 30 reps from at least 2 athletes
  before trusting a model.
- **Pose quality.** E1 and E4 depend on S8 joints, and S6 pose has been unreliable. The bar-only
  fallbacks keep the stage running, but E1's definition changes when it falls back (knee crossing
  vs scoop dip). The output says which was used.
- **The bar track is weakest at the turnover.** E2 and E3 sit right where S5 reports unmeasurable
  yaw and collar-only frames. §3.3 bounds the damage. If that isn't enough, §7.2 will show it.
- **Filter choice moves E2.** Peak velocity shifts with the low-pass cutoff. The cutoff is recorded,
  and §7.2 is rerun whenever it changes.
- **Coaching definitions differ** (see Open questions). Settling them after labeling means
  relabeling.

---

## Open questions

1. **Is there a separate transition phase?** Many coaches split a transition (scoop, double knee
   bend) between the first and second pull. Adding it later means relabeling, so decide before
   §2's first pass.
2. **What is the catch (E3)?** This spec uses the bar's lowest point after its peak. Alternatives:
   first contact / elbow lockout, or the bottom of the hips. For a power snatch these differ the
   most.
3. **Phase list.** This spec uses four phases plus `setup` and `finish`. The request listed both
   "first pull, second pull, turnover" and the four phases with recover. Confirm four.
4. **Low-pass cutoff for `v_bar`.** 8 Hz here. S8 asks the same question for joints (4–10 Hz in the
   literature). One shared answer would be better.
5. **Stage number and home.** S9 in `preprocess`, or the start of the Phase Analyzer as its own
   pipeline. This follows S8's answer.
6. **Labeling tool.** A powerflow-ui page (video plus `h_bar` curve, keys to drop E0–E4), or an
   existing tool such as CVAT/Label Studio exporting to §2.2.


---

## Feedback (2026-10-22)

1. Also write the clean and jerk. Its phases are:
   1. first pull
   2. second pull
   3. turnover
   4. clean recovery
   5. dip
   6. recovery
