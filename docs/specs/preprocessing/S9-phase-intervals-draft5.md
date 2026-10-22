# S9 Phase — Snatch and Clean & Jerk Phase Intervals from the Bar Track and Joints

Status: **draft 5** | Supersedes: [`S9-phase-intervals-draft4.md`](S9-phase-intervals-draft4.md) (feedback at its bottom) | Scope: GitHub issue [#95](https://github.com/tony-yehui-lou/PowerFlow/issues/95) (Phase Time Interval Detection) under [#94](https://github.com/tony-yehui-lou/PowerFlow/issues/94) (Phase), epic [#77](https://github.com/tony-yehui-lou/PowerFlow/issues/77) (Detection) | Sibling: [#96](https://github.com/tony-yehui-lou/PowerFlow/issues/96) Phase Issue Detection | Updated: 2026-10-22

The stage number **S9** is a proposal: the next free number after S8 Kinematics. In the
architecture this is the **Phase Analyzer** with its **Phase Rules**. Whether it lives in
`preprocess` or starts an analysis pipeline is S8's open question too; this spec assumes
`preprocess` for now.

**Changes from draft 4:**

1. **The snatch's last phase is `recovery`**, renamed from `recover`. It now matches the clean &
   jerk's `clean_recovery` and `jerk_recovery`. Open question 7 is closed.

Carried over: everything else in draft 4, including the clean & jerk's seven phases and eight events
(E0–E7), the `dip` / `drive` split at E5 dip bottom, and `jerk_recovery`.

## Purpose

Split each rep into its lift's phases, plus `setup` before the lift and `finish` after it.

**Snatch** (§1.2), five events E0–E4:

| Phase | From | To |
|---|---|---|
| `setup` | clip start | E0 liftoff |
| `first_pull` | E0 liftoff | E1 bar passes knee |
| `second_pull` | E1 bar passes knee | E2 full extension |
| `turnover` | E2 full extension | E3 catch |
| `recovery` | E3 catch | E4 stand |
| `finish` | E4 stand | clip end |

**Clean & jerk** (§1.3), eight events E0–E7:

| Phase | From | To |
|---|---|---|
| `setup` | clip start | E0 liftoff |
| `first_pull` | E0 liftoff | E1 bar passes knee |
| `second_pull` | E1 bar passes knee | E2 full extension |
| `turnover` | E2 full extension | E3 clean catch |
| `clean_recovery` | E3 clean catch | E4 dip start |
| `dip` | E4 dip start | E5 dip bottom |
| `drive` | E5 dip bottom | E6 jerk catch |
| `jerk_recovery` | E6 jerk catch | E7 stand |
| `finish` | E7 stand | clip end |

The stage finds the **events**. The phases are the intervals between them. It does not classify
frames one at a time. #96 reads these intervals to decide where to look for faults.

Two detectors share one output contract:

- **v1, rules (§3).** Built first. Mostly reads the bar track, with knee, hip and shoulder heights
  where a rule needs them.
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
- every session but 10 September (11 July, 22 August);
- the clean and the jerk recorded as separate lifts. A `CnJ` capture is one clip holding both;
- fault detection (#96);
- the labeling UI itself. This spec fixes the label file contract (§2.2). The UI gets its own spec
  in `powerflow-ui/docs/`.

---

## 1. The events

### 1.1 Time and signals

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

### 1.2 Snatch, and the pull both lifts share

| Event | What it is | v1 definition (§3 has the thresholds) |
|---|---|---|
| **E0 Liftoff** | the plates leave the floor | the first frame where `h_bar` is above the rest height by `LIFT_MM` and keeps rising for `LIFT_HOLD`; then step back to the last frame where `v_bar ≤ V_REST` |
| **E1 Bar passes knee** | the end of the first pull | the first frame after E0 where `h_bar ≥ h_knee`. Fallback when the knee is unavailable: the local minimum of `v_bar` between E0 and E2 (the scoop dip) |
| **E2 Full extension** | the end of the pull's upward drive, where the lifter starts pulling under (not the jerk's `drive` phase) | the frame of peak upward `v_bar` between E1 and the bar's first peak height |
| **E3 Catch** | the lifter is under the bar and the bar stops descending | the first frame after the bar's first peak where `v_bar` rises through 0 (the bar's lowest point after its peak), with `h_bar` above `h_shoulder` |
| **E4 Stand** | standing, with the bar overhead | the first frame after E3 where `|v_bar| < V_STILL` for `STILL_HOLD`, with `h_hip` within `STAND_FRAC` of its maximum after E3 |

E0, E1 and E2 are the same events in the clean & jerk, found by the same rules.

**Missed snatch.** The rep counts as missed if, after E3, `h_bar` drops below `h_shoulder` before E4.
Then E4 is `absent`, `recovery` runs to the drop frame, and the `missed` flag is set. A non-null
`lift.result` overrides this guess and is recorded as the source.

**Why E3 is the bottom, not first contact.** In a squat snatch the lifter meets the bar and the two
ride down together. The bar's lowest point is easy to measure and repeatable. The moment of first
contact is neither. For a power snatch, the bottom and first contact are close together. This is a
default, and the Open questions revisit it.

### 1.3 Clean & jerk

After the shared pull (E0–E2), the bar is caught on the shoulders, stood up, then jerked overhead.

| Event | What it is | v1 definition (§3 has the thresholds) |
|---|---|---|
| **E3 Clean catch** | the lifter is under the bar in the front rack and the bar stops descending | the first frame after the bar's first peak where `v_bar` rises through 0, with `h_bar` between `h_hip` and `h_shoulder + RACK_TOL` |
| **E4 Dip start** | the jerk begins: the bar starts down from the standing rack | first find the **rack stand**: after E3, the first stretch where `|v_bar| < V_STILL` for `RACK_HOLD` with `h_hip` within `STAND_FRAC` of its maximum before the dip. E4 is the first frame after the rack stand where `h_bar` is `DIP_MM` below the rack-stand height and still falling; then step back to the last frame where `|v_bar| ≤ V_REST` |
| **E5 Dip bottom** | the bottom of the dip: the bar stops going down and the drive begins | the first frame after E4 where `v_bar` rises through 0 (the bar's lowest point in the dip), with `h_bar` at least `DIP_MM` below the rack-stand height |
| **E6 Jerk catch** | the lifter is under the bar overhead and the bar stops descending | find the bar's next peak after E5. E6 is the first frame after that peak where `v_bar` rises through 0, with `h_bar` above `h_shoulder` |
| **E7 Stand** | feet recovered, standing still with the bar overhead | the snatch E4 rule, applied after E6 |

**What each phase holds:**
- `turnover` ends with the bar on the shoulders, not overhead.
- `clean_recovery` runs from the bottom of the clean, through standing up, to the end of any pause
  in the rack. The pause is not its own phase (see Open questions).
- `dip` is the bar going down from the rack: the knees bend and the bar sinks with the lifter.
- `drive` runs from the bottom of the dip, through the drive up, to the jerk catch. It holds the
  drop under the bar too: the list has no phase for it (see Open questions).
- `jerk_recovery` is standing up out of the split or squat jerk, with the bar overhead, until still.

**Missed clean & jerk.** Two ways to miss:
- **Missed clean.** After E3, `h_bar` drops below `h_hip` before a rack stand is found. E4–E7 are
  `absent`, `clean_recovery` runs to the drop frame, and the `missed_clean` flag is set.
- **Missed jerk.** After E6, `h_bar` drops below `h_shoulder` before E7. E7 is `absent`,
  `jerk_recovery` runs to the drop frame, and `missed_jerk` is set. A bar that never reaches an
  overhead catch after E5 leaves E6 and E7 `absent`, with `drive` running to the drop frame.

A non-null `lift.result` overrides both guesses, as for the snatch.

**Why E4 is the dip start, not the rack stand.** The user's phase list has no phase between standing
up and the dip, so the boundary goes where the jerk visibly begins. The rack stand is still
computed, because E4's search starts from it, and it is reported as an auxiliary event (§5).

---

## 2. Labels

### 2.1 What gets labeled

The event times per rep, nothing per frame: five for a snatch, eight for a clean & jerk. Ordered
events can't produce phases out of order, they take seconds to label instead of minutes, and error
is measured in milliseconds per event (§7).

Labels are in **epoch ms**, so they stay valid if a capture is re-cut or its frames are renumbered.

**Corpus at draft 3:**
- **Snatch:** the 7 captures from 10 September: 30kg1, 30kg2, 40kg1, 45kg1, 50kg1, 50kg2 and
  50kg3. Each is `layout: single_camera`, one rep at about 60 fps (50kg2 reports 59.71), with S8
  `joint_hva.json` on disk.
- **Clean & jerk:** none. 10 September has no `CnJ` capture. Until one is in scope, the clean & jerk
  rules are checked on synthetic curves only (§7.1).

That is 7 snatch reps, all one athlete. Enough to measure v1 and to tune its snatch thresholds. It
is not enough to train v2 (§4.4).

### 2.2 File contract

`data/annotations/phase/v1/<captureId>/phases.json`, one per labeled capture, written once per pass:

```json
{
  "schemaVersion": "1.1.0",
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

- `lift` is `snatch` or `clean_and_jerk`. It fixes which event keys must be present:

  | `lift` | Event keys, in order |
  |---|---|
  | `snatch` | `liftoff`, `barPassesKnee`, `fullExtension`, `catch`, `stand` |
  | `clean_and_jerk` | `liftoff`, `barPassesKnee`, `fullExtension`, `cleanCatch`, `dipStart`, `dipBottom`, `jerkCatch`, `stand` |

- `certainty` is `sure`, `unsure` or `absent`. An `absent` event has `epochMs: null`.
- `result` is `made`, `missed` or `unknown` for a snatch. For a clean & jerk it is `made`,
  `missed_clean`, `missed_jerk` or `unknown`.
- Each pass writes its own file, `phases.pass<N>.json`. A pass is never edited after it is written.
  `phases.json` points at the adjudicated set: the pass to use, or a merge recorded with both
  sources.
- `schemaVersion` 1.1.0 adds `clean_and_jerk`. A 1.0.0 file is a snatch file and stays valid.

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

| Name | Value | Lift | Meaning |
|---|---|---|---|
| `LIFT_MM` | 20 mm | both | rise above the rest height that counts as off the floor |
| `LIFT_HOLD` | 83 ms (5 frames at 60 Hz) | both | how long the rise must continue |
| `V_REST` | 50 mm/s | both | speed below which the bar counts as still at rest |
| `V_STILL` | 50 mm/s | both | speed below which the bar counts as still (overhead or in the rack) |
| `STILL_HOLD` | 500 ms | both | how long it must stay still for the final stand |
| `STAND_FRAC` | 0.05 | both | how close to its maximum the hip height must be to count as standing |
| `KNEE_SIGMA_MAX` | 50 mm | both | knee σ above which the E1 rule falls back |
| `RACK_TOL` | 100 mm | C&J | how far above `h_shoulder` the bar may sit and still count as in the rack |
| `RACK_HOLD` | 300 ms | C&J | how long the bar must be still for the rack stand |
| `DIP_MM` | 30 mm | C&J | drop below the rack-stand height that counts as the dip |

Holds are durations, not frame counts. The task converts them to frames with each capture's real
`tMs` cadence. All values are config, recorded in the provenance (§5), and tuned against the labels
(§7.3). The clean & jerk thresholds stay at these starting values until there are clean & jerk
labels.

### 3.2 Order and search windows

Search for each event only inside the window its neighbours leave.

Both lifts:
- E0, from the end of the rest window;
- E2, the bar's first peak and E3, all after E0;
- E1, between E0 and E2.

Snatch: E4, after E3.

Clean & jerk: the rack stand, after E3; E4, after the rack stand; E5, after E4; the bar's next
peak, then E6, after E5; E7, after E6.

If an event isn't found, it is `absent` and every phase that depends on it is null. Nothing is
guessed. Durations are checked against starting plausible ranges, which the label statistics will
replace:

| Phase | Lift | Plausible duration |
|---|---|---|
| `first_pull` | both | 0.30–1.20 s |
| `second_pull` | both | 0.08–0.40 s |
| `turnover` | both | 0.15–0.60 s |
| `clean_recovery` | C&J | 0.80–8.00 s (it includes the rack pause) |
| `dip` | C&J | 0.10–0.50 s |
| `drive` | C&J | 0.20–0.70 s (it includes the drop under the bar) |

A duration outside its range sets the `implausible_duration` flag. It never moves an event.

### 3.3 Bar track gaps

S5's draft 5 measurements show the bar's yaw can't be measured through the turnover. There, B is
selected, and some frames are collar-only or `missing`. The jerk's drop under the bar is the same
kind of fast, rotating-arms moment, and is expected to have the same gaps.

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

1. **Left-to-right HMM per lift**, with one state per phase in the order of the Purpose tables:
   six for the snatch, nine for the clean & jerk. Emissions are Gaussian over the features.
   Viterbi decoding gives the boundaries directly. The pull states (`setup` through
   `second_pull`) can share emission parameters across the two lifts, so clean & jerk reps help
   train the snatch pull and the reverse. It needs very little data, and its order constraint
   matches the problem.
2. **Event-heatmap temporal network.** A small 1D dilated conv net outputs one Gaussian heatmap per
   event, decoded under the same order constraint. It needs more data than the HMM.

The HMM goes first. The network is tried only once the corpus reaches the size in §4.4.

### 4.3 Training code

`src/powerflow_pipeline/training/phase/`, next to the existing `training/` packages. Each training
run writes its corpus manifest (captures, lift, label passes used), its features configuration and
its leave-one-capture-out results. Models that pass §4.4 are registered with the Model Hub, and S9
loads them through it, never from a training output folder.

### 4.4 Adoption gate

v2 replaces v1 **per lift and per event** only when, under leave-one-capture-out cross-validation:

- the median absolute error is lower than v1's;
- the 90th-percentile error is not worse; and
- that lift's corpus has at least **30 labeled reps** from at least **2 athletes**.

The shared pull events (E0–E2) may count reps from both lifts towards the 30. Until the gate is met,
S9 runs v1, and v2 is reported in the training manifest only.

---

## 5. Output

`data/s9_phase_output/<captureId>/phase.json`:

```json
{
  "schemaVersion": "1.1.0",
  "captureId": "10 September/50kg3",
  "lift": "snatch",
  "method": "docs/specs/preprocessing/S9-phase-intervals-draft5.md",
  "detector": {"liftoff": "rules-v1", "barPassesKnee": "rules-v1-fallback", "...": "..."},
  "frames": {"count": 433, "startEpochMs": 0, "tMs": [0, 17, "..."]},
  "events": {
    "liftoff": {"frame": 141, "tMs": 2350, "epochMs": 0, "sigmaMs": 17, "status": "found"},
    "stand":   {"frame": null, "tMs": null, "epochMs": null, "sigmaMs": null, "status": "absent"}
  },
  "auxiliary": {},
  "phases": [
    {"phase": "setup", "startFrame": 0, "endFrame": 141},
    {"phase": "first_pull", "startFrame": 141, "endFrame": 178}
  ],
  "result": {"value": "made", "source": "rules"},
  "flags": ["bar_gap_bridged"],
  "provenance": {"thresholds": {}, "filterCutoffHz": 8, "inputs": {"barPose": "", "jointHva": ""}}
}
```

- `lift` comes from `lift_type_in_source_file`: `snch` gives `snatch`, `CnJ` gives
  `clean_and_jerk`. Any other value is rejected in the manifest, not guessed.
- `events` uses the keys of §2.2 for that lift, so a phase file and a label file compare key by key.
- `auxiliary` holds events that are computed but bound no phase. For a clean & jerk this is
  `rackStand`, in the same shape as an event. For a snatch it is empty.
- `endFrame` is exclusive. Phases never overlap, and together they cover every frame from 0 to
  `count`. A phase bounded by an `absent` event is left out, and its frames belong to no phase.
- `status` is `found`, `fallback` or `absent`.
- `result.value` uses the §2.2 values for that lift. `result.source` is `metadata` or `rules`.
- `flags` can be: `missed` (snatch), `missed_clean`, `missed_jerk` (clean & jerk),
  `bar_gap_bridged`, `knee_unavailable`, `implausible_duration`.
- A capture from outside 10 September is rejected in the manifest as out of scope for draft 3.

The run manifest (a Prefect markdown artifact, like every stage) reports per capture: the lift,
each event with its status and σ, the phase durations, flags, rejected captures, and the thresholds
used.

---

## 6. Code layout

- `data/preprocess/phase_events.py` holds the pure functions: signal prep (filter, velocity,
  gap bridging), each event rule, interval assembly, the plausibility checks. It has no I/O. The
  pull rules (E0–E2) are written once. Each lift adds its own rules after E2 and its own phase
  table.
- `data/preprocess/tasks/phase.py` holds the `@task`: it reads `bar_pose.json` (or recomputes the
  track), `joint_hva.json` and the metadata, picks the lift, and writes `phase.json` through staging
  like every stage.
- `data/preprocess/phase_eval.py` loads labels and scores `phase.json` against them (§7). The
  pipeline never imports it. It is called from a CLI subcommand.
- Thresholds go in `config.py` under a `phase` block, with a `clean_and_jerk` sub-block for the
  thresholds only it uses.

---

## 7. Validation

### 7.1 Synthetic (unit tests)

**Snatch.** Build `h_bar` from piecewise curves with known event times: rest, a rise accelerating
to a peak velocity, the drop under, the catch dip, the rise, then still.

**Clean & jerk.** The same pull, then a catch with the bar at shoulder height, the rise to a rack
stand, a still rack pause, the dip to its bottom, the drive to a second peak, the drop under into
the jerk catch, the rise, then still overhead.

For both, check that:
- each rule recovers its event to within one frame;
- the event order holds;
- adding noise at the measured S5 rest-residual σ moves each event by no more than its `sigmaMs`;
- a gap across a catch either widens σ (short gap) or makes the catch `absent` (long gap);
- a dropped frame (a gap in `tMs`) leaves the event *times* unchanged.

Then, per lift:
- **Snatch:** the missed-lift curve sets `missed` and leaves E4 `absent`.
- **Clean & jerk:**
  - a bar dropped from the rack sets `missed_clean` and leaves E4–E7 `absent`;
  - a bar dropped from overhead sets `missed_jerk` and leaves E7 `absent`;
  - a dip shallower than `DIP_MM` finds neither E4 nor E5;
  - a clean & jerk with no rack pause (dip straight after standing) still finds the rack stand and
    E4;
  - a rack pause with small re-grip bounces below `DIP_MM` doesn't trigger E4;
  - a snatch curve given `lift: clean_and_jerk` finds no clean catch (the bar is above the
    shoulders), and the reverse finds no snatch catch.

### 7.2 Against the labels: the primary real-data test

Per lift and per event, over the corpus:
- the median and 90th-percentile absolute error, in ms;
- the share of events `found` when the label is `sure`.

Each result is reported next to the **human floor** (§2.3). Starting targets, to be re-set once the
human floor is measured:

| Event | Median abs. error | 90th percentile |
|---|---|---|
| E0, E2, snatch E3, clean catch, dip bottom, jerk catch | ≤ 33 ms (2 frames at 60 Hz) | ≤ 67 ms |
| E1, dip start, stand | ≤ 67 ms | ≤ 133 ms |

At draft 3 only the snatch rows can be measured, because there are no clean & jerk labels (§2.1).

### 7.3 Tuning without overfitting

Thresholds are tuned with leave-one-capture-out cross-validation, the same as §4.4. The spec records
the chosen values with the cross-validated error, never the error on the reps they were tuned on.

### 7.4 Consistency

- **Hand phases in S5.** The windows S5's draft 5 marked by eye on 50kg3 (Pull 150–200, Turnover
  205–255) should broadly agree. That is a sanity check, not ground truth.

---

## 8. Risks

- **Seven reps, one athlete, one session, one camera position.** Thresholds tuned here may not carry
  over to another lifter, another camera height or another bar speed. That is why §4.4 asks for 30
  reps from at least 2 athletes before trusting a model.
- **Clean & jerk rules are untested on real data.** Its thresholds are guesses until clean & jerk
  captures are in scope and labeled. Synthetic curves (§7.1) check the logic, not the values.
- **Clean & jerk clips are long.** Rack pauses vary from almost nothing to several seconds, so a
  clip cut too tight may lose the jerk. The 22 August captures show the spread: their frame counts
  run from 76 (about 1.3 s, too short to hold a whole clean & jerk) to 763.
- **Pose quality.** E1, the stands and the rack checks depend on S8 joints, and S6 pose has been
  unreliable. The bar-only fallbacks keep the stage running, but E1's definition changes when it
  falls back (knee crossing vs scoop dip). The output says which was used. The rack and overhead
  checks (`h_bar` against `h_shoulder`) have no fallback yet. If the shoulder is unavailable, those
  events are `absent`.
- **The bar track is weakest at the turnover and the jerk catch.** The catches sit right where S5
  reports unmeasurable yaw and collar-only frames. §3.3 bounds the damage. If that isn't enough,
  §7.2 will show it.
- **Filter choice moves E2.** Peak velocity shifts with the low-pass cutoff. The cutoff is recorded,
  and §7.2 is rerun whenever it changes.
- **Coaching definitions differ** (see Open questions). Settling them after labeling means
  relabeling.

---

## Open questions

1. **Is there a separate transition phase?** Many coaches split a transition (scoop, double knee
   bend) between the first and second pull. Adding it later means relabeling, so decide before
   §2's first pass. It applies to both lifts.
2. **What is the catch?** This spec uses the bar's lowest point after its peak, for the snatch catch,
   the clean catch and the jerk catch. Alternatives: first contact / elbow lockout, or the bottom of
   the hips. They differ most for a power snatch, a power clean and a push jerk.
3. **Snatch phase list.** This spec uses four phases plus `setup` and `finish`. The request listed
   both "first pull, second pull, turnover" and the four phases with recover. Confirm four.
4. **Where does `drive` end?** Here it runs to the jerk catch, so it holds the drop under the bar.
   The alternative ends it at the jerk's peak upward bar velocity (full extension, as E2 does for
   the pull), with the drop under as its own phase. That would add one event and one phase.
5. **Should the rack pause be its own phase?** Here it sits inside `clean_recovery`, and the rack
   stand is reported as an auxiliary event so the split can be added later without relabeling.
6. **Where do clean & jerk captures come from?** 10 September has none. 22 August has 4 `CnJ`
   captures (48kgCnJ1, 75kgCnJ1, 77kgCnJ1, 82kgCnJ2). They have S4 crop output, but no S5 barbell
   annotations (v1 covers only the 22 August snatches), no S6 pose and no S8 kinematics output yet.
   The options are to bring them into scope, or to record new clean & jerks on the 10 September
   setup.
7. **Low-pass cutoff for `v_bar`.** 8 Hz here. S8 asks the same question for joints (4–10 Hz in the
   literature). One shared answer would be better.
8. **Stage number and home.** S9 in `preprocess`, or the start of the Phase Analyzer as its own
   pipeline. This follows S8's answer.
9. **Labeling tool.** A powerflow-ui page (video plus `h_bar` curve, keys to drop the events), or an
   existing tool such as CVAT/Label Studio exporting to §2.2.



---

## Feedback (2026-10-22)

1. Don't use speed alone. Also compute a rough range for where the bar's height should be at each
   position, and use both speed and height to decide each event.
2. (From the first real-data run of the draft 5 rules on 10 September, agreed with the above.) With
   `V_STILL` 50 mm/s and `STILL_HOLD` 500 ms no rep finds its stand, so lowering the bar afterwards
   reads as a miss. 30kg2 holds the bar overhead for about 250 ms, and 50kg3's overhead speed
   jitters up to about 150 mm/s. 150 mm/s and 200 ms find the stand on 4 of the 6 reps with a bar.
