# S9 Phase — Snatch and Clean & Jerk Phase Intervals from the Bar Track and Joints

Status: **draft 9** | Supersedes: [`S9-phase-intervals-draft8.md`](S9-phase-intervals-draft8.md) (feedback at its bottom) | Scope: GitHub issue [#95](https://github.com/tony-yehui-lou/PowerFlow/issues/95) (Phase Time Interval Detection) under [#94](https://github.com/tony-yehui-lou/PowerFlow/issues/94) (Phase), epic [#77](https://github.com/tony-yehui-lou/PowerFlow/issues/77) (Detection) | Sibling: [#96](https://github.com/tony-yehui-lou/PowerFlow/issues/96) Phase Issue Detection | Pipeline: **`analyze`** ([phase split](../../superpowers/specs/2026-10-22-pipeline-phase-split-design.md)) | Updated: 2026-10-23

**S9** runs in the **`analyze`** pipeline, after S8 Kinematics. In the architecture this is the
**Phase Analyzer** with its **Phase Rules**.

**Changes from draft 8:**

1. **`phase.json` publishes the bar track** (§6, new `barTrack`). Until now S9 built `h_bar` and
   `v_bar` (§1.1), measured every event against them, and threw them away. The product UI's graph
   view (`powerflow-ui/docs/analyse-screen-v5.md`) plots the bar's height over the lift's phases,
   and `powerflow-runtime` (`powerflow-runtime/docs/lift-api.md`) reads it from here. The track
   is published by S9, not S5 or S8, because S9's track is the one the events, bands and `barMm`
   values are measured on. A graph of it lines up with the phases by construction. A second copy
   built anywhere else could drift from them.
2. **`schemaVersion` 1.3.0.** It adds `barTrack` and changes nothing else, so a 1.2 reader still
   works.
3. **`method` names this draft.** The code's `METHOD_SPEC` still named draft 7 under draft 8.

**Changes from draft 7 (carried over from draft 8):**

1. **S9 lives in `analyze`** (user, 2026-10-22; was Open question 8). `preprocess` ends at S4,
   `detect` holds S5–S7, and `analyze` holds S8 and S9. S9 reads S5, S6 and S8 only from their
   published trees, and runs alone with `powerflow analyze --only s9`, so a rule change re-runs
   without loading any detector. §7's code layout moves to `analyze/`. The `phase` threshold
   block moves to `AnalyzeConfig`. `bar_geometry` moves to `common/`.
2. **v2's model stays in `analyze`** (§5.3). It is a small sequence model over per-frame
   features, not a vision model. `analyze` reads no video and runs no detector.
3. **Inputs from S8** point at `S8-joint-hva-draft4.md`. The fields S9 reads have not changed.

**Changes from draft 6 (carried over from draft 7):**

1. **The end of the lift is height alone.** E4 (snatch) and E7 (clean & jerk) are the first frame
   after the catch where the bar reaches the `overhead` band. There is no speed rule and no hold:
   30kg1 and 40kg1 never hold the bar still overhead before lowering it, and draft 6 called both
   made lifts missed (§8.1). `STILL_HOLD` is gone. This is a stopgap to be revisited (Open
   question 11).
2. **S8's knee counts only inside the knee band.** E1 takes S8's knee crossing only if the bar's
   height there is inside the `knee` band. Otherwise it falls back to the knee band's centre and
   sets the new `knee_out_of_band` flag. 50kg2's E1 came from a bad S8 knee at 303 mm, below the
   band (403–604 mm).
3. **The draft 6 run is recorded** in §8.1.

Carried over: everything else in drafts 6 and 7, including both lifts' phases and events, the other height
gates, and the rack stand's speed rule and hold.

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

- **v1, rules (§4).** Built first. Reads the bar track and the lifter's bone lengths, plus S8's knee
  height for E1.
- **v2, learned (§5).** Trained on the labels in §3. It replaces v1 for an event only when it beats
  v1 on held-out reps (§5.4).

**In:**
- S5's bar centre height per frame (`bar_pose.json`). Until that file exists, recompute it in the
  task with `bar_geometry.solve_track`, measured from S8's floor (`floor.yFloorM`);
- S8's `joint_hva.json` ([`S8-joint-hva-draft4.md`](S8-joint-hva-draft4.md) §6):
  - `bones` and `ankleHeight`: each bone's `usedM`, measured or population (§2.1);
  - the knee heights with σ and flags, for E1;
  - `frames.tMs`;
- S6's `wrist_refinement.json`: `halfGripMm`, for the overhead band;
- the capture's raw `metadata.yaml`: `lift.type` (`snch` or `CnJ`) and `lift.result`;
- for evaluation and training only: the phase labels (§3).

**Out:** one `phase.json` per capture (§6): event frames and times, phase intervals, per-event
uncertainty and source, the bands used, and flags.

**Out of scope:**
- every session but 10 September (11 July, 22 August);
- the clean and the jerk recorded as separate lifts. A `CnJ` capture is one clip holding both;
- fault detection (#96);
- the labeling UI itself. This spec fixes the label file contract (§3.2). The UI gets its own spec
  in `powerflow-ui/docs/`.

---

## 1. The events

### 1.1 Time and signals

Each event is one instant you can see on video and measure from the signals. All times are
`startEpochMs + tMs[i]`, never `i / fps`, because `frames.fps` is nominal and dropped frames leave
gaps in `tMs`.

Signals (all in S8's floor frame, heights in mm):
- `h_bar(t)` is the bar centre height and `v_bar(t)` its vertical velocity. Velocity comes from a
  zero-phase low-pass filter (default cutoff 8 Hz, see Open questions) on the height, then a central
  difference over the real `tMs` spacing.
- `h_knee(t)`: the mean of left and right knee `heightM`, weighted by `1/σ²`. Frames flagged
  `uncertain` or with σ above `JOINT_SIGMA_MAX` are left out.
- The **bands** of §2: a fixed height range per position for this lifter, the same on every frame.
  `in(band)` means `h_bar` is inside it, and `above(band)` / `below(band)` mean past its upper or
  lower edge.

Each event has a **speed rule** and a **height gate**. The speed rule finds the candidate frame. The
gate accepts or rejects it. A rejected candidate makes the event `absent`; the search does not move
on to a later candidate.

### 1.2 Snatch, and the pull both lifts share

| Event | What it is | Speed rule | Height gate |
|---|---|---|---|
| **E0 Liftoff** | the plates leave the floor | the first frame where `h_bar` is above the rest height by `LIFT_MM` and keeps rising for `LIFT_HOLD`; then step back to the last frame where `v_bar ≤ V_REST` | none: the rest height is the bar's own floor (flag only, `floor` band) |
| **E1 Bar passes knee** | the end of the first pull | none: a height crossing | the first frame after E0 where `h_bar ≥ h_knee` (S8), accepted only if `in(knee)` there. Fallback when S8's knee is unavailable, or its crossing is outside the band: where `h_bar` first reaches the `knee` band's centre |
| **E2 Full extension** | the end of the pull's upward drive, where the lifter starts pulling under (not the jerk's `drive` phase) | the frame of peak upward `v_bar` between E1 and the bar's first peak | flag only: `in(hip)` |
| **E3 Catch** | the lifter is under the bar and the bar stops descending | the first frame after the bar's first peak where `v_bar` rises through 0, after falling faster than `V_REST` | `above(hip)`: the bar is overhead. In a deep squat it sits below the standing shoulders (50kg3: 1533 mm, against a rack band reaching 1585 mm), but well above the standing hips |
| **E4 Stand** | the bar reaches overhead: the lift is done | none (draft 7 stopgap, Open question 11) | the first frame after E3 where `h_bar` is at or above the `overhead` band's lower edge |

E0, E1 and E2 are the same events in the clean & jerk, found by the same rules.

**Flags, not gates, on the pull.** E0 and E2 are reliable from speed alone. If their bar height is
outside their band, the event stands and `height_out_of_band` is set, so a bad band or a bad bar
track shows up without losing the event.

**Missed snatch.** After E3, if `h_bar` goes `below(hip)` before E4 is found, the rep counts as
missed. Then E4 is `absent`, `recovery` runs to that frame, and the `missed` flag is set. Lowering
the bar after E4 is the end of a made lift, not a miss. A `lift.result` that names a result
overrides the rules and is recorded as the source (§6).

**Why E3 is the bottom, not first contact.** In a squat snatch the lifter meets the bar and the two
ride down together. The bar's lowest point is easy to measure and repeatable. The moment of first
contact is neither. For a power snatch, the bottom and first contact are close together. This is a
default, and the Open questions revisit it.

### 1.3 Clean & jerk

After the shared pull (E0–E2), the bar is caught on the shoulders, stood up, then jerked overhead.

| Event | What it is | Speed rule | Height gate |
|---|---|---|---|
| **E3 Clean catch** | the lifter is under the bar in the front rack and the bar stops descending | as the snatch E3 | between the `hip` band's lower edge and the `rack` band's upper edge: the bar on the shoulders at any squat depth, not overhead |
| **E4 Dip start** | the jerk begins: the bar starts down from the standing rack | first find the **rack stand**: after E3, the first stretch where `|v_bar| < V_STILL` for `RACK_HOLD`. E4 is the first frame after the rack stand where `h_bar` is `DIP_MM` below the rack-stand height and still falling; then step back to the last frame where `|v_bar| ≤ V_REST` | the rack stand: `in(rack)` on every frame of its hold |
| **E5 Dip bottom** | the bottom of the dip: the bar stops going down and the drive begins | the first frame after E4 where `v_bar` rises through 0 | `h_bar` at least `DIP_MM` below the rack-stand height |
| **E6 Jerk catch** | the lifter is under the bar overhead and the bar stops descending | after E5, the bar's next peak; then the first frame after it where `v_bar` rises through 0, after falling faster than `V_REST` | `above(rack)`. A split or squat jerk catches below the standing overhead band, so `in(overhead)` would be too strict |
| **E7 Stand** | the bar reaches overhead after the jerk: the lift is done | none, as the snatch E4 | the snatch E4 rule, after E6 |

**What each phase holds:**
- `turnover` ends with the bar on the shoulders, not overhead.
- `clean_recovery` runs from the bottom of the clean, through standing up, to the end of any pause
  in the rack. The pause is not its own phase (see Open questions).
- `dip` is the bar going down from the rack: the knees bend and the bar sinks with the lifter.
- `drive` runs from the bottom of the dip, through the drive up, to the jerk catch. It holds the
  drop under the bar too: the list has no phase for it (see Open questions).
- `jerk_recovery` is standing up out of the split or squat jerk, with the bar overhead, until still.

**Missed clean & jerk.** Two ways to miss:
- **Missed clean.** After E3, `h_bar` goes `below(hip)` before the rack stand is found. E4–E7 are
  `absent`, `clean_recovery` runs to that frame, and the `missed_clean` flag is set.
- **Missed jerk.** After E6, `h_bar` goes `below(rack)` before E7. E7 is `absent`, `jerk_recovery`
  runs to that frame, and `missed_jerk` is set. If the jerk is never caught (E6 `absent` after the
  bar's peak) and the bar goes `below(rack)`, E6 and E7 are `absent`, `drive` runs to that frame,
  and `missed_jerk` is set.

Lowering the bar after E7 is the end of a made lift, not a miss. A `lift.result` that names a result
overrides both guesses, as for the snatch.

**Why E4 is the dip start, not the rack stand.** The phase list has no phase between standing up and
the dip, so the boundary goes where the jerk visibly begins. The rack stand is still computed,
because E4's search starts from it, and it is reported as an auxiliary event (§6).

---

## 2. Height bands

### 2.1 Where the expected heights come from

S8 publishes every bone length it used, measured from `metadata.yaml`'s `athlete:` block where the
field is filled, population otherwise, in `bones.<name>.usedM` and `ankleHeight.usedM`. S9 reads
those, so its bands and S8's skeleton agree on the lifter's body. Left and right bones are averaged.

| Symbol | Source |
|---|---|
| `a` | `ankleHeight.usedM` |
| `t` | mean of `leftTibia`, `rightTibia` |
| `f` | mean of `leftFemur`, `rightFemur` |
| `T` | mean of `leftTorso`, `rightTorso` |
| `u`, `l` | means of the upper arms and of the forearms |
| `w` | `shoulderWidth` |
| `g` | S6's `halfGripMm`: wrist to bar centre, measured on the hands-on-bar frames |

### 2.2 The bands

Each band is a centre and a tolerance, as fractions of the centre:

| Band | Centre | Tolerance (below / above) | Used by |
|---|---|---|---|
| `floor` | the rest height (median of the opening frames) | ±`LIFT_MM` | E0 flag |
| `knee` | `a + t` | −20% / +20% | E1 fallback |
| `hip` | `a + t + f` | −20% / +20% | E2 flag, snatch catch gate, clean catch gate, missed snatch and clean |
| `rack` | `a + t + f + T` | −15% / +10% | clean catch gate, rack stand, jerk catch gate, missed jerk |
| `overhead` | `a + t + f + T + r`, with `r = √((u + l)² − (g − w/2)²)` | −10% / +10% | the stands |

- **The overhead reach `r`.** The wrists are `g` from the bar centre and the shoulders `w/2`, so each
  arm leans out by `g − w/2`, and its vertical reach is the rest of its length. There is no hand
  term, because S6 measures `g` at the wrist.
- **Missing grip.** Without `halfGripMm`, `g = w/2` (arms vertical), the overhead band's lower
  tolerance widens to −25%, and `grip_assumed` is set.
- **Missing bones.** S8 always publishes `usedM`, measured or population. If S8's document has no
  `bones`, every band is unknown, every gate fails, and every event that needs one is `absent`.
- **Bands overlap.** The tolerances are wide and the rack band's lower edge can sit below the hip
  band's upper edge. Each gate names its own band, so an overlap never makes one rule match
  another's position.

### 2.3 Measured on 10 September (first check)

The 10 September lifter's bones are all `measured`: `a` 105 mm, `t` 398, `f` 423, `T` 515, `u` 290,
`l` 280, `w` 440.

| Band | Centre | Observed |
|---|---|---|
| `knee` | 503 mm | S8's knee standing after the lift, 30kg2: 435–515 mm |
| `rack` (shoulder) | 1441 mm | S8's shoulders standing overhead: 1428–1445 mm (30kg2), 1440–1484 mm (50kg3) |
| `overhead` | 1959 mm (30kg2, `g` 457), 1953 mm (50kg3, `g` 471) | bar standing overhead: 1964–1972 mm (30kg2), 1862–1916 mm (50kg3) |

Both reps' overhead heights fall inside ±10%. 50kg3 sits 2–5% under its centre: the arms may not be
fully locked, or the grip is wider than S6's measurement.

---

## 3. Labels

### 3.1 What gets labeled

The event times per rep, nothing per frame: five for a snatch, eight for a clean & jerk. Ordered
events can't produce phases out of order, they take seconds to label instead of minutes, and error
is measured in milliseconds per event (§8).

Labels are in **epoch ms**, so they stay valid if a capture is re-cut or its frames are renumbered.

**Corpus at draft 6:**
- **Snatch:** the 7 captures from 10 September: 30kg1, 30kg2, 40kg1, 45kg1, 50kg1, 50kg2 and
  50kg3. Each is `layout: single_camera`, one rep at about 60 fps (50kg2 reports 59.71), with S8
  `joint_hva.json` and S5 `barbell.json` on disk.
- **Clean & jerk:** none. 10 September has no `CnJ` capture. Until one is in scope, the clean & jerk
  rules are checked on synthetic curves only (§8.2).

That is 7 snatch reps, all one athlete. Enough to measure v1 and to tune its snatch thresholds and
bands. It is not enough to train v2 (§5.4).

### 3.2 File contract

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

### 3.3 Protocol

1. Label from `rgb.mp4` at full frame rate with `h_bar` and the bands drawn under the video. The
   labeler judges by eye. The curve is there to help find the frame, not to decide it.
2. Use the definitions in §1, written out in the labeling guide with one reference frame per event
   (a guide in the same style as `S5-barbell-annotation-guide.md`).
3. **Two passes, at least three days apart**, or a second labeler. The gap between passes per event
   is the **human floor**: the best accuracy any detector can be held to (§8.3).
4. Label before looking at v1's output, so the labels aren't biased toward the rules.

---

## 4. v1 — Rules

### 4.1 Starting thresholds

| Name | Value | Lift | Meaning |
|---|---|---|---|
| `LIFT_MM` | 20 mm | both | rise above the rest height that counts as off the floor |
| `LIFT_HOLD` | 83 ms (5 frames at 60 Hz) | both | how long the rise must continue |
| `V_REST` | 50 mm/s | both | speed below which the bar counts as at rest, and how fast it must fall before a bottom |
| `V_STILL` | 150 mm/s | C&J | speed below which the bar counts as still in the rack |
| `JOINT_SIGMA_MAX` | 50 mm | both | knee σ above which E1 falls back to the knee band |
| `RACK_HOLD` | 300 ms | C&J | how long the bar must be still for the rack stand |
| `DIP_MM` | 30 mm | C&J | drop below the rack-stand height that counts as the dip |

The band tolerances of §2.2 are thresholds too. `STAND_FRAC` and `RACK_TOL` are gone: the overhead
and rack bands do their job.

Holds are durations, not frame counts. The task converts them to frames with each capture's real
`tMs` cadence. All values are config, recorded in the provenance (§6), and tuned against the labels
(§8.4). The clean & jerk thresholds stay at these starting values until there are clean & jerk
labels.

### 4.2 Order and search windows

Search for each event only inside the window its neighbours leave.

Both lifts:
- E0, from the clip start;
- E2, the bar's first peak and E3, all after E0;
- E1, between E0 and the bar's first peak.

Snatch: E4, after E3 and before any miss.

Clean & jerk: the rack stand, after E3 and before any miss; E4, after the rack stand; E5, after E4;
the bar's next peak, then E6, after E5; E7, after E6 and before any miss.

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

### 4.3 Bar track gaps

S5's draft 5 measurements show the bar's yaw can't be measured through the turnover. There, B is
selected, and some frames are collar-only or `missing`. The jerk's drop under the bar is the same
kind of fast, rotating-arms moment, and is expected to have the same gaps.

- Gaps up to `MAX_GAP` (100 ms) are bridged for `v_bar` only, by linear interpolation of `h_bar`,
  and flagged `bar_gap_bridged`.
- An event that falls inside a bridged gap gets its σ widened to half the gap.
- A longer gap at an event makes that event `absent`.

### 4.4 Uncertainty

Each event reports `sigmaMs`: half the width of the window where the rule's condition is within
its tolerance (for peaks, where `v_bar` is within 5% of its peak), floored at half a frame.

---

## 5. v2 — Learned detector

Built only after v1 has been measured against the labels and there are enough labels (§5.4).

### 5.1 Features per frame

All features are relative to the rep, so different loads and athletes compare:
- `h_bar`, `v_bar` and the bar's vertical acceleration;
- **`h_bar` as a fraction of each band's centre** (§2): the knee, hip, rack and overhead ratios.
  These replace draft 5's "heights divided by the standing hip height", because they place the bar
  against every position at once;
- knee, hip, shoulder and wrist heights from S8 over the same scale, with σ as a separate input
  channel;
- knee and hip angles from S8 positions where both bones are `measured` or `fitted`;
- frame validity masks: bar `missing`, joint `uncertain`.

IMU and odometry describe the **phone**, not the lifter, so they are left out.

### 5.2 Candidates

1. **Left-to-right HMM per lift**, with one state per phase in the order of the Purpose tables:
   six for the snatch, nine for the clean & jerk. Emissions are Gaussian over the features.
   Viterbi decoding gives the boundaries directly. The pull states (`setup` through
   `second_pull`) can share emission parameters across the two lifts, so clean & jerk reps help
   train the snatch pull and the reverse. It needs very little data, and its order constraint
   matches the problem.
2. **Event-heatmap temporal network.** A small 1D dilated conv net outputs one Gaussian heatmap per
   event, decoded under the same order constraint. It needs more data than the HMM.

The HMM goes first. The network is tried only once the corpus reaches the size in §5.4.

### 5.3 Training code

`src/powerflow_pipeline/training/phase/`, next to the existing `training/` packages. Each training
run writes its corpus manifest (captures, lift, label passes used), its features configuration and
its leave-one-capture-out results. Models that pass §5.4 are registered with the Model Hub, and S9
loads them through it, never from a training output folder. The model is a sequence model over
per-frame features, not a vision model, so loading it keeps `analyze` free of video and
detectors.

### 5.4 Adoption gate

v2 replaces v1 **per lift and per event** only when, under leave-one-capture-out cross-validation:

- the median absolute error is lower than v1's;
- the 90th-percentile error is not worse; and
- that lift's corpus has at least **30 labeled reps** from at least **2 athletes**.

The shared pull events (E0–E2) may count reps from both lifts towards the 30. Until the gate is met,
S9 runs v1, and v2 is reported in the training manifest only.

---

## 6. Output

`data/s9_phase_output/<captureId>/phase.json`:

```json
{
  "schemaVersion": "1.3.0",
  "captureId": "10 September/50kg3",
  "lift": "snatch",
  "method": "docs/specs/preprocessing/S9-phase-intervals-draft9.md",
  "detector": {"liftoff": "rules-v1", "barPassesKnee": "rules-v1-fallback", "...": "..."},
  "frames": {"count": 433, "startEpochMs": 0, "tMs": [0, 17, "..."]},
  "bands": {
    "overhead": {"centreMm": 1953, "lowMm": 1758, "highMm": 2148},
    "...": "..."
  },
  "events": {
    "liftoff": {"frame": 141, "tMs": 2350, "epochMs": 0, "sigmaMs": 17, "status": "found",
                "barMm": 247},
    "stand":   {"frame": null, "tMs": null, "epochMs": null, "sigmaMs": null, "status": "absent",
                "barMm": null}
  },
  "auxiliary": {},
  "phases": [
    {"phase": "setup", "startFrame": 0, "endFrame": 141},
    {"phase": "first_pull", "startFrame": 141, "endFrame": 178}
  ],
  "barTrack": {
    "rawMm":          [226.4, 226.1, null, "..."],
    "heightMm":       [226.3, 226.2, 226.4, "..."],
    "velocityMmPerS": [-1.2, 0.4, 3.0, "..."],
    "bridgedFrames":  [2]
  },
  "result": {"value": "made", "source": "rules"},
  "flags": ["bar_gap_bridged"],
  "provenance": {"thresholds": {}, "filterCutoffHz": 8, "inputs": {"jointHva": "", "wristRefinement": ""}}
}
```

- `lift` comes from the raw `lift.type`: `snch` gives `snatch`, `CnJ` gives `clean_and_jerk`. Any
  other value is a manifest warning and no document, not a guess.
- `events` uses the keys of §3.2 for that lift, so a phase file and a label file compare key by key.
  Each event carries `barMm`, the bar's height there, so a reader can see it against the bands.
- `bands` holds every band of §2.2 that could be computed, with its centre and edges in mm.
- `auxiliary` holds events that are computed but bound no phase. For a clean & jerk this is
  `rackStand`, in the same shape as an event. For a snatch it is empty.
- `endFrame` is exclusive. Phases never overlap, and together they cover every frame from 0 to
  `count`. A phase bounded by an `absent` event is left out, and its frames belong to no phase.
- `status` is `found`, `fallback` or `absent`.
- `result.value` uses the §3.2 values for that lift. `result.source` is `metadata` or `rules`.
  The raw operator template's `lift.result` is the placeholder list `[success, fail]`; only a
  string counts. `success` reads as `made`, and for a snatch `fail` reads as `missed`.
- `flags` can be:
  - `missed` (snatch), `missed_clean`, `missed_jerk` (clean & jerk);
  - `bar_gap_bridged`, `knee_unavailable`, `implausible_duration`;
  - **`knee_out_of_band`**: S8's knee crossing was outside the knee band, so E1 used the band (§1.2);
  - **`height_out_of_band`**: a flag-only event (E0, E2) whose bar height is outside its band;
  - **`grip_assumed`**: no `halfGripMm`, so the overhead band assumed vertical arms (§2.2).
- **`barTrack`** is the bar centre's height above S8's floor, on the `frames` axis. Each array has
  `frames.count` entries, one per frame (struct-of-arrays, as the data-storage spec requires).
  Frame `i` is at `tMs[i]`. Values are rounded to 0.1, like `barMm`.
  - `rawMm` is `h_bar` as S5 and S8 give it, before §4.3 and before the filter. It is `null`
    where S5 could not place the bar, or where S8 has no floor.
  - `heightMm` is `h_bar` after gap bridging (§4.3) and the low-pass (§1.1). This is the track
    every rule reads. For every event with a frame, `barMm` equals `heightMm[frame]`. It is `null`
    where the gap was too long to bridge.
  - `velocityMmPerS` is `v_bar` (§1.1), in mm/s. It is `null` wherever `heightMm` is, and on any run
    of fewer than two frames.
  - `bridgedFrames` lists the frames whose `heightMm` came from bridging rather than S5. It is
    ascending, and its length is `provenance.bridgedFrames`.
  - With no bar track at all (no S5 output, or no floor in `joint_hva.json`), every array is all
    `null` and `bridgedFrames` is empty. The key is never left out.
- `schemaVersion` 1.2.0 added `bands` and `barMm`. 1.3.0 adds `barTrack`.

The run manifest (a Prefect markdown artifact, like every stage) reports per capture: the lift,
each event with its status and σ, the phase durations, the bands, flags, and the thresholds used.

---

## 7. Code layout

S9 is a stage of the `analyze` pipeline (`data/analyze/`). Until the phase split lands, these
files sit under `data/preprocess/`.

- `data/analyze/phase_events.py` holds the pure functions: the bands from bone lengths, signal
  prep (filter, velocity, gap bridging), each event rule, interval assembly, the plausibility
  checks. It has no I/O. The pull rules (E0–E2) are written once. Each lift adds its own rules after
  E2 and its own phase table. `Detection` carries the track `detect` built: the raw heights, the
  bridged and filtered heights, the velocity and the bridged frames, taken from `_Track` (which
  keeps its input for this). `to_document` writes them as `barTrack`. The track is not built a
  second time to publish it. `SCHEMA_VERSION` is `1.3.0`, and `METHOD_SPEC` names this draft.
- `data/analyze/tasks/phase.py` holds the `@task`. It reads, all from published trees:
  - S5's `barbell.json` (and solves the track with `common/bar_geometry.py`);
  - S8's `joint_hva.json`;
  - S6's `wrist_refinement.json`;
  - S0's `metadata.yaml`.

  It picks the lift and writes `phase.json` through staging like every stage. It runs inside
  `powerflow analyze` after S8, or alone with `--only s9` against an existing S8 tree. A capture
  with no published S5, S6 or S8 output is rejected with that reason in the manifest.
- `data/analyze/phase_eval.py` loads labels and scores `phase.json` against them (§8). The
  pipeline never imports it. It is called from a CLI subcommand of `powerflow analyze`.
- Thresholds go in `analyze/config.py` (`AnalyzeConfig`) under a `phase` block, with a
  `clean_and_jerk` sub-block for the thresholds only it uses, and a `bands` sub-block for the
  tolerances.

---

## 8. Validation

### 8.1 Real-data runs on 10 September

#### Draft 5 rules

The draft 5 rules, run read-only on the published S5 and S8 outputs:

| Capture | Bar coverage | E0–E3 | Stand with draft 5's still thresholds | With 150 mm/s and 200 ms |
|---|---|---|---|---|
| 30kg1 | 100% | found | absent: "missed" | absent: "missed" |
| 30kg2 | 100% | found | absent: "missed" | found: made |
| 40kg1 | 100% | found | absent: "missed" | absent: "missed" |
| 45kg1 | 28% | all absent | — | — |
| 50kg1 | 100% | found | absent: "missed" | found: made |
| 50kg2 | 98% | found | absent: "missed" | found: made |
| 50kg3 | 97% | found | absent: "missed" | found: made |

- **50kg3's stand is frame 290**, exactly where S5's hand-marked overhead hold starts (290–370).
  Its turnover is 200–235, against S5's hand window 205–255.
- **Two reps still read as missed.** On 30kg1 and 40kg1, the missed check used S8's shoulders. The
  draft 6 hip-band check replaces it, and §8.4 measures whether that is enough. 30kg1's
  rest-window height spread is 57 mm, against 1–3 mm on the others: its bar track is suspect in the
  opening second.
- **45kg1** has a bar on only 28% of its frames (S5 warns about this capture), so every event is
  `absent`. That is the intended behaviour.
- **50kg2's E1** comes only 15 frames after liftoff, giving a 0.25 s first pull that is flagged
  `implausible_duration`. The knee band cross-check is meant to catch this.
- The rest-window height spread is 1.1–3.4 mm on the clean captures. The synthetic noise test (§8.2)
  uses 2 mm.

#### Draft 6 rules

The draft 6 code, run read-only on the same outputs:

| Capture | Stand | Result | Bar at E1 | Flags |
|---|---|---|---|---|
| 30kg1 | absent | "missed" | 475 mm | `missed` |
| 30kg2 | frame 221, 1970 mm | made | 475 mm | — |
| 40kg1 | absent | "missed" | 491 mm | `missed` |
| 45kg1 | — | unknown (no bar) | — | — |
| 50kg1 | frame 276, 1917 mm | made | 491 mm | `implausible_duration` |
| 50kg2 | frame 386, 1927 mm | made | **303 mm** | `height_out_of_band`, `implausible_duration` |
| 50kg3 | frame 290, 1868 mm | made | 495 mm | — |

- **30kg1 and 40kg1 are made lifts.** 30kg1's bar reaches about 1950 mm overhead, then drifts down
  at 66–188 mm/s for about 250 ms before it is lowered. 40kg1's holds for about 130 ms. Neither
  meets "still below 150 mm/s for 200 ms". Draft 7's height-only stand finds both.
- **50kg2's E1 at 303 mm** is S8's knee crossing, below the knee band (403–604 mm). Draft 7's knee
  gate replaces it with the band crossing.
- **The bar heights at each event sit where the bands expect.** The catches are 1491–1663 mm, above
  the hip band. The stands are 1868–1970 mm, inside the overhead band. E2 is 1013–1116 mm, at or
  just above the hip band's 1111 mm top.

### 8.2 Synthetic (unit tests)

**Snatch.** Build `h_bar` from piecewise curves with known event times: rest, a rise accelerating
to a peak velocity, the drop under, the catch dip, the rise, then still. The bands are built from
bone lengths that put the curve's positions inside them.

**Clean & jerk.** The same pull, then a catch with the bar at shoulder height, the rise to a rack
stand, a still rack pause, the dip to its bottom, the drive to a second peak, the drop under into
the jerk catch, the rise, then still overhead.

For both, check that:
- each rule recovers its event to within one frame;
- the event order holds;
- 2 mm of height noise moves each event by no more than its `sigmaMs` (or two frames);
- a gap across a catch either widens σ (short gap) or makes the catch `absent` (long gap);
- a dropped frame (a gap in `tMs`) leaves the event *times* unchanged;
- a bar that stops below overhead never reaches the stand;
- a bar that only touches the overhead band's lower edge on its way up is a stand at that frame;
- a bar lowered after the stand leaves the lift `made`;
- an E2 outside the hip band keeps E2 and sets `height_out_of_band`;
- with S8's knee missing, E1 is the knee-band crossing, with status `fallback`;
- with S8's knee crossing outside the knee band, E1 is the knee-band crossing, with status
  `fallback` and the `knee_out_of_band` flag;
- with no `halfGripMm`, the overhead band widens and `grip_assumed` is set.

Then, per lift:
- **Snatch:** the missed-lift curve sets `missed` and leaves E4 `absent`.
- **Clean & jerk:**
  - a bar dropped from the rack sets `missed_clean` and leaves E4–E7 `absent`;
  - a bar dropped from overhead sets `missed_jerk` and leaves E7 `absent`;
  - a short rack pause (just over `RACK_HOLD`) still finds the rack stand and E4;
  - a rack pause with small re-grip bounces below `DIP_MM` doesn't trigger E4;
  - a dip shallower than `DIP_MM` finds neither E4 nor E5;
  - a clean catch with the bar overhead (above the rack band) is rejected, and so is a snatch
    catch with the bar at the hips.

The bar track (§6), on both lifts:
- every `barTrack` array has `frames.count` entries;
- for every event with a frame, `barMm == heightMm[frame]`;
- a 3-frame gap under `MAX_GAP` is `null` in `rawMm`, filled in `heightMm`, and listed in
  `bridgedFrames`. A gap longer than `MAX_GAP` is `null` in all three arrays;
- a constant-velocity synthetic bar gives `velocityMmPerS` equal to that velocity, within 1%,
  away from a run's ends;
- with no S5 output, every array is all `null` and the events are as before.

Real data: re-run S9 alone (`--stages s9 --overwrite`) over 10 September and 20 October. Every
`phase.json` is 1.3.0. Apart from `schemaVersion`, `method` and the new `barTrack`, each one
matches its 1.2.0 version key for key, since publishing the track changes no rule.

### 8.3 Against the labels: the primary real-data test

Per lift and per event, over the corpus:
- the median and 90th-percentile absolute error, in ms;
- the share of events `found` when the label is `sure`.

Each result is reported next to the **human floor** (§3.3). Starting targets, to be re-set once the
human floor is measured:

| Event | Median abs. error | 90th percentile |
|---|---|---|
| E0, E2, snatch E3, clean catch, dip bottom, jerk catch | ≤ 33 ms (2 frames at 60 Hz) | ≤ 67 ms |
| E1, dip start, stand | ≤ 67 ms | ≤ 133 ms |

At draft 6 only the snatch rows can be measured, because there are no clean & jerk labels (§3.1).

### 8.4 Tuning without overfitting

Thresholds and band tolerances are tuned with leave-one-capture-out cross-validation, the same as
§5.4. The spec records the chosen values with the cross-validated error, never the error on the
reps they were tuned on.

**Bands from the labels.** Once the labels exist, measure `h_bar` divided by the band's centre at
each labeled event. Each band's tolerance becomes the observed range plus a margin. With one
athlete, the formulas of §2.2 set the centres and the labels set the widths.

### 8.5 Consistency

- **Hand phases in S5.** The windows S5's draft 5 marked by eye on 50kg3 (Pull 150–200, Turnover
  205–255, Overhead hold 290–370) should broadly agree. That is a sanity check, not ground truth.

---

## 9. Risks

- **Seven reps, one athlete, one session, one camera position.** Thresholds and bands tuned here may
  not carry over to another lifter, another camera height or another bar speed. That is why §5.4
  asks for 30 reps from at least 2 athletes before trusting a model.
- **Bands are only as good as the bone lengths.** Measured bones (10 September) put the bands within
  a few percent (§2.3). Population bones may not, and S8's armspan check already warns that the
  10 September lifter's implied armspan (1.97 m) disagrees with the measured one (1.76 m). If the
  bands sit wrong, the gates fail and events go `absent`, which is visible but costs events.
- **Clean & jerk rules are untested on real data.** Its thresholds and the jerk grip are guesses
  until clean & jerk captures are in scope and labeled. The jerk uses a narrower grip than the
  snatch, so its overhead band sits higher. S6 measures `g` per capture, so this is covered if S6's
  grip is right.
- **Clean & jerk clips are long.** Rack pauses vary from almost nothing to several seconds, so a
  clip cut too tight may lose the jerk. The 22 August captures show the spread: their frame counts
  run from 76 (about 1.3 s, too short to hold a whole clean & jerk) to 763.
- **Pose quality.** E1 still reads S8's knee first. When it falls back to the knee band, its
  definition changes from the bent knee's actual height to the standing knee height, and the output
  says which was used.
- **The bar track is weakest at the turnover and the jerk catch.** The catches sit right where S5
  reports unmeasurable yaw and collar-only frames. §4.3 bounds the damage. If that isn't enough,
  §8.3 will show it.
- **Filter choice moves E2.** Peak velocity shifts with the low-pass cutoff. The cutoff is recorded,
  and §8.3 is rerun whenever it changes. Sharp synthetic events shorter than about 100 ms already
  land about a frame late at 8 Hz.
- **Coaching definitions differ** (see Open questions). Settling them after labeling means
  relabeling.

---

## Open questions

1. **Is there a separate transition phase?** Many coaches split a transition (scoop, double knee
   bend) between the first and second pull. Adding it later means relabeling, so decide before
   §3's first pass. It applies to both lifts.
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
8. **Stage number and home.** *Closed in draft 8:* S9, in the `analyze` pipeline after S8. The
   number is kept so the other questions keep their numbers.
9. **Labeling tool.** A powerflow-ui page (video plus `h_bar` curve and bands, keys to drop the
   events), or an existing tool such as CVAT/Label Studio exporting to §3.2.
10. **Gates or scores?** v1 uses each band as a hard gate. A softer version would score each
    candidate frame by how far it sits from the band's centre, combined with its speed score, and
    pick the best. That tolerates a band set slightly wrong, at the cost of rules that are harder to
    read. v2's band-ratio features (§5.1) already carry the soft version.
11. **What ends the lift?** Draft 7 ends it when the bar first reaches the overhead band, with no
    speed rule, as a stopgap. That calls a press-out or a bar that reaches overhead and is then
    dropped a made lift. Options: the bar's peak inside the overhead band; the bar inside the band
    for a short time, with no speed limit; or a hold measured from the labels once they exist.

---

## Feedback (2026-10-23)

1. After the lift ends, cut the rest off: keep everything from the setup through the end of
   recovery, plus a margin, and drop what follows.
2. S9 records this as a lift window. It leaves every published file untouched.
3. The `finish` phase goes.
4. For a missed lift the window ends where the bar drops out of its band, plus the margin. With no
   stand and no miss, the whole clip is kept and flagged.
5. Write the next draft with that, then implement it.
