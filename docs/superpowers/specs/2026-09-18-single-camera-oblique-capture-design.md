# Single-Camera Oblique Capture (22 August)

Date: 2026-09-18
Status: **draft — not implemented**
Supersedes: `2026-08-29-dual-capture-layout-design.md` (that document's §0, §5 and §6 are
stale against the raw data as it stands today; its §1–§4 are adopted here largely unchanged)

## Context

`data/raw/` holds two capture shapes. Every stage of `preprocess` was written against the
July shape, and a run over `22 August` produces **zero output today**.

| | `11 July` | `22 August` |
|---|---|---|
| Path shape | `<date>/<session>/<camera>/` streams | `<date>/<liftType>/<trial>/` streams |
| Cameras per lift | 2 — `Side`, `Front` | **1**, no role name |
| Camera placement | one square-on front, one square-on side | **one oblique (~45°) three-quarter view** |
| Operator `metadata.yaml` | at **session** level (parent of the capture) | at **trial** level (inside the capture) |
| Date-level `meta.yaml` template | present | absent |
| Captures | 4 | 29 (24 `Snch`, 5 `CnJ`) |
| Rig stability | per session | **one tripod, unmoved all day** (measured — §4) |

Two things are different, and the earlier design doc only addressed the first:

1. **Shape** — path depth, camera identity, and where `metadata.yaml` lives. Pure plumbing.
2. **Geometry and scene** — one camera at an oblique azimuth, in a small gym with spectators
   inside the frame and inside the near foreground. This is not plumbing: it changes what the
   floor annotation has to be, what a rectified frame means downstream, and which of S5's
   assumptions still hold.

Intended outcome: one command over `../data/raw` processes both roots, mixed layouts included,
with no new flag to get wrong — and an August capture that reaches S4 is geometrically sound,
not merely un-rejected.

---

## 0. What actually happens today (measured, 2026-09-18)

A dry run over a two-capture August tree (`CnJ/48kgCnJ1`, `Snch/107kgSnch1`):

```
processed 0 camera(s), rejected 2
- 22 August/CnJ:  session has no Side camera
- 22 August/Snch: session has no Side camera
```

**S0 Ingest passes** for both captures — discovery finds `<date>/<group>/<trial>` as a
`<date>/<session>/<camera>` triple, the stream validation is layout-agnostic, and
`record.json` is written. The first hard stop is `flow.py:_session_cut_interval`, which
requires a camera literally named `Side`. Nothing downstream of that has ever run on August
data, so every defect below S1 is latent, not observed-failing.

Where it breaks, confirmed against the current source:

| Location | Break on August |
|---|---|
| `flow.py:180` (`_session_cut_interval`) | hard-codes `(date, session, "Side")` → every group rejected |
| `flow.py:68`, `:150` (`_group_cameras`) | groups by `(date, session)` → all 24 `Snch` trials become one session's "cameras" |
| `tasks/cut.py:117` (`resolve_cut_interval`) | reads `<raw>/<date>/<session>/metadata.yaml` = `22 August/Snch/metadata.yaml`, which does not exist |
| `retilt.py:57` (`_region_prefix`) | accepts only camera names `front`/`side`; raises `no floor-region convention for camera '107kgSnch1'` |
| `tasks/retilt.py:235` | rebuilds the raw metadata path as `raw_root/date/session/metadata.yaml` — same wrong path as cut |
| `tasks/metadata.py:43` (`SESSION_NAME`) | matches `cnj_45kg_Set1`, never `110kgSnch1`; destination is `<root>/<date>/<session>/metadata.yaml` = one file for 24 trials |
| `flow.py:199` (`_sessions_without_a_camera`) | infers session-vs-camera from `scan_id` **segment count**; both layouts give 3 segments, so the inference is wrong |
| `common/models.py:208` (`CameraName`) | `Literal["Side", "Front"]` — `PoseDocument` raises `Input should be 'Side' or 'Front'` for any August capture id (verified) |
| raw `video:` blocks (all 29) | mis-keyed, y-inverted, **and pointed at a speaker cabinet rather than floor** — see §4 |

---

## 1. Re-validated data survey — what changed since 2026-08-29

The August raw tree has been partly cleaned since the earlier design doc was written. Three of
its four stated data defects are **gone**; do not re-fix them.

| Defect claimed on 2026-08-29 | Status on 2026-09-18 |
|---|---|
| Six captures share the lift window 22.40–29.57 s | **Fixed.** All 29 windows are now distinct. |
| `CnJ/82kgCnJ2`'s window overruns its own recording by 5.35 s | **Fixed.** No window exceeds its capture duration; the largest end (`82kgCnJ1`, 68.5 s) sits inside a 72.35 s recording. |
| All 29 files record `type: snch`, including the five under `CnJ/` | **Fixed.** The five `CnJ/*` files say `type: CnJ`; the 24 `Snch/*` say `type: snch`. |
| The `video:` floor block is malformed | **Still broken, and worse than described** — see §4. |

Verified clean across all 29 captures: `rgb.mp4` 1920×1440 landscape with a `creation_time`
tag, depth `uint16` 256×192, confidence `{0,1,2}` 256×192, identical `odometry.csv` /
`imu.csv` headers, bare 3×3 `camera_matrix.csv`, and `n_rgb == n_depth == n_confidence ==
n_odometry_rows` for every capture. **Stream formats need no work.**

Two captures have a short lift window — `Snch/58kgSnch2` (1.64 s) and `Snch/70kgSnch2`
(1.75 s). Both are plausible snatch durations. **Record as a manifest warning below a
configurable floor; never reject on duration alone.**

---

## 2. Identity model and layout detection

Adopted from `2026-08-29-dual-capture-layout-design.md` §1–§3 **unchanged**. Restated here
only as far as this document depends on it; that document remains the detailed reference.

- Replace the `(date, session, camera)` triple with a single `CaptureUnit` carrying `source`,
  `relative`, `capture_id`, `metadata_path`, `group_id`, `role`, `layout`. `relative` is the
  single source of truth for output paths, so "output mirrors the raw path exactly" falls out
  for both shapes with no branching in `ingest`/`cut`/`orient`/`retilt`/`crop`.
- **Layout is auto-detected per capture** by *where the operator `metadata.yaml` lives*, never
  by a name match: parent has one → `MULTI_CAMERA` (role `front`/`side` from the directory
  name); the capture itself has one → `SINGLE_CAMERA` (role `single`, its own group); neither
  → reject, naming both paths checked. Parent is tested first.
- `flow.py` groups by `group_id`; the cut-interval owner is the member with `role == "side"`
  for a multi-camera group, and the capture itself for a single-camera group.
- `resolve_cut_interval` and `retilt_camera` take `metadata_path` directly instead of
  rebuilding it from `raw_root / date / session`.
- `_sessions_without_a_camera` stops inferring group-vs-capture from segment count; the flow
  tracks surviving and rejected `group_id`s explicitly.
- The `lift_start_time_side_in_ms` / `lift_end_time_side_in_ms` names stay. They are
  identifiers, not assertions about a camera; all 29 August files already use them.
- **No new CLI flag.** `--only <glob>` against `capture_id` is promoted from "optional
  convenience" to **required scope** — a full 29-capture run is hours of re-encoding, and §4's
  annotation loop needs to re-run single captures repeatedly.

---

## 3. Camera role, view geometry, and what retilt does *not* fix

This section is new. The earlier doc asserted the August work involved "no geometry, timing,
or codec changes." That is true of the *stream formats* and false of the *scene*.

### 3.1 Role is not view

`role` (`front` / `side` / `single`) answers one question: **which fields to read out of
`metadata.yaml`**. It is a lookup key. It says nothing about where the camera was pointed.

An August capture's role is `single`; its **view is oblique** — a three-quarter view at
roughly 45° of azimuth to the lifter's frontal plane, per the capture operator. July's two
cameras are square-on front and square-on side.

### 3.2 Azimuth must be declared, not inferred

An attempt was made to recover the view azimuth from the data — detect shoulder landmarks,
back-project both shoulders through depth, de-tilt with S3's own rectifying rotation, and take
the angle of the shoulder line in the floor plane (0° = square-on front, 90° = square-on
side). Medians over ~12 sampled frames per capture: July Front **8°**, July Side **20°**,
August `107kgSnch1` **41°**, August `48kgCnJ1` **44°**. The August numbers are close to the
operator's "45°", but July Side's 20° is plainly wrong for a square-on side view, and every
interquartile range spans 40–70°. Depth at a moving athlete's shoulders is too noisy, and the
athlete rotates during the lift.

**Decision: the view is an operator annotation, with no algorithmic fallback** — the same rule
the floor region already follows, for the same reason. Add to the `video:` block:

```yaml
video:
  view: oblique            # front | side | oblique
  view_azimuth_deg: 45     # optional; degrees from square-on front, positive = camera to the lifter's left
```

Both fields are **optional and non-blocking**: a capture missing them is processed normally,
with `view: null` recorded. They are carried through `metadata.yaml` into every stage root and
into `retilt_sidecar.json` as provenance. Nothing in S0–S5 branches on them today. They exist
so the training layer can partition by view instead of guessing from a directory name, and so
the omission is visible rather than silent.

### 3.3 What survives S3, and what that means downstream

S3 Retilt removes **tilt and roll** and explicitly leaves **yaw** untouched (`4-retilt.md`
§4 — a single plane does not constrain rotation about the vertical). Therefore:

- **Valid for August today.** Anything measured along the floor normal: joint height above the
  floor, bar height, vertical velocity and acceleration, and `PoseDocument.position[1]`. The
  floor-anchored frame's *vertical* axis is well defined, and §4's measurements show it is
  recovered to better than 1.5° on 27 of 29 captures.
- **Not valid for August today.** Anything that reads the *horizontal* components as though
  they were anatomical. `PoseDocument`'s `position[0]` / `position[2]` are in a frame whose
  azimuth is the camera's, which for August is ~45° off the lifter's sagittal plane. A
  "horizontal bar displacement" computed from an August capture the way it would be from
  July's Side camera is a projection of the true sagittal displacement onto a 45°-rotated
  axis, which is wrong by an unrecorded factor and *looks* plausible.

**Decision: do not add a yaw-alignment stage in this spec.** It needs a per-capture azimuth
that is measured, not declared (§3.2 shows declaration is all we have), and it would change
the meaning of every existing July output. Instead:

- `PoseDocument` gains **`frame_azimuth: "camera"`** — an explicit statement that the
  horizontal axes are the camera's, not the athlete's. Today every capture is `"camera"`; a
  future yaw-alignment stage would emit `"sagittal"` and nothing older would silently change
  meaning.
- The restriction is stated in `1-ingestion_orient.md`'s invariant table as a scope note on
  I3, and in `PoseDocument`'s docstring alongside the existing "two cameras of one session are
  not in a common frame" note.

Named follow-up, out of scope here: **S6 Yaw alignment** — derive the platform azimuth from
the operator-annotated platform rectangle (the taped square is a known-rectangular ground
feature, visible in every August frame, and its two vanishing directions in the rectified
floor plane give the azimuth without depending on the athlete at all).

---

## 4. The floor region — the real August blocker

This section supersedes `2026-08-29-dual-capture-layout-design.md` §0 and §6. That document
identified two defects in the August `video:` blocks. There is a third, and it is the one that
matters.

All 29 files carry the identical block, byte for byte:

```yaml
video:
  front_floor_region_bottom_left_in_pixels: (0.65204, 0.66118)
  floor_region_top_right_in_pixels: (0.79080, 0.86117)
```

1. **Mismatched key prefixes** — `front_` on one key, unprefixed on the other. Neither the
   `front`/`side` convention nor §2's `single` role reads this pair.
2. **Inverted on y** — `y` runs top→bottom, so `FloorRegion.validate_extent` requires
   `y1 < y0`; here `y0 = 0.66118 < y1 = 0.86117`.
3. **The rectangle is not floor.** Rendered onto the portrait frame, the format-corrected
   rectangle lands squarely on a **PA speaker cabinet and a bundle of cables** in the near
   foreground. This was not visible from the YAML; it took drawing the rectangle on a frame.

Defect 3 is the one that decides the work. Fixing only 1 and 2 produces a capture that reaches
S3 and is then rejected — or, worse, one that squeaks past the gates with a wrong plane.

### 4.1 Measured: the current rectangle vs. real floor

Method: pool depth over frames sampled across each capture's own lift window, using the
pipeline's own `select_floor_pixels` → `backproject` → `fit_plane` → `tilt_roll_from_normal`,
with correctly rotated portrait intrinsics (`rotate_intrinsics` CW, then `depth_intrinsics`).
The method reproduces July's published `retilt_sidecar.json` values to within the sampling
difference (Side: measured tilt −8.79° / RMS 0.78 cm vs published −8.892° / 0.814 cm; Front:
−7.50° / 0.87 cm vs −8.013° / 0.498 cm).

| Capture | Rectangle | points | RMS | tilt | roll | camera height | gate |
|---|---|---|---|---|---|---|---|
| `Snch/107kgSnch1` | current, format-corrected | 10 732 | **7.12 cm** | 10.8° | **−41.2°** | 0.66 m | **reject (RMS)** |
| `CnJ/48kgCnJ1` | current, format-corrected | 37 821 | **10.26 cm** | 11.0° | **−36.0°** | 0.59 m | **reject (RMS)** |
| `Snch/107kgSnch1` | candidate A | 11 872 | 0.58 cm | −1.31° | −1.24° | 1.15 m | pass |
| `CnJ/48kgCnJ1` | candidate A | 14 596 | 1.14 cm | −1.22° | −0.89° | 1.13 m | pass |

Candidate **A** = `(0.20, 0.78)`–`(0.55, 0.68)`: the black rubber mat in the near foreground,
between the camera and the platform.

### 4.2 Measured: candidate A over all 29 captures

| | value |
|---|---|
| Captures passing every S3 gate | **27 of 29** |
| RMS residual, median / max over the 27 | 0.79 cm / 1.55 cm (gate: 2.00 cm) |
| tilt, median / range over the 27 | −1.10° / [−1.73°, −0.09°] |
| roll, median / range over the 27 | −1.07° / [−2.54°, +0.24°] |
| camera height above floor, median / range | 1.13 m / [1.09 m, 1.16 m] |
| Pooled floor points, minimum | 5 783 (gate: 500) |
| Warp-invalid border after rectification | **0–46 rows of 1920** (July, for comparison: 238–268) |

Two conclusions follow.

**The rig really was static all day.** Camera height varies by 7 cm and tilt by 1.6° across 27
captures spanning the whole session. One rectangle is therefore *geometrically* reasonable —
the camera never moved.

**Rectification is nearly free on August.** Because the camera sat almost level (tilt ≈ −1°
against July's ≈ −8°), S3's homography throws away 0–46 rows instead of July's 238–268. S4's
`crop_max_crop_fraction = 0.25` gate is nowhere near being reached, and no tunable needs
raising. (August's S4 output will be *larger* than July's — roughly 1420×1860 against July's measured
1404×1640 to 1420×1672 —
which widens the existing cross-clip dimension spread that `6-cropping.md` already flags as an
open question against invariant I5. Noted, not solved here.)

### 4.3 But the scene is not static — two captures are occluded

`Snch/50kgSnch1` and `Snch/55kgSnch1` fail with candidate A (RMS 9.09 cm and 11.05 cm). Rendering the frames shows why: **a spectator is sitting directly
in front of the camera, and their head fills the near-foreground floor patch.**

- `55kgSnch1` recovers with rectangle **D** = `(0.38, 0.78)`–`(0.62, 0.68)`: RMS 1.69 cm,
  tilt −0.92°, roll −0.81°, height 1.14 m. Clean.
- `50kgSnch1` does **not** recover. Ten trial rectangles across the whole frame were fitted;
  the best is 2.96 cm, still above the 2.00 cm gate, and most land between 3.5 cm and 8.6 cm.

This is the finding that decides the annotation policy.

### 4.4 Decisions

- **The floor region is annotated per capture**, in each trial's own `metadata.yaml`, with
  both keys **unprefixed** (role `single`), y-order correct, normalized `[0,1]`, written as
  parenthesised strings. A `side_`-prefixed spelling is accepted as an alias, because the lift
  window in the same file is already spelled `..._side_in_ms`. **No `front_` alias** — the
  existing `front_` key is an authoring mistake, not an alternate spelling.
- **One rectangle may be *copied* to all 29 as a starting point** — §4.2 justifies that — but
  copying is a starting point, not the annotation. Each capture is verified from its own
  `retilt_sidecar.json` before it is accepted.
- **Recommended starting rectangle** (measured, §4.2):
  ```yaml
  video:
    floor_region_bottom_left_in_pixels: (0.20, 0.78)
    floor_region_top_right_in_pixels: (0.55, 0.68)
  ```
- **`Snch/55kgSnch1` gets rectangle D** — `(0.38, 0.78)` / `(0.62, 0.68)`.
- **`Snch/50kgSnch1` is expected to be rejected** with `plane fit RMS ... exceeds
  retilt_max_plane_rms_m`, and that is the correct outcome. Whoever annotates should make one
  hand-picked attempt against the frame; if nothing clears 2 cm, the capture is dropped.
  28 of 29 is the expected yield. **Do not raise `retilt_max_plane_rms_m` to rescue it** —
  the gate is what caught the speaker cabinet.
- **No inheritance, no fallback, no full-frame default.** A capture whose region is absent,
  malformed, or degenerate is rejected by its exact field name, exactly as July's is.
  *Rejected alternative:* fitting the plane once per day from a clean capture and reusing it
  for the occluded ones. It is defensible given §4.2's stability, but it makes a capture's
  geometry depend on a *different* capture with nothing in the output saying so, and it would
  have silently "rescued" `50kgSnch1` with a plane nobody verified against its own pixels.
- **`FloorRegion.validate_extent` does not gain a y-swap**, and the code does not get more
  lenient anywhere. The raw files are corrected; raw stays write-once, so the correction is an
  authoring edit to `metadata.yaml`, which is operator-authored, not pipeline output.

### 4.5 New validation: day-level plane consistency (warn-only)

§4.3 produced a fit that **passed every existing gate and was still wrong**: candidate B on
`55kgSnch1` gives tilt −44.89°, roll +37.10°, height 0.72 m, RMS 0.85 cm — a tight fit to the
wrong surface, sliding under the 45° angle gate by a tenth of a degree. RMS alone cannot catch
this; it is a *fit quality* measure, not a *fit correctness* measure.

For a single-camera group whose rig is static across a capture day, there is a cheap
correctness check the multi-camera layout never offered: **the day's captures must agree with
each other.**

- After all captures in one `<date>/` have been retilted, compute the median `tilt_deg`,
  `roll_deg`, and `floor_offset_m` across them.
- Raise a **manifest warning** for any capture deviating by more than
  `retilt_group_tilt_tolerance_deg` (proposed **3°**) or
  `retilt_group_height_tolerance_m` (proposed **0.10 m**). Both defaults sit comfortably
  outside §4.2's observed spread (1.6° and 0.07 m) and comfortably inside the failure
  (44° and 0.4 m).
- Record `group_tilt_median_deg`, `group_height_median_m`, and this capture's deviations in
  `retilt_sidecar.json`.

**Warn-only, not a rejection**, following the precedent `4-retilt.md` §7 set for the
odometry-gravity check: it is a heuristic over a population, it assumes a rig that did not
move, and it should be calibrated against a second single-camera capture day before it is
allowed to fail a run. It applies to `SINGLE_CAMERA` groups sharing a date; it is skipped for
July, whose four cameras are four different poses.

---

## 5. Metadata output

Adopted from `2026-08-29-dual-capture-layout-design.md` §5, with two corrections.

- **Destination**: `root / capture.metadata_relative` for each of `config.stage_roots`. July
  keeps `<date>/<session>/metadata.yaml`; August writes
  `<date>/<liftType>/<trial>/metadata.yaml`, beside its streams. Five stage roots, not four —
  `stage_roots` already includes `crop_root`.
- **Name parsing**: add `^(?P<weight>\d+(?:\.\d+)?)kg(?P<type>[A-Za-z]+)(?P<attempt>\d+)$`
  alongside `SESSION_NAME`, matching `110kgSnch1` and `82kgCnJ2`. Same
  `derived_from_session_name` block, keyed by attempt rather than set.
- **Lift type still comes from the group folder** (`Snch` / `CnJ`), and the file's own `type:`
  is carried through separately, never overwritten. **Corrected from the earlier doc:** the
  claim that "all 29 August files record `type: snch`" is no longer true (§1) — the two now
  agree everywhere. Keep the folder as the derived value anyway, so the rule does not depend
  on a field that was demonstrably unreliable a month ago.
- `_observed()` emits `role` and `capture_id` in place of `record.camera`; the body's
  `date`/`session` keys become `capture_id`/`group_id`.
- The date-level `meta.yaml` is absent for August; `load_meta_template` already returns
  `status: "absent"`. No change.
- Carry `view` / `view_azimuth_deg` (§3.2) through when present.

---

## 6. S5 Pose under a single oblique camera

S5 is the only stage with a **hard type-level blocker**, and one substantive accuracy risk.

### 6.1 Required: `CameraName` cannot name an August capture

`common/models.py:208` declares `CameraName = Literal["Side", "Front"]`, and
`PoseDocument.camera` uses it. Constructing a document for an August capture raises
`Input should be 'Side' or 'Front'` (verified). `tasks/pose.py:130`'s
`cast(CameraName, record.camera)` hides this from `mypy` but not from pydantic at runtime.

**Decision:** `PoseDocument` carries `role` (`"side" | "front" | "single"`) plus `capture_id`,
replacing the `camera` literal. `pose_storage.pose_path` already takes the session path whole
"because captures are not a uniform depth on disk" and its docstring already names
`22 August/CnJ/82kgCnJ1` — the storage layer is ready; only the document's field is not.
Bumping `POSE_SCHEMA_VERSION` is part of this change.

### 6.2 Risk: subject selection with spectators in frame

`MediaPipePoseDetector` runs with `num_poses=1`, which makes MediaPipe return **one** pose and
gives the caller no way to tell whether it is the athlete.

Measured on portrait frames with `num_poses=5`:

| Capture | frame | candidate poses found |
|---|---|---|
| `22 August/Snch/107kgSnch1` | 1300 / 1350 / 1400 | 4 / 4 / **5** |
| `11 July/30kg_Set1/Side` | 2100 / 2200 / 2300 | 2 / 2 / 2 |

On the three August frames sampled, `num_poses=1` did pick the athlete (bbox height 0.12–0.17
of frame height, consistently the tallest candidate). But **July frame 2300 picked a
bystander** — bbox height 0.08 at `cx=0.39`, while the athlete sat at `cx=0.56` with bbox
height 0.15. So this is a **pre-existing defect that August amplifies**, not an August defect:
5 candidates instead of 2 means five times the chances to latch onto the wrong one, with no
signal in the output when it does.

**Decision: out of scope for this spec, tracked as a named follow-up** — "S5 subject
selection": detect with `num_poses > 1`, then select per capture with an explicit,
recorded rule (temporal continuity plus proximity to the operator-annotated platform region),
and write the number of candidates and the selected index into the pose output. It is a change
to S5's contract that affects July equally, and folding it into a layout-and-annotation spec
would couple two unrelated risks.

**What this spec does require:** August pose output is **not accepted on gate-passing alone**.
The acceptance check in §9 includes rendering the detected skeleton over the source video for
at least three August captures and confirming by eye that the tracked person is the lifter.

### 6.3 Unaffected

Depth back-projection, the floor-frame lift, and `floor_offset_m` all work unchanged — August
depth at the floor patch is dense and confident (§4.2: 5 783–47 256 pooled points per capture),
and `lookup_patch_depth_m`'s confidence ladder already handles the lower-confidence returns
that a 6 m working distance produces. `PlaneFit.floor_offset_m` is populated for August
exactly as for July.

---

## 7. Configuration

New tunables, all with proposed defaults pending calibration against a second single-camera
day:

| Name | Default | Purpose |
|---|---|---|
| `retilt_group_tilt_tolerance_deg` | 3.0 | §4.5 day-level consistency warning |
| `retilt_group_height_tolerance_m` | 0.10 | §4.5 day-level consistency warning |
| `cut_min_window_s` | 1.0 | §1 short-window warning (warn, never reject) |

No existing default changes. In particular `retilt_max_plane_rms_m` stays at 0.02 m (§4.4) and
`crop_max_crop_fraction` stays at 0.25 (§4.2 shows August uses ~2% of it).

---

## 8. Documentation to update

- `1-ingestion_orient.md` §2 — currently states the input is one date folder containing
  exactly `Front` and `Side`. Add the single-camera shape, and the §3.3 scope note on I3
  (yaw is not corrected; horizontal axes are the camera's).
- `2-cut.md` §§1–3 — the interval owner is chosen by role, not by the literal name `Side`.
- `4-retilt.md` §1 — the `single` role's unprefixed keys, the `view` fields, and §4.5's
  day-level consistency check in §7's validation list.
- `S5-pose-model-recommendation.md` — the `role`/`capture_id` document change (§6.1) and the
  subject-selection follow-up (§6.2).
- `README.md` — both layouts, both output shapes, run commands, and `--only`.

---

## 9. Test plan

TDD per `CLAUDE.md`; coverage floor stays 90%. Geometry is verified numerically against
synthetic fixtures, never by observing that a transform ran.

- `tests/conftest.py` — `make_camera` gains a `layout` knob building the August shape
  (`<date>/<group>/<trial>/` streams, `metadata.yaml` inside the capture);
  `make_session_metadata` gains an unprefixed-`floor_regions` mode and optional `view` fields.
- `tests/unit/test_preprocess_discovery.py` — the classification table: parent-metadata →
  `MULTI_CAMERA`; own-metadata → `SINGLE_CAMERA`; neither → rejected naming both paths checked;
  parent wins when both exist; one mixed raw root yielding both kinds in a single walk.
- `tests/unit/test_preprocess_cut.py` — a single-camera group derives its interval from its own
  `metadata.yaml` and its own `creation_time`. Every existing July rejection case still passes.
  A window shorter than `cut_min_window_s` warns and does not reject.
- `tests/unit/test_preprocess_retilt_math.py` — `role="single"` reads unprefixed keys; the
  `side_` alias resolves; `front`/`side` unchanged. Plus the three real-data regressions from
  §4: a `front_`-prefixed bottom-left paired with an unprefixed top-right on a `single` capture
  rejects naming the missing **unprefixed** field (no `front_` fallback); an inverted-y
  rectangle rejects as degenerate (no silent swap); a synthetic plane fitted at 8 cm RMS
  rejects on `retilt_max_plane_rms_m` (the speaker-cabinet case).
- `tests/unit/test_preprocess_retilt_group.py` (new) — §4.5: a group whose members agree emits
  no warning; one member at +45° tilt against the group median emits a warning **and still
  publishes**; a `MULTI_CAMERA` group is skipped entirely.
- `tests/unit/test_preprocess_metadata.py` — `110kgSnch1` and `82kgCnJ2` parse; lift type comes
  from the group folder even when the file's own `type:` disagrees; `view` fields round-trip;
  `_observed()` emits `role`/`capture_id`.
- `tests/unit/test_common_pose.py` — `PoseDocument` accepts `role="single"` with an August
  `capture_id`, and `frame_azimuth` defaults to `"camera"`.
- `tests/integration/test_preprocess_flow.py` — a second `build_capture` in the August shape
  asserting outputs land at `<date>/<liftType>/<trial>/rgb.mp4` with no camera level; one
  mixed-root run proving both layouts survive the same invocation; `metadata.yaml` present at
  `<date>/<liftType>/<trial>/metadata.yaml` under all five stage roots.

---

## 10. Verification

Gates first, from `powerflow-pipeline/`:

```bash
uv run ruff check && uv run ruff format --check
uv run mypy src
uv run pytest --cov --cov-fail-under=90
```

Then against the real data. **Expect this in two stages, not one.**

```bash
uv run powerflow preprocess --input ../data/raw \
  --records ../data/s0_ingest_output --cut ../data/s1_cut_output \
  --retilt ../data/s3_retilt_output --crop ../data/s4_crop_output \
  --output ../data/s2_orient_output --dry-run
```

**Stage 1 — layout fix, annotations untouched.** Expect `processed 33 camera(s)` reaching S3,
then **29 rejected there** with `missing floor region field:
floor_region_bottom_left_in_pixels`. That is the correct result: it proves discovery, cut and
orient handle both shapes, and isolates the remainder to the annotations. Treat `rejected 0`
at this point as a sign the raw data changed underneath the spec, not as success.

**Stage 2 — after §4's annotation pass.** Expect **32 processed, 1 rejected**, the rejection
being `22 August/Snch/50kgSnch1` on `plane fit RMS`. `31 of 33` with a *different* capture
rejected means the annotation pass introduced a rectangle nobody checked.

Per-capture checks:

```bash
# streams at the trial level, no camera directory
ls "../data/s3_retilt_output/22 August/Snch/110kgSnch1/"

# the cut interval matches the trial's own metadata.yaml
jq '{cut_start_epoch_ms, cut_end_epoch_ms, camera_creation_time}' \
  "../data/s1_cut_output/22 August/Snch/110kgSnch1/cut_sidecar.json"

# the floor fit against §4.2's measured envelope
jq '{tilt_deg, roll_deg, plane_rms_residual_m, floor_offset_m, n_floor_points,
     confidence_mode, translation_span_m, group_tilt_median_deg}' \
  "../data/s3_retilt_output/22 August/Snch/110kgSnch1/retilt_sidecar.json"
```

For an August capture, expect `tilt_deg` ≈ −1.1° ± 1°, `roll_deg` ≈ −1.1° ± 1.5°,
`floor_offset_m` ≈ 1.13 ± 0.05, `plane_rms_residual_m` < 0.016, `n_floor_points` > 5 000.
Anything outside that is a rectangle that caught something other than floor — re-annotate
rather than accept the run. A value inside the gates but outside this envelope is exactly the
§4.5 case; the manifest warning should already be pointing at it.

Then, per §6.2, render the S5 skeleton over at least three August captures and confirm the
tracked person is the lifter.

Finally, re-run the July root and diff a `retilt_sidecar.json` against the values recorded
before the change: **July output must be unchanged**. This work moves path construction and
adds an advisory check; any drift in `tilt_deg` or `plane_rms_residual_m` means geometry was
touched by accident.

---

## 11. Out of scope

- **S6 Yaw alignment** (§3.3) — rotating the floor-anchored frame into the athlete's sagittal
  plane. Needs a measured azimuth; changes the meaning of existing outputs.
- **S5 subject selection** (§6.2) — a pre-existing defect that August amplifies; affects July
  equally and belongs in its own spec.
- **Cross-clip frame dimensions** (§4.2) — August's S4 output is a different size from July's,
  widening an I5 gap `6-cropping.md` already lists as open.
- **Re-measuring the August lift windows** — they are distinct, in-range, and plausible (§1).
  The two sub-2 s windows warn; they are not corrected here.
- **The `type: snch` / folder-name reconciliation** — already consistent in the raw data (§1);
  the folder stays authoritative anyway (§5).

---

## Appendix: how the measurements were produced

Every number in §0, §1, §3.2, §4 and §6.2 was measured on 2026-09-18 against
`data/raw/11 July` and `data/raw/22 August`, using the pipeline's own functions rather than
reimplementations:

- **Floor fits** — `retilt.select_floor_pixels` → `retilt.backproject` → `retilt.fit_plane` →
  `retilt.tilt_roll_from_normal`, over depth/confidence frames sampled across each capture's
  own lift window (stride 10, capped at 32 frames), rotated CW to portrait to stand in for S2's
  output.
- **Intrinsics** — `camera_matrix.csv` read as landscape, then
  `geometry.rotate_intrinsics(..., width=1920, height=1440, rotation=CW)` and
  `retilt.depth_intrinsics(k, (1440, 1920), (192, 256))`. Using the unrotated matrix produces a
  principal point 250 px off centre and shifts every fitted tilt by ~8°; the rotated one
  reproduces July's published sidecars.
- **Warp-invalid border** — `retilt.homography` → `retilt.valid_bounds` →
  `crop.crop_fractions` on a 1440×1920 mask.
- **Pose candidates** — `pose_landmarker_full.task` via the MediaPipe Tasks API at the
  detector's own thresholds (0.05/0.05), `num_poses` 1 and 5.
- **Today's failure** — `preprocess()` invoked directly with `dry_run=True` over a two-capture
  August tree so the in-memory manifest's rejection reasons could be read (a dry run writes no
  `manifest.json`).
