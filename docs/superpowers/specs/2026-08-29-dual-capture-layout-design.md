# Dual Capture Layout

Date: 2026-08-29

## Context

`data/raw/` now holds two capture roots with **different directory shapes**. Every stage of
the preprocess pipeline was written against the July shape, so a run over `22 August`
produces zero output today: all 29 captures are rejected before any pixel is read.

| | `11 July` | `22 August` |
|---|---|---|
| Shape | `<date>/<session>/<camera>/` streams | `<date>/<liftType>/<trial>/` streams |
| Cameras per lift | 2 — `Side`, `Front` | 1, no role name |
| Operator `metadata.yaml` | at **session** level (parent of the capture) | at **trial** level (inside the capture) |
| Date-level `meta.yaml` template | present | absent |
| `video:` floor regions | `50kg_Set3` only | none yet — **operator will author them** |
| Captures | 4 | 29 (24 `Snch`, 5 `CnJ`) |

**Stream formats are identical** and need no work: RGB 1920×1440 @60 with a `creation_time`
tag, depth `uint16` 256×192, confidence `{0,1,2}` 256×192, same `odometry.csv` / `imu.csv`
headers, same bare-3×3 `camera_matrix.csv`. Verified by direct probe of both roots. The fix
is entirely about **path shape, camera identity, and where `metadata.yaml` lives** — no
geometry, timing, or codec changes.

Intended outcome: one command over `../data/raw` processes both roots, mixed layouts
included, with no new flag to get wrong.

### Where it breaks today

| Location | Break on August |
|---|---|
| `preprocess/flow.py:148` | hard-codes `camera == "Side"`; no August capture has that name → every group rejected |
| `preprocess/flow.py:67` | groups by `(date, session)` → all 24 `Snch` trials become one session's "cameras" |
| `preprocess/tasks/cut.py:118` | reads the lift window from `<date>/<session>/metadata.yaml` = `22 August/Snch/metadata.yaml`, which does not exist |
| `preprocess/retilt.py:57` | `_region_prefix` accepts only camera names `front`/`side` |
| `preprocess/tasks/metadata.py:43` | `SESSION_NAME` regex matches `cnj_45kg_Set1`, never `110kgSnch1` |
| `preprocess/flow.py:172` | `_sessions_without_a_camera` infers session-vs-camera from `scan_id` segment **count** — both layouts produce 3 segments, so the inference is now wrong |

## Decision

- **Layout is auto-detected per capture**, not passed as a flag — one command handles a
  mixed `data/raw`.
- **Output mirrors the raw path exactly.** August publishes to
  `<root>/<date>/<liftType>/<trial>/` with the streams at the trial level; July is
  unchanged at `<root>/<date>/<session>/<camera>/`. Stage trees therefore have two depths,
  matching their raw counterparts byte-for-byte in path shape.
- **Floor regions stay operator-authored per capture, with no fallback.** No inheritance,
  no full-frame default. S3 continues to reject a capture whose region is absent. The
  operator will add the `video:` block to each August `metadata.yaml` (exact keys in §6).

---

## 1. One identity model for both layouts

Replace the `(date, session, camera)` triple — which cannot name an August capture — with a
single model in `preprocess/models.py`, extending the existing `CameraDir`:

```
CaptureUnit
  source          Path   raw directory holding the streams
  relative        Path   raw-relative path -> the output path under every stage root
  capture_id      str    str(relative), the manifest id
  metadata_path   Path   the operator metadata.yaml governing THIS capture
  group_id        str    captures sharing one cut interval
  role            "side" | "front" | "single"
  layout          MULTI_CAMERA | SINGLE_CAMERA
```

`relative` is the single source of truth for output paths, so "mirror the raw path exactly"
falls out for both layouts with no branching in `ingest`/`cut`/`orient`/`retilt` — they all
already build destinations as `<stage_root> / record.relative` (e.g. `orient.py:159`,
`retilt.py:233`).

`CameraRecord` gains the same identity fields and drops its `date`/`session`/`camera`
string triple, keeping `camera_id`/`relative` as properties for manifest compatibility.

## 2. Layout detection — `preprocess/tasks/discover.py`

A directory is a **capture** when it contains the `REQUIRED_STREAMS` set already defined at
`ingest.py:31`. Walk each `<date>/` subtree for capture directories, then classify each by
**where its operator `metadata.yaml` lives** — file presence only, never a name match:

```
for each capture directory C:
    if (C.parent / "metadata.yaml").is_file():      -> MULTI_CAMERA
        group_id      = C.parent (the session)
        metadata_path = C.parent / "metadata.yaml"
        role          = "side" | "front" from C.name  (reject on any other name)
    elif (C / "metadata.yaml").is_file():           -> SINGLE_CAMERA
        group_id      = C itself (its own group)
        metadata_path = C / "metadata.yaml"
        role          = "single"
    else:
        reject, naming BOTH paths that were checked
```

Parent-metadata is tested **first** so that adding a stray per-camera file inside a July
camera directory can never silently split a two-camera session into two groups.

Applied to the real data: `11 July/30kg_Set1/Side` → parent `30kg_Set1/metadata.yaml`
exists → MULTI_CAMERA, role `side`. `22 August/Snch/110kgSnch1` → parent `Snch/` has no
metadata → own `metadata.yaml` exists → SINGLE_CAMERA, its own group.

Keep the existing dotfile / `INTERNAL_PREFIX` skipping (`discover.py:20`) — `data/raw` is
littered with `.DS_Store`, and staging directories must never be re-ingested.

## 3. Cut — generalise the interval owner

`flow.py` groups by `group_id` rather than `(date, session)`, and picks the interval owner
by **role** instead of the literal string `"Side"`:

- MULTI_CAMERA group → the member with `role == "side"`; reject the whole group if absent
  (unchanged behaviour, unchanged message).
- SINGLE_CAMERA group → the capture itself, its sole member.

`resolve_cut_interval` (`cut.py:113`) takes `metadata_path` and the owning record directly,
instead of rebuilding `raw_root / date / session / "metadata.yaml"`. Everything downstream
of that path — `read_lift_window`, `derive_cut_interval`, the capture-span sanity check —
is unchanged and already correct for a single camera.

The `lift_start_time_side_in_ms` / `lift_end_time_side_in_ms` field names stay as they are
in both layouts. They are identifiers, not assertions about a camera; all 29 August files
already use them. `CutInterval`'s `side_*` fields likewise keep their names.

`_sessions_without_a_camera` (`flow.py:172`) must stop inferring group-vs-capture from
`scan_id` segment count — under the August layout a 3-segment id is a capture, not a
camera. Track surviving and rejected `group_id`s explicitly in the flow instead.

## 4. Retilt — floor-region lookup by role

`_region_prefix` (`retilt.py:57`) becomes role-based rather than camera-name based:

| role | keys read from the `video:` block |
|---|---|
| `front` | `front_floor_region_bottom_left_in_pixels`, `front_floor_region_top_right_in_pixels` |
| `side` | `side_floor_region_*` |
| `single` | **unprefixed** `floor_region_bottom_left_in_pixels` / `..._top_right_in_pixels`, accepting the `side_`-prefixed spelling as an alias |

The alias exists because a single-camera capture's lift window is already spelled
`..._side_in_ms`; accepting both means whichever the operator types works. Rejection
messages keep naming the exact field that was missing.

`retilt_camera` (`retilt.py:221`) takes `metadata_path` instead of `raw_root`, deleting its
own reconstruction of the session path at `retilt.py:235`. Everything from
`read_floor_region` onward — `select_floor_pixels`, `fit_plane`, the homography, all the
§7 gates — is untouched.

## 5. Per-capture metadata output — `preprocess/tasks/metadata.py`

- **Destination**: `root / capture.metadata_relative` for each of `config.stage_roots`,
  where `metadata_relative` is `metadata_path` relative to `raw_root`. July keeps
  `<date>/<session>/metadata.yaml`; August writes
  `<date>/<liftType>/<trial>/metadata.yaml`, beside its streams — mirroring raw.
- **Name parsing**: add an August pattern alongside `SESSION_NAME` (`metadata.py:43`) —
  `^(?P<weight>\d+(?:\.\d+)?)kg(?P<type>[A-Za-z]+)(?P<attempt>\d+)$`, matching `110kgSnch1`
  and `82kgCnJ2`. Emit the same `derived_from_session_name` block, keyed by attempt rather
  than set.
- **Lift type comes from the group folder** (`Snch` / `CnJ`), not the metadata `type:`
  field. **All 29 August files record `type: snch`, including the five under `CnJ/`** — the
  field is unreliable. Record the folder-derived value as the derived one and carry the
  file's own value through separately rather than overwriting it; nothing is silently
  corrected.
- The date-level `meta.yaml` template is absent for August. `load_meta_template`
  (`metadata.py:46`) already treats that as `status: "absent"`, not an error — no change.

## 6. What the operator must author (August)

Into each `data/raw/22 August/<liftType>/<trial>/metadata.yaml`, appended to the existing
`lift:` / `athlete:` blocks:

```yaml
video:
  floor_region_bottom_left_in_pixels: (x0, y0)
  floor_region_top_right_in_pixels: (x1, y1)
```

Constraints enforced by `FloorRegion` (`models.py:138`) and `4-retilt.md` §1:

- Normalized `[0,1]` fractions, **not** pixels — the `_in_pixels` suffix is legacy and
  wrong. A value outside `[0,1]` is rejected, never reinterpreted.
- Written as **parenthesised strings**; `yaml.safe_load` returns `"(0, 1)"` and
  `parse_region_point` splits it. Do not write a YAML list.
- `y` runs top→bottom, so `y0` (bottom-left) must be **greater** than `y1` (top-right), and
  `x0 < x1`. A degenerate rectangle is rejected.
- Measured on the **portrait** frame — S3 runs on S2's rotated output, not the raw
  landscape image.
- One rectangle over a clean patch of floor, sized to exclude the lifter, barbell, plates
  and walls: `fit_plane` is a plain least-squares solve with no outlier rejection, so
  confident non-floor depth inside the rectangle tilts the plane.

Also worth an operator pass while editing: `11 July/30kg_Set1/metadata.yaml` currently has
no `video:` block, though the 2026-08-26 manifest shows it retilted successfully — the
block appears to have been removed since. It needs restoring for July to run whole.

## 7. CLI and docs

`cli.py` needs **no new flag** — layout is auto-detected. Optionally add `--only <glob>`
matching against `capture_id`, so a single August capture can be run without the other 28;
a full 29-capture run is hours of re-encoding. Marked optional: it is a convenience, not
part of the layout fix.

Update, matching the code: `README.md` (both layouts, both output shapes, run commands),
`docs/specs/preprocessing/1-ingestion_orient.md` §2 (currently states the input is one date
folder with exactly `Front` and `Side`), `2-cut.md` §§1–3 (interval owner by role), and
`4-retilt.md` §1 (the `single` role's unprefixed keys).

## 8. Tests

TDD per `CLAUDE.md`; coverage floor stays 90%.

- `tests/conftest.py` — `make_camera` gains a `layout` knob that builds the August shape
  (`<date>/<group>/<trial>/` streams, `metadata.yaml` inside the capture);
  `make_session_metadata` gains an unprefixed-`floor_regions` mode.
- `tests/unit/test_preprocess_discovery.py` — the classification table: parent-metadata →
  MULTI_CAMERA; own-metadata → SINGLE_CAMERA; neither → rejected naming both paths;
  parent-wins when both exist; a mixed raw root yielding both kinds in one walk.
- `tests/unit/test_preprocess_cut.py` — a single-camera group derives its interval from its
  own metadata and its own `creation_time`. Existing July rejection cases must all still pass.
- `tests/unit/test_preprocess_retilt_math.py` — `role="single"` reads unprefixed keys, the
  `side_` alias resolves, `front`/`side` are unchanged.
- `tests/unit/test_preprocess_metadata.py` — `110kgSnch1` / `82kgCnJ2` parse; lift type
  comes from the group folder even when the file says `snch` under `CnJ/`.
- `tests/integration/test_preprocess_flow.py` — a second `build_capture` in the August
  shape asserting outputs land at `<date>/<liftType>/<trial>/rgb.mp4` (no camera level),
  plus one mixed-root run proving both layouts survive the same invocation.

---

## Verification

Run from `powerflow-pipeline/`. Gates first:

```bash
uv run ruff check && uv run ruff format --check
uv run mypy src
uv run pytest --cov --cov-fail-under=90
```

Then against the real data — **dry-run first**, which validates every capture and writes
nothing:

```bash
uv run powerflow preprocess --input ../data/raw \
  --records ../data/s0_ingest_output --cut ../data/s1_cut_output \
  --retilt ../data/s3_retilt_output --output ../data/s2_orient_output --dry-run
```

Expect `processed 33 camera(s), rejected 0` — 4 July + 29 August — assuming the `video:`
blocks are in place. Then confirm nothing was misclassified:

```bash
jq '.rejected_scans' ../data/s0_ingest_output/manifest.json      # [] expected
jq -r '.scans[].scan_id' ../data/s0_ingest_output/manifest.json  # both layouts present
```

A real run over all 33 is hours of re-encoding. Prove the August path end-to-end on one
capture first (with `--only`, or by pointing `--input` at a temporary tree holding a single
trial), then check the shape and the numbers:

```bash
# streams at the trial level, no camera directory
ls ../data/s3_retilt_output/22\ August/Snch/110kgSnch1/

# the cut interval matches the trial's own metadata.yaml
jq '{cut_start_epoch_ms, cut_end_epoch_ms, camera_creation_time}' \
  ../data/s1_cut_output/22\ August/Snch/110kgSnch1/cut_sidecar.json

# the floor fit is sound: RMS well under 0.02, tilt/roll plausible, camera static
jq '{tilt_deg, roll_deg, plane_rms_residual_m, n_floor_points, confidence_mode, translation_span_m}' \
  ../data/s3_retilt_output/22\ August/Snch/110kgSnch1/retilt_sidecar.json
```

`plane_rms_residual_m` near the 0.02 ceiling, `confidence_mode: "conf1_and_2"`, or
`n_floor_points` barely over 500 all point at a floor rectangle that caught something that
is not floor — re-annotate rather than accept the run.

Finally, re-run the July root and diff a `retilt_sidecar.json` against the values recorded
before the change: **July output must be bit-identical**. The refactor moves path
construction only, and any drift in `tilt_deg` or `plane_rms_residual_m` means geometry was
touched by accident.

## Not in scope

Annotating the 29 floor regions (operator task, §6); S4 Crop; the `type: snch` values in
the August files (surfaced in the derived block, never rewritten — raw is write-once).
