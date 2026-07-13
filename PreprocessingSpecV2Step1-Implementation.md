# Preprocessing V2 — Step 1 (S0 + S1) · Technical Implementation Spec

Status: **executable** | Source: `PreprocessingSpecV2Step1.md` (draft design spec) | Written: 2026-07-13

> This document says **exactly** how to build S0 and S1. Every number in §2 was measured from
> `data/raw/`, not assumed. Where the source spec and the data disagree, the data is reported and the
> conflict is filed as a labelled issue in §11 — it is **not** silently resolved.
>
> Scope is **S0 and S1 only**. The source spec's invariant table promises I2–I5 but defines no stages
> for them (`ISSUE-14`). This spec establishes **I1** and nothing else.

---

## 1. What the two stages must deliver

| # | Invariant | Established by | In scope here |
|---|---|---|---|
| I1 | Every image is **portrait**-oriented, in the frame its intrinsics describe. | S1 | ✅ |
| I2 | Frame `k` of every stream refers to the same instant; `t=0` is a common origin. | S2 | ❌ undefined |
| I3 | Images undistorted and rectified as if level. | S3 | ❌ undefined |
| I4 | Scale constant across clips. | S4 | ❌ undefined |
| I5 | Identical pixel dimensions, same physical region. | S5 | ❌ undefined |

**Rejection is the only failure mode.** If a stage cannot establish its invariant for a camera, that
camera's session is rejected and recorded — never emitted degraded. A silently mis-oriented clip is
worse than a missing one.

---

## 2. Ground truth (measured from `data/raw/`, 2026-07-13)

### 2.1 Directory layout — as it actually is

```
data/raw/
  9 July/                       # date folder
    meta.yaml                   # NOT metadata.yaml; see ISSUE-04, ISSUE-05
    cnj_45kg_Set1/              # session folder — ABSENT FROM THE SOURCE SPEC (ISSUE-04)
      Front/                    # camera folder
        rgb.mp4
        depth/000000.png …      # zero-padded, 6 digits
        confidence/000000.png …
        camera_matrix.csv
        odometry.csv
        imu.csv
      Side/
    cnj_55kg_Set1/
      Front/
      Side/
```

The source spec says the date folder contains the two positional folders directly. It does not: a
**session** level (one per lift) sits between them, and the session name encodes the lift
(`cnj_45kg_Set1` → type `cnj`, 45 kg, set 1).

### 2.2 Stream properties

| Stream | File | Geometry | dtype | Notes |
|---|---|---|---|---|
| RGB | `rgb.mp4` | **1920×1440 landscape** | H.264 | No rotation metadata. fps varies per camera. |
| Depth | `depth/NNNNNN.png` | 256×192 landscape | `uint16` | Observed 150–4125; assumed millimetres (`ISSUE-13`). |
| Confidence | `confidence/NNNNNN.png` | 256×192 landscape | `uint8` | Values exactly `{0, 1, 2}` ✓ as specified. |
| Odometry | `odometry.csv` | — | — | `timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y` |
| IMU | `imu.csv` | — | — | `timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z`; ≈125 Hz (≈2× frame rate). |

RGB and depth share a 4:3 aspect (1920×1440 and 256×192), so depth is a uniformly scaled-down
landscape view of the same frustum. Scale factor `s = 256/1920 = 192/1440 = 0.13333`.

### 2.3 Per-camera counts — **RGB is always exactly one frame short**

| Session / camera | RGB frames | depth | confidence | odometry rows | fps |
|---|---|---|---|---|---|
| cnj_45kg_Set1/Front | 4335 | 4336 | 4336 | 4336 | 58.57 |
| cnj_45kg_Set1/Side | 5932 | 5933 | 5933 | 5933 | 59.99 |
| cnj_55kg_Set1/Front | 5486 | 5487 | 5487 | 5487 | 59.97 |
| cnj_55kg_Set1/Side | 6257 | 6258 | 6258 | 6258 | 60.00 |

Depth = confidence = odometry in every camera. RGB is short by **exactly 1**, systematically. This is
a real, reproducible property of the capture, not decoder noise (`ISSUE-03`).

### 2.4 Intrinsics — two disagreeing sources, both landscape-centred

`camera_matrix.csv` (bare 3×3, no distortion coefficients):

```
1402.2333, 0.0,       968.371
0.0,       1402.2333, 719.6846
0.0,       0.0,       1.0
```

`odometry.csv` also carries **per-frame** `fx, fy, cx, cy` — which *drift* (`fx: 1373.3534 →
1374.2737 → …`) and disagree with the static file by ≈29 px of focal length (`ISSUE-02`).
`distortion_center_x/y` are **empty in every row**; no distortion coefficients exist anywhere
(`ISSUE-06`).

**Principal point vs image centre** — recorded here because S1's correctness rests on it:

| | `cx` | `cy` |
|---|---|---|
| Reported `K` | 968.37 | 719.68 |
| Landscape centre (1920×1440) | 960.0 → **off by 8.4 px** | 720.0 → **off by 0.3 px** |
| Portrait centre (1440×1920) | 720.0 → off by 248.4 px | 960.0 → **off by 240.3 px** |

**Decision in force:** `K` is treated as **portrait-framed and passed through untouched**, per the
source spec and the project owner's explicit instruction. The measurement above contradicts that
premise and is filed as **`ISSUE-01` (critical)**. This spec implements the untouched-`K` behaviour;
it does not attempt to reconcile the contradiction.

For the record, the source spec's "real check" (back-project depth, fit floor, compare to gravity)
was executed on four frames and **cannot discriminate** — a principal-point error only shears the
cloud, so the floor stays near gravity either way: median 3.5° (landscape hypothesis) vs 2.1°
(portrait hypothesis). It is not usable as a tiebreaker.

---

## 3. Data contract

Per `CLAUDE.md` (Data Repository): data files are **write-once by one task, read-many by others**.

| | Path |
|---|---|
| Input (S0) | `data/raw/<date>/<session>/<camera>/` |
| S0 output | `data/s0_ingest_output/<date>/<session>/<camera>/record.json` |
| S1 input | `data/raw/…` (pixels) + `data/s0_ingest_output/…` (validated record) |
| S1 output | `data/s1_orient_output/<date>/<session>/<camera>/` |

`CLAUDE.md` currently states the data root as `../data`, the input as `../data/input/`, and outputs as
`../data/{task}_output`. The actual root is the in-repo `./data/` and the input is `./data/raw/`
(`ISSUE-08`). This spec uses the paths above; adjust once that is settled.

**S1 output per camera:**

```
rgb.mp4                 # portrait 1440x1920, re-encoded (ISSUE-11)
depth/NNNNNN.png        # portrait 192x256, uint16, lossless
confidence/NNNNNN.png   # portrait 192x256, uint8, values {0,1,2}, lossless
camera_matrix.csv       # BYTE-FOR-BYTE COPY of the input. Never rewritten.
odometry.csv            # copied verbatim
imu.csv                 # copied verbatim
sidecar.json            # provenance: rotation applied, K passthrough attestation, counts, preflight verdicts
```

---

## 4. Module layout

Follows the Prefect structure documented in `CLAUDE.md`:

```
src/powerflow_pipeline/data/preprocess/
  flow.py          # @flow, orchestration only
  config.py        # PreprocessConfig (pydantic)
  models.py        # Intrinsics, CameraRecord, SessionRecord, LiftMeta, AthleteMeta
  tasks/
    discover.py    # discover_sessions
    ingest.py      # S0: probe_streams, validate_camera, reconcile_meta
    orient.py      # S1: rotate_rgb, rotate_depth, rotate_confidence, copy_passthrough, preflight
```

**Reuse from `data/common/` — do not reimplement:**

| Need | Use |
|---|---|
| Run settings, output mode, `destination_for` | `common/context.py` → `RunContext`, `OutputMode` |
| Reject one camera without failing the run | `common/errors.py` → `ScanRejected` |
| Staging, atomic publish, in-place commit, `write_json` | `common/filesystem.py` |
| Audit record + Prefect markdown artifact | `common/manifest.py` → `RunManifest`, `emit_manifest` |
| Manifest records | `common/models.py` → `ScanOutcome`, `RejectedScan`, `FileOp`, `StepResult` |

**`common/discovery.py:discover_scans` does NOT fit this layout.** It looks for `meta.json` +
`frames/` in immediate child directories — a shape this capture does not have. S0 needs its own
`discover_sessions` task walking `<date>/<session>/<camera>`. Do not bend `discover_scans` to cover
both; a scan and a camera-session are different things.

---

## 5. Configuration

```python
class RotationDirection(StrEnum):
    CW = "cw"    # cv2.ROTATE_90_CLOCKWISE
    CCW = "ccw"  # cv2.ROTATE_90_COUNTERCLOCKWISE

class PreprocessConfig(BaseModel):
    rotation: RotationDirection            # NO DEFAULT — must be set explicitly. See ISSUE-09.
    frame_count_tolerance: int = 1         # RGB may be this many frames short of depth/conf (§2.3)
    depth_scale_mm: float = 1.0            # uint16 -> millimetres (ISSUE-13)
    min_confidence: Literal[0, 1, 2] = 2   # used by preflight only; S1 does not filter pixels
    rgb_crf: int = 16                      # H.264 quality for the re-encode (ISSUE-11)
    require_stopwatch_attestation: bool = True   # ISSUE-07
```

`rotation` intentionally has **no default**: picking one silently is the exact failure the source spec
warns about. The flow refuses to start without it.

---

## 6. Data contracts (`models.py`)

```python
class Intrinsics(BaseModel):
    fx: float; fy: float; cx: float; cy: float
    distortion: list[float] | None = None    # always None for this capture app (ISSUE-06)
    frame: Literal["portrait", "landscape"]  # recorded, per ISSUE-01; set to "portrait" today

class StreamCounts(BaseModel):
    rgb: int; depth: int; confidence: int; odometry: int; imu: int

class CameraRecord(BaseModel):
    date: str; session: str; camera: str      # "9 July", "cnj_45kg_Set1", "Front"
    source: Path
    rgb_width: int; rgb_height: int; fps: float
    depth_width: int; depth_height: int
    counts: StreamCounts
    n_frames: int                             # the aligned count actually usable (§7.3)
    intrinsics: Intrinsics                    # from camera_matrix.csv
    odometry_intrinsics_drift: float          # max |fx_odom - fx_static| over frames (ISSUE-02)
    stopwatch_legible: bool | None            # ISSUE-07

class SessionRecord(BaseModel):
    date: str; session: str
    lift: LiftMeta; athlete: AthleteMeta
    cameras: list[CameraRecord]
```

---

## 7. S0 · Ingest, validate & update

**In:** `data/raw/<date>/`. **Out:** one `record.json` (`CameraRecord`) per camera, plus a reconciled
`meta.yaml`. **Establishes no invariant** — it is the gate that lets S1 assume its inputs exist.

### 7.1 Algorithm

1. **Discover** (`discover_sessions`): for each `<date>/<session>/<camera>` directory, skipping names
   starting with `.` (and `.powerflow-` staging dirs, per `common/filesystem.INTERNAL_PREFIX`).
2. **Probe RGB** (`av`): open `rgb.mp4`, read `width`, `height`, `average_rate`, and count decoded
   frames. Counting requires a full decode pass — cache the count in `record.json` so S1 never repeats it.
3. **Count depth / confidence**: `sorted(glob("*.png"))`; assert filenames are a contiguous
   zero-padded range `000000 … N-1`.
4. **Read `camera_matrix.csv`** → `Intrinsics` (3×3; `fx=K[0,0]`, `fy=K[1,1]`, `cx=K[0,2]`, `cy=K[1,2]`).
5. **Read `odometry.csv`** → row count, and `max|fx_row − fx_static|` for the drift field.
6. **Read `imu.csv`** → row count.
7. **Validate** (§7.2). First failure raises `ScanRejected`; the flow records a `RejectedScan` and
   continues with other cameras.
8. **Reconcile `meta.yaml`** (§7.4).
9. **Emit** `record.json` via `common/filesystem.write_json`.

### 7.2 Validation rules — exact conditions and reason strings

| # | Condition to reject | Reason string (verbatim into `RejectedScan.reason`) |
|---|---|---|
| V1 | any of `rgb.mp4`, `depth/`, `confidence/`, `camera_matrix.csv`, `odometry.csv`, `imu.csv` missing | `missing required stream: {name}` |
| V2 | `depth/` or `confidence/` empty | `empty required stream: {name}` |
| V3 | `n_depth != n_confidence` | `depth/confidence frame count mismatch: {n_depth} vs {n_conf}` |
| V4 | depth PNG shape != confidence PNG shape (checked on frame 0 **and** frame N−1) | `depth/confidence resolution mismatch: {d_shape} vs {c_shape}` |
| V5 | `n_odometry != n_depth` | `odometry/depth frame count mismatch: {n_odom} vs {n_depth}` |
| V6 | `not (0 <= n_depth - n_rgb <= frame_count_tolerance)` | `rgb frame count outside tolerance: rgb={n_rgb} depth={n_depth} tolerance={tol}` |
| V7 | `camera_matrix.csv` unparseable or not 3×3 | `intrinsics absent or malformed` |
| V8 | `fx <= 0 or fy <= 0` | `intrinsics absent or malformed` |
| V9 | confidence PNG contains values outside `{0,1,2}` (frame 0 and N−1) | `confidence values outside {{0,1,2}}: {found}` |
| V10 | depth PNG dtype is not `uint16` | `unexpected depth dtype: {dtype}` |
| V11 | `require_stopwatch_attestation` and `stopwatch_legible` is not `True` in `meta.yaml` | `stopwatch not attested legible for camera {camera}` |

V6 encodes §2.3: today every camera is short by exactly 1, so the default tolerance of 1 admits the
real data while still catching a genuinely truncated stream. **`n_frames = min(n_rgb, n_depth)`** and
frames `[0, n_frames)` are the usable range — see `ISSUE-03` for why "the first N align" is an
assumption, not a fact.

Rejection granularity is **per camera**. A session with one bad camera keeps its good camera and
records the rejection; S2 (unwritten) decides whether a one-camera session is usable.

### 7.3 Session-level checks

Reject the **session** (all cameras) if fewer than one camera survives, with reason
`session has no usable camera`.

### 7.4 `meta.yaml` reconciliation

The source spec says: *"Read the metadata of `rgb.mp4`, and check with the metadata in
`metadata.yaml` … Update all information into `metadata.yaml`."* This is **not currently executable**:
the file is named `meta.yaml`, it contains a **type schema rather than values**
(`weight_in_kg: float`), and it holds a single `lift:` block while the date folder holds two lifts
(`ISSUE-05`). There is nothing to check `rgb.mp4` against.

Until that is resolved, S0 implements the following and **does not invent values**:

1. Parse `meta.yaml`. If every leaf is a type name rather than a value, treat it as an **uninstantiated
   template** and record `meta_status = "template"` in the record. Do not fail.
2. Write the **observed, measured** facts into a new per-session file
   `data/s0_ingest_output/<date>/<session>/meta.yaml`, never mutating the input (write-once rule):
   `cameras[].{rgb_width, rgb_height, fps, n_frames, counts, intrinsics}`.
3. Leave `lift` / `athlete` fields as `null` where the template gives no value. Session-name-derived
   values (`type=cnj`, `weight_in_kg=45`) are recorded under `derived_from_session_name` **and marked
   as derived**, never merged into `lift` as if measured.

---

## 8. S1 · Orientation repair → establishes **I1**

**In:** landscape RGB + depth + confidence, and the S0 record. **Out:** all three portrait. **`K` is
copied verbatim.**

### 8.1 The rotation is exact — no interpolation is involved

The source spec calls for **nearest-neighbour** on depth and confidence, to avoid fabricating depths
across edges and inventing a confidence `1` between `0` and `2`. A 90° rotation is a **transpose plus
a flip**: every output pixel *is* an input pixel, moved. No resampling kernel runs at all, so the
requirement is satisfied **by construction**.

Therefore:

- **Use** `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` (or `..._COUNTERCLOCKWISE`).
- **Never** use `cv2.warpAffine`, `cv2.resize`, or any `INTER_*` flag on these streams. A "90° rotation"
  expressed as a warp *does* interpolate, and would silently violate the requirement.

Pixel mapping, for the tests in §9 to assert exactly (source `W×H` → destination `H×W`):

| Direction | Destination coordinate of source `(u, v)` |
|---|---|
| CW | `u' = H − 1 − v`, `v' = u` |
| CCW | `u' = v`, `v' = W − 1 − u` |

Applied to the real geometry: RGB `1920×1440 → 1440×1920`; depth and confidence `256×192 → 192×256`.

### 8.2 Algorithm

1. **RGB**: decode `rgb.mp4` with `av`; rotate each frame; encode to portrait `rgb.mp4` (H.264,
   `crf = rgb_crf`, same fps as the source). Stream frame-by-frame — never hold 6000 frames of
   1440×1920 in memory.
2. **Depth**: for each `NNNNNN.png`, `cv2.imread(..., IMREAD_UNCHANGED)` → rotate → `cv2.imwrite`.
   Assert `uint16` in and `uint16` out.
3. **Confidence**: same, `uint8`. Assert the output value set is a subset of `{0,1,2}`.
4. **Same direction for all three streams, and for every camera.** Rotating RGB one way and depth the
   other leaves them mirror-misaligned while each looks individually correct — the exact silent
   failure the source spec calls out.
5. **`camera_matrix.csv`**: byte-for-byte copy. `odometry.csv`, `imu.csv`: verbatim copies.
6. **Sidecar**: write `sidecar.json` recording `rotation`, `k_rewritten: false`, input/output
   dimensions, `n_frames`, and the preflight verdicts (§8.3).

Only frames `[0, n_frames)` are emitted, per §7.2.

### 8.3 Preflight — run once per device, record the verdict

Both checks come from the source spec. They are recorded, not silently trusted.

- **Cheap check.** Assert `cx < W_portrait` and `cy < H_portrait`, and report the distance from
  `(cx, cy)` to the portrait centre. **On today's data this check fails its intent** — the principal
  point sits on the *landscape* centre (§2.4). Per the standing decision the pipeline proceeds and
  records `k_frame_verdict: "portrait (asserted, contradicted by data — ISSUE-01)"`.
- **Gravity check.** Back-project depth through `K`, fit the floor plane, compare its normal to
  ARKit's gravity (`+Y` up in the odometry world frame). Measured: **inconclusive** (§2.4). Record the
  angle; do not gate on it.

### 8.4 Exit validation

1. All three streams are portrait: `H > W` (RGB `1920 > 1440`; depth/confidence `256 > 192`).
2. Depth and confidence remain **pixel-identical in shape** to each other, and `{0,1,2}` is preserved.
3. RGB and depth **agree in normalized coordinates**: a landmark at `(u/W, v/H)` in portrait RGB lands
   at the same normalized position in portrait depth. The source spec asks for pixel-for-pixel
   agreement, which is impossible between 1440×1920 and 192×256 (`ISSUE-10`); normalized agreement is
   the checkable form of the same claim, and holds because both streams share a 4:3 frustum (§2.2).
4. `camera_matrix.csv` output hash == input hash.

---

## 9. Test plan (TDD — failing test first, coverage ≥ 90%)

Geometry is verified **numerically on synthetic fixtures**, never by eyeballing that a transform ran.

**Fixtures** (`tests/conftest.py`): a synthetic camera directory whose depth/confidence frames carry an
**asymmetric marker** — e.g. a single hot pixel at `(u, v) = (5, 3)` in a 256×192 depth map, and a
matching RGB block. Asymmetric so that a wrong rotation direction, a transpose, or a flip each produce
a *different* wrong answer.

| Test | Asserts |
|---|---|
| `test_cw_rotation_maps_pixels_exactly` | hot pixel `(5,3)` → `(H−1−3, 5) = (188, 5)`; shape `256×192 → 192×256` |
| `test_ccw_rotation_maps_pixels_exactly` | `(5,3)` → `(3, 250)` |
| `test_depth_dtype_and_values_survive` | output `uint16`; value set identical to input (no new values invented) |
| `test_confidence_stays_in_0_1_2` | output value set ⊆ `{0,1,2}`; no `1` appears where input had only `0`/`2` |
| `test_rgb_depth_confidence_share_one_direction` | marker lands at the same **normalized** coords in all three |
| `test_camera_matrix_is_byte_identical` | S1 output hash == input hash (guards `ISSUE-01`'s decision) |
| `test_portrait_after_rotation` | `H > W` for all three streams |
| `test_v1…v11_reject_reasons` | one test per rule in §7.2; each asserts the **exact** reason string |
| `test_rgb_short_by_one_is_accepted` | tolerance 1 admits the real 4335/4336 case |
| `test_rgb_short_by_two_is_rejected` | tolerance 1 rejects 4334/4336 |
| `test_n_frames_is_min_of_rgb_and_depth` | aligned range is `[0, 4335)` for the real Front camera |
| `test_flow_emits_manifest_artifact` | integration, under `prefect_test_harness`: rejected + processed cameras both appear in the `RunManifest` artifact |

---

## 10. Acceptance criteria for Step 1

- [ ] Every emitted RGB, depth, and confidence frame is portrait (`H > W`).
- [ ] Depth and confidence are bit-exact rotations of their inputs — no interpolation, no new values.
- [ ] All three streams rotate the same way, in every camera, in every session.
- [ ] `camera_matrix.csv` leaves S1 byte-identical to how it entered.
- [ ] Every rejected camera appears in the run manifest with an exact reason string; no camera is
      emitted degraded.
- [ ] The manifest artifact records `rotation`, `k_rewritten: false`, and both preflight verdicts.

---

## 11. Issues

Each must be closed by a human decision. `ISSUE-01`, `ISSUE-03`, `ISSUE-06`, and `ISSUE-14` block
downstream stages; the rest block or degrade Step 1 itself.

### `ISSUE-01 · K-FRAME-CONTRADICTION` — **critical, blocks S3/S4/S5**
S1's central instruction is "rotate the pixels, keep `K`", resting on the premise that `K` describes
the *portrait* frame. The data contradicts it: `cy = 719.68` sits **0.3 px** from the landscape centre
(720.0) and **240 px** from the portrait centre (960.0); `cx` likewise (8.4 px vs 248.4 px). The
per-frame intrinsics in `odometry.csv` agree. A lens decentred by 12.5% of frame height is not a real
phone camera; ARKit reports `camera.intrinsics` in the **native landscape buffer** regardless of how
the device is held. The source spec's own cheap check says that when `(cx, cy)` only makes sense
against the landscape dimensions, *"the premise is false and `K` does need the rewrite."*
**Standing decision (project owner): keep `K` untouched.** Implemented as such. If the premise is in
fact false, S3/S4/S5 are silently wrong with no visible symptom in the images. The gravity check
cannot settle it (median 3.5° vs 2.1°, inconclusive). **Resolve before any geometric stage is built.**
If it later flips: CW rewrite is `cx' = H−1−cy = 719.3`, `cy' = cx = 968.4`.

> K is landscape, please rotate also.

### `ISSUE-02 · INTRINSICS-TWO-SOURCES` — high
`camera_matrix.csv` gives a static `fx = fy = 1402.2333`. `odometry.csv` gives **per-frame** `fx` that
starts at `1373.3534` and drifts (`1374.2737`, …) — ≈29 px lower and *not constant*. Nothing says
which is authoritative, and a time-varying `K` would force every downstream geometric step to accept a
per-frame matrix. This spec uses `camera_matrix.csv` and records the drift; **confirm.**

> Find the average of the per-frame fx, and then use that instead.

### `ISSUE-03 · RGB-FRAME-COUNT-OFF-BY-ONE` — high, affects I2
RGB is short by exactly one frame in **all four cameras** (4335/4336, 5932/5933, 5486/5487,
6257/6258). Depth, confidence, and odometry always agree with each other. This spec assumes the
**first `n = min(n_rgb, n_depth)` frames correspond** and drops the trailing depth/confidence/odometry
row. If the app actually drops the *first* RGB frame, every stream is off by one frame (~17 ms) and I2
is violated with no visible symptom. **Confirm against the capture app** before S2.

> Delete the final frame of Depth and confidence. That is the excess frame.

### `ISSUE-04 · LAYOUT-MISMATCH` — medium
Spec §2 says the date folder holds two positional folders. It holds **session** folders, each holding
`Front`/`Side`. Spec says `metadata.yaml`; the file is `meta.yaml`, and sits at the **date** level.
Update the source spec, or the layout.

> meta.yaml is the template for metadata.yaml. If metadata.yaml does not exist in the folder, follow the template to create a corresponding metadata.yaml, and use the data in rgb.mp4 to fill in.

### `ISSUE-09 · ROTATION-DIRECTION-UNDETERMINED` — high
CW vs CCW cannot be derived from any metadata (`rgb.mp4` carries no rotation tag). It must be fixed by
looking at one frame **once**, then frozen in config for all cameras and all three streams. Config
therefore has **no default** — the flow refuses to run until it is set. Getting this wrong yields
upside-down lifters that still pass every automated check in §8.4.

> rotate clockwise by 90 degrees.

### `ISSUE-10 · S1-VALIDATION-IMPOSSIBLE-AS-WRITTEN` — medium
S1's validation asks that the streams "agree pixel-for-pixel — a landmark visible in RGB sits at the
same coordinates in depth and confidence". RGB is 1440×1920 and depth is 192×256 after rotation, so
identical pixel coordinates are impossible. Restated here as agreement in **normalized** coordinates
(§8.4.3). Confirm that is the intended meaning.

> do not validate this as one to one mapping. There is no requirement to do so, the depth is just for calculation later for distance and angle.

### `ISSUE-11 · RGB-OUTPUT-ENCODING` — medium
S1 must write rotated RGB somewhere. Re-encoding to H.264 (`crf=16`, proposed) costs **one lossy
generation** on top of the capture's own compression; lossless PNG frames cost tens of GB per session
(≈6000 × 1440×1920); FFV1 is lossless but large and slow to decode in training. Proposed default:
**H.264 `crf=16`**. **Needs sign-off** — it is irreversible for downstream training data.

> Explain the proposed default a bit more clearly to me in the chat.

### `ISSUE-12 · TIMESTAMPS-AND-FPS` — medium, feeds I2
`rgb.mp4` has no per-frame timestamp file; the only per-frame clock is `odometry.csv:timestamp`
(device uptime, ≈370169 s — **not** epoch), joined by frame index. fps is not a constant 60: measured
58.57, 59.99, 59.97, 60.00 across the four cameras. The common `t = 0` origin required by I2 is
undefined until S2 exists.

> Follow odometry timestamps