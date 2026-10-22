# S3 Retilt: gravity-seeded floor detection (no floor annotation)

Date: 2026-10-21
Status: **draft — not implemented**
Supersedes: `2026-10-18-retilt-single-frame-design.md` (never implemented; see §9)
Depends on: the odometry-gravity axis-convention fix (§1), which ships first

## Context

S3 Retilt fits one floor plane per camera and derives two things from it:

1. **Tilt and roll.** The plane's normal gives the rectifying rotation (`4-retilt.md` §4–§5).
2. **`floor_offset_m`.** This is the camera's height above the floor, read by S7 Lift
   (`tasks/lift.py:28`, `to_floor_frame`) and by S8 as its fallback floor anchor
   (`tasks/kinematics.py:78`, `bar_geometry.floor_anchor`).

Today the plane is fit only to pixels inside an **operator-annotated rectangle** read from
`metadata.yaml` (`4-retilt.md` §1). This has three problems:

- **Manual work.** Every camera of every session needs someone to draw a rectangle and type
  four parenthesised strings correctly. The rectangle on disk for 10 September is August's,
  copied over and never checked (`2026-10-18-retilt-single-frame-design.md`, "Out of scope").
- **Wrong surfaces.** The rectangle has already landed on a PA speaker cabinet and on a
  spectator's head (`4-retilt.md` §1 and §7). Neither failure is visible in the frame.
- **Occlusion.** If the lifter or the bar crosses the rectangle in any sampled frame, those
  pixels go into the fit unflagged. The single-frame spec was a workaround for this.

The operator draws a rectangle only to answer "which pixels are floor". This spec answers
that question from data the capture already has: the phone's gravity vector, and the fact
that the floor is the lowest large horizontal surface in view.

## Decision

Replace the annotated rectangle with a two-step automatic floor search:

1. **Seed from gravity.** Take "up" from ARKit odometry. Back-project every confident depth
   pixel of the sampled frames and measure each point's height along "up". The floor shows
   up as the **lowest significant peak** in the height histogram.
2. **Refine from depth.** Fit a plane to the points near that peak, then repeatedly re-select
   the points close to the fitted plane and refit. The refined plane, not gravity, becomes
   the `PlaneFit` that S3 already consumes. Tilt, roll, the homography and
   `floor_offset_m` are then computed exactly as today.

Gravity only decides **which** points are floor. Depth still decides **where** the floor is,
so S3 keeps its sub-degree precision. Gravity then turns into a hard check on the result
(§6).

`metadata.yaml`'s `*_floor_region_*` fields are no longer read. If present, they are ignored,
not an error.

### Why this works: prototype on real data

A scratch prototype of §2–§5 (frame stride 10, at most 32 frames, confidence 2 only, 1 cm
bins, ±3 cm seed band, ±2 cm refine band, 3 iterations) was run on every capture with S2
output on disk on 2026-10-21: 11 captures from 11 July and 10 September. The 22 August
S2/S3 outputs had been removed by then, so August is not covered (see Verification).

"Old" below means the published annotated-rectangle sidecar.

| capture | floor share of points | refined vs old normal | offset new − old | RMS | refined vs gravity |
|---|---|---|---|---|---|
| 10 Sep 30kg1 | 0.79 | 0.58° | +3.1 cm | 0.71 cm | 3.20° |
| 10 Sep 30kg2 | 0.78 | 0.37° | +2.8 cm | 0.73 cm | 0.82° |
| 10 Sep 40kg1 | 0.75 | 0.44° | +3.1 cm | 0.73 cm | 1.04° |
| 10 Sep 45kg1 | 0.64 | 2.70° | −3.0 cm | 0.83 cm | 0.25° |
| 10 Sep 50kg1 | 0.64 | 2.57° | +5.1 cm | 0.83 cm | 0.73° |
| 10 Sep 50kg2 | 0.57 | 2.12° | +6.0 cm | 0.73 cm | 1.71° |
| 10 Sep 50kg3 | 0.66 | 0.72° | +4.3 cm | 0.83 cm | 0.80° |
| 11 Jul 30kg_Set1/Front | 0.92 | 0.82° | −1.8 cm | 0.70 cm | 0.74° |
| 11 Jul 30kg_Set1/Side | 0.88 | 0.71° | −1.4 cm | 0.86 cm | 3.50° |
| 11 Jul 50kg_Set3/Front | 0.94 | 0.84° | −1.8 cm | 0.54 cm | 1.06° |
| 11 Jul 50kg_Set3/Side | 0.89 | 0.92° | +1.0 cm | 0.66 cm | 2.77° |

What the prototype showed:

- **The floor dominates every capture.** Every capture had exactly one histogram peak
  holding ≥ 5% of points. It held 41–94% of all confident points.
- **The refined planes are as tight as the annotated ones.** RMS was 0.54–0.86 cm, far
  inside the 2 cm gate.
- **July agrees with the annotated fit** to < 1° and < 2 cm.
- **September is more self-consistent than the annotated fit.** It disagrees with the old fit
  by 2–3° on three captures. Those captures used the unverified rectangle copied from August,
  and they were shot from one unmoved tripod, so their floors should match. Spread of "up"
  across the 7 captures (median / max from the day's mean):

  | source | spread |
  |---|---|
  | annotated rectangle | 1.01° / 1.82° |
  | gravity alone | 0.83° / 1.92° |
  | **gravity-seeded, depth-refined** | **0.37° / 0.69°** |

  Camera-height range across the day dropped from 8.1 cm to 5.3 cm.
- **The result is insensitive to the seed band.** Widening it from ±3 cm to ±10 cm changed
  the normal by ≤ 0.26° and the offset by ≤ 0.8 cm.
- **Pooling frames helps.** Fitting frame 0 alone moved the result by up to 0.8° and 1.8 cm,
  so pooling stays (§2).
- **Confidence 1 hurts slightly.** Admitting confidence-1 pixels moved the result by up to
  0.7° and 2.6 cm, so confidence 2 stays the default (§2).

## 1. Prerequisite: correct the odometry gravity vector

`retilt.gravity_down_camera` currently applies an ARKit→CV axis flip, `diag(1, −1, −1)`.
Across all 39 captures on disk on 2026-10-20, that version disagreed with the fitted floor by
2–20°. The disagreement grew with tilt, which is what a mirrored tilt axis looks like.
Removing the flip and keeping `R_wcᵀ` and S2's 90° CW image rotation brought the disagreement
down to a median of 1.4° and a max of 4.0°.

This fix was found by searching 64 axis-convention variants against real data, not derived.
The fix should therefore come with a justification from Stray Scanner's odometry convention,
and it ships as its own change before this one. Once fixed, this spec defines:

```text
R_wc     = quat_to_matrix(qx, qy, qz, qw)             # per odometry row
g_cam    = R_z(90° CW) @ R_wcᵀ @ (0, −1, 0)            # "down", S2 portrait camera frame
u        = −g_cam / ‖g_cam‖                            # "up": away from the floor, −Y-ish
```

`u` is averaged over the **sampled frames' rows** and then renormalised. The camera is
static, so this only makes the result independent of which row was first. On real data,
averaging did not shrink the per-capture bias, so nothing more is claimed for it.

## 2. Point cloud: the whole frame, no rectangle

- **Frame sampling** is unchanged: `retilt_sample_stride` (10), capped at
  `retilt_max_sampled_frames` (32). The camera-translation gate (`retilt_max_translation_m`)
  is unchanged.
- **Pixel selection.** Take every pixel of each sampled frame with `confidence == 2` and
  `depth > 0`.
  - **Fallback.** If the pooled confidence-2 cloud has fewer than `retilt_min_floor_points`
    points, rebuild it with confidence ∈ {1, 2}. The sidecar's `confidence_mode` then reads
    `"conf1_and_2"`, keeping that field's existing meaning (`4-retilt.md` §1).
  - **Removed.** The "largest 4-connected confidence-2 component ≥ 1/6 of the region" rule
    existed only to judge whether a small rectangle was clean enough. It has no meaning over
    a whole frame, so it is removed along with `retilt_conf2_area_fraction`.
- **Back-projection** is unchanged: `depth_intrinsics` + `backproject`, metres, CV camera
  axes (`4-retilt.md` §2).
- **Downsampling.** If the pooled cloud exceeds `retilt_max_pooled_points` (200 000),
  downsample with the existing strided rule. Real captures produce 350–580 k points.

## 3. Seed: lowest significant height peak

```text
h_i = p_i · u                       # metres; points below the camera are negative
```

1. Histogram `h` with bin width `retilt_floor_bin_m` (**0.01 m**) over `[min h, max h]`.
   Smooth the counts with a 3-bin moving average. The smoothed counts are used only to
   locate local maxima.
2. **Significant peak.** A local maximum of the smoothed counts whose **raw** counts in the
   ±2 bins around it hold at least `retilt_floor_min_peak_fraction` (**0.15**) of all points.
3. **Seed height.** `h0` is the centre of the **lowest** significant peak, i.e. the most
   negative one.
4. If there is no significant peak, the camera is **rejected**: `"no horizontal surface
   holds ≥ 15% of confident depth points; floor not visible"`.

**Why the lowest peak, not the largest.**
- Nothing real sits below the floor, so the lowest substantial horizontal surface is the
  floor. A larger peak higher up (a bench, a stage, a table filling the frame) would be
  wrong.
- On every real capture the lowest peak and the largest peak were the same peak.
- The peak-mass threshold handles the main risk of "lowest": sparse points *below* the floor
  from specular multipath off a shiny floor. Those never reach 15%.

**Why the threshold is 15%.** Real floors held 41–94%, with ≥ 0.41 inside the ±2-bin
window. 15% leaves room for tighter framing. It is high enough that a speaker top or a
plate face, being small and not the lowest, is never chosen.

**Why the peak is broad.** Gravity's 1–4° error tilts the measuring axis. Points on a floor
spanning depth `D` then spread over about `D · tan ε` in `h`: 3° over 4 m ≈ 21 cm. The peak
is broad but still unmistakable, and §4 removes the spread.

## 4. Refine: fit to depth, not to gravity

1. Seed inliers are points with `|h_i − h0| < retilt_floor_seed_band_m` (**0.03 m**).
2. Fit a plane to the inliers with the **existing `fit_plane`** (`torch.linalg.lstsq`,
   `Y = aX + bZ + c`, normal oriented toward the camera). This needs no new plane maths and
   keeps the same `PlaneFit` semantics for normal, RMS and `floor_offset_m`.
3. **Re-select.** The new inliers are points of the **full cloud** within
   `retilt_floor_inlier_band_m` (**0.02 m**) perpendicular distance of that plane. Refit.
4. Repeat step 3 `retilt_floor_refine_iterations` (**3**) times, or stop early if the inlier
   set does not change.

The final `PlaneFit` replaces today's rectangle fit. Its fields mean what they meant before:
`n_points` is the final inlier count and `rms_residual_m` is the final inliers' RMS. From
here `retilt_camera` runs as today: `tilt_roll_from_normal` → `rectifying_rotation` →
homographies → warp → sidecar.

**Why refine at all.** Gravity has a fixed bias of up to ~4° per capture. The refined plane
is what made September's spread drop from 0.83°/1.92° to 0.37°/0.69°. A band of a few cm
around a seed that is off by a few degrees still picks out a real strip of floor, and each
refit realigns the plane to the floor's true orientation. The ±10 cm seed band converged to
the same answer (≤ 0.26°).

**Why occlusion stops mattering.**
- The lifter, the bar and the plates are not horizontal planes at floor height. They fall
  outside the ±2 cm band and never enter the fit.
- What does fall inside the band (shoe soles, plate rims touching the floor) is a tiny share
  of a floor that is 40–90% of the cloud.

## 5. Ambiguous floors (platforms, mats)

A weightlifting platform or a thick mat creates a second horizontal surface a few cm above
the gym floor. The feet stand on it, and the plates rest on it.

If a second significant peak lies within `retilt_floor_ambiguity_m` (**0.30 m**) above the
chosen one, record both in the sidecar and raise a manifest **warning**:
`"floor ambiguous: horizontal surfaces at h0 and h1; chose the lower"`.

This is a warning, not a rejection. No capture so far has two peaks, so there is no data on
which surface downstream needs. S8's rest-plate anchor (`bar_geometry.floor_anchor`) already
measures the surface the plates rest on, and is the natural tie-breaker if this ever
triggers (see Open questions).

## 6. Validation and rejection

| gate | rule | status |
|---|---|---|
| translation | `retilt_max_translation_m` (0.05 m) across sampled frames | unchanged |
| no floor | no significant peak (§3) | **new, reject** |
| too few points | final inliers < `retilt_min_floor_points` (500) | unchanged, now counts refined inliers |
| not planar | final RMS > `retilt_max_plane_rms_m` (0.02 m) | unchanged |
| implausible attitude | `\|tilt\|` or `\|roll\|` > 45° | unchanged |
| **gravity disagreement** | angle(refined normal, `u`) > `retilt_gravity_tolerance_deg` | **promoted: warn → reject** |
| day consistency | `assess_plane_consistency`: tilt 3°, height 0.10 m | unchanged, warn-only |
| ambiguous floor | §5 | **new, warn** |

**Promoting the gravity check to a rejection.**
- **Why.** Before this change, a person had looked at the frame and picked the floor. Now
  nobody has. The gravity check is the only gate that catches a clean plane fit on a sloped
  surface: a ramp, a tilted mat, the side of a plinth.
- **Why it can be trusted now.** The check was warn-only because its axis convention was
  unproven (`4-retilt.md` §7). §1 fixes the convention, and after the fix it measures
  ≤ 4.0° on 39 captures over three days.
- **The default rises from 5° to 6°** to leave a margin above that observed maximum.

**Removed gates.** The rectangle-validation rejections no longer apply because the fields
are not read: missing fields, unparseable `"(x, y)"`, degenerate region, out of bounds.

## 7. Output contract

All existing sidecar fields stay, with unchanged meaning. Downstream readers are untouched:
S4 Crop reads `valid_bounds_px`, and S7/S8 read `floor_offset_m`. Changes:

| field | change |
|---|---|
| `floor_source` | **new**: `"gravity_seeded"`. Distinguishes these sidecars from older annotated ones still on disk. |
| `gravity_up_cam` | **new**: `u` from §1 |
| `floor_seed_height_m` | **new**: `h0` |
| `floor_peaks` | **new**: every significant peak as `[height_m, mass_fraction]`, lowest first |
| `floor_inlier_fraction` | **new**: final inliers / pooled cloud |
| `floor_refine_iterations` | **new**: iterations actually run |
| `gravity_agreement_deg` | unchanged meaning; now computed with the corrected `u` |
| `gravity_check_mode` | `"warn"` → `"reject"` |
| `confidence_mode` | unchanged values; meaning per §2 |
| `region_px` (dry-run `derived`) | **removed**: there is no region left to plan from |

`derived` in the manifest gains `floor_source`, `floor_inlier_fraction` and `floor_peaks`
next to the existing `tilt_deg`, `roll_deg` and `plane_rms_residual_m`.

`metadata.yaml`: retilt **no longer reads it**. `record.metadata_path` stays, because S1 Cut
reads it for the lift window. Existing `*_floor_region_*` keys are left in place and
ignored; nothing needs editing in the raw data.

## 8. Code changes (for the implementation plan, not done here)

Everything is in `powerflow-pipeline/src/powerflow_pipeline/data/preprocess/`:

- **`retilt.py`**
  - Fix `gravity_down_camera` (§1).
  - Add pure functions:
    - `gravity_up(quats) -> np.ndarray`
    - `height_peaks(heights, bin_m, min_fraction) -> list[tuple[float, float]]`
    - `refine_floor(points, up, h0, seed_band_m, inlier_band_m, iterations) -> PlaneFit`
  - Delete `read_floor_region`, `region_prefixes`, `parse_region_point` and
    `select_floor_pixels`, and `FloorRegion` in `models.py`. Their only caller is
    `retilt_camera`. Check for other callers before deleting.
- **`tasks/retilt.py`**
  - Drop `_read_metadata` and the region lookup.
  - Replace `_pool_floor_points` with a whole-frame version (§2).
  - Call `height_peaks` → `refine_floor`.
  - Turn the gravity warning into a `ScanRejected`.
  - Write the new sidecar fields.
- **`config.py`**
  - Remove `retilt_conf2_area_fraction`.
  - Add `retilt_floor_bin_m`, `retilt_floor_min_peak_fraction`, `retilt_floor_seed_band_m`,
    `retilt_floor_inlier_band_m`, `retilt_floor_refine_iterations` and
    `retilt_floor_ambiguity_m`.
  - Raise `retilt_gravity_tolerance_deg` to 6.0.
- **Docs**
  - Per the repo's draft convention, write `docs/specs/preprocessing/4-retilt-draft2.md`
    rather than editing `4-retilt.md`. It replaces §1 and the "operator-annotated, no
    algorithmic fallback" assumption, rewrites §3 per this spec, and moves the gravity
    paragraph in §7 from warn to reject.
  - Mark `2026-10-18-retilt-single-frame-design.md` superseded.

## 9. Relationship to the single-frame spec

`2026-10-18-retilt-single-frame-design.md` existed to stop occluded frames contaminating a
rectangle fit. Without a rectangle there is nothing to contaminate (§4). Pooled sampling
stays the only mode, and it measurably beat frame-0-only in the prototype. That spec should
not be implemented.

## Configuration

| Name | Default | Purpose |
|---|---|---|
| `retilt_floor_bin_m` | 0.01 | Height histogram bin width (§3) |
| `retilt_floor_min_peak_fraction` | 0.15 | Share of points a peak's ±2 bins must hold to count (§3) |
| `retilt_floor_seed_band_m` | 0.03 | Half-width around `h0` for the first fit (§4) |
| `retilt_floor_inlier_band_m` | 0.02 | Perpendicular inlier band for each refit (§4) |
| `retilt_floor_refine_iterations` | 3 | Maximum refits (§4) |
| `retilt_floor_ambiguity_m` | 0.30 | A second peak within this height above the floor triggers a warning (§5) |
| `retilt_gravity_tolerance_deg` | 5.0 → **6.0** | Now a rejection (§6) |
| `retilt_conf2_area_fraction` | — | **Removed** (§2) |

## Out of scope

- **Picking a platform over the gym floor automatically.** §5 only warns. Choosing needs data
  that does not exist yet.
- **Reconciling `floor_offset_m` with S8's rest-plate anchor inside S3.** S3 runs before bar
  detection (S5), so it cannot. S8 already compares the two.
- **Removing the floor-region keys from raw `metadata.yaml`.** Raw data is write-once, and
  the keys are harmless once ignored.
- **A `--retilt-floor-source annotated` override.** Keeping both paths doubles the code and
  the tests to preserve exactly what this change removes. The comparison baseline is the
  annotated sidecars already published (see Verification).

## Test plan

TDD per `CLAUDE.md`, coverage ≥ 90%. Geometry is verified numerically on synthetic
fixtures.

**Fixture change.** `conftest._render_floor_plane` bakes a tilt into depth, but `make_camera`
writes odometry that does not encode that tilt. Every retilt fixture must now write the
**matching quaternion**: the inverse of §1's chain applied to the baked-in normal. It must
also take an optional `gravity_error_deg` so gravity can be deliberately biased.

**Unit tests for the pure functions**, in `test_preprocess_retilt_math.py`:
- `gravity_up` round-trips: a quaternion built from a known normal returns that normal. This
  covers §1's fix with a hand-derived case, not just the fitted convention.
- `height_peaks`:
  - finds a single floor peak;
  - picks the **lower** of two peaks when the upper is larger (floor 30%, table 50%);
  - ignores a below-floor cluster under 15%;
  - returns nothing for uniformly spread heights.
- `refine_floor` recovers a known plane exactly (noise-free: < 0.01° and < 1 mm) when seeded
  with `u` biased by **0°, 2° and 4°**. This is the property the whole design rests on.
- `refine_floor` ignores a box ("the lifter") and a vertical wall in the cloud: same plane to
  the same tolerance.

**Task tests**, in `test_preprocess_retilt.py` (rewrite the rectangle-based tests):
- `test_retilt_camera_recovers_baked_in_tilt`: unchanged assertion, but with **no
  floor-region keys** in `metadata.yaml`, and gravity biased by 3°.
- Floor-region keys present but pointing at a non-floor area: output is identical to having
  no keys. This proves the keys are ignored.
- An occluding object across the floor in later sampled frames: tilt and roll still within
  tolerance. This is the old single-frame spec's motivating case, now passing in pooled mode.
- A frame with no floor (wall only): `ScanRejected` with the §3 message.
- A plane sloped 10° relative to gravity: `ScanRejected` with the gravity-disagreement message.
- A 10 cm platform peak above the floor: published, with an ambiguity warning in the manifest
  and both peaks in `floor_peaks`.
- Not enough confidence-2 points: `confidence_mode == "conf1_and_2"`.
- The sidecar has every new field from §7, and `gravity_check_mode == "reject"`.
- The existing translation, tilt-bound, RMS and dry-run tests keep their assertions, rebuilt
  on whole-frame fixtures.
- The rectangle tests (missing, unparseable, degenerate region) are deleted, along with
  `test_preprocess_retilt_models.py`'s `FloorRegion` cases.

**Other tests**
- `test_preprocess_retilt_group.py`: unchanged.
- `tests/integration/test_preprocess_flow.py`: the small fixture session runs end to end with
  its `metadata.yaml` floor regions removed.

## Verification

Gates, from `powerflow-pipeline/`:

```bash
uv run ruff check && uv run ruff format --check
uv run mypy src
uv run pytest --cov --cov-fail-under=90
```

Real data, for the user to run. S3 outputs are write-once, so move the current annotated
sidecars aside first; they are the comparison baseline.

```bash
mv ../data/s3_retilt_output ../data/s3_retilt_output_annotated
```

```bash
uv run powerflow preprocess --input ../data/raw --records ../data/s0_ingest_output \
  --cut ../data/s1_cut_output --output ../data/s2_orient_output \
  --retilt ../data/s3_retilt_output --crop ../data/s4_crop_output --overwrite
```

Acceptance:

1. **No new rejections** on 11 July and 10 September.
2. **11 July is within 1°/2 cm of the annotated sidecars.** This is the only day whose
   rectangle was checked by hand.
3. **10 September's day-level spread is ≤ 0.7° max** with **no** `assess_plane_consistency`
   warnings. This reproduces the prototype.
4. **`gravity_agreement_deg` ≤ 4°** everywhere.
5. **22 August** must be regenerated through S0–S2 first, since its outputs were removed by
   2026-10-21. It is then the first day this method meets unseen: 27 captures from one
   tripod. Expect the day spread to stay at or below the annotated fit's 0.34°/0.88°.
6. **Downstream impact.** Diff S7/S8 outputs for one capture per day. September's
   `floor_offset_m` moves by 3–6 cm, and absolute heights from the floor should move by the
   same amount and nothing else.

## Open questions

- **Platform vs floor (§5).** Which surface should `floor_offset_m` describe when both are
  visible? This needs a capture with a platform.
- **Is 6° tight enough?** Calibrated on 39 captures over three days with one phone. A second
  phone model may have a different IMU-to-depth extrinsic bias.
- **How much tilt error can downstream tolerate?** The refined plane's ~0.4° day spread is
  about 0.7 cm of height error per metre of lateral distance. That is probably fine, but no
  downstream spec states a requirement yet.

---

## Implementation notes (2026-10-21, branch `s3-retilt-gravity-floor`)

Implemented as specified. The gravity fix (§1) ships on the same branch rather than as a
separate change. Where the implementation departs from the text above, or learned something:

- **Iterations never settle on real data.** On noisy real data the inlier set keeps flickering
  at the 2 cm band edge, so every real capture used all 3 refits. Against 10 refits, 3 differ
  by < 0.05° / 1 mm on 9 of 11 captures and by at most 0.28° / 9 mm on the other two. The
  default of 3 is kept.
- **Read-only re-run of the shipped functions over the 11 captures on disk** reproduces the
  prototype table:
  - exactly one peak per capture, holding 42–94% of points;
  - RMS 0.53–0.87 cm;
  - gravity agreement ≤ 3.3°;
  - 10 September spread median 0.40° / max 0.66°, camera-height range 5.6 cm.

  No pipeline run was made; the Verification section above is still the user's to run.
- **Test fixtures.**
  - `make_camera` now writes the odometry quaternion that matches a rendered floor
    (`conftest.gravity_quaternion`, inverted by hand rather than through the code under test),
    plus an optional `gravity_error_deg`.
  - The retilt task tests render 64×48 depth instead of 16×12. At 16×12 only ~60 points
    survive, which is too few for a 3°-biased seed to keep a peak, while 64×48 recovers tilt
    and roll to 0.001° at up to 4° of bias.
- **Fixtures no longer write floor regions.** Every test fixture's `floor_regions` argument was
  removed, so the whole suite runs without an annotation. The one integration test that forced
  an S3 rejection by stripping the region now forces it through `retilt_min_floor_points`.
- **`scripts/fix_august_floor_regions.py`** is now obsolete: it writes keys S3 no longer reads.
  It is left in place.
