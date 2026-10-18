# S3 Retilt: optional single-frame floor fit

Date: 2026-10-18
Status: **draft — not implemented**

## Context

S3 Retilt fits one floor plane per camera by pooling depth points from **multiple** sampled
frames of the cut clip: `_sample_indices` (`tasks/retilt.py:66-72`) takes every
`retilt_sample_stride`-th frame (default 10), capped at `retilt_max_sampled_frames` (default
32) — frames `0, 10, 20, ..., 310`. Because S1 Cut has already trimmed the clip to the
operator-declared lift window (typically a few seconds), that span is not "the first 5 seconds
of a long clip" — on a short lift it can be most or all of the cut clip, i.e. most or all of
the actual lifting motion.

`select_floor_pixels` (`retilt.py:145-184`) has no way to tell a floor pixel from a non-floor
pixel other than "inside the operator-annotated rectangle". If the lifter, the bar, or a plate
passes through that rectangle in *any* sampled frame, that frame's occluding pixels are
back-projected and pooled into the plane fit exactly as if they were floor — there is no
outlier rejection in `fit_plane` (`retilt.py:201-231`), only a single ordinary-least-squares
pass over the whole pooled cloud. A rectangle that is clean at the start of a lift and
occluded by its end contaminates the fit without the pipeline detecting it before the
`retilt_max_plane_rms_m` gate, and a fit that catches a *consistent* near-plane obstruction can
still pass that gate while describing the wrong surface — exactly the failure mode
`2026-09-18-single-camera-oblique-capture-design.md` §4.5 already documents for a tight-but-
wrong fit.

`10 September`'s captures are the motivating case: a single oblique camera per trial, annotated
with a floor rectangle close to the camera (per `4-retilt.md`'s guidance), where the lifter's
feet and the bar's path are expected to cross nearer the camera as the lift progresses. The
first frame of the cut clip — timestamped at the operator's own `lift_start_time_side_in_ms`,
before the pull begins — is the one frame in the window least likely to have the lifter
standing on the annotated patch.

## Decision

Add an **opt-in** single-frame fitting mode, selected per run, rather than changing the
default. Two reasons against a blanket change:

1. **11 July and 22 August are already calibrated against pooled sampling.** The implementation
   plan's Task 9 recorded real `tilt_deg`/`roll_deg`/RMS numbers for July under pooled
   sampling, and the oblique-capture design's §4.2 recorded the same for 27 of 29 August
   captures (RMS median 0.79 cm). Those cameras are static for the whole clip and nothing in
   their raw data suggests occlusion; pooling more points only tightens their fit. Changing
   the default would silently change already-verified output for data that doesn't have the
   problem this spec is solving.
2. **Single-frame fitting is strictly less robust on its own terms** — one frame's point cloud
   instead of up to 32 frames' worth, no cross-frame averaging of per-pixel sensor noise, and
   the existing camera-translation check (`translation_span_m`, `retilt.py:363-371`) becomes
   vacuous (a single sample has no second point to compare, so it's always `0.0`). That's an
   acceptable trade for a capture where pooling risks contamination, not a universal
   improvement.

So: a new `retilt_single_frame` switch, default `False` (today's pooled behaviour, byte-for-byte
unchanged), settable per run via `--retilt-single-frame` so a 10 September run can opt in
without touching July or August.

### What "first frame" means

Frame index `0` of the record `retilt_camera` receives — i.e. S2's output, which S1 Cut has
already trimmed to start at the operator's `lift_start_time_side_in_ms`. This is the same frame
index `0` that pooled mode already includes first; single-frame mode only stops after it
instead of continuing to index 10, 20, ....

### Behaviour under `retilt_single_frame=True`

- `_sample_indices` returns `[0]` unconditionally — `retilt_sample_stride` and
  `retilt_max_sampled_frames` are ignored (documented as such; not an error to pass them
  together, since a future run may toggle the flag without editing every other tunable).
- Region selection, back-projection, and the plane fit (`select_floor_pixels` →
  `backproject` → `fit_plane`) run exactly as today, just over one frame's points instead of
  the pooled cloud. No new maths.
- `retilt_min_floor_points` (default 500) and `retilt_max_plane_rms_m` (default 2 cm) are
  **not relaxed**. A single clean frame over a reasonably sized rectangle has produced
  thousands of points in every real capture measured so far (§4.2's smallest pooled count
  across *multiple* frames was 5 783; one frame of the same rectangle is a few hundred to a
  few thousand). If a specific capture's single frame doesn't clear the existing gates, it is
  rejected with the same message pooled mode already uses — no silent fallback to pooled
  sampling. Silently falling back would hide exactly the failure this mode exists to surface.
- The camera-translation check still runs; with one sampled index it always measures `0.0` and
  therefore never rejects. This is correct, not a gap: the check exists to catch the camera
  moving *between* sampled frames, and there is only one.
- The gravity-agreement check (warn-only, §7 of `4-retilt.md`) is unaffected — it already reads
  only the first sampled frame's odometry quaternion (`tasks/retilt.py:313-316`), pooled or not.

### Sidecar and manifest

`retilt_sidecar.json` gains one field, `frame_sample_mode: "pooled" | "first"`, next to the
existing `sample_indices`/`n_frames_sampled` (which already report `[0]`/`1` correctly under
single-frame mode with no change). Carried into `StepResult.derived` as well, so it shows up in
the run manifest next to `tilt_deg`/`roll_deg`/`plane_rms_residual_m` — anyone reading a
manifest can tell which captures were fit which way without opening every sidecar.

### Day-level consistency check (§4.5 of the oblique-capture design)

`assess_plane_consistency` compares a single-camera day's captures' `tilt_deg`/`roll_deg`/
`floor_offset_m` against their shared median — unaffected by this change; it reads the derived
values regardless of how they were fit. A day that mixes pooled and single-frame captures (not
expected, but not forbidden) would compare them on equal footing, which is correct: the
invariant being checked is "did the rig move," not "how was the floor sampled."

## Configuration

| Name | Default | Purpose |
|---|---|---|
| `retilt_single_frame` | `False` | Fit the floor plane from frame 0 alone instead of pooling up to `retilt_max_sampled_frames` frames. |

No existing default changes.

## Out of scope

- **Per-frame outlier rejection in `fit_plane`** (e.g. RANSAC) — would let pooled mode tolerate
  an occluded frame automatically, and is a reasonable follow-up, but it's a bigger change to
  the core maths than this spec's problem requires, and it wouldn't help a capture where the
  rectangle is occluded in *every* frame but the first.
- **Auto-selecting the least-occluded frame** — would need some occlusion signal (e.g. a
  depth discontinuity inside the rectangle) that nothing in the pipeline computes today, and
  risks silently picking a frame nobody checked, which is exactly what the operator-annotated,
  no-algorithmic-fallback rule (`4-retilt.md` §1) exists to avoid.
- **Changing the default for any existing capture day.** July and August stay on pooled
  sampling unless a future run explicitly opts them in.
- **Re-measuring 10 September's floor region.** Separate, prerequisite work — the rectangle
  currently on disk is August's, copied over, and needs its own verification against 10
  September's camera placement regardless of which sampling mode is used.

## Test plan

TDD per `CLAUDE.md`; coverage floor stays 90%. Geometry verified numerically against synthetic
fixtures, matching every existing test in this module — never by observing that a transform ran.

- `tests/unit/test_cli.py` — `--retilt-single-frame` parses and threads through to
  `PreprocessConfig.retilt_single_frame`; omitting it leaves the default `False`. Mirror
  `test_preprocess_command_accepts_retilt_root`'s shape.
- `tests/unit/test_preprocess_retilt.py`:
  - `_sample_indices(n_frames, config)` returns `[0]` when `retilt_single_frame=True`,
    regardless of `retilt_sample_stride`/`retilt_max_sampled_frames` values — a direct unit
    test of the function, not routed through `retilt_camera`.
  - **The motivating case.** Build a fixture via `make_camera`/`make_session_metadata` (same
    helpers `_make_floor_session` already uses) where frame 0's annotated rectangle is clean
    floor but a later sampled frame (e.g. index 20) has the rectangle's confidence/depth
    overwritten to simulate an occluding body — confidence set outside `{0,1,2}`'s valid floor
    reading or depth values well off the fitted plane. Assert:
    - with `retilt_single_frame=False` (default), the contaminated frame measurably pulls the
      fit — either the RMS gate rejects it, or (the more important assertion) the recovered
      `tilt_deg`/`roll_deg` drift from the baked-in truth by more than the tolerance the clean
      case asserts.
    - with `retilt_single_frame=True`, the fit recovers the baked-in `tilt_deg`/`roll_deg` to
      the same tolerance `test_retilt_camera_recovers_baked_in_tilt` already uses, unaffected
      by the later frame's contamination.
  - `retilt_sidecar.json` (or the returned `StepResult`/sidecar dict in the test, following
    this file's existing pattern of asserting on the written JSON) carries
    `frame_sample_mode: "first"` when the flag is set, `"pooled"` otherwise, for both the
    default-config case and an explicit `retilt_single_frame=False` case.
  - Single-frame mode still rejects on `retilt_min_floor_points` and
    `retilt_max_plane_rms_m` using the same fixtures `test_translation_beyond_tolerance_is_rejected`
    / `test_tilt_beyond_bound_is_rejected` already build, just with `retilt_single_frame=True`
    and `retilt_sample_stride`/`retilt_max_sampled_frames` left at their defaults (proving
    those two tunables are genuinely ignored, not merely harmless).
  - Camera-translation check: with `retilt_single_frame=True`, `translation_span_m` in the
    written sidecar is `0.0` even when the same odometry perturbation
    `test_translation_beyond_tolerance_is_rejected` uses is applied — proving the check is
    inert under one sample rather than silently still catching movement some other way.
- `tests/integration/test_preprocess_flow.py` — one flow-level run with
  `config.retilt_single_frame=True` over the existing small fixture session, asserting
  `"retilt"` still appears in `steps` and the capture still publishes; a second run confirms
  the default (`retilt_single_frame=False`, or omitted) produces byte-identical
  `retilt_sidecar.json` to today's — the regression guarantee that nothing changes without the
  flag.

## Verification

Gates first, from `powerflow-pipeline/`:

```bash
uv run ruff check && uv run ruff format --check
uv run mypy src
uv run pytest --cov --cov-fail-under=90
```

Then against real data, **after** 10 September's floor region has been re-measured against its
own S2 output (separate, prerequisite work — see "Out of scope"):

```bash
# Baseline: today's pooled behaviour, for comparison.
uv run powerflow preprocess --input ../data/raw --records ../data/s0_ingest_output \
  --cut ../data/s1_cut_output --output ../data/s2_orient_output \
  --retilt ../data/s3_retilt_output --crop ../data/s4_crop_output \
  --only '10 September/*' --overwrite

jq '{tilt_deg, roll_deg, plane_rms_residual_m, n_floor_points, frame_sample_mode}' \
  "../data/s3_retilt_output/10 September/30kg1/retilt_sidecar.json"

# Single-frame mode over the same captures.
uv run powerflow preprocess --input ../data/raw --records ../data/s0_ingest_output \
  --cut ../data/s1_cut_output --output ../data/s2_orient_output \
  --retilt ../data/s3_retilt_output --crop ../data/s4_crop_output \
  --only '10 September/*' --retilt-single-frame --overwrite

jq '{tilt_deg, roll_deg, plane_rms_residual_m, n_floor_points, frame_sample_mode}' \
  "../data/s3_retilt_output/10 September/30kg1/retilt_sidecar.json"
```

Compare the two per capture. Expect `frame_sample_mode` to differ and `n_floor_points` to drop
(one frame's worth vs. up to 32), but `tilt_deg`/`roll_deg`/`plane_rms_residual_m` to stay
close — if single-frame mode disagrees sharply with pooled mode on a capture where pooled mode
*passed* its gates, that is evidence pooled mode was contaminated exactly as this spec predicts,
and single-frame mode's result should be the one trusted. If a capture is rejected under one
mode but not the other, read the rejection reason before deciding which is right — a rejection
is not automatically the wrong answer.

Finally, re-run 11 July without the flag and diff `retilt_sidecar.json` against the values
recorded in `2026-08-26-retilt-implementation.md` Task 9: **unchanged**, confirming the default
path was not touched.
