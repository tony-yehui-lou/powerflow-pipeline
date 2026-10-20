# S5 Barbell — Landmark Detection, As Built

Status: **implemented, first model trained** | Scope: GitHub issue [#121](https://github.com/tony-yehui-lou/PowerFlow/issues/121) under epic [#59](https://github.com/tony-yehui-lou/PowerFlow/issues/59) | Updated: 2026-10-19

This records what was actually built and measured, and supersedes the target and labelling method
proposed in [`S5-barbell-model-recommendation.md`](S5-barbell-model-recommendation.md). That
document remains the decision record for *why* the bar is detected rather than the athlete, and
why subject identity is never learned; its choice of **the plate face** as the detection target,
and of **SAM 2 propagation** as the labelling route, were both replaced for reasons recorded in §6.

---

## 1. What the detector finds

Two classes, both metal parts of the bar itself:

| Class | What it is | Why it is there |
|---|---|---|
| **cap** | the end face of the sleeve, seen as a ring or ellipse | the true bar tip, the sharpest landmark when its face is toward the camera |
| **collar** | the bar's stepped inner shoulder, where the shaft meets the plates | visible on the end whose cap is hidden, and set against a coloured plate rather than a grey floor |

Neither is the plate. A bar looks the same whatever is loaded on it and whatever gym it is in,
where a bumper changes colour between plates and a gym floor is covered in other bumpers.

**At most one cap is ever visible.** The two caps face in opposite directions along the axis, so
only one can point at the camera, and when the bar is fronto-parallel neither does. The far
**collar** is likewise hidden behind its own plates. So a frame typically offers **one cap and one
collar, on opposite ends** — which is exactly the pair the geometry needs.

## 2. Metric scale

Scale comes from the fixed distance between two landmarks. Which pair is available decides which
constant applies:

| Pair | Distance | Same on both bar types? | Availability |
|---|---|---|---|
| collar ↔ collar | **1310 mm** | **yes** | rare — the far collar is usually occluded |
| cap ↔ opposite collar | **1755 mm** (men's) / 1660 mm (women's) | no | **the usual case** |
| cap ↔ cap | 2200 / 2010 mm | no | never |

Because the usual pair is bar-type dependent, the bar type must be declared. It is read from the
capture's **raw** `metadata.yaml` (`lift.barbell_type`, e.g. `male IWF`) and written into the
annotation document, because the preprocess metadata task carries a fixed field set and drops it
on the way to S4 — along with `weight_in_kg` and `type`, which arrive as `null` (see §7).

**A cap paired with the collar on its own end is only 445 mm apart.** Pairing logic that matches
same-end landmarks produces a bar roughly 4× too short, and every downstream number is wrong while
still looking plausible. Pairing must use the opposite end, and should assert the result against
the declared constant.

## 3. Corpus

`10 September`, seven captures, one gym, one athlete, one bar, a tripod that did not move:

| | |
|---|---|
| Captures | 7 (`30kg1 30kg2 40kg1 45kg1 50kg1 50kg2 50kg3`) |
| Frames labelled | 280 (40 evenly spaced per capture) |
| Cap boxes | 238 |
| Collar boxes | 280 |
| Blurred boxes | 33 |
| Frames with nothing visible | 0 |
| Bar type | `male IWF` on all seven |

Boxes are stored at **full resolution** (~1416×1770) whatever scale they were drawn at, because
the detector trains at native resolution where a cap is ~30×40 px rather than half that.

Splits are **whole captures, never frames** — two frames of one lift from a fixed tripod are
nearly the same picture, so a frame-level split scores itself:

| Split | Captures | Images | Boxes (cap / collar) |
|---|---|---|---|
| train | 30kg1, 30kg2, 45kg1, 50kg1, 50kg2 | 200 | 165 / 200 |
| val | 40kg1 | 40 | 36 / 40 |
| test | 50kg3 | 40 | 37 / 40 |

`45kg1` is deliberately in train: its cap faced the camera on only 9 of 40 frames, which makes it
poor validation material and useful hard training material.

## 4. Model

A torchvision detector fine-tuned from COCO weights, with only the classification head replaced
(background / cap / collar). Seven architectures are registered and swap with one flag —
`fasterrcnn_resnet50_fpn{,_v2}`, `fasterrcnn_mobilenet_v3_large{,_320}_fpn`,
`retinanet_resnet50_fpn{,_v2}`, `fcos_resnet50_fpn` — all verified to build, run a forward pass,
and compute a loss with the replaced head.

First run: **`fasterrcnn_resnet50_fpn_v2`**, 20 epochs, AdamW at 5e-4, batch 2, cosine schedule,
~275 s per epoch on MPS.

Two settings matter more than the architecture:

- **Native resolution** (`min_size=1416`). Torchvision's default would resize to an 800 px short
  side and shrink a 30 px cap to ~17 px.
- **Horizontal flip on** (`p=0.5`). A cap is still a cap mirrored, and flipping deliberately
  breaks this rig's "cap is image-left of collar" regularity, which is a property of the tripod
  rather than of barbells.

**Evaluation is not COCO mAP.** It is recall and precision at IoU 0.5 plus **median centre error
in pixels, reported separately for sharp and blurred boxes** — centre error is what a bar path
consumes, and the sharp/blurred split is the only honest way to quote a number when the fast
frames are the ones that matter.

## 5. Results

Validation (`40kg1`, held out):

| | best (epoch 6) | final (epoch 20) |
|---|---|---|
| Recall | **0.987** | 0.947 |
| Precision | 0.852 | 0.923 |
| Cap recall | 1.000 | 0.944 |
| Collar recall | 0.975 | 0.950 |
| Median centre error, sharp | 2.48 px | **2.23 px** |
| Median centre error, blurred | 2.35 px | 3.61 px |

At this footage's scale (a 450 mm plate across ~170 px, so ~2.6 mm/px), **2.2 px is about 6 mm**.

End-to-end on `50kg3`, the untouched test capture, every frame, 0.43 s/frame on MPS:

| | |
|---|---|
| Frames | 433 |
| Both landmarks found | **413** |
| Collar missed | **0** |
| Cap missed | 20 — all with the bar at 0–15% of its travel, after the lift |
| Confidence | cap median 0.996 (min 0.66), collar median 0.980 (min 0.898) |
| Frame-to-frame motion | cap 2.9 px/frame median, collar 2.3 — maxima at the drop, genuine motion |

**No landmark is missed during the lift itself.** The cap misses are end-of-clip frames with the
bar at rest, and the collar is found on every frame of the capture — which is the payoff for
having a second landmark visible where the first is not.

## 6. Why the recommendation's target and method were replaced

Both changes were forced by measurement, and both are worth keeping written down:

- **Plate face → bar landmarks.** Plates change colour between weights and a gym floor is full of
  other plates; a chrome sleeve and collar are the same object everywhere. Switching the class
  also removed the obligation to label every plate in the room, which had been the dominant
  labelling cost.
- **SAM 2 propagation → hand-drawn boxes.** SAM 2 tracks the whole barbell reliably (398 frames,
  zero lost) but **loses a ~15–30 px sleeve at the turnover**: measured area jump of 3.4× and an
  84 px centre leap onto the far plate. Refinement did not repair it — adding a prompt on an
  already-propagated frame produced a byte-identical mask, and registering two conditioning frames
  before a forward pass raises `maskmem_features in conditioning outputs cannot be empty`. A click
  near a sleeve also returns a *part* or the whole bar depending on footage, never a clean sleeve.
  Hand-drawing ~500 boxes on sampled frames costs about half a day and sidesteps all of it; a
  detector has no memory and cannot drift.
- **No ignore regions, and no unlabelled visible instances.** Anything present and unlabelled
  trains as background, and a fixed tripod puts it at identical pixels in every frame.

## 7. Known gaps

- **One gym, one athlete, one bar, one tripod.** `50kg3` was held out but is visually near
  identical to the training captures. This validates the method; it does not measure
  generalisation. Footage from a second gym is the only thing that will.
- **Checkpoint selection is wrong.** `best.pt` is chosen on recall, which picked epoch 6 — worse
  precision and worse sharp centre error than epoch 17. Centre error is the better criterion.
- **Blurred accuracy degrades with training**, 2.35 px at epoch 6 to 3.61 px at epoch 20, while
  sharp improves. With 23 blurred boxes in training this is unsurprising, and it matters because
  blurred frames are the fast ones. Weighting them up, or early stopping, is the fix.
- **Pairing, scale and bar path are not built.** `predictions.json` carries per-frame boxes and
  scores; nothing yet pairs them, applies 1755 mm, or emits the UI's `bar` block.
- **The preprocess metadata task drops fields.** September's raw `barbell_type`, `weight_in_kg`
  and `type` do not reach S4; the annotator reads raw directly as a workaround. The flat
  `<date>/<trial>/` layout also leaves `derived_from_session_name` empty.
- **The trained checkpoint predates a label fix.** Four collar boxes were drawn on the plate hub
  rather than the collar (0.16–0.35× the class median) and were corrected after training; two were
  in the val and test splits.

## 8. Tooling

All under `scripts/`, none added to the pipeline's runtime dependencies:

| Script | Does |
|---|---|
| `bar_landmark_label.py` | draw cap/collar boxes on sampled frames; magnifier, blur flag, autosave, resume |
| `bar_landmark_qc.py` | coverage, composition, box-size outliers, side consistency, trajectory jumps |
| `export_landmarks_coco.py` | COCO instances split by whole captures, carrying bar type and blur |
| `train_landmark_detector.py` | fine-tune any of seven torchvision detectors; `--list-models` |
| `predict_landmarks.py` | run a checkpoint over a capture, render `predictions.mp4` + `.json` |

`sam2_plate_label.py`, `plate_labels_qc.py` and `migrate_static_to_tracked.py` belong to the
superseded plate method and the `v1` corpus. They are kept for reproducibility, not for use.

Data lives under `../data/annotations/barbell/`: `v1` the plate corpus, `v3` the landmark corpus,
`v3/coco` the exported splits, `v3/runs` the training runs.
