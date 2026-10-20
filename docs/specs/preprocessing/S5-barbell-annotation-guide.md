# S5 Barbell — Human Annotation Guide

Status: **current method** | Scope: the labelling half of [#121](https://github.com/tony-yehui-lou/PowerFlow/issues/121) | Updated: 2026-10-19

How to produce the cap/collar dataset by hand. Written to be followed literally, including by the
version of you that comes back to this in three months.

What the labels are for, and what was measured with them, is in
[`S5-barbell-landmark-detection.md`](S5-barbell-landmark-detection.md). An earlier revision of this
guide described tracking **plates** with SAM 2 propagation; that method was replaced and the
reasons are recorded in §6 of that document.

---

## 0. The idea

**The model learns what a bar's metal landmarks look like. It never learns whose bar it is.**

A 25 kg plate on the floor and the same plate on a lifted bar are pixel-identical, so subject-hood
is not a visual property and no amount of labelling can teach it. It is supplied from outside the
model — by the annotator here, by the user's tap in the app.

Two classes, because between them something is always visible:

| Class | What | Where |
|---|---|---|
| **cap** | the sleeve's end face, as a ring or ellipse | the end pointing toward the camera — **at most one, sometimes neither** |
| **collar** | the bar's stepped inner shoulder, shaft meeting plates | the end whose cap is hidden; the far collar sits behind its own plates |

Neither is the plate.

---

## 1. Three rules

Get these wrong and the model learns your inconsistency rather than the bar.

1. **Box a cap only when its face reads as a ring or an ellipse.** The side of the chrome cylinder
   is not a cap.
2. **Box a collar when the stepped metal between shaft and plate is distinguishable**, even partly
   behind a hand — say over half of it. The hands grip inboard of the collars, so this case comes
   up constantly and needs deciding once rather than per frame.
3. **Motion blur is labelled, not skipped, and the box covers the whole smear.** The fast frames
   are where peak velocity is measured, so a model never shown a blurred cap fails exactly where
   it matters. For linear blur the smear's centre is the position at mid-exposure, which is the
   position a bar path wants — one box serves both. Mark them with `b`.

A frame with nothing visible is a **real label**, not a skipped frame. Press `n` on it.

---

## 2. Which frames

Label **S4 crop output** — `../data/s4_crop_output/<capture>/rgb.mp4`. That is the image the
detector sees at inference; labelling anything else labels a coordinate space the model never
encounters.

**40 evenly spaced frames per capture** is the default and was enough for the first model. The
clip is already cut to the lift, so even spacing covers every phase, and consecutive frames from
a tripod that never moved are near-duplicates that teach nothing.

**Hold out whole captures, never frames.** A frame-level split scores itself.

---

## 3. Labelling

```bash
uv run python scripts/bar_landmark_label.py "10 September/50kg1"
```

No model, no checkpoint, no `--with` — it starts instantly in the project env.

| Key | Does |
|---|---|
| `2` / `1` | choose class: collar (the default, drawn most) / cap |
| left drag | draw a box of the active class |
| `b` | toggle the blurred flag — sticky, because blur comes in runs |
| `u` | undo the last box on this frame |
| `n` / space | next frame; empty means "nothing visible here" |
| `p` | previous frame |
| `g` | toggle the magnifier |
| `q` | save and quit |

**The magnifier matters.** A 30 px landmark is 15 px on screen at the default display scale; a 3×
inset follows the cursor and parks away from it so corners can be placed precisely.

Boxes are stored at **full resolution** whatever scale you label at. It **autosaves after every
frame and resumes** — re-running skips frames already decided, `--redo` revisits them with your
existing boxes drawn so you can add or correct.

The **bar type** is read from the capture's raw `metadata.yaml` (`lift.barbell_type`) and recorded
with the labels, because metric scale depends on it. If it is missing the tool warns rather than
guessing.

---

## 4. Checking

```bash
uv run python scripts/bar_landmark_qc.py                      # every labelled capture
uv run python scripts/bar_landmark_qc.py "10 September/50kg1" # just one
```

It reports, and names the frames for each:

- **Coverage** — sampled frames still undecided, frames deliberately empty.
- **Composition** — caps and collars per frame, and whether a frame carries a pair of known
  separation (two collars at 1310 mm, or a cap and the opposite collar at 1755 mm on a men's bar).
- **Box size** — boxes far off the class median, ignoring blurred ones where a longer smear is
  correct. This is what caught four boxes drawn on the plate's centre hub instead of the collar,
  at 0.16–0.35× the median: invisible in any summary, obvious in a crop.
- **Side consistency** — if the cap is image-left of the collar on most frames and image-right on
  a few, those few are the wrong end. On a clean capture this reads 40/40.
- **Trajectory** — landmarks leaping between neighbouring samples, with whether the blur flag is
  set, since that is what distinguishes a mislabel from genuinely fast motion.

It exits non-zero when it finds a problem, so it batches.

---

## 5. Export

```bash
uv run python scripts/export_landmarks_coco.py --val 40kg1 --test 50kg3
```

COCO instances under `../data/annotations/barbell/v3/coco/`, split by whole captures. Each image
carries its bar type and pair distance; each box keeps its `blurred` flag so a trainer can
down-weight it and an evaluator can report sharp and blurred separately.

Pick a validation capture where the cap is often visible. `45kg1` showed its cap on only 9 of 40
frames, which makes it poor validation material and good hard training material.

---

## 6. Effort

Measured over the September corpus: **40 frames per capture, a few seconds a box, about half an
hour per capture including the QC pass.** Seven captures produced 280 frames and 518 boxes.

Hand-drawing is not a fallback here. Propagation was tried first and abandoned — a detector sees
one frame at a time and cannot drift, so per-frame labels bought by hand are worth more than
propagated ones that quietly wander onto the wrong object.

---

## 7. Not covered

- **The inference-time tap.** How a user identifies their bar in the app is a separate issue.
- **A second gym.** Everything here assumes one fixed rig. New footage needs its own sample, not
  more frames from these seven.
- **Pairing, scale and bar path.** The labels and the detector stop at boxes; nothing yet pairs a
  cap with the opposite collar or converts pixels to metres.
