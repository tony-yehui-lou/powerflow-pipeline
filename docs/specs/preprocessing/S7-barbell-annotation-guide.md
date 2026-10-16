# S7 Barbell — Human Annotation Guide

Status: **draft** | Scope: the labelling half of [#121](https://github.com/tony-yehui-lou/PowerFlow/issues/121), feeding C5 of [`S7-barbell-model-recommendation.md`](S7-barbell-model-recommendation.md) | Updated: 2026-10-16

This is the procedure for producing the plate-detection dataset by hand. It is written to be
followed literally by whoever sits down to annotate, including the version of you that comes back
to this in three months.

**Assumed throughout:** all equipment in the corpus is IWF-standard — every plate from 10 kg to
25 kg is **450 mm ± 1 mm** in diameter, and bar length is the IWF standard (2200 mm men's /
2010 mm women's). Nothing about plate size needs to be recorded per capture.

---

## 0. The one idea this document is built on

**The model learns what a plate looks like. It never learns whose bar it is.**

Those are two different problems, and conflating them is what makes a detector that works in one
gym and nowhere else:

| | Learned? | How it is decided |
|---|---|---|
| "Where are the plate faces?" | **Yes** — this is the detector | Appearance. Generic, transfers to any gym. |
| "Which pair is the subject's bar?" | **No** | At inference: the user taps their bar (separate issue, designed with the app UI). In this corpus: a human recorded it at annotation time. |

A 25 kg plate on the floor and the same plate on a lifted bar are pixel-identical. Subject-hood is
not a visual property, so no quantity of labelling can teach it — attempting to would only teach
position-in-frame, which is the rig-specific shortcut that breaks elsewhere.

Three consequences that shape everything below:

1. **Every plate in the frame is a positive example, tracked through every frame.** There is no
   "background plate". There are no ignore regions. A plate the detector is not shown is a plate it
   learns to suppress.
2. **The subject's bar is recorded as a flag, not as a class.** It sits beside the annotations,
   feeds downstream lift-analysis training and evaluation, and never reaches the detector's loss.
3. **Nothing is assumed static.** Other athletes lift mid-capture, plates get carried off, new ones
   appear. Every plate is propagated frame by frame, so movement is captured rather than assumed
   away.

---

## 1. What a label is

**The target is the plate face, not the barbell.** One annotation = one plate face = one closed
elliptical region bounded by the **outer edge of the plate's rim**.

Three definitions that decide whether this dataset is worth anything. Get them wrong and the model
learns your inconsistency:

1. **The boundary is the outer rim**, not the coloured inner ring, not the printed weight badge,
   not the steel insert. The 450 mm certificate refers to the outer rim, so any other boundary
   silently breaks the metric scale downstream.
2. **A "bar end" is a plate centre.** The value the UI's `leftEnd`/`rightEnd` ultimately carries is
   the centre of the fitted ellipse, which lies on the bar's axis. You never annotate the bar's
   physical tip, and you never annotate the shaft.
3. **Label every plate face in the frame** — the subject's, floor plates, rack plates, another
   athlete's loaded bar, all of them, all tracked. They are plates; showing the model that
   identical objects are sometimes background is the fastest way to make a detector unreliable.

### Attributes on every annotation

| Attribute | Values | Why it exists |
|---|---|---|
| `subject` | bool | **The flag that replaces guesswork.** True for the two plates on the bar the capture is about. Never reaches the detector's loss; it is ground truth for the selection step and for lift-analysis training. |
| `sleeve` | `left`, `right`, `none` | `none` = not on the subject's bar. Left/right are **image-left / image-right**, not the athlete's anatomical left/right — the camera is oblique and the athlete turns. |
| `stack_index` | `0`, `1`, `2`, … | `0` = the plate **against the bar's inner shoulder**, counting outward. Only meaningful when `sleeve` is left/right. |
| `occluded` | `none`, `partial`, `heavy` | `partial` = rim broken by a hand/leg but over half the rim traceable; `heavy` = under half. |
| `truncated` | bool | Rim leaves the frame. Common on July Front. |
| `blurred` | bool | Motion blur has smeared the rim beyond a crisp edge. Set it generously — these frames are the evaluation set's whole point. |
| `edge_on` | bool | The face is turned so far from the camera that no ellipse can be traced (rack plates seen side-on). Flagged, and filtered out of training rather than teaching the model a shape it cannot see. |

### Why `stack_index` is worth the extra clicks

The distance between the two **outermost** plates changes with how the bar is loaded, so it is not
a constant and can't be used as a self-check. The **innermost** plate on each sleeve sits flush
against the bar's inner shoulder, and that shoulder-to-shoulder distance is fixed by the bar —
1310 mm on an IWF men's bar. So `stack_index = 0` annotations give C4 a genuinely constant
separation to validate a track against (up to half the plate's thickness, which the colour and the
recorded lift weight determine). Ten seconds of extra clicking per capture buys a free correctness
check on every lift the pipeline ever processes.

---

## 2. Which frames to annotate

Annotate **S4 crop output**, never raw — `../data/s4_crop_output/<capture>/rgb.mp4`. That is the
exact image the detector sees at inference; labelling anything else means labelling a coordinate
space the model never encounters.

The corpus is what **reached S4**, counted 2026-09-21: **23 August Snch + 5 August CnJ + 2 July
Front = 30** (the July Side pair is out of scope, and the raw August tree's 29th capture did not
survive preprocessing).

Propagation labels every frame, so the question is not which frames to *label* but which to
**sample for training**. A tripod that never moved makes consecutive frames nearly redundant:

| Split | Captures | Frames sampled each | Total | Purpose |
|---|---|---|---|---|
| **Train** | 24 (19 Snch, 4 CnJ, 1 July Front) | 12 stratified + 8 hard | ~480 | Fine-tuning |
| **Validation** | 2 (2 Snch) | 12 stratified | ~24 | Early stopping, threshold choice |
| **Test** | 4 (2 Snch, 1 CnJ, 1 July Front) | 12 stratified + 8 hard + eval GT | ~100 | Reported numbers, touched once |

**Hold out whole captures, never frames.** Two frames from the same lift are almost the same
picture; splitting within a capture inflates your test score into fiction.

**The 12 stratified frames** span the lift window evenly, with at least one each at: setup (bar on
floor), first pull, transition/knee pass, second pull at full extension, turnover, catch, recovery,
and bar returned to floor.

**The 8 hard frames** are chosen *because* they are hard: the fastest-moving frames of the second
pull (worst blur), frames where a hand or thigh cuts the rim, frames where the athlete's body
crosses in front of the far plate, and on July Front, frames where a plate runs off the frame edge.
Over-representing these is deliberate — the mean over all frames is dominated by easy setup frames
and tells you nothing about the moment peak velocity is measured.

**One capture is not a lift.** `22 August/Snch/58kgSnch2` shows 6 cm of bar travel across its whole
50-frame clip: its cut window caught the setup and missed the lift. Keep it out of the splits until
S1's window for it is fixed.

---

## 3. Pass A — label every plate, track them all

[SAM 2](https://github.com/facebookresearch/sam2) (Apache-2.0, code and checkpoints) takes a click
on an object in one frame and propagates its mask through the whole clip. For a rigid,
high-contrast disc this is close to free accuracy, and it costs the same per plate whether the
plate moves or not — which is why nothing here is assumed static.

**Where it runs:** `scripts/sam2_plate_label.py`, on this machine, against
`../data/s4_crop_output/<capture>/rgb.mp4`. No Docker, no notebook, no cloud upload — the footage
never leaves the laptop.

**Install:** nothing is installed into the pipeline. `transformers` ships SAM 2 as
`Sam2VideoModel`, and `uv run --with` layers it over the project environment for the length of one
command without touching `pyproject.toml`, so **`powerflow-pipeline`'s dependency list does not
grow** and `CLAUDE.md`'s "ask before adding dependencies" rule is not engaged. `torch` with working
MPS is already in the project venv. The first run downloads `transformers` (~50 MB) and the SAM 2.1
small checkpoint (~180 MB, into the HuggingFace cache).

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --with transformers --with torchvision \
    python scripts/sam2_plate_label.py "22 August/Snch/42kgSnch1"
```

**`--with torchvision` is required**, and easy to miss: the project pins `torch` but not
`torchvision`, while SAM 2's image processor imports it. Without it the run gets all the way
through the 184 MB checkpoint download before failing on
`Sam2ImageProcessor requires the Torchvision library`.

On an M3 Pro it runs on **MPS**; `PYTORCH_ENABLE_MPS_FALLBACK=1` lets any op Metal lacks drop to
CPU instead of aborting. Pass `--device cpu` if MPS produces visibly wrong masks — SAM 2 is trained
on CUDA, and its authors do not test Apple silicon.

### Using the window

A window opens at 1:1 on frame 0 (around 704×944 at the default `--scale 0.5`), with a banner
showing the frame index, the active plate, and the key bindings. **Keyboard focus must be on that
window, not the terminal.**

1. **Pick a prompt frame.** `]` steps forward 5 frames, `[` back. You want a frame where as many
   plates as possible are unobstructed. Changing frame **clears your clicks** on purpose: a click
   means "this pixel, on this frame".
2. **Click the subject's two bar plates first** — plate 1, `tab`, plate 2. They are objects 1 and 2
   by convention, and that convention is what the `subject` flag keys off.
3. **Press `n` and click the next plate.** Repeat for every other plate face in the frame: floor
   plates, the rack, another athlete's bar. `n` before each one, or the clicks pile onto the plate
   you are already on.
4. **`space` to preview.** Runs the model on this frame only and tints every mask.
5. **Refine with right-click**, which adds a *negative* point — "not this". Use it where a mask
   swallowed the neighbouring plates of a stack, crept along the bar, or grabbed a shin. `r` clears
   the frame and starts over.
6. **`enter` to propagate.** The window closes; the terminal prints `fwd frame N` then
   `back frame N`. Propagation runs forward from the prompt frame and then backward, so a mid-clip
   prompt frame still labels the whole clip.
7. **`q` quits without writing** — safe any time before `enter`.

### What it writes

Under `../data/annotations/barbell/v1/<capture>/`:

- `masks/<frame>.png` — one indexed mask per frame. `0` background, `1…N` the plates, **ids stable
  across every frame of the capture**.
- `sam2_run.json` — the checkpoint, device, decode scale, prompt frame, every click, and
  `subject_object_ids` (by convention `[1, 2]`).
- `overlay.mp4` — for eyeballing before trusting any of it.

Write-once: it refuses to relabel a capture in place. A redo means bumping to `v2`.

### Measured on this machine (2026-09-21, two bar plates only)

- Both masks landed on the plates and held through propagation as the bar rose — plate-1 centre
  tracked 549 → 533 → 515 px over frames 78 → 90.
- **~1.05 s/frame** on MPS, `sam2.1-hiera-small`, 704×944. Cost scales with the number of tracked
  objects, so a capture with six plates costs proportionally more than the ~3 minutes two plates
  took. Budget accordingly and run it unattended.
- `cv2.fitEllipse` on the resulting mask contour returned a centre within 0.4 px of the mask
  centroid — the C2 path working on real data.

**One behaviour to correct at prompt time:** on a sleeve loaded with several plates, a single click
tends to return **the whole stack as one mask**. The 450 mm ruler still holds (every IWF plate
shares that diameter), but the mask is inflated along the bar axis, so the fitted minor axis is not
the plate's. Fix it with right-click negative points on the neighbouring plates.

---

## 4. Pass B — QC, attributes, and the frames a human must see

```bash
uv run python scripts/plate_labels_qc.py "22 August/Snch/42kgSnch1" \
    --stack-index 1:0,2:0 --contact-sheet
```

It cleans each mask to its largest contour, fits the ellipse, derives `sleeve`, `occluded`,
`truncated`, `blurred` and `edge_on`, marks objects 1–2 as `subject`, runs the §5 checks, and
writes `plates.json`. `--contact-sheet` writes one JPEG per flagged frame into `review/`, each with
the **fitted ellipse** drawn on (not the mask) so a biased fit is visible at a glance.
`--force` replaces an existing `plates.json`, which is safe because it is derived.

Watch `overlay.mp4` first — it is the only check the numbers cannot replace, and it is where you
decide `--stack-index`.

**Review only the flagged frames.** On the first real capture this was 27 of 175, all one
contiguous run during the recovery. You are deciding one thing per frame: *is the mask wrong, or is
the plate genuinely occluded?* A genuinely occluded plate is correctly labelled and correctly
flagged — leave it; the `occluded` attribute is what training and evaluation need. Only a mask on
the wrong object sends you back to Pass A.

---

## 5. Quality control before anyone trains on it

Run all five. Each catches a different class of mistake, and all are cheap:

1. **Overlay flipbook.** Page through the annotated frames with masks drawn on. Bad labels are
   obvious to the eye in a way they never are in a JSON diff.
2. **Major-axis smoothness.** Plot each track's fitted major axis against frame index. A plate's
   apparent size changes slowly; **every spike is a mislabelled frame**.
3. **Scale sanity.** Convert the major axis to mm/px against 450 mm. On August footage this lands
   at **6.5–7.9 mm/px**. An order-of-magnitude miss means the rim convention slipped.
4. **Subject-bar excursion.** Convert the subject pair's vertical travel to metres. Real snatches
   in this corpus measure **1.06–1.72 m**. A value far outside that band means either the track is
   wrong or the capture is not a lift.
5. **Inter-annotator agreement.** Re-annotate 30 frames a week later with Pass C's protocol and
   compute the centre disagreement. **Target ≤ 2 px median (≈1 cm).** If humans disagree by 5 px,
   no model trained on this can be evaluated below 5 px.

Also assert mechanically, at export: every `sleeve: left/right` annotation has a `stack_index`;
exactly two objects per capture carry `subject: true`; left/right assignments never swap between
consecutive frames; and object ids are stable across the capture.

---

## 6. Pass C — evaluation ground truth, by rim clicks

For the test captures, do **not** eyeball the plate's centre. A human is bad at judging the centre
of a foreshortened ellipse and good at clicking a point on a visible rim. So:

> Click **5 or more points spread around the visible rim** → fit an ellipse to those points →
> the fitted centre is the ground-truth bar end.

Spread the points around the whole arc, not clustered on one side. On a `partial`-occluded plate,
5 points on the visible 60% of the rim still pin the ellipse well; on `heavy`, mark it and skip.
This protocol is what makes a ≲2 px (≈1 cm) accuracy claim meaningful.

**What the test set has to score** is not plate mAP in isolation. The deployed system is given the
subject's bar (by a user tap, designed separately) and must then hold it for the whole lift. So the
headline metric is **end-to-end**: *given the subject's bar identified at one frame, is it tracked
correctly for the rest of the lift, and how far are its centres from ground truth?* Build the
ground truth to answer that question.

---

## 7. Effort

Rough, for one person, per capture:

| Step | Cost |
|---|---|
| Pass A clicking (all plates, one frame) | 1–3 minutes |
| Pass A propagation | ~1 s per frame per object, unattended |
| Watching the overlay | ~2 minutes |
| Pass B, reviewing flagged frames only | ~5 minutes |

The clicking is the short part and the propagation needs nobody watching. This is the entire reason
the recommendation picks a fine-tuned detector over a zero-shot model: SAM 2 does the drawing, the
human only judges.

---

## 8. What this does not cover

- **Automating frame sampling and the COCO export.** Neither exists yet. The export must carry
  `subject` through, and must not let it reach the detector's loss as a class.
- **The inference-time tap.** How the user identifies their bar in the app is a separate issue,
  designed with the UI. Nothing in this document depends on how it is built — only on the fact that
  the detector is not responsible for the decision.
- **Re-annotation for a second gym.** Everything here assumes the two existing fixed rigs. New
  footage from a different room needs its own stratified sample, not more frames from these 30.
