# Retired: view-dependent frame scaling

Status: **retired** | Updated: 2026-08-27

## What this was

A proposed **Scaling** stage between Retilt and Cropping. It would scale each camera's frames
by a per-view physical reference — barbell length between the plates for a front-facing video,
plate diameter for a side-facing video — so that one pixel represented the same real-world
distance across every clip. This was meant to establish invariant **I4**: *"Scale is constant:
one pixel means the same physical distance in every clip."*

## Why it was retired

Writing this stage's implementation spec surfaced that it has **no data source**. Both halves
of the measurement it depends on are missing:

- **Where the reference is in the frame.** There is no barbell or plate annotation anywhere.
  S3 Retilt's floor region works because an operator manually drew it and recorded the
  coordinates in `metadata.yaml`'s `video:` block; nobody has done the equivalent for a barbell
  or a plate, and the block holds only floor-region fields today.
- **How big the reference is in real life.** Neither the real session `metadata.yaml` nor the
  date-level `meta.yaml` template records a barbell or plate physical size. `lift.weight_in_kg`
  — which could in principle look up a standard competition-plate diameter — is an unfilled
  template placeholder in every capture inspected.

Supplying both would mean either hand-annotating every existing and future capture, or training
a barbell/plate detector — a substantial project of its own, whose errors would silently
corrupt downstream geometry with no visible symptom in the frame, exactly the failure mode
these specs otherwise refuse to pass through.

## The measurement that made retiring it safe

Downstream consumers were confirmed to work off **relative lengths within a clip**, not
absolute cross-video pixel comparisons — so a per-camera scale factor was never actually load-
bearing for that use. Measured across the four cameras in `data/raw/11 July` (S3-retilted
output, `../data/s3_retilt_output/`), using each camera's median depth and RGB-resolution `fx`:

| camera | fx (px) | median depth | px/mm at that depth |
|---|---|---|---|
| 30kg_Set1/Front | 1343.00 | 5.69 m | 0.2360 |
| 30kg_Set1/Side | 1336.02 | 6.79 m | 0.1967 |
| 50kg_Set3/Front | 1344.31 | 5.98 m | 0.2248 |
| 50kg_Set3/Side | 1339.03 | 6.34 m | 0.2111 |

Focal lengths are effectively identical (same phone), so all variation is camera-to-subject
distance. The overall 1.20× spread is dominated by Front-vs-Side, an inherent viewpoint
difference no scalar rescale corrects. **Restricted to the same view across sessions, the
spread is only 5–7%** — trivially absorbed by ordinary scale augmentation during training, and
far smaller than the error a barbell/plate detector would plausibly introduce.

## What replaced it

- **Metric scale is carried by the depth stream and intrinsics**, not by pixel size. Real-world
  distances (bar velocity, bar path, plate positions) are computed from depth and
  `camera_matrix.csv` directly; the image was never the right place to encode scale, and
  nothing downstream needed it to be. See the restated invariant **I4** in
  `1-ingestion_orient.md`.
- **The trim this stage would also have owned did not go away with it.** S3 Retilt's rotation
  leaves a warp-invalid border with no source pixel (12–14% of frame height on real captures,
  and it *varies per camera* — a camera-identity leak worth removing on its own). That trim
  moved into **S4 Crop** (`6-cropping.md`), which already computes and applies one rectangle
  per camera; it now intersects S3's valid-content box with its own motion guard band and
  crops once, at no extra decode/re-encode cost.

## What would have to change to revive this

1. A reliable source for the reference measurement — either an operator annotation added to
   `metadata.yaml`'s `video:` block (mirroring the floor-region fields), or a trained detector
   with a measured, acceptable error rate.
2. A re-examination of whether pixel-space scale normalization is still worth doing at all,
   given depth already supplies metric truth for every real-world-unit use case identified so
   far — reviving this stage without that check would be solving a problem that may no longer
   exist.
