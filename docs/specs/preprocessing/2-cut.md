# Cut and cross-camera time alignment

Status: **draft** | Updated: 2026-07-14

## Purpose

Add a **Cut** stage between Ingest and Orientation:

```text
S0 Ingest → S1 Cut → S2 Orient
```

The stage uses the Side camera's lift window as the source of truth, converts that window to
real-world epoch time, and applies the same epoch window to both the Side and Front captures. Its
output is two camera directories that cover the same lift interval.

## Metadata contract

`metadata.yaml` contains the Side-camera lift window:

```yaml
lift_start_time_side_in_ms: <milliseconds>
lift_end_time_side_in_ms: <milliseconds>
```

These fields are interpreted as millisecond offsets from the creation time of the Side `rgb.mp4`.
The fields must satisfy:

```text
0 <= lift_start_time_side_in_ms < lift_end_time_side_in_ms
```

Let `side_video_created_at_epoch_ms` be the Side RGB video's creation time, converted to Unix epoch
milliseconds. The Cut stage derives one shared interval:

```text
cut_start_epoch_ms = side_video_created_at_epoch_ms + lift_start_time_side_in_ms
cut_end_epoch_ms   = side_video_created_at_epoch_ms + lift_end_time_side_in_ms
```

The two derived epoch timestamps are written to the output metadata together with the source fields
and the timestamp source used. They become the sole cut bounds for both cameras.

## Procedure

1. Validate the Side lift-window metadata and read the Side RGB creation time.
2. Derive `cut_start_epoch_ms` and `cut_end_epoch_ms`.
3. Cut the Side RGB video to that interval.
4. Cut every **time-indexed** Side stream to the same interval:
   - `depth/`;
   - `confidence/`;
   - time-indexed CSV files, including `odometry.csv` and `imu.csv`.
5. Use the same derived epoch interval to cut the Front RGB video, depth frames, confidence frames,
   and time-indexed CSV files.
6. Preserve non-time-indexed calibration files, such as `camera_matrix.csv`, unchanged.

For a frame or record timestamped exactly on a bound, include it. For a stream whose timestamps do
not land exactly on a bound, retain only samples whose timestamps fall within the closed interval
`[cut_start_epoch_ms, cut_end_epoch_ms]`; never fabricate samples through resampling. Depth and
confidence must be selected as matched pairs, so neither stream can gain or lose an item independently.

## Output contract

The output directory contains trimmed Front and Side captures. Every retained time-indexed item has a
timestamp inside the shared epoch interval, and the output metadata records:

- `cut_start_epoch_ms` and `cut_end_epoch_ms`;
- the Side RGB creation time used to derive them;
- the source metadata values;
- retained timestamp ranges and item counts for every trimmed stream.

## Validation and rejection

Reject the session when any of the following is true:

- either Side lift-window field is missing, non-numeric, negative, or out of order;
- the Side RGB creation time is absent, ambiguous, or cannot be converted to epoch milliseconds;
- the derived interval lies outside the Side capture;
- either camera has no RGB frames in the interval;
- a depth frame lacks its corresponding confidence frame after cutting;
- a required time-indexed CSV has no records in the interval; or
- timestamps are unavailable for a stream that must be cut.

The stage must cut by timestamps, not by matching frame indices or assuming a fixed frame rate.
Side is cut first because it defines the lift window; Front is then cut against the exact same
real-world epoch bounds.
