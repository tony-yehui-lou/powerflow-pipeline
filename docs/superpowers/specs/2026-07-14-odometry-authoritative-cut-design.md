# Odometry-Authoritative Cut Design

Date: 2026-07-14

## Context

The Cut stage currently derives RGB frame epochs from `rgb.mp4` presentation timestamps while
deriving depth and confidence epochs from `odometry.csv`. In the real captures these clocks drift:
the `cnj_45kg_Set1/Front` RGB and odometry clocks differ by 16.22 seconds at their last paired
frame. Applying the same epoch window independently therefore retains 2,835 RGB frames but only
2,704 depth frames and rejects the camera.

The capture contract pairs RGB, depth, confidence, and odometry by frame index. Each camera has one
more trailing depth/confidence/odometry item than RGB, and `odometry.csv:timestamp` is the
authoritative per-frame clock.

## Decision

For each camera, Cut will map odometry timestamps into epoch time and select the odometry indices
inside the shared closed interval `[cut_start_epoch_ms, cut_end_epoch_ms]`. It will apply those same
indices to RGB, depth, confidence, and odometry.

The paired index range is limited to the available RGB frames. A trailing depth, confidence, or
odometry entry without an RGB counterpart is not emitted. IMU remains independently selected by
timestamp because it has a different sampling rate and is not frame-index paired.

MP4 presentation timestamps remain encoding metadata but are not used as the real-world frame
clock for Cut.

## Data Flow

1. Derive the shared epoch cut interval from the Side RGB creation time and Side lift offsets.
2. For each camera, anchor its first odometry timestamp to that camera's RGB creation time.
3. Convert every odometry timestamp to epoch milliseconds.
4. Restrict candidate indices to the paired range shared by RGB, depth, confidence, and odometry.
5. Select paired indices whose odometry epochs lie inside the shared interval.
6. Use the selected indices for RGB decoding, depth and confidence copying, and odometry filtering.
7. Select IMU rows independently using their timestamps and the same epoch interval.
8. Record the authoritative clock and retained ranges in `cut_sidecar.json` and the run manifest.

## Validation and Errors

Ingest continues to validate that depth and confidence counts match, odometry and depth counts
match, and the configured RGB shortfall tolerance is satisfied. Cut rejects a camera when the
shared interval contains no paired RGB/depth frame or no IMU row.

Cut will not force two independently selected streams to have equal counts. Equal output counts
follow from applying one timestamp-derived index selection to all paired frame streams.

## Testing

A regression test will model the production failure by giving RGB presentation timestamps a
different duration from the paired odometry clock. Before the fix, it must fail with the same
RGB/depth count rejection. After the fix, it must show that:

- RGB, depth, confidence, and odometry retain identical counts and corresponding indices;
- the retained odometry epochs fall inside the requested closed interval;
- MP4 PTS drift does not change the selected paired indices; and
- IMU is still selected independently by timestamp.

The focused Cut tests, complete unit and integration suites, static checks, and the real pipeline
command must pass before completion is reported.

## Scope

This change is limited to Cut clock selection and its provenance. It does not resample streams,
interpolate missing frames, change the shared interval calculation, or attempt to equalize frame
counts between Front and Side cameras.
