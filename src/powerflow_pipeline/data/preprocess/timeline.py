"""Pure epoch-time maths for S1 Cut. No I/O, no Prefect, no side effects.

Frame-paired streams use the device's `odometry.csv:timestamp` monotonic clock; RGB,
depth, confidence, and odometry share each selected odometry index. IMU uses the same clock
domain but is independently sampled. Both are bridged to Unix epoch milliseconds by anchoring
the first odometry row to the camera's `creation_time`. IMU uses that *same* odometry anchor,
so the real sub-second offset between the two streams' starts is preserved rather than each
stream separately zeroed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime


def creation_time_to_epoch_ms(creation_time: str) -> int:
    """Convert an ISO-8601 `Z` creation tag to epoch milliseconds, sub-second preserved.

    Raises `ValueError` on anything that does not parse -- the caller decides how that
    becomes a rejection.
    """

    if not creation_time:
        raise ValueError("creation time is empty")
    dt = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
    return round(dt.timestamp() * 1000)


def derive_cut_interval(
    side_created_epoch_ms: int, lift_start_ms: int, lift_end_ms: int
) -> tuple[int, int]:
    """`cut_start_epoch_ms, cut_end_epoch_ms` from the Side creation time and lift window."""

    return side_created_epoch_ms + lift_start_ms, side_created_epoch_ms + lift_end_ms


def epoch_ms_series(
    created_epoch_ms: int, anchor_s: float, samples_s: Sequence[float]
) -> list[int]:
    """Map a series of clock-domain samples onto epoch milliseconds.

    `anchor_s` is the sample that maps exactly onto `created_epoch_ms`; every other sample
    is offset by its elapsed distance from that anchor, on the *same* clock.
    """

    return [created_epoch_ms + round((sample - anchor_s) * 1000) for sample in samples_s]


def select_closed_indices(epochs: Sequence[int], start_ms: int, end_ms: int) -> list[int]:
    """Indices whose epoch falls in the closed interval `[start_ms, end_ms]`.

    Filters, never fabricates: a gap in `epochs` (a dropped frame) simply yields a gap in
    the result, it is never filled by interpolation or resampling.
    """

    return [index for index, epoch in enumerate(epochs) if start_ms <= epoch <= end_ms]
