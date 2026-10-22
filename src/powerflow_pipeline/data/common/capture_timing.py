"""When each frame of a published stream was taken, for every stage that publishes per frame.

Later stages don't carry epoch time forward, so a stage that needs it reaches back to S1's own
output tree; and S1 rebased the stream onto a zero-based, odometry-derived timeline that every
pixel-only stage after it (S2-S4) leaves untouched, so the container's own PTS are exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import av


def frame_offsets_ms(rgb_path: Path) -> list[int]:
    """Each frame's offset from the first, in milliseconds, read from `rgb_path`'s own PTS."""

    offsets: list[int] = []
    with av.open(str(rgb_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            assert frame.pts is not None, f"{rgb_path}: frame carries no PTS"
            assert frame.time_base is not None, f"{rgb_path}: frame carries no time_base"
            offsets.append(round(float(frame.pts * frame.time_base) * 1000))
    return offsets


def capture_start_epoch_ms(cut_root: Path, relative: Path) -> int:
    """The true epoch of frame 0, from the `cut_sidecar.json` S1 wrote for this camera."""

    sidecar = json.loads((cut_root / relative / "cut_sidecar.json").read_text())
    start_ms: int = sidecar["retained"]["rgb"]["epoch_range"][0]
    return start_ms
