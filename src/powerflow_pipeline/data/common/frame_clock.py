"""When each frame of a capture was taken: S4's PTS for the offsets, S1's sidecar for the epoch.

Shared by every stage that publishes a per-frame document (S6 Pose, S9 Bar HVA), so their
`frames` blocks are timed identically. Never derive timing from a frame index and an fps: the
captures are variable frame rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import av


def frame_offsets_ms(rgb_path: Path) -> list[int]:
    """Each frame's offset from the first, in milliseconds, read from `rgb_path`'s own PTS.

    S1 Cut rebased this stream onto a zero-based, odometry-derived timeline (`trim_rgb`); S2-S4
    only touch pixels, never PTS, so the container still carries that exact timeline here.
    """

    offsets: list[int] = []
    with av.open(str(rgb_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            assert frame.pts is not None, f"{rgb_path}: frame carries no PTS"
            assert frame.time_base is not None, f"{rgb_path}: frame carries no time_base"
            offsets.append(round(float(frame.pts * frame.time_base) * 1000))
    return offsets


def capture_start_epoch_ms(cut_root: Path, relative: Path) -> int:
    """The true epoch of frame 0, from the `cut_sidecar.json` S1 wrote for this camera.

    Later stages don't carry epoch time forward, so this reaches back to S1's own output tree
    rather than something copied down the pipeline.
    """

    sidecar = json.loads((cut_root / relative / "cut_sidecar.json").read_text())
    start_ms: int = sidecar["retained"]["rgb"]["epoch_range"][0]
    return start_ms
