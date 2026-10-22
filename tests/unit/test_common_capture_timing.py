"""Frame timing shared by S5 and S6: PTS offsets from the video, the epoch anchor from S1."""

from __future__ import annotations

import json
from pathlib import Path

from powerflow_pipeline.data.common.capture_timing import capture_start_epoch_ms


def test_the_epoch_anchor_is_the_start_of_the_retained_rgb_range(tmp_path: Path) -> None:
    sidecar = tmp_path / "10 September" / "50kg3" / "cut_sidecar.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({"retained": {"rgb": {"epoch_range": [1_700, 1_900]}}}))

    assert capture_start_epoch_ms(tmp_path, Path("10 September/50kg3")) == 1_700
