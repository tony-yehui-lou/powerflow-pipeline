"""S5's barbell detections and the declared bar type, as S6's wrist correction reads them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from powerflow_pipeline.data.preprocess import barbell_input as bi


def write_predictions(path: Path, frames: dict[str, object], size: tuple[int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"frame_size": [size[1], size[0]], "predictions": frames}))
    return path


def box(cx: float, cy: float, score: float) -> dict[str, object]:
    return {"box": [cx - 10, cy - 10, cx + 10, cy + 10], "score": score}


class TestReadDetections:
    def test_returns_box_centres_per_frame_padded_to_the_capture(self, tmp_path: Path) -> None:
        path = write_predictions(
            tmp_path / "predictions.json",
            {"0": {"cap": box(100, 200, 0.99), "collar": box(800, 600, 0.95)}, "1": {}},
            (1416, 1744),
        )
        caps, collars = bi.read_detections(path, n_frames=3, frame_size=(1416, 1744))
        assert caps == [(100.0, 200.0), None, None]
        assert collars == [(800.0, 600.0), None, None]

    def test_low_scores_are_dropped(self, tmp_path: Path) -> None:
        path = write_predictions(
            tmp_path / "predictions.json",
            {"0": {"cap": box(100, 200, 0.31), "collar": box(800, 600, 0.95)}},
            (1416, 1744),
        )
        caps, collars = bi.read_detections(path, n_frames=1, frame_size=(1416, 1744))
        assert caps == [None]
        assert collars == [(800.0, 600.0)]

    def test_detections_in_another_image_space_are_refused(self, tmp_path: Path) -> None:
        path = write_predictions(tmp_path / "predictions.json", {}, (1440, 1920))
        with pytest.raises(ValueError, match="1440x1920"):
            bi.read_detections(path, n_frames=1, frame_size=(1416, 1744))

    def test_detections_path_mirrors_the_capture(self, tmp_path: Path) -> None:
        got = bi.detections_path(tmp_path, Path("10 September/50kg3"))
        assert got == tmp_path / "10 September" / "50kg3" / "predictions.json"


class TestBarType:
    def test_reads_the_declared_bar(self, tmp_path: Path) -> None:
        path = tmp_path / "metadata.yaml"
        path.write_text("lift:\n  type: snch\n  barbell_type: male IWF\n")
        assert bi.read_bar_type(path) == "male IWF"

    def test_undeclared_or_unreadable_is_none(self, tmp_path: Path) -> None:
        path = tmp_path / "metadata.yaml"
        path.write_text("lift:\n  type: snch\n")
        assert bi.read_bar_type(path) is None
        assert bi.read_bar_type(tmp_path / "missing.yaml") is None
