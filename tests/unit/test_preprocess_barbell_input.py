"""S5's barbell detections and the declared bar type, as S6's wrist correction reads them."""

from __future__ import annotations

from pathlib import Path

import pytest

from powerflow_pipeline.data.common.barbell_storage import write_barbell
from powerflow_pipeline.data.common.models import (
    BarbellDetection,
    BarbellDocument,
    BarbellProvenance,
    Frames,
    PixelSpace,
)
from powerflow_pipeline.data.preprocess import barbell_input as bi


def box(cx: float, cy: float, score: float) -> BarbellDetection:
    return BarbellDetection(box=(cx - 10, cy - 10, cx + 10, cy + 10), score=score)


def write_detections(
    path: Path,
    caps: list[BarbellDetection | None],
    collars: list[BarbellDetection | None],
    size: tuple[int, int] = (1416, 1744),
) -> Path:
    n = len(caps)
    write_barbell(
        path,
        BarbellDocument(
            capture_id="10 September/50kg3",
            role="single",
            stage="s4_crop",
            pixel_space=PixelSpace(width=size[0], height=size[1]),
            frames=Frames(count=n, fps=60.0, start_epoch_ms=0, t_ms=tuple(range(n))),
            landmarks={"cap": tuple(caps), "collar": tuple(collars)},
            detector=BarbellProvenance(
                architecture="fasterrcnn_resnet50_fpn_v2",
                checkpoint="m.pt",
                checkpoint_sha256="ab" * 32,
                training_epoch=6,
                record_threshold=0.3,
                device="cpu",
            ),
        ),
    )
    return path


class TestReadDetections:
    def test_returns_box_centres_per_frame(self, tmp_path: Path) -> None:
        path = write_detections(
            tmp_path / "barbell.json",
            [box(100, 200, 0.99), None, None],
            [box(800, 600, 0.95), None, None],
        )

        got = bi.read_detections(path, n_frames=3, frame_size=(1416, 1744))

        assert got.caps == [(100.0, 200.0), None, None]
        assert got.collars == [(800.0, 600.0), None, None]
        assert got.checkpoint_sha256 == "ab" * 32

    def test_scores_below_the_consumer_gate_are_dropped(self, tmp_path: Path) -> None:
        path = write_detections(
            tmp_path / "barbell.json", [box(100, 200, 0.31)], [box(800, 600, 0.95)]
        )

        got = bi.read_detections(path, n_frames=1, frame_size=(1416, 1744))

        assert got.caps == [None]
        assert got.collars == [(800.0, 600.0)]

    def test_detections_in_another_image_space_are_refused(self, tmp_path: Path) -> None:
        path = write_detections(tmp_path / "barbell.json", [None], [None], size=(1440, 1920))

        with pytest.raises(ValueError, match="1440x1920"):
            bi.read_detections(path, n_frames=1, frame_size=(1416, 1744))

    def test_detections_of_another_length_are_refused(self, tmp_path: Path) -> None:
        path = write_detections(tmp_path / "barbell.json", [None, None], [None, None])

        with pytest.raises(ValueError, match="2 frames, the capture has 3"):
            bi.read_detections(path, n_frames=3, frame_size=(1416, 1744))

    def test_a_malformed_document_is_a_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "barbell.json"
        path.write_text('{"frame_size": [1744, 1416], "predictions": {}}')

        with pytest.raises(ValueError):
            bi.read_detections(path, n_frames=1, frame_size=(1416, 1744))

    def test_detections_path_mirrors_the_capture(self, tmp_path: Path) -> None:
        got = bi.detections_path(tmp_path, Path("10 September/50kg3"))

        assert got == tmp_path / "10 September" / "50kg3" / "barbell.json"


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
