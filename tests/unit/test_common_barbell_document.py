"""`BarbellDocument`: S5's published per-frame cap and collar detections (barbell.json)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from powerflow_pipeline.data.common.barbell_storage import (
    barbell_path,
    read_barbell,
    write_barbell,
)
from powerflow_pipeline.data.common.errors import PublishError
from powerflow_pipeline.data.common.models import (
    BarbellDetection,
    BarbellDocument,
    BarbellProvenance,
    Frames,
    PixelSpace,
)


def provenance(**overrides: Any) -> BarbellProvenance:
    values: dict[str, Any] = {
        "architecture": "fasterrcnn_resnet50_fpn_v2",
        "checkpoint": "../data/models/s5_barbell/frcnn-v2-2026-10-19.pt",
        "checkpoint_sha256": "ab" * 32,
        "training_epoch": 6,
        "record_threshold": 0.3,
        "device": "cpu",
        "versions": {"torch": "2.13.0", "torchvision": "0.28.0"},
    }
    return BarbellProvenance(**(values | overrides))


def document(n: int = 3, **overrides: Any) -> BarbellDocument:
    hit = BarbellDetection(box=(10.0, 20.0, 30.0, 40.0), score=0.99)
    values: dict[str, Any] = {
        "capture_id": "10 September/50kg3",
        "role": "single",
        "stage": "s4_crop",
        "pixel_space": PixelSpace(width=100, height=200),
        "frames": Frames(count=n, fps=60.0, start_epoch_ms=1_000, t_ms=tuple(range(0, 17 * n, 17))),
        "landmarks": {"cap": (hit,) + (None,) * (n - 1), "collar": (hit,) * n},
        "detector": provenance(),
    }
    return BarbellDocument(**(values | overrides))


class TestBarbellDetection:
    def test_a_box_must_have_positive_extent(self) -> None:
        with pytest.raises(ValidationError, match="positive extent"):
            BarbellDetection(box=(10.0, 20.0, 10.0, 40.0), score=0.5)

    def test_a_score_must_be_a_probability(self) -> None:
        with pytest.raises(ValidationError):
            BarbellDetection(box=(0.0, 0.0, 1.0, 1.0), score=1.5)

    def test_the_centre_is_the_box_midpoint(self) -> None:
        assert BarbellDetection(box=(10.0, 20.0, 30.0, 60.0), score=0.5).centre == (20.0, 40.0)


class TestBarbellDocument:
    def test_round_trips_through_the_camel_case_wire_form(self) -> None:
        original = document()

        wire = original.to_document()

        assert wire["schemaVersion"] == "1.0.0"
        assert wire["pixelSpace"] == {"width": 100, "height": 200}
        assert wire["landmarks"]["cap"][1] is None
        assert wire["detector"]["checkpointSha256"] == "ab" * 32
        assert wire["detector"]["rule"] == "highest-scoring detection per class"
        assert BarbellDocument.from_document(wire) == original

    def test_every_class_must_span_every_frame(self) -> None:
        hit = BarbellDetection(box=(10.0, 20.0, 30.0, 40.0), score=0.99)
        with pytest.raises(ValidationError, match="'collar' holds 2 frames, expected 3"):
            document(landmarks={"cap": (None, None, None), "collar": (hit, hit)})

    def test_every_declared_class_must_be_present(self) -> None:
        with pytest.raises(ValidationError, match="landmarks must hold exactly"):
            document(landmarks={"cap": (None, None, None)})

    def test_coverage_counts_frames_with_each_class(self) -> None:
        doc = document()

        assert doc.frames_with("cap") == 1
        assert doc.frames_with("collar") == 3
        assert doc.frames_with_both() == 1


class TestStorage:
    def test_path_mirrors_the_capture_under_the_root(self, tmp_path: Path) -> None:
        got = barbell_path(tmp_path, Path("10 September/50kg3"))

        assert got == tmp_path / "10 September" / "50kg3" / "barbell.json"

    def test_write_then_read_returns_the_same_document(self, tmp_path: Path) -> None:
        path = tmp_path / "x" / "barbell.json"

        write_barbell(path, document())

        assert read_barbell(path) == document()

    def test_write_is_write_once(self, tmp_path: Path) -> None:
        path = tmp_path / "barbell.json"
        write_barbell(path, document())

        with pytest.raises(PublishError, match="refusing to overwrite"):
            write_barbell(path, document())
