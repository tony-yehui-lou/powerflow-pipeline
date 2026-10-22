"""S5 · Barbell detection -> runs an injected detector on S4's cropped RGB and publishes one
`BarbellDocument` per camera. The detector is stubbed: this covers the plumbing, publishing and
manifest, never the model (that is `test_barbell_model.py`'s slow smoke test).
"""

from __future__ import annotations

from pathlib import Path

import av
import pytest

from powerflow_pipeline.data.common.barbell_storage import barbell_path, read_barbell
from powerflow_pipeline.data.common.capture_models import CameraRecord
from powerflow_pipeline.data.common.capture_timing import frame_offsets_ms
from powerflow_pipeline.data.common.models import (
    BarbellClass,
    BarbellDetection,
    BarbellProvenance,
)
from powerflow_pipeline.data.detect.config import DetectConfig
from powerflow_pipeline.data.detect.tasks.barbell import detect_barbell_camera
from tests.conftest import MakeCamera, MakeSessionMetadata
from tests.unit.test_detect_pose import _build_crop_record, _make_pose_session
from tests.unit.test_detect_pose import make_config as _pose_make_config

PROVENANCE = BarbellProvenance(
    architecture="fasterrcnn_resnet50_fpn_v2",
    checkpoint="models/s5_barbell/m.pt",
    checkpoint_sha256="cd" * 32,
    training_epoch=6,
    record_threshold=0.3,
    device="cpu",
)


def make_config(tmp_path: Path, **overrides: object) -> DetectConfig:
    overrides.setdefault("barbell_root", tmp_path / "s5_barbell_output")
    return _pose_make_config(tmp_path, **overrides)


class _StubDetector:
    """A cap on the first frame only, a collar on every frame."""

    provenance = PROVENANCE

    def __init__(self, *, short_by: int = 0) -> None:
        self.calls: list[tuple[Path, int]] = []
        self.short_by = short_by

    def detect(
        self, rgb_path: Path, n_frames: int
    ) -> dict[BarbellClass, list[BarbellDetection | None]]:
        self.calls.append((rgb_path, n_frames))
        cap = BarbellDetection(box=(1.0, 1.0, 3.0, 3.0), score=0.99)
        collar = BarbellDetection(box=(4.0, 4.0, 6.0, 7.0), score=0.9)
        n = n_frames - self.short_by
        return {"cap": [cap] + [None] * (n - 1), "collar": [collar] * n}


def _session(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_session_metadata: MakeSessionMetadata,
    **overrides: object,
) -> tuple[DetectConfig, CameraRecord]:
    raw = _make_pose_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, **overrides)
    return config, _build_crop_record(raw, config)


def _published(config: DetectConfig, record: CameraRecord) -> Path:
    assert config.barbell_root is not None
    return barbell_path(config.barbell_root, record.relative)


# --- the happy path ------------------------------------------------------------------


def test_publishes_a_barbell_document_in_s4_pixel_space(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)

    returned, _ = detect_barbell_camera.fn(record, config, _StubDetector())

    assert returned == record
    document = read_barbell(_published(config, record))
    assert document.capture_id == record.capture_id
    assert document.role == record.role
    assert document.stage == "s4_crop"
    assert (document.pixel_space.width, document.pixel_space.height) == (
        record.rgb_width,
        record.rgb_height,
    )
    assert document.frames.count == record.n_frames
    assert document.landmarks["cap"][0] is not None
    assert all(hit is None for hit in document.landmarks["cap"][1:])
    assert document.detector == PROVENANCE


def test_timing_comes_from_the_video_pts_and_s1s_epoch_anchor(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    import json

    config, record = _session(tmp_path, make_camera, make_session_metadata)

    detect_barbell_camera.fn(record, config, _StubDetector())

    document = read_barbell(_published(config, record))
    assert list(document.frames.t_ms) == frame_offsets_ms(record.source / "rgb.mp4")
    cut = json.loads((config.cut_root / record.relative / "cut_sidecar.json").read_text())
    assert document.frames.start_epoch_ms == cut["retained"]["rgb"]["epoch_range"][0]


def test_calls_the_detector_with_the_s4_rgb(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)
    detector = _StubDetector()

    detect_barbell_camera.fn(record, config, detector)

    assert detector.calls == [(record.source / "rgb.mp4", record.n_frames)]


def test_copies_the_operator_metadata_beside_the_document(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)

    detect_barbell_camera.fn(record, config, _StubDetector())

    copied = _published(config, record).parent / "metadata.yaml"
    assert copied.read_text() == record.metadata_path.read_text()


def test_no_overlay_unless_asked(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)

    detect_barbell_camera.fn(record, config, _StubDetector())

    assert not (_published(config, record).parent / "overlay.mp4").exists()


def test_renders_an_overlay_with_every_frame_when_asked(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(
        tmp_path, make_camera, make_session_metadata, barbell_render_overlay=True
    )

    _, step = detect_barbell_camera.fn(record, config, _StubDetector())

    overlay = _published(config, record).parent / "overlay.mp4"
    with av.open(str(overlay)) as container:
        frames = sum(1 for _ in container.decode(container.streams.video[0]))
    assert frames == record.n_frames
    assert any(op.dst == overlay for op in step.file_ops)


# --- the manifest --------------------------------------------------------------------


def test_derived_values_are_prefixed_so_they_cannot_collide_with_s6s(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)

    _, step = detect_barbell_camera.fn(record, config, _StubDetector())

    assert step.derived["barbell_frames_with_cap"] == 1
    assert step.derived["barbell_frames_with_collar"] == record.n_frames
    assert step.derived["barbell_frames_with_both"] == 1
    assert step.derived["barbell_median_cap_score"] == pytest.approx(0.99)
    assert step.derived["barbell_median_collar_score"] == pytest.approx(0.9)
    assert step.derived["barbell_checkpoint_sha256"] == "cd" * 32
    assert step.derived["barbell_device"] == "cpu"
    assert step.derived["barbell_reused"] is False
    assert step.derived["barbell_s_per_frame"] >= 0
    assert all(key.startswith("barbell_") for key in step.derived)


def test_low_coverage_warns_but_never_rejects(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)

    _, step = detect_barbell_camera.fn(record, config, _StubDetector())

    assert _published(config, record).is_file()
    [warning] = step.warnings
    assert warning.startswith(f"cap on 1/{record.n_frames} frames")
    assert "below 50%" in warning


def test_coverage_at_the_threshold_does_not_warn(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)
    config = config.model_copy(update={"barbell_warn_cap_coverage": 1 / record.n_frames})

    _, step = detect_barbell_camera.fn(record, config, _StubDetector())

    assert step.warnings == []


# --- publish, reuse, overwrite, dry run ----------------------------------------------


def test_dry_run_plans_without_a_detector_and_writes_nothing(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)
    config = config.model_copy(update={"dry_run": True})

    _, step = detect_barbell_camera.fn(record, config, None)

    assert not _published(config, record).exists()
    assert [op.op for op in step.file_ops] == ["write", "publish"]
    assert step.file_ops[0].src == record.source / "rgb.mp4"


def test_existing_output_is_reused_without_running_the_detector(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)
    detect_barbell_camera.fn(record, config, _StubDetector())
    before = _published(config, record).read_text()
    detector = _StubDetector()

    _, step = detect_barbell_camera.fn(record, config, detector)

    assert detector.calls == []
    assert _published(config, record).read_text() == before
    assert step.derived == {"barbell_reused": True}
    assert step.file_ops == []


def test_overwrite_replaces_existing_output(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)
    detect_barbell_camera.fn(record, config, _StubDetector())
    stale = _published(config, record).parent / "stale.txt"
    stale.write_text("from an older run")
    config = config.model_copy(update={"overwrite": True})
    detector = _StubDetector()

    _, step = detect_barbell_camera.fn(record, config, detector)

    assert len(detector.calls) == 1
    assert not stale.exists()  # the whole capture directory was replaced, not merged
    assert step.derived["barbell_reused"] is False


def test_a_detector_that_breaks_its_contract_publishes_nothing(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)

    with pytest.raises(ValueError, match="frames"):
        detect_barbell_camera.fn(record, config, _StubDetector(short_by=1))

    assert config.barbell_root is not None
    destination = _published(config, record).parent
    assert not destination.exists()
    leftovers = destination.parent.iterdir() if destination.parent.exists() else iter(())
    assert not any(p.name.startswith(".powerflow-") for p in leftovers)


def test_a_failure_while_staging_leaves_no_staging_behind(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    config, record = _session(tmp_path, make_camera, make_session_metadata)
    record.metadata_path.unlink()

    with pytest.raises(FileNotFoundError):
        detect_barbell_camera.fn(record, config, _StubDetector())

    destination = _published(config, record).parent
    assert not destination.exists()
    assert not any(p.name.startswith(".powerflow-") for p in destination.parent.iterdir())
