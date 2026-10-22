"""The detect pipeline under a real Prefect backend: S5, S6 and S7 on what preprocess published."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from powerflow_pipeline.data.common.models import (
    HUMAN_SKELETON,
    BarbellClass,
    BarbellDetection,
    BarbellProvenance,
    JointId,
)
from powerflow_pipeline.data.common.pose_storage import pose_path, read_pose, skeleton_path
from powerflow_pipeline.data.detect import flow as detect_module
from powerflow_pipeline.data.detect.barbell_model import CheckpointCard, card_path, sha256_file
from powerflow_pipeline.data.detect.config import DetectConfig
from powerflow_pipeline.data.detect.flow import detect
from powerflow_pipeline.data.detect.pose_model import JointPixelSeries
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.flow import preprocess
from tests.conftest import MakeCamera
from tests.integration.test_preprocess_flow import build_capture
from tests.integration.test_preprocess_flow import make_config as make_preprocess_config


class StubDetector:
    """A fixed, valid 2D detection for every frame -- stands in for a real `Detector2D`.

    The pixel is `(0, 0)`, the top-left corner, which the synthetic fixture gives usable depth,
    so S7 can lift it.
    """

    def detect(self, rgb_path: Path, n_frames: int) -> dict[JointId, JointPixelSeries]:
        series = JointPixelSeries(
            pixel_position=tuple((0, 0) for _ in range(n_frames)),
            confidence=tuple(0.9 for _ in range(n_frames)),
        )
        return {joint: series for joint in HUMAN_SKELETON.joints}


class StubBarbellDetector:
    """A cap and a collar on every frame -- stands in for a real `BarbellDetector`."""

    provenance = BarbellProvenance(
        architecture="fasterrcnn_resnet50_fpn_v2",
        checkpoint="m.pt",
        checkpoint_sha256="ef" * 32,
        training_epoch=6,
        record_threshold=0.3,
        device="cpu",
    )

    def __init__(self) -> None:
        self.calls = 0

    def detect(
        self, rgb_path: Path, n_frames: int
    ) -> dict[BarbellClass, list[BarbellDetection | None]]:
        self.calls += 1
        cap = BarbellDetection(box=(1.0, 1.0, 3.0, 3.0), score=0.99)
        collar = BarbellDetection(box=(4.0, 4.0, 6.0, 7.0), score=0.95)
        return {"cap": [cap] * n_frames, "collar": [collar] * n_frames}


def published(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any, make_session_metadata: Any
) -> PreprocessConfig:
    """Run preprocess on the two-session fixture: two cameras reach S4, three are rejected."""

    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_preprocess_config(tmp_path, raw)
    preprocess(config)
    return config


def make_config(upstream: PreprocessConfig, **overrides: Any) -> DetectConfig:
    return DetectConfig(
        record_root=upstream.record_root,
        cut_root=upstream.cut_root,
        retilt_root=upstream.retilt_root,
        crop_root=upstream.crop_root,
        **overrides,
    )


def _promoted_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint with a valid card. The weights are never loaded: the detector is stubbed."""

    checkpoint = tmp_path / "models" / "s5_barbell" / "m.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    card = CheckpointCard(
        architecture="fasterrcnn_resnet50_fpn_v2",
        classes=("cap", "collar"),
        training_epoch=6,
        training_run="r",
        source_checkpoint="r/best.pt",
        sha256=sha256_file(checkpoint),
    )
    card_path(checkpoint).write_text(json.dumps(card.to_document()))
    return checkpoint


@pytest.fixture
def upstream(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any, make_session_metadata: Any
) -> PreprocessConfig:
    return published(tmp_path, make_camera, make_meta_template, make_session_metadata)


# --- S6 pose and S7 lift: each opt-in, and independently ---------------------------------


def test_the_run_detects_pose_when_a_detector_is_given(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    config = make_config(upstream, pose_root=tmp_path / "s6_pose_output")

    manifest = detect(config, detector=StubDetector())

    assert [scan.scan_id for scan in manifest.scans] == [
        "9 July/cnj_45kg_Set1/Front",
        "9 July/cnj_45kg_Set1/Side",
    ]
    assert all(scan.steps == ["pose"] for scan in manifest.scans)
    assert manifest.rejected_scans == []  # S0-S4's rejections stay in preprocess's manifest
    assert config.pose_root is not None
    assert skeleton_path(config.pose_root).is_file()
    assert (config.pose_root / "manifest.json").is_file()
    for scan in manifest.scans:
        date, session, camera = scan.scan_id.split("/", 2)
        document = read_pose(pose_path(config.pose_root, Path(date) / session, camera))
        # S6 stops in pixel space: no depth was read, so no positions were computed.
        assert not document.has_positions


def test_the_run_lifts_pose_when_a_lift_root_is_given(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    config = make_config(
        upstream, pose_root=tmp_path / "s6_pose_output", lift_root=tmp_path / "s7_lift_output"
    )

    manifest = detect(config, detector=StubDetector())

    assert all(scan.steps == ["pose", "lift"] for scan in manifest.scans)
    assert config.lift_root is not None and config.pose_root is not None
    assert skeleton_path(config.lift_root).is_file()
    assert (config.lift_root / "manifest.json").is_file()  # the last stage's tree
    for scan in manifest.scans:
        date, session, camera = scan.scan_id.split("/", 2)
        detected = read_pose(pose_path(config.pose_root, Path(date) / session, camera))
        lifted = read_pose(pose_path(config.lift_root, Path(date) / session, camera))
        assert not detected.has_positions
        assert lifted.has_positions


def test_s7_lifts_an_earlier_runs_detections_without_a_detector(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    """The point of splitting the stages: re-run the depth half without re-detecting."""

    detect(make_config(upstream, pose_root=tmp_path / "s6_pose_output"), detector=StubDetector())
    lift_only = make_config(
        upstream, pose_root=tmp_path / "s6_pose_output", lift_root=tmp_path / "s7_lift_output"
    )

    manifest = detect(lift_only)  # no detector at all

    assert manifest.scans
    assert all(scan.steps == ["lift"] for scan in manifest.scans)
    assert lift_only.lift_root is not None
    for scan in manifest.scans:
        date, session, camera = scan.scan_id.split("/", 2)
        assert read_pose(pose_path(lift_only.lift_root, Path(date) / session, camera)).has_positions


def test_only_restricts_detect_to_matching_captures(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    config = make_config(upstream, pose_root=tmp_path / "s6_pose_output", only="*/Side")

    manifest = detect(config, detector=StubDetector())

    assert [scan.scan_id for scan in manifest.scans] == ["9 July/cnj_45kg_Set1/Side"]


def test_detect_refuses_a_run_with_nothing_to_do(upstream: PreprocessConfig) -> None:
    with pytest.raises(ValueError, match="nothing to run"):
        detect(make_config(upstream))


def test_detect_refuses_a_missing_s4_tree(tmp_path: Path) -> None:
    config = DetectConfig(
        record_root=tmp_path / "s0",
        cut_root=tmp_path / "s1",
        retilt_root=tmp_path / "s3",
        crop_root=tmp_path / "s4",
        pose_root=tmp_path / "s6",
    )

    with pytest.raises(ValueError, match="not a directory"):
        detect(config, detector=StubDetector())
    assert not (tmp_path / "s6").exists()


def test_detect_rejects_a_detector_without_a_pose_root(upstream: PreprocessConfig) -> None:
    with pytest.raises(ValueError, match="pose_root"):
        detect(make_config(upstream), detector=StubDetector())


def test_detect_rejects_a_lift_root_without_a_pose_root(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    with pytest.raises(ValueError, match="nothing to lift"):
        detect(make_config(upstream, lift_root=tmp_path / "s7_lift_output"))


# --- S5 barbell: runs before S6, which reads what it published ---------------------------


def test_the_run_publishes_barbell_detections_before_pose(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    config = make_config(
        upstream,
        barbell_root=tmp_path / "s5_barbell_output",
        pose_root=tmp_path / "s6_pose_output",
    )

    manifest = detect(config, detector=StubDetector(), barbell_detector=StubBarbellDetector())

    assert manifest.scans
    assert config.barbell_root is not None
    for scan in manifest.scans:
        assert scan.steps == ["barbell", "pose"]
        assert scan.derived["barbell_frames_with_both"] > 0
        assert (config.barbell_root / scan.scan_id / "barbell.json").is_file()
        # S6 found this run's detections; it stops only at the undeclared bar type.
        assert not any("no barbell detections" in w for w in scan.warnings)
        assert any("barbell type" in w for w in scan.warnings)


def test_the_flow_builds_the_detector_from_the_checkpoint(
    tmp_path: Path, upstream: PreprocessConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What lets the Prefect deployment run S5: nothing but config crosses into the flow."""

    config = make_config(
        upstream,
        barbell_root=tmp_path / "s5_barbell_output",
        barbell_checkpoint=_promoted_checkpoint(tmp_path),
    )
    built: list[DetectConfig] = []
    detector = StubBarbellDetector()

    def build(config: DetectConfig) -> StubBarbellDetector:
        built.append(config)
        return detector

    monkeypatch.setattr(detect_module, "build_barbell_detector", build)

    manifest = detect(config)

    assert len(built) == 1  # once per run, shared by every capture
    assert detector.calls == len(manifest.scans) > 0
    assert all(scan.steps == ["barbell"] for scan in manifest.scans)
    assert (tmp_path / "s5_barbell_output" / "manifest.json").is_file()


def test_a_dry_run_plans_s5_without_loading_a_model(
    tmp_path: Path, upstream: PreprocessConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(
        upstream,
        barbell_root=tmp_path / "s5_barbell_output",
        barbell_checkpoint=_promoted_checkpoint(tmp_path),
        dry_run=True,
    )

    def refuse(config: DetectConfig) -> None:
        raise AssertionError("a dry run must not load the model")

    monkeypatch.setattr(detect_module, "build_barbell_detector", refuse)

    manifest = detect(config)

    assert manifest.scans
    assert all(scan.status == "planned" and "barbell" in scan.steps for scan in manifest.scans)
    assert not (tmp_path / "s5_barbell_output").exists()


def test_without_a_checkpoint_s5_does_not_run(tmp_path: Path, upstream: PreprocessConfig) -> None:
    config = make_config(
        upstream, barbell_root=tmp_path / "s5_barbell_output", pose_root=tmp_path / "s6"
    )

    manifest = detect(config, detector=StubDetector())

    assert all("barbell" not in scan.steps for scan in manifest.scans)


def test_detect_rejects_a_barbell_detector_without_a_barbell_root(
    upstream: PreprocessConfig,
) -> None:
    with pytest.raises(ValueError, match="barbell_root is unset"):
        detect(make_config(upstream), barbell_detector=StubBarbellDetector())


def test_detect_rejects_a_checkpoint_without_a_barbell_root(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    config = make_config(upstream, barbell_checkpoint=_promoted_checkpoint(tmp_path))

    with pytest.raises(ValueError, match="barbell_root is unset"):
        detect(config)


def test_detect_rejects_a_checkpoint_without_a_card_before_any_work(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    checkpoint = tmp_path / "m.pt"
    checkpoint.write_bytes(b"weights")
    config = make_config(
        upstream, barbell_root=tmp_path / "s5_barbell_output", barbell_checkpoint=checkpoint
    )

    with pytest.raises(FileNotFoundError, match="has no card"):
        detect(config)
    assert not (tmp_path / "s5_barbell_output").exists()
