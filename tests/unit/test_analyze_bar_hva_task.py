"""S10 · Bar HVA as a task: reads every input off another stage's published output, writes
`bar_hva.json`, and never rejects a capture. The maths is `test_analyze_tikhonov`'s and the
document `test_analyze_bar_hva`'s; this covers the plumbing on the synthetic S0-S4 capture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from powerflow_pipeline.data.analyze.config import AnalyzeConfig
from powerflow_pipeline.data.analyze.tasks import bar_hva as task
from powerflow_pipeline.data.analyze.tasks.bar_hva import bar_hva_path, solve_bar_hva_camera
from powerflow_pipeline.data.common.bar_geometry import BarPose
from powerflow_pipeline.data.common.capture_models import CameraRecord
from powerflow_pipeline.data.common.errors import PublishError
from tests.conftest import MakeCamera, MakeSessionMetadata
from tests.unit.test_analyze_kinematics_task import make_config
from tests.unit.test_detect_pose import (
    _bar_scene,
    _build_crop_record,
    _declare_bar,
    _make_pose_session,
    _write_detections,
)


@pytest.fixture
def capture(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> tuple[AnalyzeConfig, CameraRecord]:
    raw = _make_pose_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(
        tmp_path, barbell_root=tmp_path / "s5", bar_hva_root=tmp_path / "s10_hva_output"
    )
    return config, _build_crop_record(raw, config)


def _destination(config: AnalyzeConfig, record: CameraRecord) -> Path:
    assert config.bar_hva_root is not None
    return bar_hva_path(config.bar_hva_root, record)


def _publish_bar(config: AnalyzeConfig, record: CameraRecord) -> None:
    _declare_bar(record)
    _write_detections(config, record, _bar_scene(record, record.n_frames))


def test_dry_run_plans_the_write_and_writes_nothing(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture
    _publish_bar(config, record)
    config = config.model_copy(update={"dry_run": True})

    _, step = solve_bar_hva_camera.fn(record, config)

    assert [op.op for op in step.file_ops] == ["write"]
    assert not _destination(config, record).exists()


def test_without_detections_it_warns_and_writes_nothing(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture

    _, step = solve_bar_hva_camera.fn(record, config)

    assert any("bar HVA not solved" in w and "no barbell detections" in w for w in step.warnings)
    assert not _destination(config, record).exists()


def test_an_undeclared_bar_is_not_guessed(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture
    _write_detections(config, record, _bar_scene(record, record.n_frames))

    _, step = solve_bar_hva_camera.fn(record, config)

    assert any("barbell type" in w for w in step.warnings)
    assert not _destination(config, record).exists()


def test_publishes_bar_hva_timed_by_s4_and_s1_and_reports_to_the_manifest(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture
    _publish_bar(config, record)

    _, step = solve_bar_hva_camera.fn(record, config)

    document = json.loads(_destination(config, record).read_text())
    assert document["captureId"] == record.capture_id
    assert document["frames"]["count"] == record.n_frames
    sidecar = json.loads((config.cut_root / record.relative / "cut_sidecar.json").read_text())
    assert document["frames"]["startEpochMs"] == sidecar["retained"]["rgb"]["epoch_range"][0]
    assert document["frames"]["tMs"][0] == 0
    assert document["bar"]["type"] == "male IWF"
    assert document["settings"]["vMaxMps"] == 5.0
    assert document["settings"]["aMaxMps2"] == 25.0
    assert document["floor"]["anchor"] == "floor-plane"  # S3's; too few frames for a rest anchor
    # The fixture's camera is scaled 1/30, so 2 px of landmark noise leaves no yaw measurable
    # and S5's bar is placed on no frame: published, and said so.
    assert document["com"]["heightM"] == [None] * record.n_frames
    assert any("placed on no frame" in w for w in step.warnings)


def test_placed_bars_reach_the_document_on_s3s_floor(
    capture: tuple[AnalyzeConfig, CameraRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, record = capture
    _publish_bar(config, record)
    bars = [BarPose((0.0, 1000.0, 2000.0), (1.0, 1000.0, 3000.0), (0.5, 1000.0, 2500.0), 38.0)]
    monkeypatch.setattr(task, "bar_track", lambda *_: (bars * record.n_frames, []))

    _, step = solve_bar_hva_camera.fn(record, config)

    document = json.loads(_destination(config, record).read_text())
    y_floor = document["floor"]["yFloorM"]
    assert document["com"]["heightM"] == [pytest.approx(y_floor - 1.0)] * record.n_frames
    # Five frames: placed and published, too short to differentiate.
    assert document["com"]["segments"][0]["status"] == "too_short"
    assert step.derived["bar_hva_status"] == {"too_short": 1}
    assert any("too_short" in w for w in step.warnings)


def test_a_frame_clock_that_disagrees_with_the_record_is_not_solved(
    capture: tuple[AnalyzeConfig, CameraRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, record = capture
    _publish_bar(config, record)
    monkeypatch.setattr(task, "frame_offsets_ms", lambda _path: [0, 17])

    _, step = solve_bar_hva_camera.fn(record, config)

    assert any("frame_count_mismatch" in w for w in step.warnings)
    assert not _destination(config, record).exists()


def test_refuses_to_overwrite_unless_asked(
    capture: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = capture
    _publish_bar(config, record)
    solve_bar_hva_camera.fn(record, config)

    with pytest.raises(PublishError):
        solve_bar_hva_camera.fn(record, config)
    solve_bar_hva_camera.fn(record, config.model_copy(update={"overwrite": True}))
