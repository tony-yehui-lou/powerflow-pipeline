"""S9 · Phase as a task: reads S8's published `joint_hva.json`, S5's bar and S6's grip, writes
`phase.json`, and never rejects a capture. The rules are `test_analyze_phase_events`'s; this
covers the plumbing on the synthetic S0-S4 capture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from powerflow_pipeline.data.analyze.config import AnalyzeConfig
from powerflow_pipeline.data.analyze.tasks import phase as phase_task
from powerflow_pipeline.data.analyze.tasks.kinematics import (
    kinematics_path,
    solve_kinematics_camera,
)
from powerflow_pipeline.data.analyze.tasks.phase import detect_phases_camera, phase_path
from powerflow_pipeline.data.common.capture_models import CameraRecord
from powerflow_pipeline.data.common.errors import PublishError
from powerflow_pipeline.data.common.pose_storage import write_sidecar
from tests.unit.test_analyze_kinematics_task import (  # noqa: F401 -- `capture` is a fixture
    _add_athlete,
    _publish_pose,
    capture,
)
from tests.unit.test_analyze_phase_events import SNATCH_V, _curve


@pytest.fixture
def phased(capture: tuple[AnalyzeConfig, CameraRecord]) -> tuple[AnalyzeConfig, CameraRecord]:  # noqa: F811
    config, record = capture
    root = config.kinematics_root
    assert root is not None
    return config.model_copy(update={"phase_root": root.parent / "s9_phase_output"}), record


def _declare_lift(record: CameraRecord, lift_type: str) -> None:
    text = record.metadata_path.read_text()
    record.metadata_path.write_text(text.replace("lift:\n", f"lift:\n  type: {lift_type}\n", 1))


def _published(config: AnalyzeConfig, record: CameraRecord) -> dict[str, Any]:
    assert config.phase_root is not None
    return json.loads(phase_path(config.phase_root, record).read_text())  # type: ignore[no-any-return]


def test_dry_run_plans_the_write_and_writes_nothing(
    phased: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = phased
    _, step = detect_phases_camera.fn(record, config.model_copy(update={"dry_run": True}))

    assert config.phase_root is not None
    assert [op.op for op in step.file_ops] == ["write"]
    assert not phase_path(config.phase_root, record).exists()


def test_without_s8_kinematics_it_warns_and_writes_nothing(
    phased: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = phased
    _, step = detect_phases_camera.fn(record, config)

    assert any("no S8 kinematics" in w for w in step.warnings)
    assert config.phase_root is not None
    assert not phase_path(config.phase_root, record).exists()


def _solve_s8(config: AnalyzeConfig, record: CameraRecord) -> None:
    _publish_pose(config, record)
    _add_athlete(record)
    solve_kinematics_camera.fn(record, config)


@pytest.mark.parametrize("lift_type", [None, "deadlift"])
def test_a_lift_type_it_does_not_know_is_not_guessed(
    phased: tuple[AnalyzeConfig, CameraRecord], lift_type: str | None
) -> None:
    config, record = phased
    _solve_s8(config, record)
    if lift_type is not None:
        _declare_lift(record, lift_type)

    _, step = detect_phases_camera.fn(record, config)

    assert any("lift.type" in w for w in step.warnings)
    assert config.phase_root is not None
    assert not phase_path(config.phase_root, record).exists()


def test_publishes_phase_json_from_the_published_inputs(
    phased: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = phased
    _solve_s8(config, record)
    _declare_lift(record, "CnJ")

    _, step = detect_phases_camera.fn(record, config)

    document = _published(config, record)
    assert document["schemaVersion"] == "1.2.0"
    assert document["captureId"] == record.capture_id
    assert document["lift"] == "clean_and_jerk"
    assert set(document["bands"]) >= {"knee", "hip", "rack", "overhead"}
    assert list(document["events"])[-1] == "stand"
    assert "rackStand" in document["auxiliary"]
    # No wrist_refinement.json from S6: the arms are assumed vertical, and the output says so.
    assert "grip_assumed" in document["flags"]
    assert step.derived["phase_lift"] == "clean_and_jerk"
    # S8 found no barbell detections, so neither does S9, and the manifest says so.
    assert any("no barbell detections" in w for w in step.warnings)


def _joint_hva(config: AnalyzeConfig, record: CameraRecord, n: int) -> None:
    """A hand-built S8 document: the 10 September lifter's bones, a floor, and no knees."""

    def bone(m: float) -> dict[str, object]:
        return {"usedM": m, "source": "measured"}

    joints = {
        name: {"heightM": [None] * n, "heightSigmaM": [None] * n, "flags": [[]] * n}
        for name in ("leftKnee", "rightKnee")
    }
    document = {
        "captureId": record.capture_id,
        "frames": {"count": n, "startEpochMs": 1_000_000, "tMs": _curve(SNATCH_V).t_ms},
        "floor": {"yFloorM": 1.364},
        "ankleHeight": {"usedM": 0.105},
        "bones": {
            **{f"{s}Tibia": bone(0.398) for s in ("left", "right")},
            **{f"{s}Femur": bone(0.423) for s in ("left", "right")},
            **{f"{s}Torso": bone(0.515) for s in ("left", "right")},
            **{f"{s}UpperArm": bone(0.290) for s in ("left", "right")},
            **{f"{s}Forearm": bone(0.280) for s in ("left", "right")},
            "shoulderWidth": bone(0.440),
        },
        "joints": joints,
    }
    assert config.kinematics_root is not None
    write_sidecar(kinematics_path(config.kinematics_root, record), document)


def test_a_snatch_curve_publishes_every_event_and_phase(
    phased: tuple[AnalyzeConfig, CameraRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, record = phased
    curve = _curve(SNATCH_V)
    _joint_hva(config, record, len(curve.t_ms))
    _declare_lift(record, "snch")
    assert config.pose_root is not None
    pose_dir = config.pose_root / record.relative
    write_sidecar(pose_dir / "wrist_refinement.json", {"halfGripMm": 457.1, "near": "rightWrist"})
    monkeypatch.setattr(phase_task, "bar_heights", lambda *_: (curve.bar, []))

    _, step = detect_phases_camera.fn(record, config)

    document = _published(config, record)
    events = document["events"]
    assert all(e["status"] in ("found", "fallback") for e in events.values())
    assert events["barPassesKnee"]["status"] == "fallback"  # no S8 knee: the knee band
    assert events["catch"]["epochMs"] == 1_000_000 + events["catch"]["tMs"]
    assert round(document["bands"]["overhead"]["centreMm"]) == 1959
    assert [p["phase"] for p in document["phases"]] == [
        "setup", "first_pull", "second_pull", "turnover", "recovery", "finish",
    ]  # fmt: skip
    assert document["result"] == {"value": "made", "source": "rules"}
    assert "grip_assumed" not in document["flags"]
    assert step.derived["phase_events_found"] == "5/5"


def test_without_bones_there_are_no_bands(
    phased: tuple[AnalyzeConfig, CameraRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, record = phased
    curve = _curve(SNATCH_V)
    _joint_hva(config, record, len(curve.t_ms))
    assert config.kinematics_root is not None
    path = kinematics_path(config.kinematics_root, record)
    document = json.loads(path.read_text())
    del document["bones"]
    path.unlink()
    write_sidecar(path, document)
    _declare_lift(record, "snch")
    monkeypatch.setattr(phase_task, "bar_heights", lambda *_: (curve.bar, []))

    _, step = detect_phases_camera.fn(record, config)

    assert any("no bone lengths" in w for w in step.warnings)
    published = _published(config, record)
    assert set(published["bands"]) == {"floor"}
    assert published["events"]["catch"]["status"] == "absent"


def test_without_a_floor_there_are_no_bar_heights(
    phased: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = phased
    heights, warnings = phase_task.bar_heights(record, config, None)
    assert heights == [None] * record.n_frames
    assert any("no floor" in w for w in warnings)


def test_reads_the_raw_lift_type_and_result(tmp_path: Path) -> None:
    path = tmp_path / "metadata.yaml"
    assert phase_task.read_lift(path) == (None, None)
    path.write_text("lift:\n  type: snch\n  result: [success, fail]\n")
    assert phase_task.read_lift(path) == ("snch", ["success", "fail"])
    path.write_text("athlete: {}\n")
    assert phase_task.read_lift(path) == (None, None)


def test_refuses_to_overwrite_unless_asked(
    phased: tuple[AnalyzeConfig, CameraRecord],
) -> None:
    config, record = phased
    _solve_s8(config, record)
    _declare_lift(record, "snch")
    detect_phases_camera.fn(record, config)

    with pytest.raises(PublishError):
        detect_phases_camera.fn(record, config)
    detect_phases_camera.fn(record, config.model_copy(update={"overwrite": True}))
