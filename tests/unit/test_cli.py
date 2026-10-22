"""The `powerflow` entry point exposes the data-pipeline command group."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from powerflow_pipeline.data.cli import app
from powerflow_pipeline.data.preprocess.config import (
    BarbellDevice,
    PoseDetector,
    PreprocessConfig,
)

runner = CliRunner()

# `cli._build_detector` imports RTMPose lazily, so it is replaced at its own module.
RTMPOSE_DETECTOR = "powerflow_pipeline.data.preprocess.rtmpose_detector.RTMPoseDetector"


def test_help_lists_the_command_group() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "PowerFlow data pipelines" in result.stdout


def test_bare_invocation_shows_help_rather_than_a_traceback() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2  # conventional "no command given" usage exit
    assert "Usage" in result.stdout


def test_preprocess_command_accepts_retilt_root(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    result = runner.invoke(
        app,
        [
            "preprocess",
            "--input",
            str(raw),
            "--records",
            str(tmp_path / "s0"),
            "--cut",
            str(tmp_path / "s1"),
            "--retilt",
            str(tmp_path / "s3"),
            "--crop",
            str(tmp_path / "s4"),
            "--output",
            str(tmp_path / "s2"),
        ],
    )

    assert result.exit_code == 0, result.output


def test_preprocess_command_skips_pose_when_pose_flag_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_if_constructed(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("no pose detector should be constructed without --pose")

    monkeypatch.setattr("powerflow_pipeline.data.cli.MediaPipePoseDetector", _fail_if_constructed)
    monkeypatch.setattr(RTMPOSE_DETECTOR, _fail_if_constructed)
    raw = tmp_path / "raw"
    raw.mkdir()

    result = runner.invoke(
        app,
        [
            "preprocess",
            "--input",
            str(raw),
            "--records",
            str(tmp_path / "s0"),
            "--cut",
            str(tmp_path / "s1"),
            "--retilt",
            str(tmp_path / "s3"),
            "--crop",
            str(tmp_path / "s4"),
            "--output",
            str(tmp_path / "s2"),
        ],
    )

    assert result.exit_code == 0, result.output


def test_preprocess_command_accepts_pose_root_and_builds_a_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class _BuiltDetector:
        """Satisfies `Detector2D` structurally -- the flow's own parameter validation
        rejects anything that doesn't, which is the contract worth asserting here."""

        def __init__(self, variant: str, **kwargs: Any) -> None:
            calls.append(f"rtmpose-{variant}")

        def detect(self, rgb_path: Path, n_frames: int) -> dict[str, object]:
            return {}

    monkeypatch.setattr(RTMPOSE_DETECTOR, _BuiltDetector)
    raw = tmp_path / "raw"
    raw.mkdir()

    result = runner.invoke(
        app,
        [
            "preprocess",
            "--input",
            str(raw),
            "--records",
            str(tmp_path / "s0"),
            "--cut",
            str(tmp_path / "s1"),
            "--retilt",
            str(tmp_path / "s3"),
            "--crop",
            str(tmp_path / "s4"),
            "--output",
            str(tmp_path / "s2"),
            "--pose",
            str(tmp_path / "s6"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["rtmpose-balanced"]  # RTMPose, balanced, without any detector flag


def _preprocess_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "preprocess",
        "--input",
        str(tmp_path / "raw"),
        "--records",
        str(tmp_path / "s0"),
        "--cut",
        str(tmp_path / "s1"),
        "--retilt",
        str(tmp_path / "s3"),
        "--crop",
        str(tmp_path / "s4"),
        "--output",
        str(tmp_path / "s2"),
        "--pose",
        str(tmp_path / "s6"),
        *extra,
    ]


class _StubDetector:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _StubMediaPipe:
    pass


def _capture_flow_call(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, Any]]:
    """Record `(config, detector)` as the CLI hands them to the flow, running neither."""

    calls: list[tuple[Any, Any]] = []

    def _record(config: Any, detector: Any = None) -> Any:
        calls.append((config, detector))
        return type("Manifest", (), {"scans": [], "rejected_scans": []})()

    monkeypatch.setattr("powerflow_pipeline.data.cli.MediaPipePoseDetector", _StubMediaPipe)
    monkeypatch.setattr(RTMPOSE_DETECTOR, _StubDetector)
    monkeypatch.setattr("powerflow_pipeline.data.cli.preprocess_flow", _record)
    return calls


def test_pose_without_lift_runs_detection_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path))

    assert result.exit_code == 0, result.output
    config, detector = calls[0]
    assert isinstance(detector, _StubDetector)
    assert config.lift_root is None


def test_lift_flag_enables_s7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path, "--lift", str(tmp_path / "s7")))

    assert result.exit_code == 0, result.output
    config, detector = calls[0]
    assert config.lift_root == tmp_path / "s7"
    assert isinstance(detector, _StubDetector)  # S6 still runs: --skip-detect was not given


def test_skip_detect_builds_no_detector_so_s7_can_reuse_an_earlier_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(
        app, _preprocess_args(tmp_path, "--lift", str(tmp_path / "s7"), "--skip-detect")
    )

    assert result.exit_code == 0, result.output
    config, detector = calls[0]
    assert detector is None  # no checkpoint is loaded at all
    assert config.pose_root == tmp_path / "s6"  # still names the tree S7 reads
    assert config.lift_root == tmp_path / "s7"


def test_barbell_flag_points_s6_at_the_barbell_detections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path, "--barbell", str(tmp_path / "s5")))

    assert result.exit_code == 0, result.output
    config, _ = calls[0]
    assert config.barbell_root == tmp_path / "s5"
    assert config.barbell_min_score == 0.5


def test_without_barbell_flag_s6_wrists_are_left_as_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path))

    assert result.exit_code == 0, result.output
    config, _ = calls[0]
    assert config.barbell_root is None


def test_kinematics_flag_enables_s8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(
        app,
        _preprocess_args(tmp_path, "--kinematics", str(tmp_path / "s8")),
    )

    assert result.exit_code == 0, result.output
    config, _ = calls[0]
    assert config.kinematics_root == tmp_path / "s8"


def test_without_kinematics_flag_s8_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path))

    assert result.exit_code == 0, result.output
    config, _ = calls[0]
    assert config.kinematics_root is None


def test_phase_flag_enables_s9(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path, "--phase", str(tmp_path / "s9")))

    assert result.exit_code == 0, result.output
    config, _ = calls[0]
    assert config.phase_root == tmp_path / "s9"


def test_barbell_checkpoint_flags_reach_the_config_and_no_detector_is_built_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(
        app,
        _preprocess_args(
            tmp_path,
            "--barbell",
            str(tmp_path / "s5"),
            "--barbell-checkpoint",
            str(tmp_path / "m.pt"),
            "--barbell-device",
            "cpu",
            "--barbell-overlay",
        ),
    )

    assert result.exit_code == 0, result.output
    config, _ = calls[0]
    assert config.barbell_checkpoint == tmp_path / "m.pt"
    assert config.barbell_device == BarbellDevice.CPU
    assert config.barbell_render_overlay is True


def test_without_barbell_checkpoint_s5_does_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path, "--barbell", str(tmp_path / "s5")))

    assert result.exit_code == 0, result.output
    config, _ = calls[0]
    assert config.barbell_checkpoint is None
    assert config.barbell_device == BarbellDevice.AUTO
    assert config.barbell_render_overlay is False


def test_rtmpose_is_the_default_pose_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path))

    assert result.exit_code == 0, result.output
    config, detector = calls[0]
    assert config.pose_detector == PoseDetector.RTMPOSE
    assert isinstance(detector, _StubDetector)


def test_mediapipe_is_still_selectable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path, "--pose-detector", "mediapipe"))

    assert result.exit_code == 0, result.output
    config, detector = calls[0]
    assert config.pose_detector == PoseDetector.MEDIAPIPE
    assert isinstance(detector, _StubMediaPipe)


def test_the_config_defaults_to_rtmpose_for_callers_that_bypass_the_cli(tmp_path: Path) -> None:
    config = PreprocessConfig(
        raw_root=tmp_path,
        record_root=tmp_path,
        cut_root=tmp_path,
        retilt_root=tmp_path,
        crop_root=tmp_path,
        output_root=tmp_path,
    )

    assert config.pose_detector == PoseDetector.RTMPOSE
