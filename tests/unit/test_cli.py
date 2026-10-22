"""The `powerflow` entry point: one command per pipeline, plus `all` to chain them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from powerflow_pipeline.data.analyze.config import AnalyzeConfig, AnalyzeStage
from powerflow_pipeline.data.cli import app
from powerflow_pipeline.data.detect.config import BarbellDevice, DetectConfig, PoseDetector
from powerflow_pipeline.data.preprocess.config import PreprocessConfig

runner = CliRunner()

# `cli._build_detector` imports RTMPose lazily, so it is replaced at its own module.
RTMPOSE_DETECTOR = "powerflow_pipeline.data.detect.rtmpose_detector.RTMPoseDetector"


def test_help_lists_the_command_group() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "PowerFlow data pipelines" in result.stdout
    for command in ("preprocess", "detect", "analyze", "all", "phase-eval"):
        assert command in result.stdout


def test_bare_invocation_shows_help_rather_than_a_traceback() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2  # conventional "no command given" usage exit
    assert "Usage" in result.stdout


def _upstream(tmp_path: Path) -> list[str]:
    """The S0-S4 roots detect and analyze rebuild each camera from."""

    return [
        "--records",
        str(tmp_path / "s0"),
        "--cut",
        str(tmp_path / "s1"),
        "--retilt",
        str(tmp_path / "s3"),
        "--crop",
        str(tmp_path / "s4"),
    ]


class _Manifest:
    def __init__(self, pipeline: str) -> None:
        self.pipeline = pipeline
        self.scans: list[object] = []
        self.rejected_scans: list[object] = []


def _capture(monkeypatch: pytest.MonkeyPatch, name: str) -> list[tuple[Any, ...]]:
    """Record what the CLI hands the flow `name`, running nothing."""

    calls: list[tuple[Any, ...]] = []

    def _record(*args: Any, **kwargs: Any) -> _Manifest:
        calls.append((*args, *kwargs.values()))
        return _Manifest(name)

    monkeypatch.setattr(f"powerflow_pipeline.data.cli.{name}_flow", _record)
    return calls


class _StubDetector:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _StubMediaPipe:
    pass


@pytest.fixture
def detectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("powerflow_pipeline.data.cli.MediaPipePoseDetector", _StubMediaPipe)
    monkeypatch.setattr(RTMPOSE_DETECTOR, _StubDetector)


# --- preprocess: S0-S4 only ---------------------------------------------------------------


def _preprocess_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "preprocess",
        "--input",
        str(tmp_path / "raw"),
        *_upstream(tmp_path),
        "--output",
        str(tmp_path / "s2"),
        *extra,
    ]


def test_preprocess_runs_on_an_empty_raw_root(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path))

    assert result.exit_code == 0, result.output
    assert "preprocess: processed 0 camera(s)" in result.output


def test_preprocess_no_longer_takes_detect_or_analyze_roots(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()

    for flag in ("--pose", "--barbell", "--kinematics", "--phase", "--lift"):
        result = runner.invoke(app, _preprocess_args(tmp_path, flag, str(tmp_path / "x")))
        assert result.exit_code == 2, flag  # usage error: no such option


def test_preprocess_hands_its_config_to_the_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture(monkeypatch, "preprocess")

    result = runner.invoke(app, _preprocess_args(tmp_path, "--only", "9 July/*", "--overwrite"))

    assert result.exit_code == 0, result.output
    [(config,)] = calls
    assert isinstance(config, PreprocessConfig)
    assert config.raw_root == tmp_path / "raw"
    assert config.crop_root == tmp_path / "s4"
    assert config.only == "9 July/*"
    assert config.overwrite is True


# --- detect: S5, S6, and S7 off by default ------------------------------------------------


def _detect_args(tmp_path: Path, *extra: str) -> list[str]:
    return ["detect", *_upstream(tmp_path), "--pose", str(tmp_path / "s6"), *extra]


@pytest.mark.usefixtures("detectors")
def test_pose_without_lift_runs_detection_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture(monkeypatch, "detect")

    result = runner.invoke(app, _detect_args(tmp_path))

    assert result.exit_code == 0, result.output
    [(config, detector)] = calls
    assert isinstance(config, DetectConfig)
    assert config.crop_root == tmp_path / "s4"
    assert config.pose_root == tmp_path / "s6"
    assert isinstance(detector, _StubDetector)
    assert config.lift_root is None  # S7 is off unless asked for


def test_without_pose_no_detector_is_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_if_constructed(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("no pose detector should be constructed without --pose")

    monkeypatch.setattr("powerflow_pipeline.data.cli.MediaPipePoseDetector", _fail_if_constructed)
    monkeypatch.setattr(RTMPOSE_DETECTOR, _fail_if_constructed)
    calls = _capture(monkeypatch, "detect")

    result = runner.invoke(
        app,
        ["detect", *_upstream(tmp_path), "--barbell", str(tmp_path / "s5"),
         "--barbell-checkpoint", str(tmp_path / "m.pt")],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    [(_, detector)] = calls
    assert detector is None


def test_rtmpose_balanced_is_built_without_any_detector_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []

    class _BuiltDetector:
        def __init__(self, variant: str, **kwargs: Any) -> None:
            built.append(f"rtmpose-{variant}")

    monkeypatch.setattr(RTMPOSE_DETECTOR, _BuiltDetector)
    _capture(monkeypatch, "detect")

    result = runner.invoke(app, _detect_args(tmp_path))

    assert result.exit_code == 0, result.output
    assert built == ["rtmpose-balanced"]


@pytest.mark.usefixtures("detectors")
def test_lift_flag_enables_s7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch, "detect")

    result = runner.invoke(app, _detect_args(tmp_path, "--lift", str(tmp_path / "s7")))

    assert result.exit_code == 0, result.output
    [(config, detector)] = calls
    assert config.lift_root == tmp_path / "s7"
    assert isinstance(detector, _StubDetector)  # S6 still runs: --skip-detect was not given


@pytest.mark.usefixtures("detectors")
def test_skip_detect_builds_no_detector_so_s7_can_reuse_an_earlier_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture(monkeypatch, "detect")

    result = runner.invoke(
        app, _detect_args(tmp_path, "--lift", str(tmp_path / "s7"), "--skip-detect")
    )

    assert result.exit_code == 0, result.output
    [(config, detector)] = calls
    assert detector is None  # no checkpoint is loaded at all
    assert config.pose_root == tmp_path / "s6"  # still names the tree S7 reads


@pytest.mark.usefixtures("detectors")
def test_barbell_flags_reach_the_config_and_no_s5_detector_is_built_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture(monkeypatch, "detect")

    result = runner.invoke(
        app,
        _detect_args(
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
    [(config, _)] = calls
    assert config.barbell_root == tmp_path / "s5"
    assert config.barbell_checkpoint == tmp_path / "m.pt"
    assert config.barbell_device == BarbellDevice.CPU
    assert config.barbell_render_overlay is True
    assert config.barbell_min_score == 0.5


@pytest.mark.usefixtures("detectors")
def test_without_barbell_flags_s5_does_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture(monkeypatch, "detect")

    result = runner.invoke(app, _detect_args(tmp_path))

    assert result.exit_code == 0, result.output
    [(config, _)] = calls
    assert config.barbell_root is None
    assert config.barbell_checkpoint is None
    assert config.barbell_device == BarbellDevice.AUTO


@pytest.mark.usefixtures("detectors")
def test_mediapipe_is_still_selectable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch, "detect")

    result = runner.invoke(app, _detect_args(tmp_path, "--pose-detector", "mediapipe"))

    assert result.exit_code == 0, result.output
    [(config, detector)] = calls
    assert config.pose_detector == PoseDetector.MEDIAPIPE
    assert isinstance(detector, _StubMediaPipe)


def test_the_config_defaults_to_rtmpose_for_callers_that_bypass_the_cli(tmp_path: Path) -> None:
    config = DetectConfig(
        record_root=tmp_path, cut_root=tmp_path, retilt_root=tmp_path, crop_root=tmp_path
    )

    assert config.pose_detector == PoseDetector.RTMPOSE
    assert config.lift_root is None


# --- analyze: S8 and S9 -------------------------------------------------------------------


def _analyze_args(tmp_path: Path, *extra: str) -> list[str]:
    return ["analyze", *_upstream(tmp_path), "--pose", str(tmp_path / "s6"), *extra]


def test_analyze_hands_its_roots_to_the_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture(monkeypatch, "analyze")

    result = runner.invoke(
        app,
        _analyze_args(
            tmp_path,
            "--barbell",
            str(tmp_path / "s5"),
            "--kinematics",
            str(tmp_path / "s8"),
            "--phase",
            str(tmp_path / "s9"),
        ),
    )

    assert result.exit_code == 0, result.output
    [(config,)] = calls
    assert isinstance(config, AnalyzeConfig)
    assert config.pose_root == tmp_path / "s6"
    assert config.kinematics_root == tmp_path / "s8"
    assert config.phase_root == tmp_path / "s9"
    assert config.runs_kinematics and config.runs_phase  # both stages by default


def test_stages_selects_s9_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch, "analyze")

    result = runner.invoke(
        app,
        _analyze_args(
            tmp_path,
            "--kinematics",
            str(tmp_path / "s8"),
            "--phase",
            str(tmp_path / "s9"),
            "--stages",
            "s9",
        ),
    )

    assert result.exit_code == 0, result.output
    [(config,)] = calls
    assert config.stages == (AnalyzeStage.PHASE,)
    assert not config.runs_kinematics  # S8's tree is read, not rewritten
    assert config.runs_phase


def test_analyze_requires_the_pose_tree(tmp_path: Path) -> None:
    result = runner.invoke(app, ["analyze", *_upstream(tmp_path)])

    assert result.exit_code == 2
    assert "--pose" in result.output


# --- all: the three in order --------------------------------------------------------------


@pytest.mark.usefixtures("detectors")
def test_all_runs_the_three_pipelines_in_order_over_the_standard_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    calls = {name: _capture(monkeypatch, name) for name in ("preprocess", "detect", "analyze")}
    for name, recorded in calls.items():
        original = recorded

        def _note(*args: Any, _name: str = name, _calls: Any = original, **kwargs: Any) -> Any:
            order.append(_name)
            _calls.append((*args, *kwargs.values()))
            return _Manifest(_name)

        monkeypatch.setattr(f"powerflow_pipeline.data.cli.{name}_flow", _note)

    result = runner.invoke(app, ["all", "--data", str(tmp_path), "--only", "10 September/*"])

    assert result.exit_code == 0, result.output
    assert order == ["preprocess", "detect", "analyze"]
    [(pre,)] = calls["preprocess"]
    [(det, detector)] = calls["detect"]
    [(ana,)] = calls["analyze"]
    assert pre.raw_root == tmp_path / "raw"
    assert det.crop_root == pre.crop_root == ana.crop_root == tmp_path / "s4_crop_output"
    assert det.pose_root == ana.pose_root == tmp_path / "s6_pose_output"
    assert det.barbell_checkpoint is None  # S5 is read as published, not re-run
    assert det.lift_root is None
    assert isinstance(detector, _StubDetector)
    assert ana.phase_root == tmp_path / "s9_phase_output"
    assert pre.only == det.only == ana.only == "10 September/*"
