"""The `powerflow` entry point exposes the data-pipeline command group."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from powerflow_pipeline.data.cli import app

runner = CliRunner()


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
        raise AssertionError("MediaPipePoseDetector should not be constructed without --pose")

    monkeypatch.setattr("powerflow_pipeline.data.cli.MediaPipePoseDetector", _fail_if_constructed)
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

        def __init__(self) -> None:
            calls.append("detector-built")

        def detect(self, rgb_path: Path, n_frames: int) -> dict[str, object]:
            return {}

    monkeypatch.setattr("powerflow_pipeline.data.cli.MediaPipePoseDetector", _BuiltDetector)
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
            str(tmp_path / "s5"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["detector-built"]


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
        str(tmp_path / "s5"),
        *extra,
    ]


class _StubDetector:
    pass


def _capture_flow_call(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, Any]]:
    """Record `(config, detector)` as the CLI hands them to the flow, running neither."""

    calls: list[tuple[Any, Any]] = []

    def _record(config: Any, detector: Any = None) -> Any:
        calls.append((config, detector))
        return type("Manifest", (), {"scans": [], "rejected_scans": []})()

    monkeypatch.setattr("powerflow_pipeline.data.cli.MediaPipePoseDetector", _StubDetector)
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


def test_lift_flag_enables_s6(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(app, _preprocess_args(tmp_path, "--lift", str(tmp_path / "s6")))

    assert result.exit_code == 0, result.output
    config, detector = calls[0]
    assert config.lift_root == tmp_path / "s6"
    assert isinstance(detector, _StubDetector)  # S5 still runs: --skip-detect was not given


def test_skip_detect_builds_no_detector_so_s6_can_reuse_an_earlier_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _capture_flow_call(monkeypatch)
    (tmp_path / "raw").mkdir()

    result = runner.invoke(
        app, _preprocess_args(tmp_path, "--lift", str(tmp_path / "s6"), "--skip-detect")
    )

    assert result.exit_code == 0, result.output
    config, detector = calls[0]
    assert detector is None  # no checkpoint is loaded at all
    assert config.pose_root == tmp_path / "s5"  # still names the tree S6 reads
    assert config.lift_root == tmp_path / "s6"
