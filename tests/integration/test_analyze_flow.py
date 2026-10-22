"""The analyze pipeline under a real Prefect backend: S8 and S9 on what detect published."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from powerflow_pipeline.data.analyze.config import AnalyzeConfig, AnalyzeStage
from powerflow_pipeline.data.analyze.flow import analyze
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from tests.conftest import MakeCamera
from tests.integration.test_detect_flow import StubDetector, detect, published
from tests.integration.test_detect_flow import make_config as make_detect_config


@pytest.fixture
def upstream(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any, make_session_metadata: Any
) -> PreprocessConfig:
    """Preprocess, then S6 on both surviving cameras into `<tmp>/s6_pose_output`."""

    config = published(tmp_path, make_camera, make_meta_template, make_session_metadata)
    detect(make_detect_config(config, pose_root=tmp_path / "s6_pose_output"), StubDetector())
    return config


def make_config(tmp_path: Path, upstream: PreprocessConfig, **overrides: Any) -> AnalyzeConfig:
    return AnalyzeConfig(
        record_root=upstream.record_root,
        cut_root=upstream.cut_root,
        retilt_root=upstream.retilt_root,
        crop_root=upstream.crop_root,
        pose_root=tmp_path / "s6_pose_output",
        **overrides,
    )


def test_the_run_solves_kinematics_when_a_kinematics_root_is_given(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    config = make_config(tmp_path, upstream, kinematics_root=tmp_path / "s8_kinematics_output")

    manifest = analyze(config)

    assert [scan.scan_id for scan in manifest.scans] == [
        "9 July/cnj_45kg_Set1/Front",
        "9 July/cnj_45kg_Set1/Side",
    ]
    assert all(scan.steps == ["kinematics"] for scan in manifest.scans)
    assert config.kinematics_root is not None
    assert (config.kinematics_root / "manifest.json").is_file()
    for scan in manifest.scans:
        assert (config.kinematics_root / scan.scan_id / "joint_hva.json").is_file()


def test_the_run_finds_phases_when_a_phase_root_is_given(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    config = make_config(
        tmp_path,
        upstream,
        kinematics_root=tmp_path / "s8_kinematics_output",
        barbell_root=tmp_path / "s5_barbell_output",
        phase_root=tmp_path / "s9_phase_output",
    )

    manifest = analyze(config)

    assert manifest.scans
    assert (tmp_path / "s9_phase_output" / "manifest.json").is_file()
    for scan in manifest.scans:
        assert scan.steps == ["kinematics", "phase"]
        # The synthetic capture declares no lift.type, so S9 warns rather than guessing.
        assert any("lift.type" in w for w in scan.warnings)


def test_s9_reruns_alone_against_an_earlier_s8_tree(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    """The point of the split: a rule change re-runs S9 without solving S8 again."""

    roots = {
        "kinematics_root": tmp_path / "s8_kinematics_output",
        "barbell_root": tmp_path / "s5_barbell_output",
        "phase_root": tmp_path / "s9_phase_output",
    }
    analyze(make_config(tmp_path, upstream, stages=(AnalyzeStage.KINEMATICS,), **roots))
    solved = {
        path: path.stat().st_mtime_ns
        for path in (tmp_path / "s8_kinematics_output").rglob("joint_hva.json")
    }

    manifest = analyze(make_config(tmp_path, upstream, stages=(AnalyzeStage.PHASE,), **roots))

    assert len(solved) == 2
    assert all(scan.steps == ["phase"] for scan in manifest.scans)
    assert {path: path.stat().st_mtime_ns for path in solved} == solved  # S8 untouched
    for scan in manifest.scans:
        # S9 read the earlier S8 tree: it got as far as the capture's missing lift.type.
        assert not any("no S8 kinematics" in w for w in scan.warnings)
        assert any("lift.type" in w for w in scan.warnings)


def test_s8_warns_when_detect_published_no_pose_for_a_camera(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    (tmp_path / "s6_pose_output" / "9 July" / "cnj_45kg_Set1" / "Front" / "pose.json").unlink()
    config = make_config(tmp_path, upstream, kinematics_root=tmp_path / "s8_kinematics_output")

    manifest = analyze(config)

    front = next(scan for scan in manifest.scans if scan.scan_id.endswith("Front"))
    assert any("no S6 pose" in w for w in front.warnings)
    assert not (tmp_path / "s8_kinematics_output" / front.scan_id / "joint_hva.json").exists()


def test_a_dry_run_writes_nothing(tmp_path: Path, upstream: PreprocessConfig) -> None:
    config = make_config(
        tmp_path, upstream, kinematics_root=tmp_path / "s8_kinematics_output", dry_run=True
    )

    manifest = analyze(config)

    assert manifest.scans
    assert all(scan.status == "planned" for scan in manifest.scans)
    assert not (tmp_path / "s8_kinematics_output").exists()


def test_analyze_refuses_a_run_with_nothing_to_do(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    with pytest.raises(ValueError, match="nothing to run"):
        analyze(make_config(tmp_path, upstream))

    # A stage selected without its root does not run either.
    config = make_config(
        tmp_path,
        upstream,
        kinematics_root=tmp_path / "s8_kinematics_output",
        stages=(AnalyzeStage.PHASE,),
    )
    with pytest.raises(ValueError, match="nothing to run"):
        analyze(config)


def test_analyze_rejects_a_phase_root_without_a_kinematics_root(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    config = make_config(
        tmp_path, upstream, barbell_root=tmp_path / "s5", phase_root=tmp_path / "s9_phase_output"
    )

    with pytest.raises(ValueError, match="no kinematics to read"):
        analyze(config)


def test_analyze_rejects_a_phase_root_without_a_barbell_root(
    tmp_path: Path, upstream: PreprocessConfig
) -> None:
    config = make_config(
        tmp_path,
        upstream,
        kinematics_root=tmp_path / "s8_kinematics_output",
        phase_root=tmp_path / "s9_phase_output",
    )

    with pytest.raises(ValueError, match="no bar to segment"):
        analyze(config)
