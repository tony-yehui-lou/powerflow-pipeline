"""The whole preprocess pipeline under a real Prefect backend: ingest, cut, then orient."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from prefect.client.orchestration import get_client
from typer.testing import CliRunner

from powerflow_pipeline.data.cli import app
from powerflow_pipeline.data.common.manifest import artifact_key
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.flow import preprocess
from tests.conftest import MakeCamera


def build_capture(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any, make_session_metadata: Any
) -> Path:
    """Two sessions. cnj_45kg has both cameras and valid lift window; cnj_55kg is broken."""

    raw = tmp_path / "raw"
    make_meta_template(raw)

    # cnj_45kg_Set1: a healthy session with a Side lift window.
    make_camera(raw, session="cnj_45kg_Set1", camera="Front", rgb_frames=5, depth_frames=6)
    make_camera(raw, session="cnj_45kg_Set1", camera="Side", rgb_frames=5, depth_frames=6)
    make_session_metadata(raw, session="cnj_45kg_Set1", lift_start_ms=16, lift_end_ms=80)

    # cnj_55kg_Set1: one camera is missing its IMU, the other its depth.
    make_camera(raw, session="cnj_55kg_Set1", camera="Front", omit=["imu"])
    make_camera(raw, session="cnj_55kg_Set1", camera="Side", omit=["depth"])
    make_session_metadata(raw, session="cnj_55kg_Set1", lift_start_ms=16, lift_end_ms=80)

    return raw


def make_config(tmp_path: Path, raw: Path, **overrides: Any) -> PreprocessConfig:
    return PreprocessConfig(
        raw_root=raw,
        record_root=tmp_path / "s0_ingest_output",
        cut_root=tmp_path / "s1_cut_output",
        output_root=tmp_path / "s2_orient_output",
        **overrides,
    )


async def _artifact_keys() -> list[str | None]:
    async with get_client() as client:
        return [artifact.key for artifact in await client.read_artifacts()]


def test_the_run_cuts_and_rotates_valid_cameras(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert [scan.scan_id for scan in manifest.scans] == [
        "9 July/cnj_45kg_Set1/Front",
        "9 July/cnj_45kg_Set1/Side",
    ]
    assert all("cut" in scan.steps for scan in manifest.scans)
    assert all("orient" in scan.steps for scan in manifest.scans)
    rejections = {rejected.scan_id: rejected.reason for rejected in manifest.rejected_scans}
    assert rejections["9 July/cnj_55kg_Set1/Front"] == "missing required stream: imu.csv"
    assert rejections["9 July/cnj_55kg_Set1/Side"] == "missing required stream: depth"
    # cnj_55kg has both cameras rejected at ingest; the session-level rollup fires from
    # _sessions_without_a_camera since no camera survived to trigger _session_cut_interval.
    assert rejections["9 July/cnj_55kg_Set1"] == "session has no usable camera"


def test_task_logs_identify_their_source_and_output_paths(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)
    source = raw / "9 July" / "cnj_45kg_Set1" / "Front"
    cut_destination = config.cut_root / "9 July" / "cnj_45kg_Set1" / "Front"
    output_destination = config.output_root / "9 July" / "cnj_45kg_Set1" / "Front"

    caplog.set_level(logging.INFO)
    preprocess(config)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    record_path = config.record_root / source.relative_to(raw) / "record.json"
    assert f"source={source} output={record_path}" in messages
    assert f"source={source} output={cut_destination}" in messages
    assert f"source={cut_destination} output={output_destination}" in messages


def test_the_manifest_artifact_records_cut_and_rotation(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert artifact_key("preprocess") in asyncio.run(_artifact_keys())

    markdown = manifest.to_markdown()
    assert "cut_start_epoch_ms" in markdown
    assert "cut_end_epoch_ms" in markdown
    assert "k_rewritten=True" in markdown
    assert "session has no usable camera" in markdown

    written = json.loads((config.record_root / "manifest.json").read_text())
    assert written["scans"][0]["derived"]["n_frames"] > 0


def test_a_surviving_session_is_published_whole(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)

    preprocess(config)

    # S2 orient output: the final portrait streams.
    session = config.output_root / "9 July" / "cnj_45kg_Set1"
    for camera in ("Front", "Side"):
        published = {path.name for path in (session / camera).iterdir()}
        assert published == {
            "rgb.mp4",
            "depth",
            "confidence",
            "camera_matrix.csv",
            "camera_matrix_source.csv",
            "odometry.csv",
            "imu.csv",
            "sidecar.json",
        }

    # S1 cut output: trimmed landscape streams + cut_sidecar.
    cut_session = config.cut_root / "9 July" / "cnj_45kg_Set1"
    for camera in ("Front", "Side"):
        published = {path.name for path in (cut_session / camera).iterdir()}
        assert "cut_sidecar.json" in published
        assert "rgb.mp4" in published

    # Every stage's output carries the session metadata.
    for root in config.stage_roots:
        metadata = yaml.safe_load((root / "9 July" / "cnj_45kg_Set1" / "metadata.yaml").read_text())
        assert [camera["camera"] for camera in metadata["cameras"]] == ["Front", "Side"]
        assert metadata["derived_from_session_name"]["weight_in_kg"] == 45.0
        # Cut epoch bounds are recorded.
        assert "cut_start_epoch_ms" in metadata["lift"]
        assert "cut_end_epoch_ms" in metadata["lift"]

    # No staging survives a completed run.
    assert not list(config.output_root.rglob(".powerflow-*"))


def test_a_dry_run_touches_nothing(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw, dry_run=True)

    manifest = preprocess(config)

    assert [scan.status for scan in manifest.scans] == ["planned", "planned"]
    assert manifest.scans[0].file_ops  # the plan is still reported
    assert not config.output_root.exists()
    assert not config.record_root.exists()
    assert not config.cut_root.exists()


def test_the_cli_runs_the_flow(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)

    s0 = tmp_path / "s0"
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    result = CliRunner().invoke(
        app,
        [
            "preprocess",
            "--input",
            str(raw),
            "--records",
            str(s0),
            "--cut",
            str(s1),
            "--output",
            str(s2),
            "--rotation",
            "cw",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "processed 2 camera(s), rejected 3" in result.output
    assert (s2 / "9 July" / "cnj_45kg_Set1" / "Front" / "sidecar.json").is_file()
    assert (s1 / "9 July" / "cnj_45kg_Set1" / "Front" / "cut_sidecar.json").is_file()
