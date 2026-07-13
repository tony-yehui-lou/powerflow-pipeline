"""The whole of Step 1, under a real Prefect backend: rotate what is valid, reject the rest."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml
from prefect.client.orchestration import get_client
from typer.testing import CliRunner

from powerflow_pipeline.data.cli import app
from powerflow_pipeline.data.common.manifest import artifact_key
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.flow import preprocess
from tests.conftest import MakeCamera


def build_capture(tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any) -> Path:
    """Two sessions, two cameras each. One camera is missing its IMU; one session has none."""

    raw = tmp_path / "raw"
    make_meta_template(raw)
    make_camera(raw, session="cnj_45kg_Set1", camera="Front")
    make_camera(raw, session="cnj_45kg_Set1", camera="Side")
    make_camera(raw, session="cnj_55kg_Set1", camera="Front", omit=["imu"])
    make_camera(raw, session="cnj_55kg_Set1", camera="Side", omit=["depth"])
    return raw


def make_config(tmp_path: Path, raw: Path, **overrides: Any) -> PreprocessConfig:
    return PreprocessConfig(
        raw_root=raw,
        record_root=tmp_path / "s0_ingest_output",
        output_root=tmp_path / "s1_orient_output",
        **overrides,
    )


async def _artifact_keys() -> list[str | None]:
    async with get_client() as client:
        return [artifact.key for artifact in await client.read_artifacts()]


def test_the_run_rotates_valid_cameras_and_rejects_the_rest(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template)
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert [scan.scan_id for scan in manifest.scans] == [
        "9 July/cnj_45kg_Set1/Front",
        "9 July/cnj_45kg_Set1/Side",
    ]
    rejections = {rejected.scan_id: rejected.reason for rejected in manifest.rejected_scans}
    assert rejections == {
        "9 July/cnj_55kg_Set1/Front": "missing required stream: imu.csv",
        "9 July/cnj_55kg_Set1/Side": "missing required stream: depth",
        # every camera of the session went, so the session goes too
        "9 July/cnj_55kg_Set1": "session has no usable camera",
    }


def test_the_manifest_artifact_records_the_rotation_and_the_k_rewrite(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template)
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert artifact_key("preprocess") in asyncio.run(_artifact_keys())

    markdown = manifest.to_markdown()
    assert "rotation=cw" in markdown
    assert "k_rewritten=True" in markdown
    assert "session has no usable camera" in markdown

    written = json.loads((config.record_root / "manifest.json").read_text())
    assert written["scans"][0]["derived"]["n_frames"] == 4


def test_a_surviving_session_is_published_whole(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template)
    config = make_config(tmp_path, raw)

    preprocess(config)

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

    # Every stage's output carries the session metadata, not just S0's.
    for root in config.stage_roots:
        metadata = yaml.safe_load((root / "9 July" / "cnj_45kg_Set1" / "metadata.yaml").read_text())
        assert [camera["camera"] for camera in metadata["cameras"]] == ["Front", "Side"]
        assert metadata["derived_from_session_name"]["weight_in_kg"] == 45.0

    # No staging survives a completed run.
    assert not list(config.output_root.rglob(".powerflow-*"))


def test_a_dry_run_touches_nothing(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template)
    config = make_config(tmp_path, raw, dry_run=True)

    manifest = preprocess(config)

    assert [scan.status for scan in manifest.scans] == ["planned", "planned"]
    assert manifest.scans[0].file_ops  # the plan is still reported
    assert not config.output_root.exists()
    assert not config.record_root.exists()


def test_the_cli_runs_the_flow(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template)

    result = CliRunner().invoke(
        app,
        [
            "preprocess",
            "--input",
            str(raw),
            "--output",
            str(tmp_path / "s1"),
            "--records",
            str(tmp_path / "s0"),
            "--rotation",
            "cw",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "processed 2 camera(s), rejected 3" in result.output
    assert (tmp_path / "s1" / "9 July" / "cnj_45kg_Set1" / "Front" / "sidecar.json").is_file()
