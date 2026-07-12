"""End-to-end behavior for the reviewable dummy pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from powerflow_pipeline.data.cli import app
from powerflow_pipeline.data.core.context import OutputMode
from powerflow_pipeline.data.core.discovery import discover_scans
from powerflow_pipeline.data.core.errors import PublishError
from powerflow_pipeline.data.dummy.pipeline import run


def test_publish_copies_scan_writes_sidecar_and_manifest(
    make_context, make_scan, tmp_path: Path
) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    make_scan(input_root, "scan_0001", meta={"marker_px": 225.0, "marker_mm": 50.0})

    manifest = run(make_context(input_root, output_root=output_root))

    copied = output_root / "scan_0001"
    assert (copied / "frames" / "000.bin").read_bytes() == bytes(8)
    sidecar = json.loads((copied / "dummy.sidecar.json").read_text())
    assert sidecar["px_per_mm"] == 4.5
    assert sidecar["crop_bounds"] == {"x0": 0, "y0": 0, "x1": 1920, "y1": 1080}
    saved_manifest = json.loads((output_root / "manifest.json").read_text())
    assert saved_manifest["pipeline"] == "dummy"
    assert saved_manifest["scans"][0]["status"] == "published"
    assert manifest.scans[0].scan_id == "scan_0001"


def test_invalid_scan_is_rejected_without_blocking_valid_scans(
    make_context, make_scan, tmp_path: Path
) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    make_scan(input_root, "scan_good")
    make_scan(input_root, "scan_bad", meta="not json")

    manifest = run(make_context(input_root, output_root=output_root))

    assert (output_root / "scan_good" / "dummy.sidecar.json").exists()
    assert not (output_root / "scan_bad").exists()
    assert manifest.rejected_scans[0].scan_id == "scan_bad"
    assert "meta.json" in manifest.rejected_scans[0].reason


def test_dry_run_does_not_create_an_output_directory(
    make_context, make_scan, tmp_path: Path
) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    make_scan(input_root, "scan_0001")

    manifest = run(make_context(input_root, output_root=output_root, dry_run=True))

    assert not output_root.exists()
    assert manifest.dry_run is True
    assert manifest.scans[0].file_ops[0].op == "copy"


def test_in_place_commit_replaces_scan_with_complete_staged_result(
    make_context, make_scan, tmp_path: Path
) -> None:
    input_root = tmp_path / "raw"
    scan = make_scan(input_root, "scan_0001")

    run(make_context(input_root, mode=OutputMode.IN_PLACE))

    assert (scan / "meta.json").exists()
    assert (scan / "dummy.sidecar.json").exists()
    assert (input_root / "manifest.json").exists()


def test_discovery_rejects_missing_metadata_and_empty_frames(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    (input_root / "no_meta" / "frames").mkdir(parents=True)
    (input_root / "empty_frames" / "frames").mkdir(parents=True)
    (input_root / "empty_frames" / "meta.json").write_text("{}")

    discovered = discover_scans(input_root)

    assert discovered.scans == []
    assert [scan.scan_id for scan in discovered.rejected_scans] == ["empty_frames", "no_meta"]


def test_publish_refuses_to_merge_into_an_existing_output(
    make_context, make_scan, tmp_path: Path
) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    make_scan(input_root, "scan_0001")
    output_root.mkdir()

    with pytest.raises(PublishError, match="already exists"):
        run(make_context(input_root, output_root=output_root))

    assert list(output_root.iterdir()) == []


def test_cli_requires_exactly_one_output_mode(make_scan, tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    make_scan(input_root, "scan_0001")
    runner = CliRunner()

    missing = runner.invoke(app, ["dummy", str(input_root)])
    both = runner.invoke(
        app, ["dummy", str(input_root), "--output", str(tmp_path / "out"), "--in-place"]
    )

    assert missing.exit_code != 0
    assert both.exit_code != 0


def test_cli_dry_run_prints_manifest(make_scan, tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    make_scan(input_root, "scan_0001")

    result = CliRunner().invoke(
        app,
        ["dummy", str(input_root), "--output", str(tmp_path / "out"), "--dry-run"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["dry_run"] is True
