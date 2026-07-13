"""The run manifest is the audit record for one pipeline invocation."""

from __future__ import annotations

import json
from pathlib import Path

from powerflow_pipeline.data.common.context import OutputMode
from powerflow_pipeline.data.common.manifest import RunManifest, artifact_key
from powerflow_pipeline.data.common.models import FileOp, RejectedScan, ScanOutcome


def _manifest(tmp_path: Path) -> RunManifest:
    return RunManifest(
        pipeline="preprocess",
        input_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
        output_mode=OutputMode.PUBLISH,
        dry_run=False,
        scans=[
            ScanOutcome(
                scan_id="scan_0001",
                source=tmp_path / "raw" / "scan_0001",
                status="published",
                steps=["copy", "sidecar"],
                derived={"px_per_mm": 4.0},
                warnings=["low marker contrast"],
                file_ops=[
                    FileOp(
                        op="copy",
                        src=tmp_path / "raw" / "scan_0001" / "meta.json",
                        dst=tmp_path / "processed" / "scan_0001" / "meta.json",
                    )
                ],
            )
        ],
        rejected_scans=[
            RejectedScan(
                scan_id="scan_0002",
                source=tmp_path / "raw" / "scan_0002",
                reason="missing required meta.json",
            )
        ],
    )


def test_manifest_writes_json_round_trip(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"

    manifest.write(path)

    payload = json.loads(path.read_text())
    assert payload["pipeline"] == "preprocess"
    assert payload["output_mode"] == "publish"
    assert payload["scans"][0]["derived"] == {"px_per_mm": 4.0}
    assert RunManifest.model_validate(payload) == manifest


def test_markdown_reports_steps_derived_values_warnings_and_rejections(tmp_path: Path) -> None:
    markdown = _manifest(tmp_path).to_markdown()

    assert "preprocess" in markdown
    assert "scan_0001" in markdown
    assert "copy, sidecar" in markdown
    assert "px_per_mm" in markdown
    assert "low marker contrast" in markdown
    assert "scan_0002" in markdown
    assert "missing required meta.json" in markdown


def test_markdown_states_when_a_run_processed_nothing(tmp_path: Path) -> None:
    empty = RunManifest(
        pipeline="preprocess",
        input_root=tmp_path,
        output_root=None,
        output_mode=OutputMode.IN_PLACE,
        dry_run=True,
    )

    markdown = empty.to_markdown()

    assert "dry run" in markdown.lower()
    assert "no scans" in markdown.lower()


def test_artifact_key_is_slugified() -> None:
    assert artifact_key("preprocess") == "preprocess-run-manifest"
    assert artifact_key("Scan_Prep v2") == "scan-prep-v2-run-manifest"
