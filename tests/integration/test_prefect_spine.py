"""The shared tasks compose into a real Prefect flow and emit a run manifest."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest
from prefect import flow
from prefect.client.orchestration import get_client

from powerflow_pipeline.data.common.context import OutputMode, RunContext
from powerflow_pipeline.data.common.discovery import discover_scans
from powerflow_pipeline.data.common.manifest import RunManifest, artifact_key, emit_manifest
from powerflow_pipeline.data.common.models import ScanOutcome
from tests.conftest import MakeScan


@flow(name="spine-smoke")
def spine_smoke(context: RunContext, manifest_path: Path) -> RunManifest:
    """Minimal flow: discover scans, record outcomes, emit the manifest."""

    discovered = discover_scans(context.input_root)
    manifest = RunManifest(
        pipeline=context.pipeline,
        input_root=context.input_root,
        output_root=context.output_root,
        output_mode=context.output_mode,
        dry_run=context.dry_run,
        scans=[
            ScanOutcome(
                scan_id=scan.scan_id,
                source=scan.source,
                status="planned",
                steps=["discover"],
            )
            for scan in discovered.scans
        ],
        rejected_scans=discovered.rejected_scans,
    )
    return emit_manifest(manifest, manifest_path)


async def _artifact_keys() -> list[str | None]:
    async with get_client() as client:
        return [artifact.key for artifact in await client.read_artifacts()]


def test_flow_discovers_scans_and_publishes_a_manifest_artifact(
    tmp_path: Path, make_scan: MakeScan, caplog: pytest.LogCaptureFixture
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_scan(raw, "scan_0001")
    make_scan(raw, "scan_0002", frames=0)  # rejected: no frames
    context = RunContext.create(
        pipeline="preprocess",
        input_root=raw,
        output_root=tmp_path / "processed",
        output_mode=OutputMode.PUBLISH,
    )
    manifest_path = tmp_path / "manifest.json"

    caplog.set_level(logging.INFO)
    manifest = spine_smoke(context, manifest_path)

    assert [scan.scan_id for scan in manifest.scans] == ["scan_0001"]
    assert [rejected.scan_id for rejected in manifest.rejected_scans] == ["scan_0002"]

    written = json.loads(manifest_path.read_text())
    assert written["pipeline"] == "preprocess"
    assert written["rejected_scans"][0]["reason"] == "missing or empty required frames directory"

    assert artifact_key("preprocess") in asyncio.run(_artifact_keys())
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert f"source={raw} output=<in-memory>" in messages
    assert f"source={raw} output={manifest_path}" in messages
