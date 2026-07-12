"""Shared fixtures for the pipeline test suite."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pytest

from powerflow_pipeline.data.core.context import OutputMode, RunContext

VALID_META: dict[str, Any] = {
    "scan_id": "scan_0001",
    "width": 1920,
    "height": 1080,
    "marker_px": 200.0,
    "marker_mm": 50.0,
}


class MakeScan(Protocol):
    def __call__(
        self,
        root: Path,
        scan_id: str,
        *,
        meta: dict[str, Any] | str | None = None,
        frames: int = 2,
    ) -> Path: ...


@pytest.fixture
def make_scan() -> MakeScan:
    """Build a scan directory: `meta.json` plus inert placeholder frames.

    `meta` overrides fields of the default valid meta, or replaces the file body
    entirely when a raw string (used for the malformed-JSON rejection case).
    """

    def _make(
        root: Path,
        scan_id: str,
        *,
        meta: dict[str, Any] | str | None = None,
        frames: int = 2,
    ) -> Path:
        scan_dir = root / scan_id
        (scan_dir / "frames").mkdir(parents=True)
        for i in range(frames):
            (scan_dir / "frames" / f"{i:03d}.bin").write_bytes(bytes([i]) * 8)

        if isinstance(meta, str):
            body = meta
        else:
            payload = {**VALID_META, "scan_id": scan_id, **(meta or {})}
            body = json.dumps(payload)
        (scan_dir / "meta.json").write_text(body)
        return scan_dir

    return _make


@pytest.fixture
def make_context() -> Callable[..., RunContext]:
    """Build a RunContext for tests; defaults to publish mode, no progress bar."""

    def _make(
        input_root: Path,
        *,
        output_root: Path | None = None,
        mode: OutputMode = OutputMode.PUBLISH,
        dry_run: bool = False,
        pipeline: str = "preprocess",
    ) -> RunContext:
        return RunContext.create(
            pipeline=pipeline,
            input_root=input_root,
            output_root=output_root,
            output_mode=mode,
            dry_run=dry_run,
        )

    return _make
