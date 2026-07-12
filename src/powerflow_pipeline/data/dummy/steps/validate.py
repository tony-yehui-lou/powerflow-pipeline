"""Validate dummy scan metadata before any output is staged."""

from __future__ import annotations

import json

from pydantic import ValidationError

from powerflow_pipeline.data.core.errors import ScanRejected
from powerflow_pipeline.data.core.models import Scan, StepResult
from powerflow_pipeline.data.dummy.config import DummyScanMetadata


def validate_metadata(scan: Scan) -> tuple[DummyScanMetadata, StepResult]:
    """Load and validate a scan's metadata, including its directory identity."""

    try:
        data = json.loads((scan.source / "meta.json").read_text(encoding="utf-8"))
        metadata = DummyScanMetadata.model_validate(data)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ScanRejected(f"invalid meta.json: {error}") from error
    if metadata.scan_id != scan.scan_id:
        raise ScanRejected("invalid meta.json: scan_id does not match scan directory")
    return metadata, StepResult(derived={"metadata_valid": True})
