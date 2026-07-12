"""Stage an exact scan copy for the dummy pipeline."""

from __future__ import annotations

from pathlib import Path

from powerflow_pipeline.data.core.filesystem import copy_tree
from powerflow_pipeline.data.core.models import FileOp, Scan, StepResult


def copy_scan(scan: Scan, staging: Path | None, target: Path, dry_run: bool) -> StepResult:
    """Copy a valid source scan into staging, or record the dry-run operation."""

    if not dry_run:
        if staging is None:
            raise RuntimeError("staging path is required outside dry-run mode")
        copy_tree(scan.source, staging)
    return StepResult(file_ops=[FileOp(op="copy", src=scan.source, dst=target)])
