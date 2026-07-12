"""Run-level manifest contract and serialization."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from powerflow_pipeline.data.core.context import OutputMode
from powerflow_pipeline.data.core.filesystem import write_json
from powerflow_pipeline.data.core.models import RejectedScan, ScanOutcome


class RunManifest(BaseModel):
    """Complete audit record for one pipeline invocation."""

    pipeline: str
    input_root: Path
    output_root: Path | None
    output_mode: OutputMode
    dry_run: bool
    scans: list[ScanOutcome] = Field(default_factory=list)
    rejected_scans: list[RejectedScan] = Field(default_factory=list)

    def write(self, path: Path) -> None:
        """Persist a JSON manifest using only JSON-compatible representations."""

        write_json(path, self.model_dump(mode="json"))
