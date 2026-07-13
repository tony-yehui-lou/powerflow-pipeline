"""Run-level manifest: the audit record for one flow invocation."""

from __future__ import annotations

import re
from pathlib import Path

from prefect import task
from prefect.artifacts import create_markdown_artifact
from pydantic import BaseModel, Field

from powerflow_pipeline.data.common.context import OutputMode
from powerflow_pipeline.data.common.filesystem import write_json
from powerflow_pipeline.data.common.models import RejectedScan, ScanOutcome


def artifact_key(pipeline: str) -> str:
    """Build the Prefect artifact key for a pipeline's manifest.

    Prefect only accepts lowercase alphanumerics and dashes in artifact keys.
    """

    slug = re.sub(r"[^a-z0-9]+", "-", pipeline.lower()).strip("-")
    return f"{slug}-run-manifest"


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

    def to_markdown(self) -> str:
        """Render the manifest for review in the Prefect UI."""

        mode = f"{self.output_mode.value}{' (dry run)' if self.dry_run else ''}"
        lines = [
            f"# {self.pipeline} run manifest",
            "",
            f"- Input: `{self.input_root}`",
            f"- Output: `{self.output_root or self.input_root}`",
            f"- Mode: {mode}",
            f"- Processed: {len(self.scans)} | Rejected: {len(self.rejected_scans)}",
            "",
        ]

        if self.scans:
            lines += [
                "## Processed scans",
                "",
                "| Scan | Status | Steps | Derived | Warnings | File ops |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for scan in self.scans:
                derived = ", ".join(f"{key}={value}" for key, value in sorted(scan.derived.items()))
                lines.append(
                    f"| {scan.scan_id} | {scan.status} | {', '.join(scan.steps)} "
                    f"| {derived or '—'} | {'; '.join(scan.warnings) or '—'} "
                    f"| {len(scan.file_ops)} |"
                )
            lines.append("")
        else:
            lines += ["No scans were processed.", ""]

        if self.rejected_scans:
            lines += ["## Rejected scans", "", "| Scan | Reason |", "| --- | --- |"]
            lines += [
                f"| {rejected.scan_id} | {rejected.reason} |" for rejected in self.rejected_scans
            ]
            lines.append("")

        return "\n".join(lines)


@task
def emit_manifest(manifest: RunManifest, path: Path | None = None) -> RunManifest:
    """Publish the manifest to the Prefect UI, and to disk when a path is given."""

    if path is not None:
        manifest.write(path)
    create_markdown_artifact(
        key=artifact_key(manifest.pipeline),
        markdown=manifest.to_markdown(),
        description=f"Run manifest for the {manifest.pipeline} pipeline.",
    )
    return manifest
