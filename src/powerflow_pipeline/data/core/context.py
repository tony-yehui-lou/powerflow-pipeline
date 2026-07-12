"""Immutable run configuration shared by pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class OutputMode(StrEnum):
    """The single allowed destination strategy for a run."""

    PUBLISH = "publish"
    IN_PLACE = "in_place"


@dataclass(frozen=True, slots=True)
class RunContext:
    """Validated execution settings for one pipeline invocation."""

    pipeline: str
    input_root: Path
    output_root: Path | None
    output_mode: OutputMode
    dry_run: bool = False

    @classmethod
    def create(
        cls,
        *,
        pipeline: str,
        input_root: Path,
        output_root: Path | None,
        output_mode: OutputMode,
        dry_run: bool = False,
    ) -> RunContext:
        input_root = input_root.resolve()
        if not input_root.is_dir():
            raise ValueError(f"input_root is not a directory: {input_root}")
        if output_mode is OutputMode.PUBLISH and output_root is None:
            raise ValueError("output_root is required for publish mode")
        if output_mode is OutputMode.IN_PLACE and output_root is not None:
            raise ValueError("output_root must not be set for in-place mode")
        return cls(
            pipeline=pipeline,
            input_root=input_root,
            output_root=output_root.resolve() if output_root is not None else None,
            output_mode=output_mode,
            dry_run=dry_run,
        )

    def destination_for(self, scan: str) -> Path:
        """Return the eventual public destination for a scan."""

        if self.output_mode is OutputMode.IN_PLACE:
            return self.input_root / scan
        if self.output_root is None:
            raise RuntimeError("publish mode requires output_root")
        return self.output_root / scan
