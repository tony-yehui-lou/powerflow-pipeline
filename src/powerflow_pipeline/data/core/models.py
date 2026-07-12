"""Pydantic models shared by every data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CropBounds(BaseModel):
    """Exclusive pixel bounds for a crop in image coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    @model_validator(mode="after")
    def validate_extent(self) -> CropBounds:
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError("crop origin must be non-negative")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("crop bounds must have a positive extent")
        return self

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


class FileOp(BaseModel):
    """One file operation planned or performed by a pipeline step."""

    op: Literal["copy", "write", "publish", "commit"]
    src: Path
    dst: Path


class Scan(BaseModel):
    """A discovered scan and the files it contributes to a pipeline run."""

    model_config = ConfigDict(frozen=True)

    scan_id: str
    source: Path
    files: tuple[Path, ...]


class StepResult(BaseModel):
    """Information one processing step contributes to the run manifest."""

    derived: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    file_ops: list[FileOp] = Field(default_factory=list)


class ScanOutcome(BaseModel):
    """The manifest record for a successfully processed scan."""

    scan_id: str
    source: Path
    status: Literal["published", "planned"]
    steps: list[str]
    derived: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    file_ops: list[FileOp] = Field(default_factory=list)


class RejectedScan(BaseModel):
    """The manifest record for a scan that failed validation."""

    scan_id: str
    source: Path
    reason: str
