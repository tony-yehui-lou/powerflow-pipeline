"""Run configuration for the preprocess pipeline (S0 ingest + S1 orient)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class RotationDirection(StrEnum):
    """The 90 degree rotation that repairs the capture app's landscape output."""

    CW = "cw"
    CCW = "ccw"


class PreprocessConfig(BaseModel):
    """Settings for one preprocess run.

    `rotation` defaults to CW: the direction cannot be derived from any metadata
    (`rgb.mp4` carries no rotation tag), so it was fixed once by looking at a frame.
    """

    raw_root: Path
    record_root: Path
    output_root: Path
    rotation: RotationDirection = RotationDirection.CW
    frame_count_tolerance: int = Field(default=1, ge=0)
    rgb_crf: int = Field(default=16, ge=0, le=51)
    require_stopwatch_attestation: bool = False
    overwrite: bool = False
    dry_run: bool = False

    @property
    def stage_roots(self) -> tuple[Path, ...]:
        """Every tree this run publishes into, in stage order.

        `metadata.yaml` is written to each: a stage's output says what the pixels are, never
        which lift they came from, and a consumer should not have to reach back into an
        earlier stage's tree to find out. A new stage adds its root here and inherits the copy.
        """

        return (self.record_root, self.output_root)
