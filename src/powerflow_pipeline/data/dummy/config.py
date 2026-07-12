"""Metadata contract for the dummy pipeline's fixture-style input scans."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt


class DummyScanMetadata(BaseModel):
    """Minimal information needed to derive scale and full-frame crop bounds."""

    model_config = ConfigDict(extra="forbid")

    scan_id: str
    width: PositiveInt
    height: PositiveInt
    marker_px: PositiveFloat
    marker_mm: PositiveFloat
