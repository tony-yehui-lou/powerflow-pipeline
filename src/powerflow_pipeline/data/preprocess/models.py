"""Records exchanged between S0 and S1, and written to disk as provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ImageFrame = Literal["portrait", "landscape"]


class Intrinsics(BaseModel):
    """A pinhole camera matrix, tagged with the image frame it describes."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: list[float] | None = None  # the capture app exports none
    frame: ImageFrame

    @classmethod
    def from_matrix(cls, matrix: list[list[float]], *, frame: ImageFrame) -> Intrinsics:
        return cls(fx=matrix[0][0], fy=matrix[1][1], cx=matrix[0][2], cy=matrix[1][2], frame=frame)

    def to_matrix(self) -> list[list[float]]:
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]


class StreamCounts(BaseModel):
    """Raw per-stream lengths, before any alignment."""

    rgb: int
    depth: int
    confidence: int
    odometry: int
    imu: int


class CameraDir(BaseModel):
    """One `<date>/<session>/<camera>` directory found by discovery."""

    date: str
    session: str
    camera: str
    source: Path

    @property
    def camera_id(self) -> str:
        """The manifest identifier for this camera."""

        return f"{self.date}/{self.session}/{self.camera}"

    @property
    def relative(self) -> Path:
        return Path(self.date) / self.session / self.camera


class CameraRecord(BaseModel):
    """S0's validated description of one camera. S1 reads it instead of re-probing."""

    date: str
    session: str
    camera: str
    source: Path
    rgb_width: int
    rgb_height: int
    fps: float
    depth_width: int
    depth_height: int
    counts: StreamCounts
    n_frames: int  # the aligned length; trailing depth/confidence/odometry entries are dropped
    intrinsics: Intrinsics  # averaged from odometry.csv (ISSUE-02), landscape frame (ISSUE-01)
    static_intrinsics: Intrinsics  # camera_matrix.csv as shipped, kept for provenance
    odometry_intrinsics_drift: float  # max |fx_row - fx_static|
    creation_time: str | None  # rgb.mp4's container tag
    stopwatch_legible: bool | None

    @property
    def camera_id(self) -> str:
        return f"{self.date}/{self.session}/{self.camera}"

    @property
    def relative(self) -> Path:
        return Path(self.date) / self.session / self.camera


class CutInterval(BaseModel):
    """The shared epoch-time window S1 Cut derives from the Side lift window.

    Both cameras are trimmed to `[cut_start_epoch_ms, cut_end_epoch_ms]`; the other fields
    are provenance, carried through to the output metadata unchanged.
    """

    cut_start_epoch_ms: int
    cut_end_epoch_ms: int
    side_creation_time: str
    side_created_epoch_ms: int
    lift_start_time_side_in_ms: int
    lift_end_time_side_in_ms: int


class LiftMeta(BaseModel):
    """The lift half of `metadata.yaml`. Absent values stay absent."""

    dateTime_epoch: int | None = None  # the template's key, kept verbatim
    weight_in_kg: float | None = None
    type: str | None = None
    result: str | None = None


class AthleteMeta(BaseModel):
    """The athlete half of `metadata.yaml`. Nothing in the capture supplies these."""

    name: str | None = None
    measureDate: str | None = None  # the template's key, kept verbatim
    height_in_cm: float | None = None
    weight_in_kg: float | None = None
    tibia_in_cm: float | None = None
    femur_in_cm: float | None = None
    torso_in_cm: float | None = None
    armspan_in_cm: float | None = None


class SessionRecord(BaseModel):
    """One lift: its metadata and the cameras that survived S0."""

    date: str
    session: str
    lift: LiftMeta
    athlete: AthleteMeta
    cameras: list[CameraRecord]
