"""Records exchanged between S0 and S1, and written to disk as provenance."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from powerflow_pipeline.data.common.models import CropBounds

ImageFrame = Literal["portrait", "landscape"]

# Which `metadata.yaml` fields a capture reads, and nothing else. A role is a lookup key,
# not a statement about where the camera was pointed: `single` says one camera governed by
# its own metadata file, never that the camera was square-on to anything.
CaptureRole = Literal["side", "front", "single"]


class CaptureLayout(StrEnum):
    """Which raw directory shape a capture was found in.

    Detected from *where its operator `metadata.yaml` lives* (`tasks/discover.py`), never
    from a directory name -- a name match would break the moment a capture day is foldered
    differently again.
    """

    MULTI_CAMERA = "multi_camera"  # `<date>/<session>/<camera>/`, metadata at the session
    SINGLE_CAMERA = "single_camera"  # `<date>/<group>/<trial>/`, metadata inside the capture


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


class CaptureUnit(BaseModel):
    """One raw capture directory, carrying the identity every stage keys off.

    Replaces the old `(date, session, camera)` triple, which cannot name a capture in the
    single-camera layout. `relative` is the single source of truth for output paths, so
    "mirror the raw path exactly" falls out for both layouts with no branching in the
    stages -- they all build destinations as `<stage_root> / record.relative`.
    """

    source: Path  # the raw directory holding the streams
    relative: Path  # raw-relative path -> the output path under every stage root
    metadata_path: Path  # the operator metadata.yaml governing THIS capture
    metadata_relative: Path  # that file's raw-relative path -> where each stage copies it
    group_id: str  # captures sharing one cut interval
    role: CaptureRole
    layout: CaptureLayout

    @property
    def capture_id(self) -> str:
        """The manifest identifier for this capture."""

        return str(self.relative)

    @property
    def camera_id(self) -> str:
        """Alias kept so manifest readers written against the two-camera layout still work."""

        return self.capture_id


class CameraRecord(BaseModel):
    """S0's validated description of one camera. S1 reads it instead of re-probing."""

    relative: Path
    metadata_path: Path
    metadata_relative: Path
    group_id: str
    role: CaptureRole
    layout: CaptureLayout
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
    def capture_id(self) -> str:
        return str(self.relative)

    @property
    def camera_id(self) -> str:
        return self.capture_id

    @property
    def date(self) -> str:
        """The capture-day directory: the first segment of every raw-relative path."""

        return self.relative.parts[0]


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

    group_id: str
    lift: LiftMeta
    athlete: AthleteMeta
    cameras: list[CameraRecord]


class FloorRegion(BaseModel):
    """A normalized floor-selection rectangle, operator-annotated per camera (4-retilt.md §1)."""

    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)  # bottom_left.y -- larger, since y grows downward
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)  # top_right.y -- smaller

    @model_validator(mode="after")
    def validate_extent(self) -> FloorRegion:
        if self.x0 >= self.x1:
            raise ValueError("floor region degenerate on x: x0 >= x1")
        if self.y1 >= self.y0:
            raise ValueError("floor region degenerate on y: y1 >= y0")
        return self

    def pixel_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        """`(col_start, row_start, col_end, row_end)` per §1's conversion."""

        return (
            round(width * self.x0),
            round(height * self.y1),
            round(width * self.x1),
            round(height * self.y0),
        )


class PlaneFit(BaseModel):
    """The floor plane fitted from pooled, back-projected depth points (4-retilt.md §3)."""

    normal: list[float]  # unit vector, camera frame, n_y < 0
    rms_residual_m: float
    n_points: int
    n_frames_sampled: int
    confidence_mode: Literal["conf2_only", "conf1_and_2"]
    # Perpendicular distance, in metres, from the camera's optical centre to the fitted
    # plane -- i.e. the camera's height above the floor. Unlike `normal`, this is invariant
    # under S3's own rectifying rotation (a rotation about the optical centre never changes
    # distances from it), so S5 Pose reads it straight off this pre-rectification fit to
    # place the floor in the *rectified* frame it actually detects joints in (issue #118).
    floor_offset_m: float


class RetiltResult(BaseModel):
    """Everything one camera's rectification produced, before it is written to disk."""

    tilt_deg: float
    roll_deg: float
    plane: PlaneFit
    gravity_agreement_deg: float | None  # None when odometry orientation is unavailable
    homography_rgb: list[list[float]]
    homography_depth: list[list[float]]
    valid_bounds_px: CropBounds
    translation_span_m: float


class PlaneConsistency(BaseModel):
    """How one capture day's fitted floor planes agree with each other.

    Only meaningful for a `SINGLE_CAMERA` day, where one unmoved tripod shot every capture:
    a fit can pass every per-capture gate in `4-retilt.md` §7 and still be wrong -- RMS
    measures how *tightly* the points fit a plane, never whether that plane is the floor.
    A rectangle that caught a spectator's head fits its own surface beautifully. The day's
    captures agreeing with each other is the cheapest available check that they did not.

    Warn-only, like the odometry-gravity check it sits beside: it assumes a rig that never
    moved, and wants calibrating against a second single-camera day before it rejects.
    """

    tilt_median_deg: float
    roll_median_deg: float
    height_median_m: float
    n_captures: int
    deviations: dict[str, dict[str, float]]  # capture_id -> tilt/roll/height deviation
    warnings: dict[str, list[str]]  # capture_id -> messages, empty for an agreeing capture
