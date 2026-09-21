"""Pydantic models shared by every data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


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
    """One file operation planned or performed by a pipeline task."""

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
    """Information one task contributes to the run manifest."""

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


# The human body data model, shared by every pipeline and by powerflow-runtime.
# Normative source: "Human Body Data Structure v2" (docs/spec, GitHub issue #115).
JointId = Literal[
    "leftAnkle",
    "rightAnkle",
    "leftKnee",
    "rightKnee",
    "leftHip",
    "rightHip",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftClavicle",
    "rightClavicle",
    "head",
]


class Position3D(BaseModel):
    """A point in the 3D Cartesian capture space. Units/origin/axes: TBD (see spec v2)."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: float


class Joint(BaseModel):
    """One tracked anatomical landmark."""

    model_config = ConfigDict(frozen=True)

    id: JointId
    position: Position3D


class Bone(BaseModel):
    """A rigid connection between two distinct joints."""

    model_config = ConfigDict(frozen=True)

    from_joint: JointId
    to_joint: JointId

    @model_validator(mode="after")
    def validate_distinct_joints(self) -> Bone:
        if self.from_joint == self.to_joint:
            raise ValueError("a bone must connect two distinct joints")
        return self


class Skeleton(BaseModel):
    """The fixed joint/bone topology of the tracked human body."""

    model_config = ConfigDict(frozen=True)

    joints: tuple[JointId, ...]
    bones: tuple[Bone, ...]

    @model_validator(mode="after")
    def validate_no_duplicate_joints(self) -> Skeleton:
        if len(set(self.joints)) != len(self.joints):
            raise ValueError("joints must not contain duplicate entries")
        return self

    @model_validator(mode="after")
    def validate_bones_reference_known_joints(self) -> Skeleton:
        known = set(self.joints)
        for bone in self.bones:
            if bone.from_joint not in known or bone.to_joint not in known:
                raise ValueError(
                    f"bone {bone.from_joint!r}-{bone.to_joint!r} references a joint not in joints"
                )
        return self


HUMAN_SKELETON = Skeleton(
    joints=(
        "leftAnkle",
        "rightAnkle",
        "leftKnee",
        "rightKnee",
        "leftHip",
        "rightHip",
        "leftShoulder",
        "rightShoulder",
        "leftElbow",
        "rightElbow",
        "leftWrist",
        "rightWrist",
        "leftClavicle",
        "rightClavicle",
        "head",
    ),
    bones=(
        Bone(from_joint="leftAnkle", to_joint="leftKnee"),
        Bone(from_joint="rightAnkle", to_joint="rightKnee"),
        Bone(from_joint="leftKnee", to_joint="leftHip"),
        Bone(from_joint="rightKnee", to_joint="rightHip"),
        Bone(from_joint="leftHip", to_joint="rightHip"),
        Bone(from_joint="leftHip", to_joint="leftShoulder"),
        Bone(from_joint="rightHip", to_joint="rightShoulder"),
        Bone(from_joint="leftShoulder", to_joint="leftElbow"),
        Bone(from_joint="rightShoulder", to_joint="rightElbow"),
        Bone(from_joint="leftElbow", to_joint="leftWrist"),
        Bone(from_joint="rightElbow", to_joint="rightWrist"),
        Bone(from_joint="leftShoulder", to_joint="leftClavicle"),
        Bone(from_joint="rightShoulder", to_joint="rightClavicle"),
        Bone(from_joint="leftClavicle", to_joint="head"),
        Bone(from_joint="rightClavicle", to_joint="head"),
    ),
)


# The stored form of the human body model: topology as a standalone document, and per-frame
# joint trajectories as struct-of-arrays.
# Normative source: "Human Body Model Data Storage v1" (docs/spec, GitHub issue #117).
SKELETON_ID: Final = "human-v2"  # versioned: stored pose is only readable against its topology
SKELETON_SCHEMA_VERSION: Final = "1.0.0"
# 2.1.0 made JointSeries.position nullable as a whole, for documents S5 produced with its
# metric lift switched off (PoseOutput.PIXELS_2D). A 2.0.0 reader must not be handed one.
POSE_SCHEMA_VERSION: Final = "2.1.0"

# Which `metadata.yaml` fields a capture reads -- a lookup key, not a claim about where the
# camera pointed. Replaces a `Literal["Side", "Front"]` camera name, which could not name a
# capture in the single-camera layout at all (pydantic rejected it at runtime).
CaptureRole = Literal["side", "front", "single"]
PipelineStage = Literal["s0_ingest", "s1_cut", "s2_orient", "s3_retilt", "s4_crop"]

PositionTriple = tuple[float, float, float]
PixelPair = tuple[int, int]

# These documents are read alongside powerflow-ui's schema-normalized family, which is camelCase,
# so the wire form is camelCase while the Python attributes stay snake_case like the rest of the
# pipeline.
_DOCUMENT_CONFIG = ConfigDict(frozen=True, alias_generator=to_camel, populate_by_name=True)


class Frames(BaseModel):
    """The single frame axis that every per-frame array in one document shares."""

    model_config = _DOCUMENT_CONFIG

    count: int = Field(ge=1)
    fps: float = Field(gt=0)  # nominal only -- real captures are variable-frame-rate
    start_epoch_ms: int
    t_ms: tuple[int, ...]  # offsets from start_epoch_ms; never derive timing from index / fps

    @model_validator(mode="after")
    def validate_offsets(self) -> Frames:
        if len(self.t_ms) != self.count:
            raise ValueError(f"t_ms must hold exactly {self.count} entries, got {len(self.t_ms)}")
        if any(offset < 0 for offset in self.t_ms):
            raise ValueError("t_ms offsets must be non-negative")
        if any(later < earlier for earlier, later in zip(self.t_ms, self.t_ms[1:], strict=False)):
            raise ValueError("t_ms offsets must not travel backwards")
        return self


class JointSeries(BaseModel):
    """One joint's trajectory across the frames of a single capture.

    A frame the detector could not place carries `None` for both positions and zero confidence;
    the arrays keep their full length so every index still refers to the same frame.

    `position` is the whole-array `None` when S5 ran its 2D detection step without the metric
    lift (`PoseOutput.PIXELS_2D`). That is a different claim from a tuple of `None`s: the tuple
    says every frame dropped out, `None` says positions were never computed for this capture.
    Collapsing the two would let a consumer read "the detector saw nothing" off a document whose
    2D detections are complete.
    """

    model_config = _DOCUMENT_CONFIG

    # metres, floor frame (see PoseDocument); None when the metric lift did not run
    position: tuple[PositionTriple | None, ...] | None
    pixel_position: tuple[PixelPair | None, ...]  # image space of PoseDocument.stage
    confidence: tuple[float, ...]

    @model_validator(mode="after")
    def validate_series(self) -> JointSeries:
        if len(self.pixel_position) != len(self.confidence):
            raise ValueError("pixelPosition and confidence must be the same length")
        if self.position is not None and len(self.position) != len(self.pixel_position):
            raise ValueError("position, pixelPosition and confidence must be the same length")

        for index, (pixel, confidence) in enumerate(
            zip(self.pixel_position, self.confidence, strict=True)
        ):
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"frame {index}: confidence must lie in [0, 1]")
            # Without positions the pixel carries the drop-out, so it takes position's role in
            # the confidence rule below.
            if self.position is None and pixel is None and confidence != 0.0:
                raise ValueError(f"frame {index}: an occluded joint must carry zero confidence")

        if self.position is None:
            return self

        for index, (point, pixel, confidence) in enumerate(
            zip(self.position, self.pixel_position, self.confidence, strict=True)
        ):
            if (point is None) != (pixel is None):
                raise ValueError(
                    f"frame {index}: position and pixelPosition must drop out together"
                )
            if point is None and confidence != 0.0:
                raise ValueError(f"frame {index}: an occluded joint must carry zero confidence")
        return self

    def __len__(self) -> int:
        # pixelPosition, not position: it is the one array every document carries.
        return len(self.pixel_position)

    def position_at(self, index: int) -> Position3D | None:
        """The stored triple for one frame, rebuilt as a `Position3D`.

        Always `None` for a 2D-only series -- there is no position to rebuild.
        """

        if self.position is None:
            return None
        point = self.position[index]
        if point is None:
            return None
        x, y, z = point
        return Position3D(x=x, y=y, z=z)


class PoseDocument(BaseModel):
    """Every tracked joint's trajectory for one camera of one capture.

    Positions are metres in a right-handed frame whose origin is the camera optical centre
    projected onto the floor plane fitted by `s3_retilt`, with +y along the floor normal. So
    `position[1]` is the joint's height above the floor, and the UI's `heightM` is a view of this
    rather than a separately stored number. The floor plane is fitted per camera, so two cameras
    of one session are not in a common frame until a fusion step exists.

    `capture_id` is the raw-relative path of the capture these joints came from -- the only
    identifier that names a capture under both raw layouts -- and `role` says which set of
    operator annotations it was processed against.
    """

    model_config = _DOCUMENT_CONFIG

    schema_version: str = POSE_SCHEMA_VERSION
    skeleton_id: str = SKELETON_ID
    capture_id: str
    role: CaptureRole
    stage: PipelineStage  # whose image space pixel_position is drawn in
    frames: Frames
    joints: dict[JointId, JointSeries]

    @model_validator(mode="after")
    def validate_joints(self) -> PoseDocument:
        missing = set(HUMAN_SKELETON.joints) - set(self.joints)
        if missing:
            raise ValueError(f"pose is missing joints: {sorted(missing)}")
        for joint, series in self.joints.items():
            if len(series) != self.frames.count:
                raise ValueError(
                    f"joint {joint!r} holds {len(series)} frames, expected {self.frames.count}"
                )
        # The metric lift is a per-run stage setting, so it either ran for this capture or it
        # did not. A document where some joints carry positions and others do not could only
        # come from a bug, and would make `has_positions` a question without an answer.
        lifted = {series.position is not None for series in self.joints.values()}
        if len(lifted) > 1:
            raise ValueError(
                "pose mixes lifted and pixel-only joints: either every joint carries positions "
                "or none does"
            )
        return self

    @property
    def has_positions(self) -> bool:
        """Whether S5's metric lift ran for this capture, i.e. `position` is populated.

        `validate_joints` guarantees the joints agree, so any one of them answers for all.
        """

        return next(iter(self.joints.values())).position is not None

    def to_document(self) -> dict[str, Any]:
        """The camelCase wire form written to `pose.json`."""

        return self.model_dump(by_alias=True, mode="json")

    @classmethod
    def from_document(cls, payload: dict[str, Any]) -> PoseDocument:
        """Rebuild from the wire form, validating every invariant on the way in."""

        return cls.model_validate(payload)


def skeleton_document(
    skeleton: Skeleton = HUMAN_SKELETON,
    skeleton_id: str = SKELETON_ID,
) -> dict[str, Any]:
    """The topology as a standalone document, generated from the normative constant.

    Nothing hand-maintains the published `skeleton.<id>.json`; it is rendered from
    `HUMAN_SKELETON` so the code and the document cannot drift apart.
    """

    return {
        "schemaVersion": SKELETON_SCHEMA_VERSION,
        "id": skeleton_id,
        "joints": list(skeleton.joints),
        "bones": [[bone.from_joint, bone.to_joint] for bone in skeleton.bones],
    }
