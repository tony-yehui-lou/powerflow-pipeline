"""S6 · Metric lift -> gives S5's pixel-space joints floor-frame positions from LiDAR depth.

S5 Pose publishes a `PoseDocument` with `position: null`; this stage reads it, back-projects
every detected pixel through S4's depth stream, and publishes the same document with positions
filled in. The split is what lets the two be re-run independently: detection is the expensive
half (RTMPose on CPU), and the depth sampler is the half still being revised -- see
docs/specs/preprocessing/S5-bone-constrained-reconstruction.md.

Everything this stage reads is another stage's published output: S5's pose document, S4's
`depth/` and `confidence/` frames, S3's floor offset, and S0's intrinsics off the crop record.
Nothing is recomputed.
"""

from __future__ import annotations

import json

from prefect import task

from powerflow_pipeline.data.common.models import FileOp, PoseDocument, StepResult
from powerflow_pipeline.data.common.pose_storage import pose_path, read_pose, write_pose
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.pose_lift import lift_joint_series


def _floor_offset_m(config: PreprocessConfig, record: CameraRecord) -> float:
    """This camera's floor height above the optical centre, from S3's own sidecar.

    `PlaneFit.floor_offset_m` is invariant under S3's rectifying rotation (a distance from the
    origin doesn't change when the axes about that origin are rotated), so it's read straight
    off S3's pre-rectification fit rather than recomputed here -- S6 has no floor region or
    depth-selection logic of its own, only 4-retilt.md's.
    """

    sidecar_path = config.retilt_root / record.relative / "retilt_sidecar.json"
    sidecar = json.loads(sidecar_path.read_text())
    offset: float = sidecar["floor_offset_m"]
    return offset


@task(retries=1)
def lift_pose_camera(
    record: CameraRecord, config: PreprocessConfig
) -> tuple[CameraRecord, StepResult]:
    """Lift one camera's S5 pose document into metres and publish it to `config.lift_root`.

    `record` is S4's output record, the same one S5 was given: it carries the intrinsics and
    the RGB/depth dimensions the back-projection needs, and `record.source` is where S4's
    `depth/` and `confidence/` frames live.
    """

    assert config.pose_root is not None  # the caller checks both before looping cameras
    assert config.lift_root is not None

    session, camera = record.relative.parent, record.relative.name
    source = pose_path(config.pose_root, session, camera)
    destination = pose_path(config.lift_root, session, camera)
    log_task_paths(source, destination)

    file_ops = [
        FileOp(op="write", src=source, dst=destination),
        FileOp(op="publish", src=record.source, dst=destination),
    ]
    if config.dry_run:
        return record, StepResult(file_ops=file_ops)

    detected = read_pose(source)
    joints = lift_joint_series(
        detected.joints,
        depth_dir=record.source / "depth",
        confidence_dir=record.source / "confidence",
        intrinsics=record.intrinsics,
        rgb_size=(record.rgb_width, record.rgb_height),
        depth_size=(record.depth_width, record.depth_height),
        floor_offset_m=_floor_offset_m(config, record),
        patch_radius=config.lift_patch_radius,
    )

    # S5's frame axis passes through untouched: it was derived from `rgb.mp4`'s own PTS and
    # this stage adds no frames, drops none, and re-times nothing.
    document = PoseDocument(
        capture_id=detected.capture_id,
        role=detected.role,
        stage=detected.stage,
        frames=detected.frames,
        joints=joints,
    )

    if config.overwrite and destination.exists():
        destination.unlink()
    write_pose(destination, document)

    placed = sum(
        1 for series in joints.values() for point in (series.position or ()) if point is not None
    )
    detected_pixels = sum(
        1 for series in detected.joints.values() for pixel in series.pixel_position if pixel
    )
    return record, StepResult(
        derived={"joint_frames_detected": detected_pixels, "joint_frames_lifted": placed},
        file_ops=file_ops,
    )
