"""S5 · Pose detection -> runs an injected joint-detection model on S4's cropped RGB and
publishes one `PoseDocument` per camera (docs/spec, GitHub issue #116).

The model itself is out of scope (issue #118): `model` is a `PoseModel` this task calls
through, so the stage's plumbing -- reading S4's output, timing the frames, and writing the
result at `pose_storage.py`'s stage-output layout (issue #117) -- can be built and tested
against a stand-in before a real model exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, cast

import av
from prefect import task

from powerflow_pipeline.data.common.models import (
    CameraName,
    FileOp,
    Frames,
    PipelineStage,
    PoseDocument,
    StepResult,
)
from powerflow_pipeline.data.common.pose_storage import (
    pose_path,
    skeleton_path,
    write_pose,
    write_skeleton,
)
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.pose_model import PoseModel

# Whose image space `model`'s pixel_position is drawn in -- S5 always reads S4's output.
POSE_SOURCE_STAGE: Final[PipelineStage] = "s4_crop"


def _frame_offsets_ms(rgb_path: Path) -> list[int]:
    """Each frame's offset from the first, in milliseconds, read from `rgb_path`'s own PTS.

    S1 Cut rebased this stream onto a zero-based, odometry-derived timeline (`trim_rgb`); S2-S4
    only touch pixels, never PTS, so the container still carries that exact timeline here.
    """

    offsets: list[int] = []
    with av.open(str(rgb_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            assert frame.pts is not None, f"{rgb_path}: frame carries no PTS"
            assert frame.time_base is not None, f"{rgb_path}: frame carries no time_base"
            offsets.append(round(float(frame.pts * frame.time_base) * 1000))
    return offsets


def _capture_start_epoch_ms(config: PreprocessConfig, record: CameraRecord) -> int:
    """The true epoch of frame 0, from the `cut_sidecar.json` S1 wrote for this camera.

    Later stages don't carry epoch time forward -- they don't need it, only S5 does -- so this
    reaches back to S1's own output tree rather than something copied down the pipeline.
    """

    sidecar_path = (
        config.cut_root / record.date / record.session / record.camera / "cut_sidecar.json"
    )
    sidecar = json.loads(sidecar_path.read_text())
    start_ms: int = sidecar["retained"]["rgb"]["epoch_range"][0]
    return start_ms


@task(retries=1)
def detect_pose_camera(
    record: CameraRecord, config: PreprocessConfig, model: PoseModel
) -> tuple[CameraRecord, StepResult]:
    """Run `model` on one camera's S4 output and publish the resulting `PoseDocument`.

    `record` is S4's output record; like S4 Crop, everything S5 reads lives in the prior
    stage's own output tree, except the capture's epoch anchor, which only S1 recorded.
    """

    assert config.pose_root is not None  # the caller checks this before looping cameras
    source = record.source
    destination = pose_path(config.pose_root, Path(record.date) / record.session, record.camera)
    log_task_paths(source, destination)

    file_ops = [
        FileOp(op="write", src=source / "rgb.mp4", dst=destination),
        FileOp(op="publish", src=source, dst=destination),
    ]
    if config.dry_run:
        return record, StepResult(file_ops=file_ops)

    joints = model.predict(rgb_path=source / "rgb.mp4", n_frames=record.n_frames)
    frames = Frames(
        count=record.n_frames,
        fps=record.fps,
        start_epoch_ms=_capture_start_epoch_ms(config, record),
        t_ms=tuple(_frame_offsets_ms(source / "rgb.mp4")),
    )
    document = PoseDocument(
        camera=cast(CameraName, record.camera),
        stage=POSE_SOURCE_STAGE,
        frames=frames,
        joints=joints,
    )

    if config.overwrite and destination.exists():
        destination.unlink()
    write_pose(destination, document)

    return record, StepResult(derived={"n_frames": record.n_frames}, file_ops=file_ops)


def ensure_skeleton_published(pose_root: Path, *, overwrite: bool) -> Path:
    """Publish `skeleton.human-v2.json` once per run; a rerun leaves it alone unless `overwrite`.

    Unlike a per-camera pose document, the topology doesn't vary by scan, so this is called
    once for the whole run rather than per camera (see `flow.py`).
    """

    path = skeleton_path(pose_root)
    if path.exists():
        if not overwrite:
            return path
        path.unlink()
    return write_skeleton(pose_root)
