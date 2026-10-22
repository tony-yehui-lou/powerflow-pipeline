"""S5 · Barbell detection -> runs an injected detector on S4's cropped RGB and publishes one
`BarbellDocument` per camera (docs/specs/preprocessing/S5-barbell-detection-stage-draft3.md).

S5 records candidates and stops: the top-scoring cap and collar per frame, in S4's pixels. It does
no geometry. S6's wrist correction and S8 read the published document -- never this task's
in-memory result -- so they can run against an earlier run's detections as well as this one's.

The detector is a `BarbellDetector` this task calls through, so everything here is testable
against a stand-in, and none of it imports torch.
"""

from __future__ import annotations

import shutil
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import av
import cv2
import numpy as np
from prefect import task

from powerflow_pipeline.data.common.barbell_storage import BARBELL_FILENAME, write_barbell
from powerflow_pipeline.data.common.capture_timing import capture_start_epoch_ms, frame_offsets_ms
from powerflow_pipeline.data.common.filesystem import (
    cleanup,
    create_staging_root,
    publish_staging,
)
from powerflow_pipeline.data.common.models import (
    BARBELL_CLASSES,
    BarbellClass,
    BarbellDetection,
    BarbellDocument,
    FileOp,
    Frames,
    PipelineStage,
    PixelSpace,
    StepResult,
)
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess.barbell_model import BarbellDetector
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord

# Whose image space every box is drawn in -- S5 always reads S4's output.
BARBELL_SOURCE_STAGE: Final[PipelineStage] = "s4_crop"
OVERLAY_FILENAME: Final = "overlay.mp4"
OVERLAY_CRF: Final = 23  # a review video, not a stage input
COLOURS: Final[dict[BarbellClass, tuple[int, int, int]]] = {
    "cap": (60, 220, 255),
    "collar": (120, 255, 120),
}


@task(retries=0)  # inference is deterministic: a retry only repeats the failure
def detect_barbell_camera(
    record: CameraRecord, config: PreprocessConfig, detector: BarbellDetector | None
) -> tuple[CameraRecord, StepResult]:
    """Detect the cap and collar on every frame of S4's `rgb.mp4`, and publish them.

    Published output already there is kept, not re-detected, unless `overwrite`: at minutes per
    capture, a rerun over a day should only fill in what's missing. `detector` is None only in a
    dry run, which never loads a model.
    """

    assert config.barbell_root is not None  # the flow checks this before looping cameras
    source = record.source / "rgb.mp4"
    destination = config.barbell_root / record.relative
    log_task_paths(record.source, destination)

    if destination.exists() and not config.overwrite:
        return record, StepResult(derived={"barbell_reused": True})

    file_ops = [
        FileOp(op="write", src=source, dst=destination / BARBELL_FILENAME),
        FileOp(op="publish", src=record.source, dst=destination),
    ]
    if config.barbell_render_overlay:
        file_ops.insert(1, FileOp(op="write", src=source, dst=destination / OVERLAY_FILENAME))
    if config.dry_run:
        return record, StepResult(file_ops=file_ops)

    assert detector is not None, "S5 needs a detector outside a dry run"
    started = time.perf_counter()
    detected = detector.detect(source, record.n_frames)
    seconds = time.perf_counter() - started

    for name in BARBELL_CLASSES:
        if len(detected[name]) != record.n_frames:
            raise ValueError(
                f"detector returned {len(detected[name])} frames of {name!r}, "
                f"expected {record.n_frames}"
            )
    document = BarbellDocument(
        capture_id=record.capture_id,
        role=record.role,
        stage=BARBELL_SOURCE_STAGE,
        pixel_space=PixelSpace(width=record.rgb_width, height=record.rgb_height),
        frames=Frames(
            count=record.n_frames,
            fps=record.fps,
            start_epoch_ms=capture_start_epoch_ms(config.cut_root, record.relative),
            t_ms=tuple(frame_offsets_ms(source)),
        ),
        landmarks={name: tuple(detected[name]) for name in BARBELL_CLASSES},
        detector=detector.provenance,
    )

    staging = create_staging_root(destination)
    try:
        write_barbell(staging / BARBELL_FILENAME, document)
        shutil.copy2(record.metadata_path, staging / "metadata.yaml")
        if config.barbell_render_overlay:
            render_overlay(source, document.landmarks, staging / OVERLAY_FILENAME)
        if destination.exists():  # only with `overwrite`; replaced whole, never merged
            shutil.rmtree(destination)
        publish_staging(staging, destination)
    except BaseException:
        cleanup(staging)
        raise

    derived = _derived(document, seconds)
    return record, StepResult(
        derived=derived, warnings=_coverage_warnings(document, config), file_ops=file_ops
    )


def _derived(document: BarbellDocument, seconds: float) -> dict[str, Any]:
    def median_score(name: BarbellClass) -> float | None:
        scores = [hit.score for hit in document.landmarks[name] if hit is not None]
        return round(statistics.median(scores), 4) if scores else None

    # Prefixed: the flow merges every stage's derived values into one dict per capture.
    return {
        "barbell_frames_with_cap": document.frames_with("cap"),
        "barbell_frames_with_collar": document.frames_with("collar"),
        "barbell_frames_with_both": document.frames_with_both(),
        "barbell_median_cap_score": median_score("cap"),
        "barbell_median_collar_score": median_score("collar"),
        "barbell_s_per_frame": round(seconds / document.frames.count, 3),
        "barbell_checkpoint_sha256": document.detector.checkpoint_sha256,
        "barbell_device": document.detector.device,
        "barbell_reused": False,
    }


def _coverage_warnings(document: BarbellDocument, config: PreprocessConfig) -> list[str]:
    """A warning per class seen on too few frames. Never a rejection: sparse is still valid."""

    thresholds: dict[BarbellClass, float] = {
        "cap": config.barbell_warn_cap_coverage,
        "collar": config.barbell_warn_collar_coverage,
    }
    total = document.frames.count
    warnings = []
    for name, threshold in thresholds.items():
        seen = document.frames_with(name)
        if seen / total < threshold:
            warnings.append(f"{name} on {seen}/{total} frames, below {threshold:.0%}")
    return warnings


def render_overlay(
    source: Path,
    landmarks: Mapping[BarbellClass, Sequence[BarbellDetection | None]],
    destination: Path,
) -> None:
    """Every frame of `source` with its boxes and scores drawn on, size and PTS unchanged.

    Frames past the end of `landmarks` are copied undrawn (a quick look may detect only a few).
    """

    with av.open(str(source)) as inp, av.open(str(destination), "w") as out:
        in_stream = inp.streams.video[0]
        out_stream = out.add_stream(
            "libx264", rate=in_stream.average_rate, options={"crf": str(OVERLAY_CRF)}
        )
        out_stream.width = in_stream.codec_context.width
        out_stream.height = in_stream.codec_context.height
        out_stream.pix_fmt = "yuv420p"
        time_base = in_stream.time_base
        assert time_base is not None, f"{source}: stream carries no time_base"
        out_stream.time_base = time_base
        out_stream.codec_context.time_base = time_base

        for index, frame in enumerate(inp.decode(in_stream)):
            canvas = np.ascontiguousarray(frame.to_ndarray(format="bgr24"))
            _draw(canvas, landmarks, index)
            out_frame = av.VideoFrame.from_ndarray(canvas, format="bgr24")
            out_frame.pts = frame.pts
            out_frame.time_base = frame.time_base
            for packet in out_stream.encode(out_frame):
                out.mux(packet)
        for packet in out_stream.encode():
            out.mux(packet)


def _draw(
    canvas: np.ndarray,
    landmarks: Mapping[BarbellClass, Sequence[BarbellDetection | None]],
    index: int,
) -> None:
    for name, series in landmarks.items():
        hit = series[index] if index < len(series) else None
        if hit is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in hit.box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOURS[name], 3)
        label = f"{name} {hit.score:.2f}"
        cv2.putText(
            canvas, label, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS[name], 2
        )
