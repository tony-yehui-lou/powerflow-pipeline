"""S9 · Bar HVA -> the bar centre's height, velocity and acceleration, by box-constrained Tikhonov
regularisation of S5's bar track (docs/specs/preprocessing/S9-hva-tikhonov-draft2.md).

Everything read here is another stage's published output: S5's barbell detections, S4's
intrinsics and PTS (off the crop record and its `rgb.mp4`), S3's floor offset, S1's epoch, and the
capture's raw `metadata.yaml` for the bar type. The bar is placed exactly as S8 places it
(`barbell_input.bar_track`, `floor_anchor`), so the bar and the joints share one floor. The
document is assembled by `bar_hva.build`; this task is the I/O around it.

A capture this stage cannot solve is never rejected: S0-S4 already published, and the reason is a
manifest warning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from prefect import task

from powerflow_pipeline.data.common.frame_clock import capture_start_epoch_ms, frame_offsets_ms
from powerflow_pipeline.data.common.models import FileOp, StepResult
from powerflow_pipeline.data.common.pose_storage import write_sidecar
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess import bar_hva, barbell_input
from powerflow_pipeline.data.preprocess.bar_geometry import floor_anchor
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.tasks.retilt import floor_offset_mm

BAR_HVA_FILENAME: Final = "bar_hva.json"


def bar_hva_path(bar_hva_root: Path, record: CameraRecord) -> Path:
    return bar_hva_root / record.relative / BAR_HVA_FILENAME


@task(retries=1)
def solve_bar_hva_camera(
    record: CameraRecord, config: PreprocessConfig
) -> tuple[CameraRecord, StepResult]:
    """Differentiate one camera's bar height and publish `bar_hva.json` to `config.bar_hva_root`.

    `record` is S4's output record: it carries the intrinsics and the image size the barbell
    detections must match, and its source holds the `rgb.mp4` whose PTS time the frames.
    """

    assert config.bar_hva_root is not None  # the flow checks both before looping cameras
    assert config.barbell_root is not None

    source = barbell_input.detections_path(config.barbell_root, record.relative)
    destination = bar_hva_path(config.bar_hva_root, record)
    log_task_paths(source, destination)
    file_ops = [FileOp(op="write", src=source, dst=destination)]
    if config.dry_run:
        return record, StepResult(file_ops=file_ops)

    bars, missing = barbell_input.bar_track(record, config)
    if missing:
        return record, StepResult(warnings=[f"bar HVA not solved: {cause}" for cause in missing])
    t_ms = frame_offsets_ms(record.source / "rgb.mp4")
    if len(t_ms) != record.n_frames:
        return record, StepResult(
            warnings=[
                f"bar HVA not solved: frame_count_mismatch, {len(t_ms)} PTS in rgb.mp4 against "
                f"{record.n_frames} frames in the record"
            ]
        )
    bar_type = barbell_input.read_bar_type(record.metadata_path)
    assert bar_type is not None  # `bar_track` placed a bar, so the type was declared

    built = bar_hva.build(
        bar_hva.Inputs(
            capture_id=record.capture_id,
            start_epoch_ms=capture_start_epoch_ms(config.cut_root, record.relative),
            t_ms=t_ms,
            bars=bars,
            anchor=floor_anchor(
                bars, floor_offset_mm=floor_offset_mm(config.retilt_root, record.relative)
            ),
            bar_type=bar_type,
            settings=bar_hva.Settings(
                v_max_mps=config.bar_hva_v_max_mps,
                a_max_mps2=config.bar_hva_a_max_mps2,
                collar_only=config.bar_hva_collar_only.value,
                tau=config.bar_hva_tau,
                kappa=config.bar_hva_kappa,
            ),
        )
    )

    if config.overwrite and destination.exists():
        destination.unlink()
    write_sidecar(destination, built.document)
    return record, StepResult(derived=built.derived, warnings=built.warnings, file_ops=file_ops)
