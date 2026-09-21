"""S4 establishes per-camera I5: one stable, common-region rectangle across every frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.common.models import CropBounds
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.crop import depth_bounds
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.tasks.crop import crop_camera
from powerflow_pipeline.data.preprocess.tasks.cut import cut_camera, resolve_cut_interval
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera, read_frame
from powerflow_pipeline.data.preprocess.tasks.orient import orient_camera
from powerflow_pipeline.data.preprocess.tasks.retilt import retilt_camera
from tests.conftest import MakeCamera, MakeSessionMetadata, sole_capture
from tests.unit.test_preprocess_ingest import make_config as _base_make_config


def make_config(tmp_path: Path, **overrides: object) -> PreprocessConfig:
    # The synthetic fixture's intrinsics are tuned to its default (tiny) frame size --
    # the default crop_safety_px=8 (sized for real ~1440px-wide frames) would swallow most
    # of a 48px-wide synthetic one. 1px still exercises the full guard-band/intersection path.
    overrides.setdefault("crop_safety_px", 1)
    return _base_make_config(tmp_path, **overrides)


def _build_retilt_record(
    raw_root: Path, config: PreprocessConfig, camera: str = "Side"
) -> CameraRecord:
    """Run S0 -> S1 -> S2 -> S3 on the one synthetic camera named `camera`."""

    camera_dir = sole_capture(raw_root, camera)
    ingested = ingest_camera.fn(camera_dir, config)
    interval = resolve_cut_interval.fn(camera_dir.metadata_path, ingested)
    cut_record, _ = cut_camera.fn(ingested, interval, config)
    orient_record, _ = orient_camera.fn(cut_record, config)
    retilt_record, _ = retilt_camera.fn(orient_record, config)
    return retilt_record


def _make_crop_session(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_session_metadata: MakeSessionMetadata,
    *,
    odometry_translation: object = None,
    **camera_overrides: object,
) -> Path:
    raw = tmp_path / "raw"
    make_camera(
        raw,
        camera="Side",
        rgb_frames=5,
        depth_frames=6,
        render_floor_plane=True,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
        odometry_translation=odometry_translation,  # type: ignore[arg-type]
        **camera_overrides,
    )
    make_session_metadata(raw, lift_start_ms=16, lift_end_ms=80, floor_regions="full_frame")
    return raw


# --- the happy path ------------------------------------------------------------------


def test_crop_camera_produces_the_expected_rectangle(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    crop_record, step = crop_camera.fn(retilt_record, config)

    assert crop_record.rgb_width % 2 == 0
    assert crop_record.rgb_height % 2 == 0
    assert crop_record.rgb_width < retilt_record.rgb_width  # S3's border was trimmed
    assert step.derived["crop_bounds_px"][2] - step.derived["crop_bounds_px"][0] == (
        crop_record.rgb_width
    )


def test_crop_camera_rewrites_intrinsics_for_the_new_origin(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    crop_record, step = crop_camera.fn(retilt_record, config)

    x0, y0, _, _ = step.derived["crop_bounds_px"]
    assert crop_record.intrinsics.fx == retilt_record.intrinsics.fx
    assert crop_record.intrinsics.fy == retilt_record.intrinsics.fy
    assert crop_record.intrinsics.cx == pytest.approx(retilt_record.intrinsics.cx - x0)
    assert crop_record.intrinsics.cy == pytest.approx(retilt_record.intrinsics.cy - y0)


def test_crop_sidecar_is_written_with_provenance(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    import json

    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    crop_record, _ = crop_camera.fn(retilt_record, config)

    sidecar = json.loads((crop_record.source / "crop_sidecar.json").read_text())
    assert sidecar["reference_frame"] == 0
    assert sidecar["k_rewritten"] is True
    assert sidecar["pixels_shifted"] is False
    assert sidecar["camera_translation_corrected"] is False
    assert "valid_bounds_px" in sidecar
    assert "crop_bounds_px" in sidecar
    assert "bound_source" in sidecar


def test_odometry_and_imu_are_copied_byte_for_byte(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    crop_record, _ = crop_camera.fn(retilt_record, config)

    assert (crop_record.source / "odometry.csv").read_bytes() == (
        retilt_record.source / "odometry.csv"
    ).read_bytes()
    assert (crop_record.source / "imu.csv").read_bytes() == (
        retilt_record.source / "imu.csv"
    ).read_bytes()


def test_cropped_depth_matches_the_input_slice_exactly(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    crop_record, step = crop_camera.fn(retilt_record, config)

    x0, y0, x1, y1 = step.derived["crop_bounds_px"]
    input_frame = read_frame(sorted((retilt_record.source / "depth").glob("*.png"))[0])
    dbounds = depth_bounds(
        CropBounds(x0=x0, y0=y0, x1=x1, y1=y1),
        (retilt_record.rgb_width, retilt_record.rgb_height),
        (retilt_record.depth_width, retilt_record.depth_height),
    )
    expected_slice = input_frame[dbounds.y0 : dbounds.y1, dbounds.x0 : dbounds.x1]
    output_frame = read_frame(sorted((crop_record.source / "depth").glob("*.png"))[0])
    assert np.array_equal(output_frame, expected_slice)


# --- §6 rejection cases -----------------------------------------------------------------


def test_missing_valid_bounds_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    import json

    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    sidecar_path = retilt_record.source / "retilt_sidecar.json"
    sidecar = json.loads(sidecar_path.read_text())
    del sidecar["valid_bounds_px"]
    sidecar_path.write_text(json.dumps(sidecar))

    with pytest.raises(ScanRejected, match="valid_bounds_px"):
        crop_camera.fn(retilt_record, config)


def test_degenerate_valid_bounds_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    import json

    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    sidecar_path = retilt_record.source / "retilt_sidecar.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["valid_bounds_px"] = [100, 0, 50, 100]  # x0 > x1
    sidecar_path.write_text(json.dumps(sidecar))

    with pytest.raises(ScanRejected, match="degenerate"):
        crop_camera.fn(retilt_record, config)


def test_translation_beyond_tolerance_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        odometry_translation=lambda i: (0.05 * i, 0.0, 0.0),  # drifts well past any tolerance
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    with pytest.raises(ScanRejected, match="translated"):
        crop_camera.fn(retilt_record, config)


def test_no_valid_depth_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)

    for path in (retilt_record.source / "confidence").glob("*.png"):
        zero = np.zeros(read_frame(path).shape, dtype=np.uint8)
        import cv2

        cv2.imwrite(str(path), zero)

    with pytest.raises(ScanRejected, match="depth"):
        crop_camera.fn(retilt_record, config)


def test_empty_intersection_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10, crop_safety_px=100_000)
    retilt_record = _build_retilt_record(raw, config)

    with pytest.raises(ScanRejected, match="overlap"):
        crop_camera.fn(retilt_record, config)


def test_excessive_crop_fraction_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10, crop_max_crop_fraction=0.001)
    retilt_record = _build_retilt_record(raw, config)

    with pytest.raises(ScanRejected, match="crop_max_crop_fraction"):
        crop_camera.fn(retilt_record, config)


# --- dry run ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_crop_session(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    retilt_record = _build_retilt_record(raw, config)
    dry_config = config.model_copy(update={"dry_run": True})

    crop_record, step = crop_camera.fn(retilt_record, dry_config)

    assert not crop_record.source.exists()
    assert step.file_ops
