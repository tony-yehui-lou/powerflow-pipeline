"""S3 establishes I3: streams rectified level with the fitted floor plane."""

from __future__ import annotations

from pathlib import Path

import pytest

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.tasks.cut import cut_camera, resolve_cut_interval
from powerflow_pipeline.data.preprocess.tasks.discover import discover_sessions
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from powerflow_pipeline.data.preprocess.tasks.orient import orient_camera
from powerflow_pipeline.data.preprocess.tasks.retilt import retilt_camera
from tests.conftest import MakeCamera, MakeSessionMetadata
from tests.unit.test_preprocess_ingest import make_config as _base_make_config


def make_config(tmp_path: Path, **overrides: object) -> PreprocessConfig:
    return _base_make_config(tmp_path, **overrides)


def _build_orient_record(
    raw_root: Path, config: PreprocessConfig, camera: str = "Side"
) -> CameraRecord:
    """Run S0 -> S1 -> S2 on the one synthetic camera named `camera`; return S2's record."""

    (camera_dir,) = [c for c in discover_sessions.fn(raw_root) if c.camera == camera]
    ingested = ingest_camera.fn(camera_dir, config)
    interval = resolve_cut_interval.fn(raw_root, camera_dir.date, camera_dir.session, ingested)
    cut_record, _ = cut_camera.fn(ingested, interval, config)
    orient_record, _ = orient_camera.fn(cut_record, config)
    return orient_record


def _make_floor_session(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_session_metadata: MakeSessionMetadata,
    *,
    floor_regions: object = "full_frame",
    **camera_overrides: object,
) -> Path:
    raw = tmp_path / "raw"
    make_camera(
        raw,
        camera="Side",
        rgb_frames=5,
        depth_frames=6,
        render_floor_plane=True,
        **camera_overrides,
    )
    make_session_metadata(
        raw,
        lift_start_ms=16,
        lift_end_ms=80,
        floor_regions=floor_regions,  # type: ignore[arg-type]
    )
    return raw


# --- the happy path ------------------------------------------------------------------


def test_retilt_camera_recovers_baked_in_tilt(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    retilt_record, step = retilt_camera.fn(orient_record, raw, config)

    assert step.derived["tilt_deg"] == pytest.approx(8.0, abs=0.5)
    assert step.derived["roll_deg"] == pytest.approx(-2.0, abs=0.5)
    assert retilt_record.n_frames == orient_record.n_frames
    assert retilt_record.source == config.retilt_root / "9 July" / "cnj_45kg_Set1" / "Side"


def test_retilt_camera_publishes_expected_files(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    retilt_camera.fn(orient_record, raw, config)

    published = config.retilt_root / "9 July" / "cnj_45kg_Set1" / "Side"
    assert {path.name for path in published.iterdir()} == {
        "rgb.mp4",
        "depth",
        "confidence",
        "camera_matrix.csv",
        "odometry.csv",
        "imu.csv",
        "retilt_sidecar.json",
    }
    assert len(list((published / "depth").glob("*.png"))) == orient_record.n_frames
    assert len(list((published / "confidence").glob("*.png"))) == orient_record.n_frames


def test_retilt_sidecar_has_the_required_fields(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    import json

    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    retilt_camera.fn(orient_record, raw, config)

    sidecar = json.loads(
        (
            config.retilt_root / "9 July" / "cnj_45kg_Set1" / "Side" / "retilt_sidecar.json"
        ).read_text()
    )
    for field in (
        "tilt_deg",
        "roll_deg",
        "floor_normal_cam",
        "plane_rms_residual_m",
        "n_floor_points",
        "n_frames_sampled",
        "confidence_mode",
        "confidence_mode_per_frame",
        "homography_rgb",
        "homography_depth",
        "gravity_agreement_deg",
        "gravity_check_mode",
        "valid_bounds_px",
        "depth_values_recomputed",
        "k_rewritten",
        "region_normalized",
        "region_px",
        "sample_indices",
        "translation_span_m",
    ):
        assert field in sidecar, field
    assert sidecar["depth_values_recomputed"] is True
    assert sidecar["k_rewritten"] is False
    assert sidecar["gravity_check_mode"] == "warn"


def test_camera_matrix_is_unchanged(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    retilt_camera.fn(orient_record, raw, config)

    published = config.retilt_root / "9 July" / "cnj_45kg_Set1" / "Side"
    assert (published / "camera_matrix.csv").read_bytes() == (
        orient_record.source / "camera_matrix.csv"
    ).read_bytes()


# --- gravity check: warn-only, never rejects ------------------------------------------


def test_gravity_disagreement_warns_but_does_not_reject(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    _, step = retilt_camera.fn(orient_record, raw, config)

    assert any("gravity disagreement" in warning for warning in step.warnings)


def test_gravity_agreement_within_tolerance_does_not_warn(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10, retilt_gravity_tolerance_deg=179.0)
    orient_record = _build_orient_record(raw, config)

    _, step = retilt_camera.fn(orient_record, raw, config)

    assert not any("gravity disagreement" in warning for warning in step.warnings)


# --- rejection rules (§7) --------------------------------------------------------------


def test_missing_floor_region_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(tmp_path, make_camera, make_session_metadata, floor_regions=None)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    with pytest.raises(ScanRejected):
        retilt_camera.fn(orient_record, raw, config)


def test_unparseable_floor_region_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_regions={
            "side_floor_region_bottom_left_in_pixels": "not-a-point",
            "side_floor_region_top_right_in_pixels": "(1, 0)",
        },
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    with pytest.raises(ScanRejected):
        retilt_camera.fn(orient_record, raw, config)


def test_degenerate_floor_region_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_regions={
            "side_floor_region_bottom_left_in_pixels": "(0.5, 0.2)",
            "side_floor_region_top_right_in_pixels": "(0.5, 0.1)",
        },
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    with pytest.raises(ScanRejected):
        retilt_camera.fn(orient_record, raw, config)


def test_too_few_floor_points_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10_000)  # far more than exist
    orient_record = _build_orient_record(raw, config)

    with pytest.raises(ScanRejected, match="floor points"):
        retilt_camera.fn(orient_record, raw, config)


def _perturb_odometry_x(path: Path, row_index: int, x: float) -> None:
    """Rewrite one row's `x` column (index 2: timestamp, frame, x, y, z, ...)."""

    lines = path.read_text().splitlines()
    header, rows = lines[0], lines[1:]
    cells = rows[row_index].split(",")
    cells[2] = f" {x}"
    rows[row_index] = ",".join(cells)
    path.write_text("\n".join([header, *rows]) + "\n")


def test_translation_beyond_tolerance_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    # Sample every frame so translation_span_m has >= 2 points to compare.
    config = make_config(
        tmp_path,
        retilt_min_floor_points=10,
        retilt_sample_stride=1,
        retilt_max_translation_m=0.01,
    )
    orient_record = _build_orient_record(raw, config)
    _perturb_odometry_x(orient_record.source / "odometry.csv", orient_record.n_frames - 1, 5.0)

    with pytest.raises(ScanRejected, match="translated"):
        retilt_camera.fn(orient_record, raw, config)


def test_tilt_beyond_bound_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10, retilt_max_tilt_deg=1.0)
    orient_record = _build_orient_record(raw, config)

    with pytest.raises(ScanRejected, match="tilt"):
        retilt_camera.fn(orient_record, raw, config)


# --- a dry run touches nothing ----------------------------------------------------------


def test_a_dry_run_publishes_nothing(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    """Unlike orient, retilt's derived values need real pixel reads (the plane fit), so
    only the region-validation half of a dry run can run without S2's output existing.
    Exercise it against a *really published* S2 output -- the concern under test is
    retilt's own dry-run contract (writes nothing), not whether upstream stages ran."""

    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        floor_tilt_deg=8.0,
        floor_roll_deg=-2.0,
        floor_distance_m=1.8,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    dry_config = config.model_copy(update={"dry_run": True})
    retilt_record, step = retilt_camera.fn(orient_record, raw, dry_config)

    assert not config.retilt_root.exists()
    assert step.file_ops
    assert retilt_record.n_frames == orient_record.n_frames
