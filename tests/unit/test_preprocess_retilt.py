"""S3 establishes I3: streams rectified level with the fitted floor plane.

The floor is found without an operator annotation: gravity seeds a height histogram over
every confident depth pixel, and depth refines the plane (2026-10-21-retilt-gravity-floor-
design.md).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from powerflow_pipeline.data.common.errors import ScanRejected
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import CameraRecord
from powerflow_pipeline.data.preprocess.tasks.cut import cut_camera, resolve_cut_interval
from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from powerflow_pipeline.data.preprocess.tasks.orient import orient_camera
from powerflow_pipeline.data.preprocess.tasks.retilt import retilt_camera
from tests.conftest import MakeCamera, MakeSessionMetadata, sole_capture
from tests.unit.test_preprocess_ingest import make_config as _base_make_config


def make_config(tmp_path: Path, **overrides: object) -> PreprocessConfig:
    return _base_make_config(tmp_path, **overrides)


def _build_orient_record(
    raw_root: Path, config: PreprocessConfig, camera: str = "Side"
) -> CameraRecord:
    """Run S0 -> S1 -> S2 on the one synthetic camera named `camera`; return S2's record."""

    camera_dir = sole_capture(raw_root, camera)
    ingested = ingest_camera.fn(camera_dir, config)
    interval = resolve_cut_interval.fn(camera_dir.metadata_path, ingested)
    cut_record, _ = cut_camera.fn(ingested, interval, config)
    orient_record, _ = orient_camera.fn(cut_record, config)
    return orient_record


def _make_floor_session(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_session_metadata: MakeSessionMetadata,
    *,
    floor_regions: object = None,
    **camera_overrides: object,
) -> Path:
    raw = tmp_path / "raw"
    make_camera(
        raw,
        camera="Side",
        rgb_frames=5,
        depth_frames=6,
        render_floor_plane=True,
        # ~1000 confident points per frame. The default 16x12 leaves ~60, too few for a
        # floor smeared by a biased gravity seed to keep a peak; a real frame has ~40 000.
        depth_size=(64, 48),
        **camera_overrides,
    )
    make_session_metadata(
        raw,
        lift_start_ms=16,
        lift_end_ms=80,
        floor_regions=floor_regions,  # type: ignore[arg-type]
    )
    return raw


def _sidecar(config: PreprocessConfig) -> dict[str, object]:
    path = config.retilt_root / "9 July" / "cnj_45kg_Set1" / "Side" / "retilt_sidecar.json"
    loaded: dict[str, object] = json.loads(path.read_text())
    return loaded


def _rewrite_frames(
    directory: Path, edit: Callable[[int, np.ndarray], np.ndarray], *, start: int = 0
) -> None:
    """Rewrite S2's published PNG frames in place, from frame `start` on."""

    for index, path in enumerate(sorted(directory.glob("*.png"))):
        if index < start:
            continue
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(path), edit(index, frame))


_TILTED = {"floor_tilt_deg": 8.0, "floor_roll_deg": -2.0, "floor_distance_m": 1.8}


# --- the happy path ------------------------------------------------------------------


def test_retilt_camera_recovers_baked_in_tilt(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        **_TILTED,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    retilt_record, step = retilt_camera.fn(orient_record, config)

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
        **_TILTED,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    retilt_camera.fn(orient_record, config)

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
    raw = _make_floor_session(tmp_path, make_camera, make_session_metadata, **_TILTED)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    retilt_camera.fn(orient_record, config)

    sidecar = _sidecar(config)
    for field in (
        "tilt_deg",
        "roll_deg",
        "floor_normal_cam",
        "floor_offset_m",
        "plane_rms_residual_m",
        "n_floor_points",
        "n_frames_sampled",
        "confidence_mode",
        "homography_rgb",
        "homography_depth",
        "gravity_agreement_deg",
        "gravity_check_mode",
        "valid_bounds_px",
        "depth_values_recomputed",
        "k_rewritten",
        "sample_indices",
        "translation_span_m",
        "floor_source",
        "gravity_up_cam",
        "floor_seed_height_m",
        "floor_peaks",
        "floor_inlier_fraction",
        "floor_refine_iterations",
    ):
        assert field in sidecar, field
    for retired in ("region_normalized", "region_px", "confidence_mode_per_frame"):
        assert retired not in sidecar, retired
    assert sidecar["depth_values_recomputed"] is True
    assert sidecar["k_rewritten"] is False
    assert sidecar["gravity_check_mode"] == "reject"
    assert sidecar["floor_source"] == "gravity_seeded"
    assert sidecar["confidence_mode"] == "conf2_only"


def test_camera_matrix_is_unchanged(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        **_TILTED,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    retilt_camera.fn(orient_record, config)

    published = config.retilt_root / "9 July" / "cnj_45kg_Set1" / "Side"
    assert (published / "camera_matrix.csv").read_bytes() == (
        orient_record.source / "camera_matrix.csv"
    ).read_bytes()


# --- no annotation needed --------------------------------------------------------------


def test_a_session_with_no_floor_region_is_retilted(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path, make_camera, make_session_metadata, floor_regions=None, **_TILTED
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    _, step = retilt_camera.fn(orient_record, config)

    assert step.derived["tilt_deg"] == pytest.approx(8.0, abs=0.5)
    assert step.derived["floor_source"] == "gravity_seeded"


def test_floor_region_keys_are_ignored(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    """A rectangle pointing nowhere near the floor changes nothing: it is never read."""

    results = []
    for name, regions in (
        ("none", None),
        (
            "bogus",
            {
                "side_floor_region_bottom_left_in_pixels": "(0.0, 0.1)",
                "side_floor_region_top_right_in_pixels": "(0.1, 0.0)",
            },
        ),
    ):
        root = tmp_path / name
        raw = _make_floor_session(
            root, make_camera, make_session_metadata, floor_regions=regions, **_TILTED
        )
        config = make_config(root, retilt_min_floor_points=10)
        _, step = retilt_camera.fn(_build_orient_record(raw, config), config)
        results.append(step.derived)

    assert results[0] == results[1]


@pytest.mark.parametrize("gravity_error_deg", [-3.0, 3.0])
def test_a_biased_gravity_seed_still_recovers_the_floor(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_session_metadata: MakeSessionMetadata,
    gravity_error_deg: float,
) -> None:
    """Real gravity is off by up to ~4 deg per capture; depth refinement removes it."""

    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        gravity_error_deg=gravity_error_deg,
        **_TILTED,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    _, step = retilt_camera.fn(orient_record, config)

    assert step.derived["tilt_deg"] == pytest.approx(8.0, abs=0.5)
    assert step.derived["roll_deg"] == pytest.approx(-2.0, abs=0.5)
    assert step.derived["gravity_agreement_deg"] == pytest.approx(abs(gravity_error_deg), abs=0.5)


def test_an_occluder_in_later_frames_does_not_move_the_fit(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    """The motivating case of the 2026-10-18 single-frame spec, now passing pooled: a
    body-like surface 0.8 m from the lens in every frame after the first."""

    raw = _make_floor_session(tmp_path, make_camera, make_session_metadata, **_TILTED)
    config = make_config(tmp_path, retilt_min_floor_points=10, retilt_sample_stride=1)
    orient_record = _build_orient_record(raw, config)

    def occlude(_: int, depth: np.ndarray) -> np.ndarray:
        out = depth.copy()
        out[:, : depth.shape[1] // 3] = 800
        return out

    _rewrite_frames(orient_record.source / "depth", occlude, start=1)

    _, step = retilt_camera.fn(orient_record, config)

    assert step.derived["tilt_deg"] == pytest.approx(8.0, abs=0.5)
    assert step.derived["roll_deg"] == pytest.approx(-2.0, abs=0.5)


def test_a_platform_above_the_floor_warns_and_keeps_the_lower_surface(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(tmp_path, make_camera, make_session_metadata, **_TILTED)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)
    distance = _TILTED["floor_distance_m"]

    def raise_nearer_half(_: int, depth: np.ndarray) -> np.ndarray:
        # A parallel plane 10 cm nearer the camera: every ray meets it at (D - 0.1) / D of
        # the distance it met the floor.
        # Only the bottom of the frame sees floor at all, so raise the nearer half of it.
        out = depth.astype(np.float64)
        floor_rows = np.nonzero((depth > 0).any(axis=1))[0]
        out[floor_rows[len(floor_rows) // 2] :] *= (distance - 0.10) / distance
        return out.round().astype(np.uint16)

    _rewrite_frames(orient_record.source / "depth", raise_nearer_half)

    _, step = retilt_camera.fn(orient_record, config)

    assert step.derived["floor_offset_m"] == pytest.approx(distance, abs=0.01)
    assert any("floor ambiguous" in warning for warning in step.warnings)
    assert len(_sidecar(config)["floor_peaks"]) == 2  # type: ignore[arg-type]


def test_too_little_confidence_2_falls_back_to_confidence_1(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(tmp_path, make_camera, make_session_metadata, **_TILTED)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)
    _rewrite_frames(
        orient_record.source / "confidence",
        lambda _, confidence: np.where(confidence == 2, 1, confidence).astype(np.uint8),
    )

    _, step = retilt_camera.fn(orient_record, config)

    assert step.derived["tilt_deg"] == pytest.approx(8.0, abs=0.5)
    assert _sidecar(config)["confidence_mode"] == "conf1_and_2"


# --- rejection rules ------------------------------------------------------------------------


def test_gravity_disagreement_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    """With nobody choosing the floor by eye, this is the gate against a clean fit to a
    sloped surface -- no longer warn-only."""

    raw = _make_floor_session(
        tmp_path, make_camera, make_session_metadata, gravity_error_deg=3.0, **_TILTED
    )
    config = make_config(tmp_path, retilt_min_floor_points=10, retilt_gravity_tolerance_deg=1.0)
    orient_record = _build_orient_record(raw, config)

    with pytest.raises(ScanRejected, match="gravity disagreement"):
        retilt_camera.fn(orient_record, config)


def test_a_frame_with_no_horizontal_surface_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    """A wall square to the lens: heights spread evenly, no peak, no floor."""

    raw = _make_floor_session(tmp_path, make_camera, make_session_metadata, **_TILTED)
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)
    _rewrite_frames(orient_record.source / "depth", lambda _, depth: np.full_like(depth, 2000))
    _rewrite_frames(orient_record.source / "confidence", lambda _, conf: np.full_like(conf, 2))

    with pytest.raises(ScanRejected, match="floor not visible"):
        retilt_camera.fn(orient_record, config)


def test_too_few_floor_points_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        **_TILTED,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10_000)  # far more than exist
    orient_record = _build_orient_record(raw, config)

    with pytest.raises(ScanRejected, match="retilt_min_floor_points"):
        retilt_camera.fn(orient_record, config)


def test_too_few_points_on_the_floor_itself_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    """Plenty of confident depth, but only half of it is the floor -- the gate counts the
    plane's own inliers, not the pooled cloud."""

    raw = _make_floor_session(tmp_path, make_camera, make_session_metadata, **_TILTED)
    config = make_config(tmp_path, retilt_min_floor_points=700)
    orient_record = _build_orient_record(raw, config)
    distance = _TILTED["floor_distance_m"]

    def raise_nearer_half(_: int, depth: np.ndarray) -> np.ndarray:
        out = depth.astype(np.float64)
        floor_rows = np.nonzero((depth > 0).any(axis=1))[0]
        out[floor_rows[len(floor_rows) // 2] :] *= (distance - 0.10) / distance
        return out.round().astype(np.uint16)

    _rewrite_frames(orient_record.source / "depth", raise_nearer_half)

    with pytest.raises(ScanRejected, match="floor points fitted"):
        retilt_camera.fn(orient_record, config)


def test_a_plane_that_is_not_flat_enough_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    """The selected surface is not actually planar -- probably not the floor (§6)."""

    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        **_TILTED,
    )
    config = make_config(
        tmp_path,
        retilt_min_floor_points=10,  # the fixture's frames are tiny
        retilt_max_plane_rms_m=1e-9,  # nothing real is this flat
    )
    orient_record = _build_orient_record(raw, config)

    with pytest.raises(ScanRejected, match="plane fit RMS"):
        retilt_camera.fn(orient_record, config)


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
        **_TILTED,
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
        retilt_camera.fn(orient_record, config)


def test_tilt_beyond_bound_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        **_TILTED,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10, retilt_max_tilt_deg=1.0)
    orient_record = _build_orient_record(raw, config)

    with pytest.raises(ScanRejected, match="tilt"):
        retilt_camera.fn(orient_record, config)


# --- a dry run touches nothing ----------------------------------------------------------


def test_a_dry_run_publishes_nothing(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: MakeSessionMetadata
) -> None:
    """Retilt's derived values all need pixel reads (the floor search), so a dry run plans
    file operations only. Exercised against a *really published* S2 output -- the concern
    under test is retilt's own dry-run contract (writes nothing), not upstream stages."""

    raw = _make_floor_session(
        tmp_path,
        make_camera,
        make_session_metadata,
        **_TILTED,
    )
    config = make_config(tmp_path, retilt_min_floor_points=10)
    orient_record = _build_orient_record(raw, config)

    dry_config = config.model_copy(update={"dry_run": True})
    retilt_record, step = retilt_camera.fn(orient_record, dry_config)

    assert not config.retilt_root.exists()
    assert step.file_ops
    assert step.derived == {}
    assert retilt_record.n_frames == orient_record.n_frames
