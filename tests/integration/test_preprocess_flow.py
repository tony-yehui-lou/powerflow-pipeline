"""The whole preprocess pipeline under a real Prefect backend: ingest, cut, then orient."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from prefect.client.orchestration import get_client
from typer.testing import CliRunner

from powerflow_pipeline.data.cli import app
from powerflow_pipeline.data.common.manifest import artifact_key
from powerflow_pipeline.data.common.models import HUMAN_SKELETON, JointId, JointSeries
from powerflow_pipeline.data.common.pose_storage import pose_path, skeleton_path
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.flow import preprocess
from powerflow_pipeline.data.preprocess.models import Intrinsics
from tests.conftest import MakeCamera


def build_capture(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any, make_session_metadata: Any
) -> Path:
    """Two sessions. cnj_45kg has both cameras and valid lift window; cnj_55kg is broken."""

    raw = tmp_path / "raw"
    make_meta_template(raw)

    # cnj_45kg_Set1: a healthy session with a Side lift window and a real, fittable floor.
    # A larger frame than the other fixtures' default: it must pool >= the default
    # `retilt_min_floor_points` (500) from a single sampled frame, and its default S4 crop
    # guard band (8px each side, sized for real ~1440px-wide frames) must stay well under
    # `crop_max_crop_fraction` (0.25) -- both for the CLI test below, which builds its config
    # straight from CLI flags with no tunable overrides. `rgb_size`/`depth_size` scale the
    # fixture's odometry intrinsics proportionally (conftest.py), so this stays a correctly
    # centred, constant-FOV "faithful miniature" at the larger size.
    floor = {
        "render_floor_plane": True,
        "floor_tilt_deg": 8.0,
        "floor_roll_deg": -2.0,
        "depth_size": (128, 96),  # landscape (W > H), like the real capture -- see DEPTH_SIZE
        "rgb_size": (128, 96),
    }
    make_camera(raw, session="cnj_45kg_Set1", camera="Front", rgb_frames=5, depth_frames=6, **floor)
    make_camera(raw, session="cnj_45kg_Set1", camera="Side", rgb_frames=5, depth_frames=6, **floor)
    make_session_metadata(
        raw,
        session="cnj_45kg_Set1",
        lift_start_ms=16,
        lift_end_ms=80,
        floor_regions="full_frame",
    )

    # cnj_55kg_Set1: one camera is missing its IMU, the other its depth.
    make_camera(raw, session="cnj_55kg_Set1", camera="Front", omit=["imu"])
    make_camera(raw, session="cnj_55kg_Set1", camera="Side", omit=["depth"])
    make_session_metadata(raw, session="cnj_55kg_Set1", lift_start_ms=16, lift_end_ms=80)

    return raw


def make_config(tmp_path: Path, raw: Path, **overrides: Any) -> PreprocessConfig:
    overrides.setdefault("retilt_min_floor_points", 10)  # the fixture's frames are tiny
    return PreprocessConfig(
        raw_root=raw,
        record_root=tmp_path / "s0_ingest_output",
        cut_root=tmp_path / "s1_cut_output",
        retilt_root=tmp_path / "s3_retilt_output",
        crop_root=tmp_path / "s4_crop_output",
        output_root=tmp_path / "s2_orient_output",
        **overrides,
    )


async def _artifact_keys() -> list[str | None]:
    async with get_client() as client:
        return [artifact.key for artifact in await client.read_artifacts()]


def test_the_run_cuts_and_rotates_valid_cameras(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert [scan.scan_id for scan in manifest.scans] == [
        "9 July/cnj_45kg_Set1/Front",
        "9 July/cnj_45kg_Set1/Side",
    ]
    assert all("cut" in scan.steps for scan in manifest.scans)
    assert all("orient" in scan.steps for scan in manifest.scans)
    assert all("retilt" in scan.steps for scan in manifest.scans)
    assert all("crop" in scan.steps for scan in manifest.scans)
    assert (config.retilt_root / "9 July" / "cnj_45kg_Set1" / "Front" / "rgb.mp4").is_file()
    assert (config.crop_root / "9 July" / "cnj_45kg_Set1" / "Front" / "rgb.mp4").is_file()
    rejections = {rejected.scan_id: rejected.reason for rejected in manifest.rejected_scans}
    assert rejections["9 July/cnj_55kg_Set1/Front"] == "missing required stream: imu.csv"
    assert rejections["9 July/cnj_55kg_Set1/Side"] == "missing required stream: depth"
    # cnj_55kg has both cameras rejected at ingest; the session-level rollup fires from
    # _sessions_without_a_camera since no camera survived to trigger _session_cut_interval.
    assert rejections["9 July/cnj_55kg_Set1"] == "session has no usable camera"


def test_task_logs_identify_their_source_and_output_paths(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)
    source = raw / "9 July" / "cnj_45kg_Set1" / "Front"
    cut_destination = config.cut_root / "9 July" / "cnj_45kg_Set1" / "Front"
    output_destination = config.output_root / "9 July" / "cnj_45kg_Set1" / "Front"

    caplog.set_level(logging.INFO)
    preprocess(config)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    record_path = config.record_root / source.relative_to(raw) / "record.json"
    assert f"source={source} output={record_path}" in messages
    assert f"source={source} output={cut_destination}" in messages
    assert f"source={cut_destination} output={output_destination}" in messages


def test_the_manifest_artifact_records_cut_and_rotation(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert artifact_key("preprocess") in asyncio.run(_artifact_keys())

    markdown = manifest.to_markdown()
    assert "cut_start_epoch_ms" in markdown
    assert "cut_end_epoch_ms" in markdown
    assert "k_rewritten=True" in markdown
    assert "session has no usable camera" in markdown

    written = json.loads((config.record_root / "manifest.json").read_text())
    assert written["scans"][0]["derived"]["n_frames"] > 0


def test_a_surviving_session_is_published_whole(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)

    preprocess(config)

    # S2 orient output: the final portrait streams.
    session = config.output_root / "9 July" / "cnj_45kg_Set1"
    for camera in ("Front", "Side"):
        published = {path.name for path in (session / camera).iterdir()}
        assert published == {
            "rgb.mp4",
            "depth",
            "confidence",
            "camera_matrix.csv",
            "camera_matrix_source.csv",
            "odometry.csv",
            "imu.csv",
            "sidecar.json",
        }

    # S1 cut output: trimmed landscape streams + cut_sidecar.
    cut_session = config.cut_root / "9 July" / "cnj_45kg_Set1"
    for camera in ("Front", "Side"):
        published = {path.name for path in (cut_session / camera).iterdir()}
        assert "cut_sidecar.json" in published
        assert "rgb.mp4" in published

    # Every stage's output carries the session metadata.
    for root in config.stage_roots:
        metadata = yaml.safe_load((root / "9 July" / "cnj_45kg_Set1" / "metadata.yaml").read_text())
        assert [camera["role"] for camera in metadata["cameras"]] == ["front", "side"]
        assert metadata["derived_from_session_name"]["weight_in_kg"] == 45.0
        # Cut epoch bounds are recorded.
        assert "cut_start_epoch_ms" in metadata["lift"]
        assert "cut_end_epoch_ms" in metadata["lift"]

    # No staging survives a completed run.
    assert not list(config.output_root.rglob(".powerflow-*"))


def test_a_dry_run_touches_nothing(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw, dry_run=True)

    manifest = preprocess(config)

    assert [scan.status for scan in manifest.scans] == ["planned", "planned"]
    assert manifest.scans[0].file_ops  # the plan is still reported
    assert not config.output_root.exists()
    assert not config.record_root.exists()
    assert not config.cut_root.exists()


def test_the_cli_runs_the_flow(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)

    s0 = tmp_path / "s0"
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    s3 = tmp_path / "s3"
    result = CliRunner().invoke(
        app,
        [
            "preprocess",
            "--input",
            str(raw),
            "--records",
            str(s0),
            "--cut",
            str(s1),
            "--retilt",
            str(s3),
            "--crop",
            str(tmp_path / "s4"),
            "--output",
            str(s2),
            "--rotation",
            "cw",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "processed 2 camera(s), rejected 3" in result.output
    assert (s2 / "9 July" / "cnj_45kg_Set1" / "Front" / "sidecar.json").is_file()
    assert (tmp_path / "s4" / "9 July" / "cnj_45kg_Set1" / "Front" / "crop_sidecar.json").is_file()
    assert (s1 / "9 July" / "cnj_45kg_Set1" / "Front" / "cut_sidecar.json").is_file()


# --- S5 pose (issue #116): opt-in, only when a `PoseModel` is supplied --------------------


class _StubPoseModel:
    """A fixed, valid pose for every frame -- stands in for the real model (issue #118)."""

    def predict(
        self,
        rgb_path: Path,
        n_frames: int,
        *,
        depth_dir: Path,
        confidence_dir: Path,
        intrinsics: Intrinsics,
        rgb_size: tuple[int, int],
        depth_size: tuple[int, int],
        floor_offset_m: float,
    ) -> dict[JointId, JointSeries]:
        series = JointSeries(
            position=tuple((0.0, 0.0, 0.0) for _ in range(n_frames)),
            pixel_position=tuple((0, 0) for _ in range(n_frames)),
            confidence=tuple(0.9 for _ in range(n_frames)),
        )
        return {joint: series for joint in HUMAN_SKELETON.joints}


def test_the_run_detects_pose_when_a_model_is_given(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw, pose_root=tmp_path / "s5_pose_output")

    manifest = preprocess(config, pose_model=_StubPoseModel())

    assert all("pose" in scan.steps for scan in manifest.scans)
    assert skeleton_path(config.pose_root).is_file()
    for scan in manifest.scans:
        date, session, camera = scan.scan_id.split("/", 2)
        assert pose_path(config.pose_root, Path(date) / session, camera).is_file()


def test_the_run_skips_pose_without_a_model(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert all("pose" not in scan.steps for scan in manifest.scans)


def test_preprocess_rejects_a_pose_model_without_a_pose_root(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)

    with pytest.raises(ValueError, match="pose_root"):
        preprocess(config, pose_model=_StubPoseModel())


# --- the single-camera layout, and a mixed raw root ---------------------------------------


def build_single_camera_capture(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_session_metadata: Any,
    *,
    raw: Path | None = None,
    date: str = "22 August",
    group: str = "Snch",
    trial: str = "110kgSnch1",
) -> Path:
    """One `<date>/<liftType>/<trial>/` capture, with its metadata inside the capture."""

    raw = tmp_path / "raw" if raw is None else raw
    floor = {
        "render_floor_plane": True,
        "floor_tilt_deg": 8.0,
        "floor_roll_deg": -2.0,
        "depth_size": (128, 96),
        "rgb_size": (128, 96),
    }
    make_camera(raw, date=date, session=group, camera=trial, rgb_frames=5, depth_frames=6, **floor)
    make_session_metadata(
        raw,
        date=date,
        session=group,
        camera=trial,
        lift_start_ms=16,
        lift_end_ms=80,
        floor_regions="full_frame_single",
    )
    return raw


def test_a_single_camera_capture_publishes_at_the_trial_level(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """Output mirrors the raw path exactly: no camera directory, because there is none."""

    raw = build_single_camera_capture(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert [scan.scan_id for scan in manifest.scans] == ["22 August/Snch/110kgSnch1"]
    assert manifest.rejected_scans == []
    trial = Path("22 August") / "Snch" / "110kgSnch1"
    assert (config.cut_root / trial / "cut_sidecar.json").is_file()
    assert (config.output_root / trial / "sidecar.json").is_file()
    assert (config.retilt_root / trial / "retilt_sidecar.json").is_file()
    assert (config.crop_root / trial / "rgb.mp4").is_file()
    # No camera level was invented anywhere.
    assert not (config.crop_root / trial / "Side").exists()


def test_a_single_camera_capture_is_cut_to_its_own_lift_window(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """It owns its interval: there is no Side camera to take the window from."""

    raw = build_single_camera_capture(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, raw)

    preprocess(config)

    sidecar = json.loads(
        (config.cut_root / "22 August" / "Snch" / "110kgSnch1" / "cut_sidecar.json").read_text()
    )
    assert sidecar["lift_start_time_side_in_ms"] == 16
    assert sidecar["lift_end_time_side_in_ms"] == 80
    assert sidecar["cut_end_epoch_ms"] > sidecar["cut_start_epoch_ms"]


def test_single_camera_metadata_lands_beside_the_streams_in_every_stage_root(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    raw = build_single_camera_capture(tmp_path, make_camera, make_session_metadata)
    config = make_config(tmp_path, raw)

    preprocess(config)

    trial = Path("22 August") / "Snch" / "110kgSnch1"
    for root in config.stage_roots:
        metadata = yaml.safe_load((root / trial / "metadata.yaml").read_text())
        assert metadata["capture_id"] == "22 August/Snch/110kgSnch1"
        assert metadata["layout"] == "single_camera"
        assert [camera["role"] for camera in metadata["cameras"]] == ["single"]


def test_one_run_processes_both_layouts(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    """The whole point of detecting the layout rather than flagging it."""

    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    build_single_camera_capture(tmp_path, make_camera, make_session_metadata, raw=raw)
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert sorted(scan.scan_id for scan in manifest.scans) == [
        "22 August/Snch/110kgSnch1",
        "9 July/cnj_45kg_Set1/Front",
        "9 July/cnj_45kg_Set1/Side",
    ]
    assert (config.crop_root / "9 July" / "cnj_45kg_Set1" / "Front" / "rgb.mp4").is_file()
    assert (config.crop_root / "22 August" / "Snch" / "110kgSnch1" / "rgb.mp4").is_file()


def test_only_restricts_the_run_to_matching_captures(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    """A full capture day is hours of re-encoding; annotation triage re-runs one at a time."""

    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    build_single_camera_capture(tmp_path, make_camera, make_session_metadata, raw=raw)
    config = make_config(tmp_path, raw, only="22 August/*")

    manifest = preprocess(config)

    assert [scan.scan_id for scan in manifest.scans] == ["22 August/Snch/110kgSnch1"]
    assert manifest.rejected_scans == []  # July was not run, not rejected


def test_a_capture_day_whose_planes_disagree_warns_without_rejecting(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """One tripod shot the whole day, so a fit that disagrees with the day is suspect.

    Warn-only: the fit passed every per-capture gate, and the check assumes a rig that
    never moved -- an assumption worth stating in the manifest, not enforcing blindly.
    """

    raw = tmp_path / "raw"
    for trial, tilt in (("70kgSnch1", 8.0), ("75kgSnch1", 8.0), ("80kgSnch1", 14.0)):
        floor = {
            "render_floor_plane": True,
            "floor_tilt_deg": tilt,
            "floor_roll_deg": -2.0,
            "depth_size": (128, 96),
            "rgb_size": (128, 96),
        }
        make_camera(
            raw,
            date="22 August",
            session="Snch",
            camera=trial,
            rgb_frames=5,
            depth_frames=6,
            **floor,
        )
        make_session_metadata(
            raw,
            date="22 August",
            session="Snch",
            camera=trial,
            lift_start_ms=16,
            lift_end_ms=80,
            floor_regions="full_frame_single",
        )
    # The odd capture's larger tilt costs a real fraction of a 96px-tall fixture frame that
    # it would not cost a 1920px one; raise the gate so S4 does not reject what S3's
    # day-level check is here to warn about.
    config = make_config(tmp_path, raw, crop_max_crop_fraction=0.5)

    manifest = preprocess(config)

    assert manifest.rejected_scans == []  # never a rejection
    warnings = {scan.scan_id: scan.warnings for scan in manifest.scans}
    odd_one_out = [w for w in warnings["22 August/Snch/80kgSnch1"] if "capture day" in w]
    assert odd_one_out, warnings["22 August/Snch/80kgSnch1"]
    assert "retilt_group_tilt_tolerance_deg" in odd_one_out[0]
    for agreeing in ("22 August/Snch/70kgSnch1", "22 August/Snch/75kgSnch1"):
        assert not [w for w in warnings[agreeing] if "capture day" in w]
    # The day's reference values are recorded on every capture that took part.
    assert "group_tilt_median_deg" in manifest.scans[0].derived


def test_a_two_camera_session_is_not_checked_for_day_level_agreement(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    """Four cameras of a July day are four different poses; disagreeing is their job."""

    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert all("group_tilt_median_deg" not in scan.derived for scan in manifest.scans)


def test_a_rejected_single_camera_capture_is_reported_once(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """A single-camera group is its capture, so a group-level rollup would double-report it."""

    raw = build_single_camera_capture(tmp_path, make_camera, make_session_metadata)
    # Strip the floor region: S3 rejects, and nothing publishes.
    metadata = raw / "22 August" / "Snch" / "110kgSnch1" / "metadata.yaml"
    metadata.write_text(metadata.read_text().split("video:")[0])
    config = make_config(tmp_path, raw)

    manifest = preprocess(config)

    assert manifest.scans == []
    (rejected,) = manifest.rejected_scans
    assert rejected.scan_id == "22 August/Snch/110kgSnch1"
    assert "no video block" in rejected.reason
