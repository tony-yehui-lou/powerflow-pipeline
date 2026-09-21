"""`meta.yaml` is a template of type names. S0 instantiates it without inventing values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from powerflow_pipeline.data.preprocess.tasks.ingest import ingest_camera
from powerflow_pipeline.data.preprocess.tasks.metadata import (
    derive_from_session_name,
    is_template,
    lift_type_from_group_folder,
    load_meta_template,
    write_group_metadata,
)
from tests.conftest import MakeCamera, sole_capture
from tests.unit.test_preprocess_ingest import make_config


def test_the_shipped_meta_yaml_is_recognised_as_a_template(
    tmp_path: Path, make_meta_template: Any
) -> None:
    """Every leaf is a type name (`weight_in_kg: float`), so there is nothing to check against."""

    path = make_meta_template(tmp_path / "raw")

    template = load_meta_template(path)

    assert is_template(template) is True


def test_a_filled_meta_yaml_is_not_a_template(tmp_path: Path, make_meta_template: Any) -> None:
    body = "lift:\n  weight_in_kg: 45.0\n  type: cnj\nathlete:\n  name: Ada\n"
    path = make_meta_template(tmp_path / "raw", body=body)

    assert is_template(load_meta_template(path)) is False


def test_a_missing_meta_yaml_is_not_an_error(tmp_path: Path) -> None:
    assert load_meta_template(tmp_path / "absent.yaml") is None


@pytest.mark.parametrize(
    ("session", "expected"),
    [
        ("cnj_45kg_Set1", {"type": "cnj", "weight_in_kg": 45.0, "set": 1}),
        ("cnj_55kg_Set1", {"type": "cnj", "weight_in_kg": 55.0, "set": 1}),
        ("snch_62.5kg_Set3", {"type": "snch", "weight_in_kg": 62.5, "set": 3}),
        ("freeform", {}),
    ],
)
def test_session_names_are_parsed_but_marked_derived(
    session: str, expected: dict[str, object]
) -> None:
    assert derive_from_session_name(session) == expected


def test_metadata_records_derived_values_separately_from_measured_ones(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    """The session name is evidence about the lift, not a measurement of it."""

    raw = tmp_path / "raw"
    make_camera(raw)
    make_meta_template(raw)
    config = make_config(tmp_path)
    camera = sole_capture(raw)
    record = ingest_camera.fn(camera, config)

    paths = write_group_metadata.fn(raw / "9 July" / "meta.yaml", [record], config)

    assert paths[0] == config.record_root / "9 July" / "cnj_45kg_Set1" / "metadata.yaml"
    metadata = yaml.safe_load(paths[0].read_text())

    assert metadata["meta_status"] == "template"
    assert metadata["derived_from_session_name"] == {
        "type": "cnj",
        "weight_in_kg": 45.0,
        "set": 1,
    }
    # Never merged into `lift` as if measured.
    assert metadata["lift"]["type"] is None
    assert metadata["lift"]["weight_in_kg"] is None
    assert metadata["lift"]["result"] is None
    # Nothing in the capture supplies these.
    assert all(value is None for value in metadata["athlete"].values())


def test_the_lift_timestamp_comes_from_the_video(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw)
    make_meta_template(raw)
    config = make_config(tmp_path)
    camera = sole_capture(raw)
    record = ingest_camera.fn(camera, config)

    paths = write_group_metadata.fn(raw / "9 July" / "meta.yaml", [record], config)
    metadata = yaml.safe_load(paths[0].read_text())

    # 2026-07-09T10:04:15Z, the creation_time tag on rgb.mp4.
    assert metadata["lift"]["dateTime_epoch"] == 1783591455


def test_every_stage_output_carries_the_same_metadata(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    """A stage's output is unreadable without the session it came from, so each root gets a copy."""

    raw = tmp_path / "raw"
    make_camera(raw)
    make_meta_template(raw)
    config = make_config(tmp_path)
    camera = sole_capture(raw)
    record = ingest_camera.fn(camera, config)

    paths = write_group_metadata.fn(raw / "9 July" / "meta.yaml", [record], config)

    assert [path.parent.parent.parent for path in paths] == list(config.stage_roots)
    assert all(path.name == "metadata.yaml" for path in paths)
    bodies = {path.read_text() for path in paths}
    assert len(bodies) == 1  # byte-identical, so no stage can drift from another


def test_the_observed_camera_facts_are_recorded(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw)
    make_meta_template(raw)
    config = make_config(tmp_path)
    camera = sole_capture(raw)
    record = ingest_camera.fn(camera, config)

    paths = write_group_metadata.fn(raw / "9 July" / "meta.yaml", [record], config)
    cameras = yaml.safe_load(paths[0].read_text())["cameras"]

    assert cameras[0]["capture_id"] == "9 July/cnj_45kg_Set1/Front"
    assert cameras[0]["role"] == "front"
    assert cameras[0]["rgb_width"] == 64
    assert cameras[0]["n_frames"] == 4
    assert cameras[0]["counts"]["depth"] == 5
    assert cameras[0]["intrinsics"]["fx"] == pytest.approx(44.8)  # mean of 44.0/46.0 over 5 rows


def test_a_dry_run_writes_no_metadata(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw)
    make_meta_template(raw)
    config = make_config(tmp_path, dry_run=True)
    camera = sole_capture(raw)
    record = ingest_camera.fn(camera, config)

    paths = write_group_metadata.fn(raw / "9 July" / "meta.yaml", [record], config)

    assert not any(path.exists() for path in paths)


def test_the_raw_meta_yaml_is_never_mutated(
    tmp_path: Path, make_camera: MakeCamera, make_meta_template: Any
) -> None:
    """Raw data is write-once: the instantiated file lands in the S0 output tree."""

    raw = tmp_path / "raw"
    make_camera(raw)
    template_path = make_meta_template(raw)
    before = template_path.read_bytes()
    config = make_config(tmp_path)
    camera = sole_capture(raw)
    record = ingest_camera.fn(camera, config)

    write_group_metadata.fn(template_path, [record], config)

    assert template_path.read_bytes() == before


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("110kgSnch1", {"type": "snch", "weight_in_kg": 110.0, "attempt": 1}),
        ("82kgCnJ2", {"type": "cnj", "weight_in_kg": 82.0, "attempt": 2}),
        ("62.5kgSnch3", {"type": "snch", "weight_in_kg": 62.5, "attempt": 3}),
    ],
)
def test_single_camera_trial_names_are_parsed_by_attempt(
    name: str, expected: dict[str, object]
) -> None:
    """The trial name reads weight-first with no separators, and numbers attempts, not sets."""

    assert derive_from_session_name(name) == expected


def test_single_camera_metadata_lands_beside_its_own_streams(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """August mirrors the raw path: the trial's own directory, not a session one level up."""

    raw = tmp_path / "raw"
    make_camera(raw, date="22 August", session="Snch", camera="110kgSnch1")
    make_session_metadata(raw, date="22 August", session="Snch", camera="110kgSnch1")
    config = make_config(tmp_path)
    record = ingest_camera.fn(sole_capture(raw), config)

    paths = write_group_metadata.fn(raw / "22 August" / "meta.yaml", [record], config)

    assert paths[0] == (config.record_root / "22 August" / "Snch" / "110kgSnch1" / "metadata.yaml")
    metadata = yaml.safe_load(paths[0].read_text())
    assert metadata["meta_status"] == "absent"  # no date-level template for August
    assert metadata["capture_id"] == "22 August/Snch/110kgSnch1"
    assert metadata["layout"] == "single_camera"
    assert metadata["derived_from_session_name"]["attempt"] == 1


def test_the_lift_type_comes_from_the_group_folder_not_the_file(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """Every August file once read `type: snch`, including the five filed under `CnJ/`.

    The folder is authoritative; the file's own value is carried through beside it so
    nothing is silently corrected.
    """

    raw = tmp_path / "raw"
    make_camera(raw, date="22 August", session="CnJ", camera="82kgCnJ2")
    metadata_path = make_session_metadata(raw, date="22 August", session="CnJ", camera="82kgCnJ2")
    metadata_path.write_text(metadata_path.read_text().replace("lift:", "lift:\n  type: snch", 1))
    config = make_config(tmp_path)
    record = ingest_camera.fn(sole_capture(raw), config)

    paths = write_group_metadata.fn(raw / "22 August" / "meta.yaml", [record], config)
    metadata = yaml.safe_load(paths[0].read_text())

    assert metadata["derived_from_session_name"]["type"] == "cnj"
    assert metadata["derived_from_session_name"]["type_source"] == "group folder"
    assert metadata["lift_type_in_source_file"] == "snch"


def test_a_two_camera_session_has_no_group_folder_lift_type(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """Only the single-camera layout files trials under a lift-type folder."""

    raw = tmp_path / "raw"
    make_camera(raw)
    make_session_metadata(raw)
    config = make_config(tmp_path)
    record = ingest_camera.fn(sole_capture(raw), config)

    assert lift_type_from_group_folder(record) is None
