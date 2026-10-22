"""detect and analyze rebuild S4's camera record from what S0-S4 published, not from memory."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from powerflow_pipeline.data.common.camera_records import (
    discover_published_cameras,
    read_published_record,
)
from powerflow_pipeline.data.common.capture_models import CameraRecord
from powerflow_pipeline.data.preprocess import flow as preprocess_module
from powerflow_pipeline.data.preprocess.flow import preprocess
from tests.conftest import MakeCamera
from tests.integration.test_preprocess_flow import (
    build_capture,
    build_single_camera_capture,
    make_config,
)


def _run_capturing_crop_records(
    monkeypatch: pytest.MonkeyPatch, config: Any
) -> dict[str, CameraRecord]:
    """Run preprocess and keep every record S4 returned in memory: the reference."""

    returned: dict[str, CameraRecord] = {}
    original = preprocess_module.crop_camera

    def recording(record: CameraRecord, config: Any) -> Any:
        crop_record, step = original(record, config)
        returned[crop_record.capture_id] = crop_record
        return crop_record, step

    monkeypatch.setattr(preprocess_module, "crop_camera", recording)
    preprocess(config)
    return returned


def test_a_published_record_equals_the_one_s4_returned(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    build_single_camera_capture(tmp_path, make_camera, make_session_metadata, raw=raw)
    config = make_config(tmp_path, raw)

    in_memory = _run_capturing_crop_records(monkeypatch, config)

    assert set(in_memory) == {
        "9 July/cnj_45kg_Set1/Front",
        "9 July/cnj_45kg_Set1/Side",
        "22 August/Snch/110kgSnch1",
    }
    for capture_id, expected in in_memory.items():
        rebuilt = read_published_record(
            Path(capture_id),
            record_root=config.record_root,
            cut_root=config.cut_root,
            crop_root=config.crop_root,
        )
        assert rebuilt == expected, capture_id


def test_discovery_finds_every_camera_s4_published(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)
    in_memory = _run_capturing_crop_records(monkeypatch, config)

    found = discover_published_cameras.fn(
        config.record_root, config.cut_root, config.crop_root, only=None
    )

    # cnj_55kg was rejected at S0, so S4 published nothing for it: it is not discovered.
    assert [record.capture_id for record in found.records] == sorted(in_memory)
    assert found.records == [in_memory[capture_id] for capture_id in sorted(in_memory)]
    assert found.rejected == []


def test_discovery_honours_the_capture_glob(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)
    preprocess(config)

    found = discover_published_cameras.fn(
        config.record_root, config.cut_root, config.crop_root, only="*/Side"
    )

    assert [record.capture_id for record in found.records] == ["9 July/cnj_45kg_Set1/Side"]


def test_a_camera_missing_an_upstream_file_is_rejected_with_a_reason(
    tmp_path: Path,
    make_camera: MakeCamera,
    make_meta_template: Any,
    make_session_metadata: Any,
) -> None:
    raw = build_capture(tmp_path, make_camera, make_meta_template, make_session_metadata)
    config = make_config(tmp_path, raw)
    preprocess(config)
    shutil.rmtree(config.record_root / "9 July" / "cnj_45kg_Set1" / "Front")

    found = discover_published_cameras.fn(
        config.record_root, config.cut_root, config.crop_root, only=None
    )

    assert [record.capture_id for record in found.records] == ["9 July/cnj_45kg_Set1/Side"]
    [rejected] = found.rejected
    assert rejected.scan_id == "9 July/cnj_45kg_Set1/Front"
    assert "record.json" in rejected.reason


def test_an_empty_crop_tree_finds_nothing(tmp_path: Path) -> None:
    found = discover_published_cameras.fn(
        tmp_path / "s0", tmp_path / "s1", tmp_path / "s4", only=None
    )

    assert found.records == []
    assert found.rejected == []
