"""Discovery walks `<date>/<session>/<camera>`, the shape the capture actually has."""

from __future__ import annotations

from pathlib import Path

from powerflow_pipeline.data.preprocess.tasks.discover import discover_sessions
from tests.conftest import MakeCamera


def test_discovers_every_camera_in_stable_order(tmp_path: Path, make_camera: MakeCamera) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, session="cnj_55kg_Set1", camera="Side")
    make_camera(raw, session="cnj_45kg_Set1", camera="Front")
    make_camera(raw, session="cnj_45kg_Set1", camera="Side")

    found = discover_sessions.fn(raw)

    assert [camera.camera_id for camera in found] == [
        "9 July/cnj_45kg_Set1/Front",
        "9 July/cnj_45kg_Set1/Side",
        "9 July/cnj_55kg_Set1/Side",
    ]
    assert found[0].date == "9 July"
    assert found[0].session == "cnj_45kg_Set1"
    assert found[0].camera == "Front"
    assert found[0].source == raw / "9 July" / "cnj_45kg_Set1" / "Front"
    assert found[0].relative == Path("9 July/cnj_45kg_Set1/Front")


def test_skips_dot_directories_and_stray_files(tmp_path: Path, make_camera: MakeCamera) -> None:
    """The real tree is littered with .DS_Store, and staging dirs must never be re-ingested."""

    raw = tmp_path / "raw"
    make_camera(raw)
    (raw / ".DS_Store").write_text("junk")
    (raw / "9 July" / ".DS_Store").write_text("junk")
    (raw / "9 July" / "meta.yaml").write_text("---\n")
    (raw / "9 July" / "cnj_45kg_Set1" / ".DS_Store").write_text("junk")
    (raw / "9 July" / "cnj_45kg_Set1" / ".powerflow-staging-abc").mkdir()
    (raw / "9 July" / ".hidden_session" / "Front").mkdir(parents=True)

    found = discover_sessions.fn(raw)

    assert [camera.camera_id for camera in found] == ["9 July/cnj_45kg_Set1/Front"]


def test_an_empty_raw_root_finds_nothing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    assert discover_sessions.fn(raw) == []
