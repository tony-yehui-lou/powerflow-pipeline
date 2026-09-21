"""Discovery finds captures in either raw shape, and says which is which.

A capture is a directory holding at least one required stream. Its layout is decided by *where
its operator `metadata.yaml` lives* -- never by a directory name, which is exactly the
inference that broke when a second capture day was foldered differently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from powerflow_pipeline.data.preprocess.models import CaptureLayout
from powerflow_pipeline.data.preprocess.tasks.discover import discover_captures
from tests.conftest import MakeCamera


def test_discovers_every_camera_in_stable_order(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, session="cnj_55kg_Set1", camera="Side")
    make_camera(raw, session="cnj_45kg_Set1", camera="Front")
    make_camera(raw, session="cnj_45kg_Set1", camera="Side")
    make_session_metadata(raw, session="cnj_45kg_Set1")
    make_session_metadata(raw, session="cnj_55kg_Set1")

    found = discover_captures.fn(raw)

    assert [capture.capture_id for capture in found.captures] == [
        "9 July/cnj_45kg_Set1/Front",
        "9 July/cnj_45kg_Set1/Side",
        "9 July/cnj_55kg_Set1/Side",
    ]
    assert found.captures[0].source == raw / "9 July" / "cnj_45kg_Set1" / "Front"
    assert found.captures[0].relative == Path("9 July/cnj_45kg_Set1/Front")
    assert found.captures[0].camera_id == "9 July/cnj_45kg_Set1/Front"


def test_metadata_at_the_parent_is_a_two_camera_session(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, camera="Front")
    make_camera(raw, camera="Side")
    metadata = make_session_metadata(raw)

    found = discover_captures.fn(raw)

    assert [capture.role for capture in found.captures] == ["front", "side"]
    for capture in found.captures:
        assert capture.layout is CaptureLayout.MULTI_CAMERA
        assert capture.group_id == "9 July/cnj_45kg_Set1"  # both share one cut interval
        assert capture.metadata_path == metadata
        assert capture.metadata_relative == Path("9 July/cnj_45kg_Set1/metadata.yaml")


def test_metadata_inside_the_capture_is_a_single_camera_trial(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw, date="22 August", session="Snch", camera="110kgSnch1")
    metadata = make_session_metadata(raw, date="22 August", session="Snch", camera="110kgSnch1")

    (capture,) = discover_captures.fn(raw).captures

    assert capture.layout is CaptureLayout.SINGLE_CAMERA
    assert capture.role == "single"
    assert capture.capture_id == "22 August/Snch/110kgSnch1"
    # Its own group: nothing else shares its lift window.
    assert capture.group_id == "22 August/Snch/110kgSnch1"
    assert capture.metadata_path == metadata
    assert capture.metadata_relative == Path("22 August/Snch/110kgSnch1/metadata.yaml")


def test_parent_metadata_wins_when_both_exist(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """A stray per-camera file must never split a two-camera session into two groups."""

    raw = tmp_path / "raw"
    make_camera(raw, camera="Front")
    make_camera(raw, camera="Side")
    make_session_metadata(raw)
    make_session_metadata(raw, camera="Front")  # the stray one

    found = discover_captures.fn(raw)

    assert {capture.group_id for capture in found.captures} == {"9 July/cnj_45kg_Set1"}
    assert [capture.role for capture in found.captures] == ["front", "side"]


def test_a_capture_with_no_operator_metadata_is_rejected_naming_both_paths(
    tmp_path: Path, make_camera: MakeCamera
) -> None:
    raw = tmp_path / "raw"
    make_camera(raw)

    found = discover_captures.fn(raw)

    assert found.captures == []
    (rejected,) = found.rejected
    assert rejected.scan_id == "9 July/cnj_45kg_Set1/Front"
    assert "9 July/cnj_45kg_Set1/metadata.yaml" in rejected.reason
    assert "9 July/cnj_45kg_Set1/Front/metadata.yaml" in rejected.reason


def test_a_session_camera_with_an_unknown_name_is_rejected(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """Under parent metadata the directory name *is* the role, so it has to be one we know."""

    raw = tmp_path / "raw"
    make_camera(raw, camera="Overhead")
    make_session_metadata(raw)

    found = discover_captures.fn(raw)

    assert found.captures == []
    (rejected,) = found.rejected
    assert "not Front or Side" in rejected.reason
    assert "'Overhead'" in rejected.reason


def test_one_walk_yields_both_layouts(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """One command over a mixed `data/raw` -- the whole point of detecting rather than flagging."""

    raw = tmp_path / "raw"
    make_camera(raw, camera="Front")
    make_camera(raw, camera="Side")
    make_session_metadata(raw)
    make_camera(raw, date="22 August", session="CnJ", camera="82kgCnJ2")
    make_session_metadata(raw, date="22 August", session="CnJ", camera="82kgCnJ2")

    found = discover_captures.fn(raw)

    assert found.rejected == []
    assert {capture.capture_id: capture.layout for capture in found.captures} == {
        "9 July/cnj_45kg_Set1/Front": CaptureLayout.MULTI_CAMERA,
        "9 July/cnj_45kg_Set1/Side": CaptureLayout.MULTI_CAMERA,
        "22 August/CnJ/82kgCnJ2": CaptureLayout.SINGLE_CAMERA,
    }


def test_skips_dot_directories_and_stray_files(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """The real tree is littered with .DS_Store, and staging dirs must never be re-ingested."""

    raw = tmp_path / "raw"
    make_camera(raw)
    make_session_metadata(raw)
    (raw / ".DS_Store").write_text("junk")
    (raw / "9 July" / ".DS_Store").write_text("junk")
    (raw / "9 July" / "meta.yaml").write_text("---\n")
    (raw / "9 July" / "cnj_45kg_Set1" / ".DS_Store").write_text("junk")
    (raw / "9 July" / "cnj_45kg_Set1" / ".powerflow-staging-abc").mkdir()
    (raw / "9 July" / ".hidden_session" / "Front").mkdir(parents=True)

    found = discover_captures.fn(raw)

    assert [capture.capture_id for capture in found.captures] == ["9 July/cnj_45kg_Set1/Front"]
    assert found.rejected == []


def test_a_directory_with_no_streams_at_all_is_not_a_capture(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """Streams define a capture, so a folder holding none of them is silently not one."""

    raw = tmp_path / "raw"
    make_camera(raw)
    make_session_metadata(raw)
    (raw / "9 July" / "cnj_45kg_Set1" / "notes").mkdir()

    found = discover_captures.fn(raw)

    assert [capture.capture_id for capture in found.captures] == ["9 July/cnj_45kg_Set1/Front"]
    assert found.rejected == []


def test_a_capture_missing_one_stream_is_still_discovered(
    tmp_path: Path, make_camera: MakeCamera, make_session_metadata: Any
) -> None:
    """S0 rejects an incomplete capture naming the stream; discovery must not hide it first.

    Requiring the *full* stream set here would turn every V1 rejection into a silent
    omission -- the capture would simply never be walked, and the run would report nothing
    at all about it.
    """

    raw = tmp_path / "raw"
    make_camera(raw, omit=["imu"])
    make_session_metadata(raw)

    found = discover_captures.fn(raw)

    assert [capture.capture_id for capture in found.captures] == ["9 July/cnj_45kg_Set1/Front"]


def test_an_empty_raw_root_finds_nothing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    found = discover_captures.fn(raw)

    assert found.captures == []
    assert found.rejected == []
