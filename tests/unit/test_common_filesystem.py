"""Staging, atomic publication, and in-place commit never leave partial results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from powerflow_pipeline.data.common.errors import PublishError
from powerflow_pipeline.data.common.filesystem import (
    cleanup,
    commit_in_place,
    copy_tree,
    create_staging_root,
    publish_staging,
    write_json,
)


def test_staging_root_is_created_beside_its_destination(tmp_path: Path) -> None:
    anchor = tmp_path / "output"

    staging = create_staging_root(anchor)

    assert staging.is_dir()
    assert staging.parent == tmp_path  # same filesystem as the destination, so rename is atomic
    assert staging.name.startswith(".powerflow-staging-")


def test_cleanup_removes_staging_and_tolerates_a_missing_path(tmp_path: Path) -> None:
    staging = create_staging_root(tmp_path / "output")

    cleanup(staging)
    cleanup(staging)  # idempotent: a published run has already moved it away

    assert not staging.exists()


def test_write_json_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    write_json(path, {"b": 2, "a": 1})

    assert path.read_text() == '{\n  "a": 1,\n  "b": 2\n}\n'
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}


def test_copy_tree_duplicates_a_scan(tmp_path: Path) -> None:
    source = tmp_path / "scan_0001"
    (source / "frames").mkdir(parents=True)
    (source / "frames" / "000.bin").write_bytes(b"\x00\x01")

    copy_tree(source, tmp_path / "staged")

    assert (tmp_path / "staged" / "frames" / "000.bin").read_bytes() == b"\x00\x01"


def test_publish_moves_the_whole_staged_tree(tmp_path: Path) -> None:
    output = tmp_path / "output"
    staging = create_staging_root(output)
    (staging / "scan_0001").mkdir()

    publish_staging(staging, output)

    assert (output / "scan_0001").is_dir()
    assert not staging.exists()


def test_publish_refuses_to_merge_into_an_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    staging = create_staging_root(output)

    with pytest.raises(PublishError, match="already exists"):
        publish_staging(staging, output)


def test_commit_in_place_replaces_existing_entries(tmp_path: Path) -> None:
    destination = tmp_path / "scan_0001"
    destination.mkdir()
    (destination / "meta.json").write_text("old")
    staging = create_staging_root(destination)
    staged = staging / "scan_0001"
    staged.mkdir()
    (staged / "meta.json").write_text("new")

    commit_in_place([(staged, destination)])

    assert (destination / "meta.json").read_text() == "new"
    assert not any(path.name.startswith(".powerflow-backup-") for path in tmp_path.iterdir())


def test_commit_in_place_rolls_back_when_a_later_replacement_fails(tmp_path: Path) -> None:
    first = tmp_path / "scan_0001"
    first.mkdir()
    (first / "meta.json").write_text("old")
    staging = create_staging_root(first)
    staged_first = staging / "scan_0001"
    staged_first.mkdir()
    (staged_first / "meta.json").write_text("new")
    missing_staged = staging / "scan_0002"  # never produced, so os.replace raises

    with pytest.raises(PublishError, match="in place"):
        commit_in_place([(staged_first, first), (missing_staged, tmp_path / "scan_0002")])

    assert (first / "meta.json").read_text() == "old"
    assert (staged_first / "meta.json").read_text() == "new"
    assert not (tmp_path / "scan_0002").exists()
