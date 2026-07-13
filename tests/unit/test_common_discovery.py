"""Scan discovery is deterministic and rejects structurally incomplete scans."""

from __future__ import annotations

from pathlib import Path

from powerflow_pipeline.data.common.discovery import discover_scans
from tests.conftest import MakeScan


def test_scans_are_discovered_in_stable_order(tmp_path: Path, make_scan: MakeScan) -> None:
    for scan_id in ("scan_0002", "scan_0001", "scan_0003"):
        make_scan(tmp_path, scan_id)

    result = discover_scans.fn(tmp_path)

    assert [scan.scan_id for scan in result.scans] == ["scan_0001", "scan_0002", "scan_0003"]
    assert result.rejected_scans == []


def test_discovered_files_are_relative_and_sorted(tmp_path: Path, make_scan: MakeScan) -> None:
    make_scan(tmp_path, "scan_0001", frames=2)

    (scan,) = discover_scans.fn(tmp_path).scans

    assert scan.source == tmp_path / "scan_0001"
    assert scan.files == (
        Path("frames/000.bin"),
        Path("frames/001.bin"),
        Path("meta.json"),
    )


def test_scan_without_meta_is_rejected(tmp_path: Path, make_scan: MakeScan) -> None:
    make_scan(tmp_path, "scan_0001")
    (tmp_path / "scan_0001" / "meta.json").unlink()

    result = discover_scans.fn(tmp_path)

    assert result.scans == []
    (rejected,) = result.rejected_scans
    assert rejected.scan_id == "scan_0001"
    assert "meta.json" in rejected.reason


def test_scan_without_frames_is_rejected(tmp_path: Path, make_scan: MakeScan) -> None:
    make_scan(tmp_path, "scan_0001", frames=0)

    result = discover_scans.fn(tmp_path)

    assert result.scans == []
    (rejected,) = result.rejected_scans
    assert "frames" in rejected.reason


def test_loose_files_and_staging_directories_are_ignored(
    tmp_path: Path, make_scan: MakeScan
) -> None:
    make_scan(tmp_path, "scan_0001")
    (tmp_path / "README.txt").write_text("not a scan")
    (tmp_path / ".powerflow-staging-abc").mkdir()

    result = discover_scans.fn(tmp_path)

    assert [scan.scan_id for scan in result.scans] == ["scan_0001"]
    assert result.rejected_scans == []
