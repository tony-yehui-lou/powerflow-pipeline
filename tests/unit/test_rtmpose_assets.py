"""Checkpoint caching for RTMPose. No network: the download itself is monkeypatched."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from powerflow_pipeline.data.preprocess import rtmpose_assets


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)


def _fake_download(tmp_path: Path, members: dict[str, bytes]) -> object:
    def _urlretrieve(url: str, destination: str | Path) -> None:
        _write_zip(Path(destination), members)

    return _urlretrieve


def test_extracts_the_single_onnx_into_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"x" * 2_000_000
    monkeypatch.setattr(
        rtmpose_assets.urllib.request,
        "urlretrieve",
        _fake_download(tmp_path, {"bundle/model.onnx": payload}),
    )

    path = rtmpose_assets.ensure_rtmpose_model("pose", "balanced", cache_dir=tmp_path)

    assert path == tmp_path / "rtmpose_balanced_pose.onnx"
    assert path.read_bytes() == payload
    # The staging file must not survive: a truncated download should never look cached.
    assert not list(tmp_path.glob("*.part"))


def test_reuses_an_already_cached_checkpoint_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = tmp_path / "rtmpose_balanced_det.onnx"
    cached.write_bytes(b"y" * 2_000_000)

    def _explode(url: str, destination: str | Path) -> None:
        raise AssertionError("should not download an already-cached checkpoint")

    monkeypatch.setattr(rtmpose_assets.urllib.request, "urlretrieve", _explode)

    assert rtmpose_assets.ensure_rtmpose_model("det", "balanced", cache_dir=tmp_path) == cached


def test_rejects_a_truncated_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rtmpose_assets.urllib.request,
        "urlretrieve",
        _fake_download(tmp_path, {"bundle/model.onnx": b"too small"}),
    )

    with pytest.raises(RuntimeError, match="truncated"):
        rtmpose_assets.ensure_rtmpose_model("pose", "balanced", cache_dir=tmp_path)


def test_rejects_an_archive_without_exactly_one_onnx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rtmpose_assets.urllib.request,
        "urlretrieve",
        _fake_download(tmp_path, {"a.onnx": b"z" * 2_000_000, "b.onnx": b"z" * 2_000_000}),
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        rtmpose_assets.ensure_rtmpose_model("pose", "balanced", cache_dir=tmp_path)


def test_rejects_an_unknown_role() -> None:
    with pytest.raises(ValueError, match="role"):
        rtmpose_assets.ensure_rtmpose_model("segmentation", "balanced")


def test_every_variant_pins_both_checkpoints_and_their_input_sizes() -> None:
    # A pinned checkpoint is provenance: it should change when someone edits the asset module,
    # never because a dependency bumped.
    for variant in ("lightweight", "balanced", "performance"):
        for role in ("det", "pose"):
            url = rtmpose_assets.checkpoint_url(role, variant)  # type: ignore[arg-type]
            assert url.startswith("https://download.openmmlab.com/")
            assert "latest" not in url
        assert set(rtmpose_assets.INPUT_SIZES[variant]) == {"det", "pose"}  # type: ignore[index]
