"""Safe staging and atomic publication helpers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from powerflow_pipeline.data.core.errors import PublishError


def create_staging_root(anchor: Path) -> Path:
    """Create staging beside its eventual destination to preserve atomic renames."""

    anchor.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".powerflow-staging-", dir=anchor.parent))


def cleanup(path: Path) -> None:
    """Remove a staging directory if a run did not publish it."""

    if path.exists():
        shutil.rmtree(path)


def copy_tree(source: Path, destination: Path) -> None:
    """Copy a complete scan into an empty staging destination."""

    shutil.copytree(source, destination)


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, human-reviewable JSON."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_staging(staging_root: Path, output_root: Path) -> None:
    """Publish one complete staged output tree without merging partial results."""

    if output_root.exists():
        raise PublishError(f"output path already exists: {output_root}")
    try:
        os.replace(staging_root, output_root)
    except OSError as error:
        raise PublishError(f"failed to publish staged output: {error}") from error


def commit_in_place(replacements: list[tuple[Path, Path]]) -> None:
    """Atomically replace staged entries, rolling back earlier replacements on failure."""

    committed: list[tuple[Path, Path, Path | None]] = []
    try:
        for staged, destination in replacements:
            backup: Path | None = None
            if destination.exists():
                backup = (
                    destination.parent / f".powerflow-backup-{uuid.uuid4().hex}-{destination.name}"
                )
                os.replace(destination, backup)
            os.replace(staged, destination)
            committed.append((staged, destination, backup))
    except OSError as error:
        for staged, destination, backup in reversed(committed):
            if destination.exists():
                os.replace(destination, staged)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
        raise PublishError(f"failed to commit staged output in place: {error}") from error
    for _, _, backup in committed:
        if backup is not None and backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
