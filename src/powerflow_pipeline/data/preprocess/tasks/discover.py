"""Find the captures in a raw capture tree, whatever shape it was foldered in.

`common/discovery.discover_scans` does not fit here: it looks for `meta.json` + `frames/`
in immediate children, and a capture is not a scan.

Two shapes exist in `data/raw` and both are walked by the same pass. A directory is a
**capture** when it holds *any* stream in `ingest.REQUIRED_STREAMS`; it is then classified
by *where its operator `metadata.yaml` lives* -- file presence only, never a name match:

- parent has one  -> `MULTI_CAMERA`, the session is the group, role from the directory name
- the capture has one -> `SINGLE_CAMERA`, the capture is its own group, role `single`
- neither -> rejected, naming both paths that were checked

Parent-metadata is tested **first**, so a stray per-camera file dropped inside a two-camera
session can never silently split it into two groups.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from prefect import task

from powerflow_pipeline.data.common.filesystem import INTERNAL_PREFIX
from powerflow_pipeline.data.common.models import RejectedScan
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess.models import CaptureLayout, CaptureRole, CaptureUnit
from powerflow_pipeline.data.preprocess.tasks.ingest import REQUIRED_STREAMS

OPERATOR_METADATA = "metadata.yaml"

# The directory names a two-camera session may use, mapped to the role they read.
CAMERA_ROLES: dict[str, CaptureRole] = {"front": "front", "side": "side"}


@dataclass(slots=True)
class CaptureDiscovery:
    """Captures ready for S0, plus the directories rejected before any pixel was read."""

    captures: list[CaptureUnit] = field(default_factory=list)
    rejected: list[RejectedScan] = field(default_factory=list)


def _child_dirs(parent: Path) -> Iterator[Path]:
    """Yield real child directories in stable order, ignoring dotfiles and staging."""

    for child in sorted(parent.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith(INTERNAL_PREFIX):
            continue
        yield child


def is_capture(directory: Path) -> bool:
    """True when `directory` holds at least one required stream.

    *Any*, not *all*, deliberately. Requiring the full set would make a capture missing its
    `imu.csv` invisible to the walk rather than rejected by S0 with the exact stream named,
    turning every V1 rejection into a silent omission -- the one failure mode these stages
    refuse everywhere else. Completeness is S0's judgement to make and to report; discovery
    only decides what is a capture-shaped directory at all.
    """

    return any((directory / name).exists() for name in REQUIRED_STREAMS)


def _capture_dirs(directory: Path) -> Iterator[Path]:
    """Walk a subtree for capture directories, never descending into one."""

    if is_capture(directory):
        yield directory
        return
    for child in _child_dirs(directory):
        yield from _capture_dirs(child)


def classify(capture: Path, raw_root: Path) -> CaptureUnit | RejectedScan:
    """Resolve one capture directory's layout, group, role, and metadata file."""

    relative = capture.relative_to(raw_root)
    parent_metadata = capture.parent / OPERATOR_METADATA
    own_metadata = capture / OPERATOR_METADATA

    if parent_metadata.is_file():
        role = CAMERA_ROLES.get(capture.name.strip().lower())
        if role is None:
            return RejectedScan(
                scan_id=str(relative),
                source=capture,
                reason=(
                    "camera directory name is not Front or Side, and its session holds the "
                    f"operator metadata: {capture.name!r}"
                ),
            )
        group = capture.parent
        return CaptureUnit(
            source=capture,
            relative=relative,
            metadata_path=parent_metadata,
            metadata_relative=parent_metadata.relative_to(raw_root),
            group_id=str(group.relative_to(raw_root)),
            role=role,
            layout=CaptureLayout.MULTI_CAMERA,
        )

    if own_metadata.is_file():
        return CaptureUnit(
            source=capture,
            relative=relative,
            metadata_path=own_metadata,
            metadata_relative=own_metadata.relative_to(raw_root),
            group_id=str(relative),
            role="single",
            layout=CaptureLayout.SINGLE_CAMERA,
        )

    return RejectedScan(
        scan_id=str(relative),
        source=capture,
        reason=(
            f"no operator {OPERATOR_METADATA}: checked "
            f"{parent_metadata.relative_to(raw_root)} and {own_metadata.relative_to(raw_root)}"
        ),
    )


@task
def discover_captures(raw_root: Path) -> CaptureDiscovery:
    """Return every capture under the raw root, in stable order, both layouts included."""

    log_task_paths(raw_root, None)
    result = CaptureDiscovery()
    for date_dir in _child_dirs(raw_root):
        for capture in _capture_dirs(date_dir):
            outcome = classify(capture, raw_root)
            if isinstance(outcome, CaptureUnit):
                result.captures.append(outcome)
            else:
                result.rejected.append(outcome)
    return result
