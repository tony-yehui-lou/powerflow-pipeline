"""Find the cameras in a raw capture tree.

`common/discovery.discover_scans` does not fit here: it looks for `meta.json` + `frames/`
in immediate children. This capture is `<date>/<session>/<camera>`, and a camera-session
is not a scan.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from prefect import task

from powerflow_pipeline.data.common.filesystem import INTERNAL_PREFIX
from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess.models import CameraDir


def _child_dirs(parent: Path) -> Iterator[Path]:
    """Yield real child directories in stable order, ignoring dotfiles and staging."""

    for child in sorted(parent.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith(INTERNAL_PREFIX):
            continue
        yield child


@task
def discover_sessions(raw_root: Path) -> list[CameraDir]:
    """Return every `<date>/<session>/<camera>` directory under the raw root."""

    log_task_paths(raw_root, None)
    return [
        CameraDir(
            date=date_dir.name,
            session=session_dir.name,
            camera=camera_dir.name,
            source=camera_dir,
        )
        for date_dir in _child_dirs(raw_root)
        for session_dir in _child_dirs(date_dir)
        for camera_dir in _child_dirs(session_dir)
    ]
