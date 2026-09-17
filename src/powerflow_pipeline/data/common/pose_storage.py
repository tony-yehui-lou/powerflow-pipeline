"""Where human body model data lands on disk, and the write-once rules it obeys.

Normative source: "Human Body Model Data Storage v1" (docs/spec, GitHub issue #117).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from powerflow_pipeline.data.common.errors import PublishError
from powerflow_pipeline.data.common.models import SKELETON_ID, PoseDocument, skeleton_document

POSE_OUTPUT_DIR: Final = "s5_pose_output"
POSE_FILENAME: Final = "pose.json"


def _session_relative(session_path: str | Path) -> Path:
    relative = Path(session_path)
    if relative.is_absolute():
        raise ValueError(f"session path must be relative to the stage root: {session_path}")
    if ".." in relative.parts:
        raise ValueError(f"session path must not escape the stage root: {session_path}")
    return relative


def pose_path(stage_root: Path, session_path: str | Path, camera: str) -> Path:
    """`<stage_root>/<session path>/<camera>/pose.json`.

    The session path is taken whole rather than as a `<date>/<session>` pair because captures are
    not a uniform depth on disk: `11 July/30kg_Set1` is two levels, `22 August/CnJ/82kgCnJ1` is
    three.
    """

    return stage_root / _session_relative(session_path) / camera / POSE_FILENAME


def skeleton_path(directory: Path) -> Path:
    """`<directory>/skeleton.<id>.json` -- one document per topology version."""

    return directory / f"skeleton.{SKELETON_ID}.json"


def dumps(payload: Any, level: int = 0) -> str:
    """Deterministic JSON that keeps per-frame arrays on one line.

    `filesystem.write_json` indents every element, which is right for the small records it
    writes but turns a 600-frame pose document into roughly a hundred thousand lines. Here only
    objects are indented; any array free of objects stays inline, so a document stays scannable
    however many frames it holds.
    """

    pad = "  " * level
    inner = "  " * (level + 1)
    if isinstance(payload, dict):
        if not payload:
            return "{}"
        entries = [
            f"{inner}{json.dumps(str(key))}: {dumps(value, level + 1)}"
            for key, value in sorted(payload.items())
        ]
        return "{\n" + ",\n".join(entries) + f"\n{pad}}}"
    if isinstance(payload, list) and any(isinstance(item, dict) for item in payload):
        entries = [f"{inner}{dumps(item, level + 1)}" for item in payload]
        return "[\n" + ",\n".join(entries) + f"\n{pad}]"
    return json.dumps(payload)


def _write_once(path: Path, payload: Any) -> None:
    """Stage outputs are written once by one task and read many times by others."""

    if path.exists():
        raise PublishError(f"refusing to overwrite: {path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(payload) + "\n", encoding="utf-8")


def write_pose(path: Path, document: PoseDocument) -> None:
    """Write one camera's pose document for one capture."""

    _write_once(path, document.to_document())


def read_pose(path: Path) -> PoseDocument:
    """Load a stored pose document, revalidating every invariant on the way in."""

    return PoseDocument.from_document(json.loads(path.read_text(encoding="utf-8")))


def write_skeleton(directory: Path) -> Path:
    """Render the topology document from `HUMAN_SKELETON` and return where it landed."""

    path = skeleton_path(directory)
    _write_once(path, skeleton_document())
    return path
