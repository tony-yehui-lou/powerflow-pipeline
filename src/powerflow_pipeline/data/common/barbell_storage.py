"""Where S5's `barbell.json` lives, and reading and writing it under the write-once rule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from powerflow_pipeline.data.common.models import BarbellDocument
from powerflow_pipeline.data.common.pose_storage import write_once

BARBELL_FILENAME: Final = "barbell.json"


def barbell_path(stage_root: Path, relative: Path) -> Path:
    """`<stage_root>/<capture path>/barbell.json`: the root mirrors the raw layout."""

    return stage_root / relative / BARBELL_FILENAME


def write_barbell(path: Path, document: BarbellDocument) -> None:
    write_once(path, document.to_document())


def read_barbell(path: Path) -> BarbellDocument:
    """Load a stored document, revalidating every invariant on the way in."""

    return BarbellDocument.from_document(json.loads(path.read_text(encoding="utf-8")))
