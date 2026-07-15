"""Instantiate the date-level `meta.yaml` template into a per-session `metadata.yaml`.

The shipped `meta.yaml` holds a *type schema*, not values (`weight_in_kg: float`), and it
sits one level above the sessions while carrying a single `lift:` block. So there is
nothing yet to check `rgb.mp4` against. S0 therefore fills in only what the capture
actually evidences, marks what it merely inferred from the session name as derived, and
leaves the rest `null`. Raw data is write-once: the output lands in the record tree.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from prefect import task

from powerflow_pipeline.data.common.task_logging import log_task_paths
from powerflow_pipeline.data.preprocess.config import PreprocessConfig
from powerflow_pipeline.data.preprocess.models import (
    AthleteMeta,
    CameraRecord,
    CutInterval,
    LiftMeta,
)

# The leaves of an uninstantiated template: type names and enum option lists.
TYPE_NAMES = {
    "long",
    "int",
    "float",
    "double",
    "string",
    "str",
    "bool",
    "boolean",
    "date",
    "datetime",
    "iso 8601 date",
}
SESSION_NAME = re.compile(r"^(?P<type>[a-z]+)_(?P<weight>\d+(?:\.\d+)?)kg_Set(?P<set>\d+)$", re.I)


def load_meta_template(path: Path) -> dict[str, Any] | None:
    """Parse the date-level `meta.yaml`. A missing file is not an error."""

    if not path.is_file():
        return None
    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else None


def _leaves(node: Any) -> list[Any]:
    if isinstance(node, dict):
        return [leaf for value in node.values() for leaf in _leaves(value)]
    return [node]


def is_template(meta: dict[str, Any] | None) -> bool:
    """True when every leaf is a type name or an option list, i.e. nothing has been filled in."""

    if not meta:
        return False
    leaves = _leaves(meta)
    return all(
        isinstance(leaf, list) or (isinstance(leaf, str) and leaf.strip().lower() in TYPE_NAMES)
        for leaf in leaves
    )


def derive_from_session_name(session: str) -> dict[str, Any]:
    """Read the lift out of `cnj_45kg_Set1`. Inference, never a measurement."""

    match = SESSION_NAME.match(session)
    if match is None:
        return {}
    return {
        "type": match["type"].lower(),
        "weight_in_kg": float(match["weight"]),
        "set": int(match["set"]),
    }


def _epoch(creation_time: str | None) -> int | None:
    """Convert `rgb.mp4`'s ISO creation tag to the template's `dateTime_epoch`."""

    if not creation_time:
        return None
    try:
        return int(datetime.fromisoformat(creation_time.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _observed(record: CameraRecord) -> dict[str, Any]:
    """The facts S0 measured, as opposed to the ones the filename suggests."""

    return {
        "camera": record.camera,
        "rgb_width": record.rgb_width,
        "rgb_height": record.rgb_height,
        "fps": record.fps,
        "depth_width": record.depth_width,
        "depth_height": record.depth_height,
        "n_frames": record.n_frames,
        "counts": record.counts.model_dump(),
        "intrinsics": record.intrinsics.model_dump(),
    }


@task
def write_session_metadata(
    template_path: Path,
    records: list[CameraRecord],
    config: PreprocessConfig,
    cut_interval: CutInterval | None = None,
) -> list[Path]:
    """Write one session's `metadata.yaml` into every stage output, inventing no values.

    One body, serialised once and written to each of `config.stage_roots`, so the stages
    cannot disagree about the session they are describing.
    """

    record = records[0]
    destinations = [
        root / record.date / record.session / "metadata.yaml" for root in config.stage_roots
    ]
    log_task_paths(template_path, destinations)
    template = load_meta_template(template_path)
    status = "absent" if template is None else "template" if is_template(template) else "filled"

    lift = LiftMeta(dateTime_epoch=_epoch(record.creation_time)).model_dump()
    if cut_interval is not None:
        lift["cut_start_epoch_ms"] = cut_interval.cut_start_epoch_ms
        lift["cut_end_epoch_ms"] = cut_interval.cut_end_epoch_ms
        lift["side_creation_time"] = cut_interval.side_creation_time
        lift["lift_start_time_side_in_ms"] = cut_interval.lift_start_time_side_in_ms
        lift["lift_end_time_side_in_ms"] = cut_interval.lift_end_time_side_in_ms

    metadata: dict[str, Any] = {
        "meta_status": status,
        "date": record.date,
        "session": record.session,
        "lift": lift,
        "athlete": AthleteMeta().model_dump(),
        "derived_from_session_name": derive_from_session_name(record.session),
        "cameras": [_observed(camera) for camera in records],
    }

    if not config.dry_run:
        body = yaml.safe_dump(metadata, sort_keys=False, default_flow_style=False)
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(body, encoding="utf-8")
    return destinations
