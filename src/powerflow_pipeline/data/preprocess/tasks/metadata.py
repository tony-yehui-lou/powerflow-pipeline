"""Instantiate the date-level `meta.yaml` template into a per-group `metadata.yaml`.

The shipped `meta.yaml` holds a *type schema*, not values (`weight_in_kg: float`), and it
sits one level above the sessions while carrying a single `lift:` block. So there is
nothing yet to check `rgb.mp4` against. S0 therefore fills in only what the capture
actually evidences, marks what it merely inferred from the directory name as derived, and
leaves the rest `null`. Raw data is write-once: the output lands in the stage trees.

The output mirrors the raw path of the operator file it describes, so a two-camera session
keeps `<date>/<session>/metadata.yaml` and a single-camera trial writes
`<date>/<liftType>/<trial>/metadata.yaml`, beside its own streams.
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
    CaptureLayout,
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
# The single-camera layout names a trial `110kgSnch1` / `82kgCnJ2`: weight first, no
# separators, and an attempt number rather than a set number.
TRIAL_NAME = re.compile(r"^(?P<weight>\d+(?:\.\d+)?)kg(?P<type>[A-Za-z]+)(?P<attempt>\d+)$")


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


def derive_from_session_name(name: str) -> dict[str, Any]:
    """Read the lift out of `cnj_45kg_Set1` or `110kgSnch1`. Inference, never a measurement."""

    if (match := SESSION_NAME.match(name)) is not None:
        return {
            "type": match["type"].lower(),
            "weight_in_kg": float(match["weight"]),
            "set": int(match["set"]),
        }
    if (match := TRIAL_NAME.match(name)) is not None:
        return {
            "type": match["type"].lower(),
            "weight_in_kg": float(match["weight"]),
            "attempt": int(match["attempt"]),
        }
    return {}


def lift_type_from_group_folder(record: CameraRecord) -> str | None:
    """The `Snch` / `CnJ` folder a single-camera trial sits in, lowercased.

    Authoritative over the file's own `type:` field. That field has been wrong before --
    every August file once read `type: snch`, including the five under `CnJ/` -- while the
    folder a capture was filed in never has been. The file's value is still carried
    through separately; nothing is silently corrected.
    """

    if record.layout is not CaptureLayout.SINGLE_CAMERA:
        return None
    parts = record.relative.parts
    return parts[-2].lower() if len(parts) >= 3 else None


def _epoch(creation_time: str | None) -> int | None:
    """Convert `rgb.mp4`'s ISO creation tag to the template's `dateTime_epoch`."""

    if not creation_time:
        return None
    try:
        return int(datetime.fromisoformat(creation_time.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _observed(record: CameraRecord) -> dict[str, Any]:
    """The facts S0 measured, as opposed to the ones the directory name suggests."""

    return {
        "capture_id": record.capture_id,
        "role": record.role,
        "layout": record.layout.value,
        "rgb_width": record.rgb_width,
        "rgb_height": record.rgb_height,
        "fps": record.fps,
        "depth_width": record.depth_width,
        "depth_height": record.depth_height,
        "n_frames": record.n_frames,
        "counts": record.counts.model_dump(),
        "intrinsics": record.intrinsics.model_dump(),
    }


def _file_lift_type(record: CameraRecord) -> str | None:
    """The raw file's own `lift.type`, carried through rather than trusted (see above)."""

    if not record.metadata_path.is_file():
        return None
    loaded = yaml.safe_load(record.metadata_path.read_text())
    if not isinstance(loaded, dict) or not isinstance(loaded.get("lift"), dict):
        return None
    value = loaded["lift"].get("type")
    return value if isinstance(value, str) else None


@task
def write_group_metadata(
    template_path: Path,
    records: list[CameraRecord],
    config: PreprocessConfig,
    cut_interval: CutInterval | None = None,
) -> list[Path]:
    """Write one group's `metadata.yaml` into every stage output, inventing no values.

    One body, serialised once and written to each of `config.stage_roots`, so the stages
    cannot disagree about the lift they are describing. The destination mirrors the raw
    path of the operator file, whatever depth that sits at.
    """

    record = records[0]
    destinations = [root / record.metadata_relative for root in config.stage_roots]
    log_task_paths(template_path, destinations)
    template = load_meta_template(template_path)
    status = "absent" if template is None else "template" if is_template(template) else "filled"

    derived = derive_from_session_name(Path(record.group_id).name)
    if (folder_type := lift_type_from_group_folder(record)) is not None:
        derived["type"] = folder_type
        derived["type_source"] = "group folder"

    lift = LiftMeta(dateTime_epoch=_epoch(record.creation_time)).model_dump()
    if cut_interval is not None:
        lift["cut_start_epoch_ms"] = cut_interval.cut_start_epoch_ms
        lift["cut_end_epoch_ms"] = cut_interval.cut_end_epoch_ms
        lift["side_creation_time"] = cut_interval.side_creation_time
        lift["lift_start_time_side_in_ms"] = cut_interval.lift_start_time_side_in_ms
        lift["lift_end_time_side_in_ms"] = cut_interval.lift_end_time_side_in_ms

    metadata: dict[str, Any] = {
        "meta_status": status,
        "capture_id": record.capture_id,
        "group_id": record.group_id,
        "layout": record.layout.value,
        "lift": lift,
        "lift_type_in_source_file": _file_lift_type(record),
        "athlete": AthleteMeta().model_dump(),
        "derived_from_session_name": derived,
        "cameras": [_observed(camera) for camera in records],
    }

    if not config.dry_run:
        body = yaml.safe_dump(metadata, sort_keys=False, default_flow_style=False)
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(body, encoding="utf-8")
    return destinations
