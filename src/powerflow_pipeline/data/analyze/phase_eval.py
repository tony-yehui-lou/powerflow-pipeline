"""S9 Phase scored against hand labels (docs/specs/preprocessing/S9-phase-intervals-draft7.md §8.3).

Read-only. The pipeline never imports this module: the `phase-eval` CLI command does. Labels are
`<labels>/<captureId>/phases.json` (the adjudicated set) and `phases.pass<N>.json` (each pass,
§3.2); `phase.json` is S9's output under `<phases>/<captureId>/`. Both are matched by epoch ms.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from powerflow_pipeline.data.analyze.phase_events import EVENTS

LABEL_FILENAME: Final = "phases.json"
PHASE_FILENAME: Final = "phase.json"

Key = tuple[str, str]  # (lift, event)


@dataclass
class EventStats:
    """One event's absolute errors in ms, and how many `sure` labels S9 found."""

    errors_ms: list[float] = field(default_factory=list)
    sure: int = 0
    sure_found: int = 0

    @property
    def median_ms(self) -> float | None:
        return statistics.median(self.errors_ms) if self.errors_ms else None

    @property
    def p90_ms(self) -> float | None:
        return float(np.percentile(self.errors_ms, 90)) if self.errors_ms else None

    @property
    def found_share(self) -> float | None:
        return self.sure_found / self.sure if self.sure else None


@dataclass
class Report:
    events: dict[Key, EventStats]
    missing: list[str]  # labeled captures with no matching phase.json


def read_labels(path: Path) -> dict[str, Any]:
    """One label document, refused unless its events are exactly its lift's (§3.2)."""

    document: dict[str, Any] = json.loads(path.read_text())
    lift = document.get("lift")
    if lift not in EVENTS:
        raise ValueError(f"{path}: lift {lift!r} is not one of {sorted(EVENTS)}")
    if tuple(document.get("events", {})) != EVENTS[lift]:
        raise ValueError(f"{path}: events must be {list(EVENTS[lift])} for a {lift}")
    return document


def _stats(lift: str) -> dict[Key, EventStats]:
    return {(lift, event): EventStats() for event in EVENTS[lift]}  # type: ignore[index]


def score(labels_root: Path, phases_root: Path) -> Report:
    """Every adjudicated label set under `labels_root` against S9's output under `phases_root`.

    A labeled event's error is counted when both sides have a time. `sure` labels also count
    towards the found share; `unsure` ones only add errors.
    """

    events: dict[Key, EventStats] = {}
    missing: list[str] = []
    for path in sorted(labels_root.rglob(LABEL_FILENAME)):
        labels = read_labels(path)
        capture, lift = labels["captureId"], labels["lift"]
        predicted_path = phases_root / capture / PHASE_FILENAME
        predicted = json.loads(predicted_path.read_text()) if predicted_path.is_file() else None
        if predicted is None or predicted.get("lift") != lift:
            missing.append(capture)
            continue
        for key, stats in _stats(lift).items():
            events.setdefault(key, stats)
        for event, label in labels["events"].items():
            stats = events[(lift, event)]
            truth = label.get("epochMs")
            found = predicted["events"].get(event, {}).get("epochMs")
            if label.get("certainty") == "sure":
                stats.sure += 1
                stats.sure_found += found is not None
            if truth is not None and found is not None:
                stats.errors_ms.append(float(abs(found - truth)))
    return Report(events, missing)


def human_floor(labels_root: Path) -> dict[Key, EventStats]:
    """Per event, the gap between a capture's first two label passes (§3.3)."""

    floor: dict[Key, EventStats] = {}
    for first in sorted(labels_root.rglob("phases.pass1.json")):
        second = first.with_name("phases.pass2.json")
        if not second.is_file():
            continue
        a, b = read_labels(first), read_labels(second)
        for key, stats in _stats(a["lift"]).items():
            floor.setdefault(key, stats)
        for event, label in a["events"].items():
            ta, tb = label.get("epochMs"), b["events"][event].get("epochMs")
            if ta is not None and tb is not None:
                floor[(a["lift"], event)].errors_ms.append(float(abs(ta - tb)))
    return floor


def _ms(x: float | None) -> str:
    return "—" if x is None else f"{x:.0f}"


def markdown(report: Report, floor: dict[Key, EventStats]) -> str:
    """The §8.3 table: per lift and event, median and p90 error, found share, and human floor."""

    lines = [
        "| Lift | Event | n | Median ms | P90 ms | Found (sure) | Human floor median ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for (lift, event), stats in report.events.items():
        share = "—" if stats.found_share is None else f"{stats.found_share:.0%}"
        human = floor.get((lift, event))
        lines.append(
            f"| {lift} | {event} | {len(stats.errors_ms)} | {_ms(stats.median_ms)} | "
            f"{_ms(stats.p90_ms)} | {share} | {_ms(human.median_ms if human else None)} |"
        )
    if report.missing:
        lines += ["", "No matching phase.json: " + ", ".join(report.missing)]
    return "\n".join(lines)
