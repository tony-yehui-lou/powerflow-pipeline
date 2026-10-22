"""S9 Phase scored against hand labels (S9-phase-intervals-draft7.md §3.2, §8.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from powerflow_pipeline.data.cli import app
from powerflow_pipeline.data.preprocess import phase_eval

SNATCH = ("liftoff", "barPassesKnee", "fullExtension", "catch", "stand")


def _label(root: Path, capture: str, times: dict[str, int | None], *, name: str = "phases.json",
           certainty: str = "sure", lift: str = "snatch") -> None:  # fmt: skip
    events = {
        k: {"epochMs": v, "certainty": "absent" if v is None else certainty}
        for k, v in times.items()
    }
    path = root / capture / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schemaVersion": "1.1.0", "captureId": capture, "lift": lift,
                                "events": events, "result": "made"}))  # fmt: skip


def _phase(root: Path, capture: str, times: dict[str, int | None], lift: str = "snatch") -> None:
    events = {
        k: {"epochMs": v, "status": "absent" if v is None else "found"} for k, v in times.items()
    }
    path = root / capture / "phase.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"captureId": capture, "lift": lift, "events": events}))


def test_scores_each_event_against_the_labels(tmp_path: Path) -> None:
    labels, phases = tmp_path / "labels", tmp_path / "phases"
    for i, capture in enumerate(["10 September/30kg1", "10 September/30kg2"]):
        _label(labels, capture, {e: 1000 * (k + 1) for k, e in enumerate(SNATCH)})
        found = {e: 1000 * (k + 1) + 10 * (i + 1) for k, e in enumerate(SNATCH)}
        found["stand"] = None if i == 1 else found["stand"]
        _phase(phases, capture, found)

    report = phase_eval.score(labels, phases)

    liftoff = report.events[("snatch", "liftoff")]
    assert liftoff.errors_ms == [10.0, 20.0]
    assert liftoff.median_ms == pytest.approx(15.0)
    assert liftoff.p90_ms == pytest.approx(19.0)
    assert liftoff.found_share == 1.0
    stand = report.events[("snatch", "stand")]
    assert stand.errors_ms == [10.0]
    assert stand.found_share == 0.5
    assert report.missing == []


def test_unsure_labels_are_scored_but_not_counted_for_found_share(tmp_path: Path) -> None:
    labels, phases = tmp_path / "labels", tmp_path / "phases"
    _label(labels, "c", {e: 1000 for e in SNATCH}, certainty="unsure")
    _phase(phases, "c", {e: None for e in SNATCH})

    stats = phase_eval.score(labels, phases).events[("snatch", "liftoff")]
    assert stats.errors_ms == []
    assert stats.found_share is None


def test_a_label_without_a_phase_file_is_reported_missing(tmp_path: Path) -> None:
    labels, phases = tmp_path / "labels", tmp_path / "phases"
    _label(labels, "10 September/40kg1", {e: 1000 for e in SNATCH})

    assert phase_eval.score(labels, phases).missing == ["10 September/40kg1"]


def test_a_lift_mismatch_is_reported_missing(tmp_path: Path) -> None:
    labels, phases = tmp_path / "labels", tmp_path / "phases"
    _label(labels, "c", {e: 1000 for e in SNATCH})
    _phase(phases, "c", {e: 1000 for e in SNATCH}, lift="clean_and_jerk")

    assert phase_eval.score(labels, phases).missing == ["c"]


def test_a_label_file_with_the_wrong_events_is_refused(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    _label(labels, "c", {"liftoff": 1000})

    with pytest.raises(ValueError, match="events"):
        phase_eval.read_labels(labels / "c" / "phases.json")


def test_the_human_floor_is_the_gap_between_two_passes(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    _label(labels, "c", {e: 1000 for e in SNATCH}, name="phases.pass1.json")
    _label(labels, "c", {e: 1030 for e in SNATCH}, name="phases.pass2.json")
    _label(labels, "d", {e: 1000 for e in SNATCH}, name="phases.pass1.json")  # one pass only

    floor = phase_eval.human_floor(labels)
    assert floor[("snatch", "catch")].errors_ms == [30.0]


def test_markdown_lists_every_event(tmp_path: Path) -> None:
    labels, phases = tmp_path / "labels", tmp_path / "phases"
    _label(labels, "c", {e: 1000 for e in SNATCH})
    _phase(phases, "c", {e: 1020 for e in SNATCH})

    text = phase_eval.markdown(phase_eval.score(labels, phases), phase_eval.human_floor(labels))
    for event in SNATCH:
        assert event in text
    assert "20" in text


def test_the_cli_prints_the_report(tmp_path: Path) -> None:
    labels, phases = tmp_path / "labels", tmp_path / "phases"
    _label(labels, "c", {e: 1000 for e in SNATCH})
    _phase(phases, "c", {e: 1020 for e in SNATCH})

    result = CliRunner().invoke(
        app, ["phase-eval", "--labels", str(labels), "--phases", str(phases)]
    )

    assert result.exit_code == 0, result.output
    assert "fullExtension" in result.output
