"""Pure epoch-time maths for the Cut stage. No I/O, no fixtures beyond plain numbers."""

from __future__ import annotations

import pytest

from powerflow_pipeline.data.preprocess.timeline import (
    creation_time_to_epoch_ms,
    derive_cut_interval,
    epoch_ms_series,
    select_closed_indices,
)


class TestCreationTimeToEpochMs:
    def test_converts_zulu_iso_with_microseconds(self) -> None:
        # 2026-07-09T10:04:15.000000Z is the tag the real rgb.mp4 carries.
        assert creation_time_to_epoch_ms("2026-07-09T10:04:15.000000Z") == 1783591455000

    def test_preserves_sub_second_precision(self) -> None:
        # The existing `_epoch` helper in metadata.py floors to whole seconds; this must not.
        assert creation_time_to_epoch_ms("2026-07-09T10:04:15.250000Z") == 1783591455250

    def test_rejects_unparseable_string(self) -> None:
        with pytest.raises(ValueError):
            creation_time_to_epoch_ms("not-a-timestamp")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError):
            creation_time_to_epoch_ms("")


class TestDeriveCutInterval:
    def test_adds_ms_offsets_to_side_creation_epoch(self) -> None:
        start, end = derive_cut_interval(
            side_created_epoch_ms=1783764255000, lift_start_ms=30090, lift_end_ms=90780
        )
        assert (start, end) == (1783764285090, 1783764345780)


class TestEpochMsSeries:
    def test_anchor_sample_maps_to_created_epoch_exactly(self) -> None:
        # Row 0 of odometry (the anchor) must land exactly on created_epoch_ms.
        series = epoch_ms_series(
            created_epoch_ms=1_000_000, anchor_s=8449.489203791, samples_s=[8449.489203791]
        )
        assert series == [1_000_000]

    def test_later_samples_advance_by_elapsed_uptime(self) -> None:
        series = epoch_ms_series(
            created_epoch_ms=1_000_000,
            anchor_s=8449.489203791,
            samples_s=[8449.489203791, 8449.505877250],
        )
        # elapsed = 0.016673459s -> 16.673459ms -> rounds to 17ms
        assert series == [1_000_000, 1_000_017]

    def test_zero_anchor_is_a_pass_through_for_pts_seconds(self) -> None:
        # RGB frames: pts is already zero-based relative time, so anchor_s=0.0 works directly.
        series = epoch_ms_series(created_epoch_ms=500, anchor_s=0.0, samples_s=[0.0, 1.5, 3.0])
        assert series == [500, 2000, 3500]

    def test_samples_before_anchor_go_negative(self) -> None:
        # IMU can start before odometry's first row; that offset must be preserved, not clamped.
        series = epoch_ms_series(created_epoch_ms=1_000_000, anchor_s=10.0, samples_s=[9.9])
        assert series == [999_900]

    def test_empty_samples_returns_empty_list(self) -> None:
        assert epoch_ms_series(created_epoch_ms=1_000_000, anchor_s=10.0, samples_s=[]) == []


class TestSelectClosedIndices:
    def test_includes_samples_exactly_on_either_bound(self) -> None:
        epochs = [90, 100, 150, 200, 210]
        assert select_closed_indices(epochs, start_ms=100, end_ms=200) == [1, 2, 3]

    def test_excludes_samples_outside_bounds(self) -> None:
        epochs = [0, 50, 300]
        assert select_closed_indices(epochs, start_ms=100, end_ms=200) == []

    def test_never_fabricates_indices_for_gaps(self) -> None:
        # An irregular stream (dropped frames) must not be resampled onto a lattice.
        epochs = [95, 101, 250, 300]
        assert select_closed_indices(epochs, start_ms=100, end_ms=260) == [1, 2]

    def test_empty_epochs_returns_empty_list(self) -> None:
        assert select_closed_indices([], start_ms=0, end_ms=100) == []
