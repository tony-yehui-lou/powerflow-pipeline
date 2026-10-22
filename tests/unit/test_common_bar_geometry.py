"""The level-bar solve S6 places the barbell with, checked against synthetic 3D bars.

Every bar is built in 3D, projected through a known camera, and handed to the solver as pixels;
every assertion is against the 3D truth it was made from.
"""

from __future__ import annotations

import math

import pytest

from powerflow_pipeline.data.common import bar_geometry as bg
from powerflow_pipeline.data.common.capture_models import Intrinsics

K = Intrinsics(fx=1336.9, fy=1336.9, cx=705.0, cy=802.6, frame="portrait")
BAR = bg.BAR_TYPES["male IWF"]


def project(p: tuple[float, float, float]) -> tuple[float, float]:
    return (K.fx * p[0] / p[2] + K.cx, K.fy * p[1] / p[2] + K.cy)


def level_bar(
    centre: tuple[float, float, float], yaw_deg: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    a = (math.sin(math.radians(yaw_deg)), 0.0, math.cos(math.radians(yaw_deg)))
    cap = tuple(c - BAR.cap_to_centre_mm * ai for c, ai in zip(centre, a, strict=True))
    beyond = BAR.cap_to_collar_mm - BAR.cap_to_centre_mm
    collar = tuple(c + beyond * ai for c, ai in zip(centre, a, strict=True))
    return cap, collar  # type: ignore[return-value]


def assert_close(got: tuple[float, ...], want: tuple[float, ...], places: int = 6) -> None:
    for g, w in zip(got, want, strict=True):
        assert g == pytest.approx(w, abs=10**-places)


class TestSolveLevel:
    @pytest.mark.parametrize(
        ("centre", "yaw"),
        [
            ((-150.0, 1150.0, 2777.0), 38.6),  # bar below the camera, at rest
            ((-150.0, -520.0, 2600.0), 38.6),  # overhead, above the camera
            ((100.0, 900.0, 3000.0), 120.0),
            ((100.0, 900.0, 3000.0), -30.0),
        ],
    )
    def test_recovers_a_level_bar(self, centre: tuple[float, float, float], yaw: float) -> None:
        cap, collar = level_bar(centre, yaw)
        pose = bg.solve_level(project(cap), project(collar), K, BAR)
        assert pose is not None
        assert_close(pose.centre_mm, centre)
        assert_close(pose.collar_mm, collar)
        assert pose.yaw_deg == pytest.approx(yaw)

    def test_an_end_at_camera_height_has_no_solution(self) -> None:
        assert bg.solve_level((500.0, K.cy), (900.0, 1200.0), K, BAR) is None

    def test_ends_straddling_camera_height_have_no_solution(self) -> None:
        assert bg.solve_level((500.0, K.cy - 40), (900.0, K.cy + 40), K, BAR) is None


class TestYaw:
    def test_sigma_is_small_away_from_camera_height_and_large_near_it(self) -> None:
        far = level_bar((-150.0, 1150.0, 2777.0), 38.6)
        near = level_bar((-150.0, 15.0, 2777.0), 38.6)
        assert bg.yaw_sigma(project(far[0]), project(far[1]), K, BAR) < 1.0
        assert bg.yaw_sigma(project(near[0]), project(near[1]), K, BAR) > 5.0

    def test_fixed_yaw_solve_recovers_a_bar_at_camera_height(self) -> None:
        centre = (-150.0, 15.0, 2777.0)
        cap, collar = level_bar(centre, 38.6)
        pose = bg.solve_fixed_yaw(project(cap), project(collar), K, BAR, 38.6)
        assert pose is not None
        assert_close(pose.centre_mm, centre)

    def test_collar_only_solve_places_the_bar_from_collar_depth_and_yaw(self) -> None:
        centre = (-150.0, -500.0, 2600.0)
        _, collar = level_bar(centre, 38.6)
        pose = bg.solve_collar_only(project(collar), K, BAR, collar[2], 38.6)
        assert_close(pose.centre_mm, centre)

    def test_outliers_are_judged_against_their_neighbours(self) -> None:
        yaws: list[float | None] = [38.0] * 10 + [159.0] + [37.0] * 10
        assert bg.yaw_outliers(yaws) == [10]

    def test_too_few_neighbours_cannot_condemn(self) -> None:
        assert bg.yaw_outliers([38.0, 159.0, None, None]) == []


class TestFill:
    def test_interior_gap_is_linear(self) -> None:
        assert bg.fill_linear([10.0, None, None, 40.0]) == [10.0, 20.0, 30.0, 40.0]

    def test_edges_hold_unless_told_not_to(self) -> None:
        assert bg.fill_linear([None, 5.0, None]) == [5.0, 5.0, 5.0]
        assert bg.fill_linear([None, 5.0, None, 7.0, None], hold_edges=False) == [
            None,
            5.0,
            6.0,
            7.0,
            None,
        ]

    def test_gaps_longer_than_max_gap_stay_empty(self) -> None:
        assert bg.fill_linear([0.0, None, None, 3.0, None, 5.0], max_gap=1) == [
            0.0,
            None,
            None,
            3.0,
            4.0,
            5.0,
        ]

    def test_unwrap_never_sweeps_the_long_way_round(self) -> None:
        assert bg.unwrap_degrees([179.0, None, -179.0]) == [179.0, None, 181.0]


class TestTrack:
    def track(
        self, centres: list[tuple[float, float, float] | None], drop_caps: set[int] = frozenset()
    ) -> list[bg.BarPose | None]:
        caps: list[tuple[float, float] | None] = []
        collars: list[tuple[float, float] | None] = []
        for i, centre in enumerate(centres):
            if centre is None:
                caps.append(None)
                collars.append(None)
                continue
            cap, collar = level_bar(centre, 38.6)
            caps.append(None if i in drop_caps else project(cap))
            collars.append(project(collar))
        return bg.solve_track(caps, collars, K, BAR)

    def test_bar_passing_camera_height_keeps_its_place(self) -> None:
        heights = [600.0, 300.0, 15.0, -300.0, -600.0]
        poses = self.track([(-150.0, y, 2777.0) for y in heights])
        assert poses[2] is not None
        assert poses[2].source == "yaw-interpolated"
        assert_close(poses[2].centre_mm, (-150.0, 15.0, 2777.0), places=3)

    def test_cap_missing_frame_takes_collar_depth_from_its_flanks(self) -> None:
        centres = [(-150.0, 1150.0 - 200.0 * i, 2777.0 + 10.0 * i) for i in range(5)]
        poses = self.track(centres, drop_caps={2})
        assert poses[2] is not None
        assert poses[2].source == "collar-only"
        assert_close(poses[2].centre_mm, centres[2], places=3)

    def test_no_flank_on_one_side_means_no_pose(self) -> None:
        centres = [(-150.0, 1150.0, 2777.0)] * 3
        poses = self.track(centres, drop_caps={2})
        assert poses[2] is None

    def test_a_cap_whose_yaw_disagrees_with_its_neighbours_is_rejected(self) -> None:
        caps: list[tuple[float, float] | None] = []
        collars: list[tuple[float, float] | None] = []
        for i in range(9):
            cap, collar = level_bar((-150.0, 1150.0 - 50.0 * i, 2777.0), 38.6)
            caps.append(project(cap))
            collars.append(project(collar))
        caps[4] = (700.0, 820.0)  # solves to a confident bar pointing back at the camera
        poses = bg.solve_track(caps, collars, K, BAR)
        assert poses[4] is not None
        assert poses[4].source == "collar-only"


class TestFloorAnchor:
    """One floor for the bar and the joints (S8-joint-hva-draft3.md §4)."""

    @staticmethod
    def pose(y: float, source: bg.PoseSource = "level") -> bg.BarPose:
        cap, collar = level_bar((-150.0, y, 2777.0), 38.6)
        return bg.BarPose(cap, collar, (-150.0, y, 2777.0), 38.6, source)

    def test_rest_plate_anchor_is_the_rest_median_plus_one_plate_radius(self) -> None:
        bars = [self.pose(1150.0 + (i % 3) - 1) for i in range(60)] + [self.pose(0.0)] * 50

        anchor = bg.floor_anchor(bars, floor_offset_mm=1340.0)

        assert anchor.method == "rest-plate"
        assert anchor.y_floor_mm == pytest.approx(1150.0 + bg.PLATE_RADIUS_MM)
        assert anchor.rest_frames == 60
        assert anchor.floor_plane_y_mm == 1340.0

    def test_interpolated_collar_frames_never_set_rest(self) -> None:
        bars = [self.pose(1150.0, "collar-only")] * 30 + [self.pose(1150.0)] * 5

        anchor = bg.floor_anchor(bars, floor_offset_mm=1340.0)

        assert anchor.method == "floor-plane"
        assert anchor.y_floor_mm == 1340.0

    def test_no_bar_and_no_plane_anchors_nothing(self) -> None:
        anchor = bg.floor_anchor([None] * 70, floor_offset_mm=None)

        assert anchor.method == "none"
        assert anchor.y_floor_mm is None
