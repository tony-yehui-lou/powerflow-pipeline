"""Pure landmark->joint mapping. No mediapipe import, no I/O -- synthetic inputs only."""

from __future__ import annotations

import pytest

from powerflow_pipeline.data.common.models import HUMAN_SKELETON
from powerflow_pipeline.data.preprocess.pose_landmark_mapping import (
    LEFT_ANKLE,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_SHOULDER,
    map_landmarks_to_joints,
)

_ALL_33_LANDMARKS = {index: (float(index), float(index) * 2, 0.9) for index in range(33)}


def test_maps_all_direct_joints_when_every_landmark_is_present() -> None:
    joints = map_landmarks_to_joints(_ALL_33_LANDMARKS)

    assert joints["leftAnkle"] == _ALL_33_LANDMARKS[LEFT_ANKLE]
    assert joints["rightAnkle"] == _ALL_33_LANDMARKS[RIGHT_ANKLE]
    assert joints["head"] == _ALL_33_LANDMARKS[NOSE]


def test_produces_every_skeleton_joint_when_all_landmarks_present() -> None:
    joints = map_landmarks_to_joints(_ALL_33_LANDMARKS)

    assert set(joints) == set(HUMAN_SKELETON.joints)


def test_clavicle_is_interpolated_between_shoulder_and_nose() -> None:
    landmarks = {
        NOSE: (0.0, 0.0, 1.0),
        LEFT_SHOULDER: (10.0, 10.0, 1.0),
        RIGHT_SHOULDER: (20.0, 10.0, 1.0),
    }

    joints = map_landmarks_to_joints(landmarks)

    # Weighted 0.7 toward the shoulder, 0.3 toward the nose.
    assert joints["leftClavicle"] == pytest.approx((7.0, 7.0, 1.0))
    assert joints["rightClavicle"] == pytest.approx((14.0, 7.0, 1.0))


def test_clavicle_confidence_is_the_minimum_of_its_two_sources() -> None:
    landmarks = {
        NOSE: (0.0, 0.0, 0.4),
        LEFT_SHOULDER: (10.0, 10.0, 0.9),
    }

    joints = map_landmarks_to_joints(landmarks)

    assert joints["leftClavicle"][2] == pytest.approx(0.4)


def test_missing_landmark_drops_the_joint_rather_than_fabricating_one() -> None:
    landmarks = dict(_ALL_33_LANDMARKS)
    del landmarks[LEFT_ANKLE]

    joints = map_landmarks_to_joints(landmarks)

    assert "leftAnkle" not in joints
    assert "rightAnkle" in joints


def test_missing_nose_drops_both_clavicles() -> None:
    landmarks = dict(_ALL_33_LANDMARKS)
    del landmarks[NOSE]

    joints = map_landmarks_to_joints(landmarks)

    assert "leftClavicle" not in joints
    assert "rightClavicle" not in joints
    assert "head" not in joints


def test_empty_landmarks_produces_no_joints() -> None:
    assert map_landmarks_to_joints({}) == {}
