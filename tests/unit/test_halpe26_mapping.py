"""Pure Halpe-26 -> JointId mapping. No rtmlib import, no I/O -- synthetic inputs only."""

from __future__ import annotations

import pytest

from powerflow_pipeline.data.common.models import HUMAN_SKELETON
from powerflow_pipeline.data.preprocess.halpe26_mapping import (
    HEAD,
    LEFT_ANKLE,
    LEFT_SHOULDER,
    N_KEYPOINTS,
    NECK,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_SHOULDER,
    map_keypoints_to_joints,
)

_ALL = {index: (float(index), float(index) * 2, 0.9) for index in range(N_KEYPOINTS)}


def test_maps_every_skeleton_joint_when_all_keypoints_present() -> None:
    joints = map_keypoints_to_joints(_ALL)

    assert set(joints) == set(HUMAN_SKELETON.joints)


def test_direct_joints_come_straight_from_their_keypoint() -> None:
    joints = map_keypoints_to_joints(_ALL)

    assert joints["leftAnkle"] == _ALL[LEFT_ANKLE]
    assert joints["rightAnkle"] == _ALL[RIGHT_ANKLE]


def test_head_uses_the_head_keypoint_not_the_nose() -> None:
    # The whole reason for choosing Halpe-26 over COCO-17: a real head point.
    joints = map_keypoints_to_joints(_ALL)

    assert joints["head"] == _ALL[HEAD]
    assert joints["head"] != _ALL[NOSE]


def test_clavicle_is_interpolated_between_neck_and_shoulder() -> None:
    keypoints = {
        NECK: (100.0, 100.0, 1.0),
        LEFT_SHOULDER: (200.0, 140.0, 1.0),
        RIGHT_SHOULDER: (0.0, 140.0, 1.0),
    }

    joints = map_keypoints_to_joints(keypoints, clavicle_shoulder_weight=0.5)

    assert joints["leftClavicle"] == pytest.approx((150.0, 120.0, 1.0))
    assert joints["rightClavicle"] == pytest.approx((50.0, 120.0, 1.0))


def test_clavicle_weight_is_a_parameter() -> None:
    keypoints = {NECK: (0.0, 0.0, 1.0), LEFT_SHOULDER: (100.0, 0.0, 1.0)}

    at_neck = map_keypoints_to_joints(keypoints, clavicle_shoulder_weight=0.0)
    at_shoulder = map_keypoints_to_joints(keypoints, clavicle_shoulder_weight=1.0)

    assert at_neck["leftClavicle"][0] == pytest.approx(0.0)
    assert at_shoulder["leftClavicle"][0] == pytest.approx(100.0)


def test_clavicle_score_is_the_minimum_of_its_two_sources() -> None:
    keypoints = {NECK: (0.0, 0.0, 0.3), LEFT_SHOULDER: (100.0, 0.0, 0.9)}

    joints = map_keypoints_to_joints(keypoints)

    assert joints["leftClavicle"][2] == pytest.approx(0.3)


def test_missing_keypoint_drops_the_joint_rather_than_fabricating_one() -> None:
    keypoints = dict(_ALL)
    del keypoints[LEFT_ANKLE]

    joints = map_keypoints_to_joints(keypoints)

    assert "leftAnkle" not in joints
    assert "rightAnkle" in joints


def test_missing_neck_drops_both_clavicles_but_keeps_the_head() -> None:
    keypoints = dict(_ALL)
    del keypoints[NECK]

    joints = map_keypoints_to_joints(keypoints)

    assert "leftClavicle" not in joints
    assert "rightClavicle" not in joints
    assert "head" in joints  # unlike BlazePose, head does not depend on the neck


def test_missing_shoulder_drops_only_that_clavicle() -> None:
    keypoints = dict(_ALL)
    del keypoints[RIGHT_SHOULDER]

    joints = map_keypoints_to_joints(keypoints)

    assert "leftClavicle" in joints
    assert "rightClavicle" not in joints


def test_empty_keypoints_produces_no_joints() -> None:
    assert map_keypoints_to_joints({}) == {}
