"""Regression evidence recorded before introducing any real read-only transport."""

from dataclasses import replace

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_dobot.errors import SafetyRejected


def test_first_waypoint_uses_latest_observation_not_previous_target(embodiment, driver, pose):
    scene = Scene(id="audit", instruction="Offline audit")
    embodiment.reset(scene)
    driver.inject_pose(replace(pose, x=0.4))
    # reset is read-only and captures the disturbed measured pose.
    observation = embodiment.reset(scene)
    assert observation.state["eef_state"][0] == 0.4
    reason = embodiment.pre_check(np.array([[0.301, 0, 0.2, 0, 0, 0, 0]]))
    assert "waypoint 1" in reason and "translation delta" in reason
    assert embodiment.audit_records[-1]["reference_pose"][0] == 0.4
    assert driver.commands == ()


def test_execution_first_delta_rechecks_translation_after_precheck(embodiment, driver, pose):
    embodiment.reset(Scene(id="audit", instruction="Offline audit"))
    action = Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0]))
    assert embodiment.pre_check(np.array([action.data])) is None
    driver.inject_pose(replace(pose, x=0.4))
    with pytest.raises(SafetyRejected, match="translation delta"):
        embodiment.step(action)
    assert driver.commands == ()
