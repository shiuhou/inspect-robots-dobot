import json
from dataclasses import replace

import numpy as np
import pytest
from inspect_robots.conformance import check_embodiment
from inspect_robots.embodiment import Embodiment
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_dobot.camera import FakeCamera
from inspect_robots_dobot.config import CameraConfig, DobotConfig
from inspect_robots_dobot.driver import FakeDobotDriver
from inspect_robots_dobot.embodiment import DobotEmbodiment
from inspect_robots_dobot.errors import (
    CameraFault,
    ConfigurationError,
    DriverFault,
    MotionNotAuthorized,
    PhaseUnavailable,
    SafetyRejected,
    SettleTimeout,
)
from inspect_robots_dobot.safety import MotionAuthority

SCENE = Scene(id="offline", instruction="Move a synthetic arm")


def test_current_contract_and_no_motion_in_lifecycle(embodiment, driver, pose):
    assert isinstance(embodiment, Embodiment)
    report = check_embodiment(embodiment.info)
    assert report.ok
    assert {issue.code for issue in report.issues} == {"zero_width"}
    observation = embodiment.reset(SCENE)
    assert observation.instruction == SCENE.instruction
    assert observation.images == {}
    assert observation.image_times == {}
    assert observation.state_time == 1
    np.testing.assert_allclose(observation.state["eef_state"], [*pose.xyz, 0, 0, 0, 0])
    np.testing.assert_allclose(observation.state["dobot_native_pose"], pose.values)
    assert observation.state["joint_pos"].shape == (6,)
    assert observation.extra["dobot"]["camera"]["available"] is False
    assert embodiment.info.action_space.semantics.gripper == "continuous"
    assert driver.commands == ()
    embodiment.close()
    assert driver.commands == ()


def test_fake_step_settles_paces_holds_orientation_and_audits(embodiment, driver, pose, clock):
    embodiment.reset(SCENE)
    result = embodiment.step(Action(np.array([0.305, 0, 0.2, 0, 0, 0, 0])))
    np.testing.assert_allclose(result.observation.state["eef_state"], [0.305, 0, 0.2, 0, 0, 0, 0])
    assert driver.get_pose().values[3:] == pose.values[3:]
    assert clock.monotonic() == pytest.approx(1.1)
    audit = result.info["dobot_motion"]
    assert audit["status"] == "settled"
    assert audit["target_native"][0] == pytest.approx(305)
    assert audit["position_residual"] == 0
    assert audit["physical_authorized"] is False
    assert audit["physical_sent"] is False
    assert audit["protocol_command"] is None
    assert audit["controller_response"] is None
    assert result.reward is None and not result.terminated
    result.info["dobot_motion"]["status"] = "mutated"
    assert embodiment.audit_records[-1]["status"] == "settled"


def test_default_authority_rejects_even_with_valid_profile(profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose, initial_joints=(0.0,) * 6, profile=profile, clock=clock
    )
    emb = DobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    emb.reset(SCENE)
    with pytest.raises(MotionNotAuthorized):
        emb.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert driver.commands == ()
    assert emb.audit_records[-1]["status"] == "failed"
    emb.close()


def test_no_profile_or_no_driver_cannot_start(profile):
    with pytest.raises(ConfigurationError, match="explicit"):
        DobotEmbodiment().reset(SCENE)
    with pytest.raises(PhaseUnavailable, match="no fake driver"):
        DobotEmbodiment(DobotConfig(profile, 10)).reset(SCENE)


def test_config_profile_mismatch_rejected(profile, driver):
    with pytest.raises(ConfigurationError, match="profiles must match"):
        DobotEmbodiment(DobotConfig(replace(profile, speed_percent=6), 10), driver=driver)


def test_first_action_large_jump_rejected_without_approver_reference(embodiment, driver):
    embodiment.reset(SCENE)
    with pytest.raises(SafetyRejected, match="translation delta"):
        embodiment.step(Action(np.array([0.4, 0, 0.2, 0, 0, 0, 0])))
    assert driver.commands == ()


def test_framework_clamped_action_is_rejected(embodiment, driver):
    embodiment.reset(SCENE)
    with pytest.raises(SafetyRejected, match="modified"):
        embodiment.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0]), {"clamped": True}))
    assert driver.commands == ()


def test_pose_drift_after_precheck_cannot_bypass_execution_checks(embodiment, driver, pose):
    embodiment.reset(SCENE)
    assert embodiment.pre_check(np.array([[0.301, 0, 0.2, 0, 0, 0, 0]])) is None
    driver.inject_pose(replace(pose, rz=pose.rz + 0.1))
    with pytest.raises(SafetyRejected, match="drift"):
        embodiment.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert driver.commands == ()


def test_timeout_stops_once_latches_fault_no_retry(profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(True),
        never_converge=True,
    )
    emb = DobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    emb.reset(SCENE)
    action = Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0]))
    with pytest.raises(SettleTimeout):
        emb.step(action)
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]
    assert emb.audit_records[-1]["stop_requested"]
    with pytest.raises(DriverFault, match="fault-latched"):
        emb.step(action)
    with pytest.raises(DriverFault, match="fault-latched"):
        emb.reset(SCENE)
    assert "fault-latched" in emb.pre_check(np.array([action.data]))
    assert len(driver.commands) == 2
    emb.close()


def test_camera_frame_must_be_after_settle(profile, driver, clock):
    config = CameraConfig(8, 6, 0.2, 0.1)
    camera = FakeCamera(config, clock)
    emb = DobotEmbodiment(DobotConfig(profile, 10, config), driver=driver, camera=camera)
    initial = emb.reset(SCENE)
    result = emb.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert result.observation.image_times["table"] > result.info["dobot_motion"]["settled_at"]
    assert result.observation.image_times["table"] > initial.image_times["table"]
    assert result.observation.extra["dobot"]["camera"]["exposure_time_verified"] is False
    emb.close()


def test_stale_camera_after_successful_settle_aborts(profile, driver, clock):
    config = CameraConfig(8, 6, 0.2, 0.1)
    camera = FakeCamera(config, clock)
    emb = DobotEmbodiment(DobotConfig(profile, 10, config), driver=driver, camera=camera)
    emb.reset(SCENE)
    camera.freeze = True
    with pytest.raises(CameraFault):
        emb.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert driver.get_pose().x == pytest.approx(0.301)
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]
    assert emb.audit_records[-1]["status"] == "failed"
    emb.close()


def test_configured_but_missing_camera_not_silently_ignored(profile, driver):
    emb = DobotEmbodiment(DobotConfig(profile, 10, CameraConfig(8, 6, 0.2, 0.1)), driver=driver)
    with pytest.raises(ConfigurationError, match="camera"):
        emb.reset(SCENE)


def test_drift_while_waiting_for_postsettle_image_aborts(profile, driver, clock, pose):
    config = CameraConfig(8, 6, 0.2, 0.1)

    class DisturbingCamera(FakeCamera):
        def latest(self, *, after=None):
            frame = super().latest(after=after)
            if after is not None:
                driver.inject_pose(replace(pose, x=0.31))
            return frame

    camera = DisturbingCamera(config, clock)
    emb = DobotEmbodiment(DobotConfig(profile, 10, config), driver=driver, camera=camera)
    emb.reset(SCENE)
    with pytest.raises(DriverFault, match="left the settled target"):
        emb.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]
    assert emb.audit_records[-1]["robot_settled"] is True
    assert emb.audit_records[-1]["final_pose"][0] == 0.31
    emb.close()


def test_orientation_drift_while_observing_aborts(profile, driver, clock, pose):
    config = CameraConfig(8, 6, 0.2, 0.1)

    class RotatingCamera(FakeCamera):
        def latest(self, *, after=None):
            frame = super().latest(after=after)
            if after is not None:
                driver.inject_pose(replace(pose, x=0.301, rz=1.0))
            return frame

    emb = DobotEmbodiment(
        DobotConfig(profile, 10, config), driver=driver, camera=RotatingCamera(config, clock)
    )
    emb.reset(SCENE)
    with pytest.raises(DriverFault, match="orientation changed"):
        emb.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    emb.close()


def test_lost_acknowledgement_stops_and_latches_without_retry(embodiment, driver, monkeypatch):
    embodiment.reset(SCENE)
    original = driver.move_linear

    def lose_ack(target):
        original(target)
        raise DriverFault("simulated lost acknowledgement after acceptance")

    monkeypatch.setattr(driver, "move_linear", lose_ack)
    with pytest.raises(DriverFault, match="lost acknowledgement"):
        embodiment.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]
    record = embodiment.audit_records[-1]
    assert record["acknowledgement_received"] is False
    assert record["stop_requested"] is True
    with pytest.raises(DriverFault, match="fault-latched"):
        embodiment.step(Action(np.array([0.302, 0, 0.2, 0, 0, 0, 0])))


def test_stop_failure_preserves_original_fault_and_records_failure(embodiment, driver, monkeypatch):
    embodiment.reset(SCENE)

    def lost_connection(target):
        driver.close()
        raise DriverFault("uncertain command result")

    monkeypatch.setattr(driver, "move_linear", lost_connection)
    with pytest.raises(DriverFault, match="uncertain command result"):
        embodiment.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert "not connected" in embodiment.audit_records[-1]["stop_error"]
    assert "not connected" in embodiment.audit_records[-1]["final_state_error"]


def test_nonfinite_rejection_is_json_serializable_and_contains_reference(embodiment):
    embodiment.reset(SCENE)
    with pytest.raises(SafetyRejected):
        embodiment.step(Action(np.array([float("nan"), 0, 0.2, 0, 0, 0, 0])))
    record = embodiment.audit_records[-1]
    assert record["requested_action"][0] == "nan"
    assert record["observation_pose"][0] == 0.3
    json.dumps(record, allow_nan=False)


def test_interrupt_during_wait_stops_and_propagates(embodiment, driver, clock, monkeypatch):
    embodiment.reset(SCENE)

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(clock, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        embodiment.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]


def test_step_before_reset_does_not_send(embodiment, driver):
    with pytest.raises(DriverFault, match="reset"):
        embodiment.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert driver.commands == ()


def test_reset_error_does_not_clear_or_enable(profile, driver):
    from inspect_robots_dobot.types import RobotMode

    driver.inject_fault(mode=RobotMode.ERROR, errors=(123,))
    emb = DobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    with pytest.raises(SafetyRejected, match="alarms"):
        emb.reset(SCENE)
    assert driver.commands == ()


def test_real_driver_injection_is_rejected(profile):
    with pytest.raises(PhaseUnavailable, match="only accepts Fake"):
        DobotEmbodiment(DobotConfig(profile, 10), driver=object())
