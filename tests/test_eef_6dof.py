"""Full agent-relative EEF safety and measured fake convergence; no hardware."""

import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_dobot.camera import FakeCamera
from inspect_robots_dobot.config import CameraConfig, DobotConfig, load_config
from inspect_robots_dobot.driver import FakeDobotDriver
from inspect_robots_dobot.embodiment import EEF_DIM_LABELS, DobotEmbodiment, build_info
from inspect_robots_dobot.errors import (
    CameraFault,
    ConfigurationError,
    DriverFault,
    MotionNotAuthorized,
    SafetyRejected,
    SettleTimeout,
)
from inspect_robots_dobot.safety import MotionAuthority, action_target
from inspect_robots_dobot.settle import wait_until_settled
from inspect_robots_dobot.transforms import (
    agent_relative_to_rotation,
    native_rotation,
    orientation_distance,
    rotation_to_dobot_native,
)
from inspect_robots_dobot.types import PoseSI, RobotMode

SCENE = Scene(id="6dof", instruction="Synthetic relative EEF tracking")


@pytest.fixture
def open_profile(profile):
    # FICTIONAL explicit fixture limits. Never installed as production defaults.
    return replace(profile, orientation_low=(-0.5, -0.3, -0.4), orientation_high=(0.5, 0.3, 0.4))


@pytest.fixture
def rig(open_profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=open_profile,
        clock=clock,
        authority=MotionAuthority(True),
    )
    emb = DobotEmbodiment(DobotConfig(open_profile, 10), driver=driver)
    emb.reset(SCENE)
    yield emb, driver
    emb.close()


def target_pose(pose, yaw=0, pitch=0, roll=0, **xyz):
    angles = rotation_to_dobot_native(
        agent_relative_to_rotation(yaw, pitch, roll, native_rotation(pose))
    )
    return replace(pose, rx=angles[0], ry=angles[1], rz=angles[2], **xyz)


def test_metadata_keeps_seven_slots_and_pins_orientation_axes(profile):
    info = build_info(DobotConfig(profile, 10))
    assert info.action_space.shape == (7,)
    assert info.action_space.semantics.dim_labels == EEF_DIM_LABELS
    assert info.action_space.semantics.control_mode == "eef_abs_pose"
    assert info.action_space.semantics.rotation_repr == "none"
    assert info.action_space.semantics.gripper == "continuous"
    np.testing.assert_array_equal(info.action_space.low[3:], [0, 0, 0, 0])
    np.testing.assert_array_equal(info.action_space.high[3:], [0, 0, 0, 1])
    assert info.action_space.semantics.max_step[3:] == (None, None, None, 0.1)
    assert info.observation_space.state.fields[0].shape == (7,)
    assert info.observation_space.state.fields[0].unit == "m+rad+normalized"


@pytest.mark.parametrize(
    "changes",
    [
        {"orientation_low": (-0.2, -0.3)},
        {"orientation_high": (0.2, float("nan"), 0.3)},
        {"orientation_low": (-4, -0.3, -0.2)},
        {"orientation_high": (0.2, 0.3, 4)},
        {"orientation_low": (0.1, 0, 0)},
        {"orientation_high": (-0.1, 0.2, 0.3)},
        {"orientation_low": (0, -math.pi / 2, 0)},
        {"orientation_high": (0, math.pi / 2, 0)},
        {"orientation_high": (0, 1.5, 0), "orientation_singularity_margin": 0.1},
        {"orientation_singularity_margin": 0},
        {"orientation_singularity_margin": float("nan")},
        {"orientation_singularity_margin": math.pi / 2},
    ],
)
def test_configuration_rejects_invalid_orientation(profile, changes):
    with pytest.raises(ConfigurationError):
        replace(profile, **changes)


def test_json_profile_orientation_is_explicit_and_old_config_pins(profile, open_profile, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"safety": asdict(open_profile), "control_hz": 10}))
    assert load_config(path).safety == open_profile
    old = asdict(profile)
    for key in ("orientation_low", "orientation_high", "orientation_singularity_margin"):
        del old[key]
    path.write_text(json.dumps({"safety": old, "control_hz": 10}))
    assert load_config(path).safety == profile


def test_zero_means_nontrivial_reference_and_reset_recaptures_measurement(rig, pose):
    emb, driver = rig
    result = emb.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert driver.get_pose().values[3:] == pose.values[3:]
    np.testing.assert_allclose(result.observation.state["eef_state"][3:], [0, 0, 0, 0])
    disturbed = target_pose(pose, yaw=0.2)
    driver.inject_pose(disturbed)
    measured = emb.reset(SCENE)
    np.testing.assert_array_equal(measured.state["eef_state"][3:], [0, 0, 0, 0])
    emb.step(Action(np.array([0.301, 0, 0.2, 0, 0, 0, 0])))
    assert orientation_distance(driver.get_pose(), disturbed) == 0


@pytest.mark.parametrize("index,label", [(3, "yaw"), (4, "pitch"), (5, "roll")])
def test_pinned_axes_reject_before_emission(embodiment, driver, index, label):
    embodiment.reset(SCENE)
    action = np.array([0.3, 0, 0.2, 0, 0, 0, 0])
    action[index] = 0.001
    assert f"{label}=" in embodiment.pre_check(action[None, :])
    with pytest.raises(SafetyRejected, match="pinned"):
        embodiment.step(Action(action))
    assert driver.commands == ()


def test_driver_independently_rejects_pinned_orientation(driver, pose):
    driver.connect()
    with pytest.raises(SafetyRejected, match="pinned"):
        driver.move_linear(target_pose(pose, yaw=0.01))
    assert driver.commands == ()


@pytest.mark.parametrize(
    "data,reason",
    [
        ([0.3, 0, 0.2, 0], "shape"),
        ([0.3, 0, 0.2, 0, 0, 0, -0.1], "outside"),
        ([0.3, 0, 0.2, 0.51, 0, 0, 0], "yaw"),
        ([0.3, 0, 0.2, 0, 0.42, 0, 0], "pitch=0.42"),
        ([0.3, 0, 0.2, 0, 0, -0.41, 0], "roll"),
        ([0.3, 0, 0.2, 0.2, 0, 0, 0], "orientation delta"),
        ([0.3, 0, 0.2, 0.07, 0.07, 0.07, 0], "orientation delta"),
        ([0.3, 0, 0.2, 0, float("inf"), 0, 0], "finite"),
        ([0.3, 0, 0.2, float("nan"), 0, 0, 0], "finite"),
    ],
)
def test_full_pose_validation_rejects_without_mutation(rig, data, reason):
    emb, driver = rig
    vector = np.array(data)
    original = vector.copy()
    assert reason in emb.pre_check(vector[None, :])
    with pytest.raises(SafetyRejected, match=reason):
        emb.step(Action(vector))
    np.testing.assert_array_equal(vector, original)
    assert driver.commands == ()


def test_precheck_later_waypoint_gives_correct_label_and_keeps_reference(rig, pose):
    emb, driver = rig
    valid = np.array([[0.3, 0, 0.2, 0.04, 0, 0, 0], [0.3, 0, 0.2, 0.08, 0.04, 0, 0]])
    assert emb.pre_check(valid) is None
    bad = np.concatenate((valid, [[0.3, 0, 0.2, 0.08, 0.42, 0, 0]]))
    reason = emb.pre_check(bad)
    assert "waypoint 3" in reason and "pitch=0.42" in reason and "0.3" in reason
    for data in valid:
        result = emb.step(Action(data))
    np.testing.assert_allclose(result.observation.state["eef_state"], valid[-1], atol=1e-14)
    assert orientation_distance(driver.get_pose(), target_pose(pose, yaw=0.08, pitch=0.04)) == 0


def test_rotational_drift_after_precheck_rechecks_measured_first_delta(rig, pose):
    emb, driver = rig
    data = np.array([0.3, 0, 0.2, 0.02, 0, 0, 0])
    assert emb.pre_check(data[None, :]) is None
    driver.inject_pose(target_pose(pose, yaw=0.3))
    with pytest.raises(SafetyRejected, match="orientation delta"):
        emb.step(Action(data))
    assert driver.commands == ()


def test_full_orientation_requires_fake_authority(open_profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose, initial_joints=(0.0,) * 6, profile=open_profile, clock=clock
    )
    driver.connect()
    with pytest.raises(MotionNotAuthorized):
        driver.move_linear(target_pose(pose, yaw=0.02))
    assert driver.commands == ()


def test_translation_arrival_does_not_mask_delayed_rotation(open_profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=open_profile,
        clock=clock,
        authority=MotionAuthority(True),
        convergence_delay=0.1,
        rotational_convergence_delay=0.3,
    )
    driver.connect()
    target = target_pose(pose, yaw=0.09, x=0.31)
    receipt = driver.move_linear(target)
    assert driver.get_pose() == pose
    clock.sleep(0.1)
    measured = driver.snapshot()
    assert measured.pose.x == 0.31
    assert measured.mode == RobotMode.RUNNING
    assert orientation_distance(pose, measured.pose) == pytest.approx(0.03)
    assert orientation_distance(measured.pose, target) == pytest.approx(0.06)
    result = wait_until_settled(driver, target, receipt, open_profile, clock)
    assert result.duration >= 0.19
    assert result.final.pose == target
    assert result.orientation_residual == 0


def test_rotation_timeout_is_visible_and_fault_latched(open_profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=open_profile,
        clock=clock,
        authority=MotionAuthority(True),
        rotational_convergence_delay=2,
    )
    emb = DobotEmbodiment(DobotConfig(open_profile, 10), driver=driver)
    emb.reset(SCENE)
    with pytest.raises(SettleTimeout):
        emb.step(Action(np.array([0.3, 0, 0.2, 0.09, 0, 0, 0])))
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]
    assert "fault-latched" in emb.pre_check(np.zeros((1, 7)))
    emb.close()


def test_controller_fault_during_rotational_settle(open_profile, pose, clock, monkeypatch):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=open_profile,
        clock=clock,
        authority=MotionAuthority(True),
        rotational_convergence_delay=0.2,
    )
    emb = DobotEmbodiment(DobotConfig(open_profile, 10), driver=driver)
    emb.reset(SCENE)
    sleep = clock.sleep

    def fault(seconds):
        sleep(seconds)
        driver.inject_fault(mode=RobotMode.ERROR, errors=(123,))

    monkeypatch.setattr(clock, "sleep", fault)
    with pytest.raises(DriverFault, match="123"):
        emb.step(Action(np.array([0.3, 0, 0.2, 0.09, 0, 0, 0])))
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]
    emb.close()


def test_postsettle_orientation_drift_inside_bounds_still_aborts(open_profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=open_profile,
        clock=clock,
        authority=MotionAuthority(True),
    )
    cfg = CameraConfig(8, 6, 0.2, 0.1)

    class DriftingCamera(FakeCamera):
        def latest(self, *, after=None):
            frame = super().latest(after=after)
            if after is not None:
                driver.inject_pose(target_pose(pose, yaw=0.08))
            return frame

    emb = DobotEmbodiment(
        DobotConfig(open_profile, 10, cfg), driver=driver, camera=DriftingCamera(cfg, clock)
    )
    emb.reset(SCENE)
    with pytest.raises(DriverFault, match="left the settled target"):
        emb.step(Action(np.array([0.3, 0, 0.2, 0.04, 0, 0, 0])))
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]
    emb.close()


def test_so3_step_wrap_and_native_encoding_are_not_raw_euler_differences(profile, clock):
    profile = replace(
        profile, orientation_low=(-math.pi, -0.3, -0.4), orientation_high=(math.pi, 0.3, 0.4)
    )
    pose = PoseSI(0.3, 0, 0.2, 0, 0, 0)
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(True),
    )
    driver.connect()
    driver.inject_pose(target_pose(pose, yaw=math.pi - 0.02))
    sample = driver.snapshot()
    target = action_target([0.3, 0, 0.2, -math.pi + 0.02, 0, 0, 0], sample, pose, profile)
    driver.move_linear(target)
    assert orientation_distance(sample.pose, driver.get_pose()) == pytest.approx(0.04)
    driver.close()


@pytest.mark.parametrize("delay", [-1, float("inf"), float("nan")])
def test_invalid_rotational_delay_rejects(profile, pose, clock, delay):
    with pytest.raises(ValueError, match="rotational"):
        FakeDobotDriver(
            initial_pose=pose,
            initial_joints=(0.0,) * 6,
            profile=profile,
            clock=clock,
            rotational_convergence_delay=delay,
        )


@pytest.mark.parametrize("change,reason", [(dict(yaw=0.6), "yaw"), (dict(z=0.04), "minimum")])
def test_settle_monitors_bounds_not_just_final_target(
    rig, pose, clock, monkeypatch, change, reason
):
    emb, driver = rig
    sleep = clock.sleep

    def drift(seconds):
        sleep(seconds)
        driver.inject_pose(target_pose(pose, **change))

    # Force a measured intermediate interval before immediate fake convergence.
    driver._rotation_delay = 0.2
    monkeypatch.setattr(clock, "sleep", drift)
    with pytest.raises(DriverFault, match=reason):
        emb.step(Action(np.array([0.3, 0, 0.2, 0.04, 0, 0, 0])))
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]


def test_measured_relative_singularity_is_not_hidden_by_state_echo(rig, pose):
    emb, driver = rig
    # Independently encode R_delta = Ry(-pi/2); pure helper deliberately excludes it.
    delta = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    rx, ry, rz = rotation_to_dobot_native(delta @ native_rotation(pose))
    driver.inject_pose(replace(pose, rx=rx, ry=ry, rz=rz))
    with pytest.raises(SafetyRejected, match="singularity"):
        emb.step(Action(np.array([0.3, 0, 0.2, 0, 0, 0, 0])))
    assert driver.commands == ()


def test_rotated_target_reachability_injection_records_no_command(rig):
    emb, driver = rig
    driver._reachable = lambda target: False
    with pytest.raises(SafetyRejected, match="reachability"):
        emb.step(Action(np.array([0.3, 0, 0.2, 0.04, 0, 0, 0])))
    assert not [c for c in driver.commands if c.name == "move_linear"]
    assert emb.audit_records[-1]["status"] == "failed"


def test_stale_camera_after_rotational_arrival_does_not_report_success(open_profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=open_profile,
        clock=clock,
        authority=MotionAuthority(True),
        rotational_convergence_delay=0.15,
    )
    cfg = CameraConfig(8, 6, 0.2, 0.1)
    camera = FakeCamera(cfg, clock)
    emb = DobotEmbodiment(DobotConfig(open_profile, 10, cfg), driver=driver, camera=camera)
    emb.reset(SCENE)
    camera.freeze = True
    with pytest.raises(CameraFault):
        emb.step(Action(np.array([0.3, 0, 0.2, 0.04, 0, 0, 0])))
    assert orientation_distance(driver.get_pose(), target_pose(pose, yaw=0.04)) == 0
    assert emb.audit_records[-1]["robot_settled"] is True
    assert emb.audit_records[-1]["status"] == "failed"
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]
    emb.close()
