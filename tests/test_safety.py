import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest
from inspect_robots.types import Action

from inspect_robots_dobot.config import CameraConfig, DobotConfig, load_config
from inspect_robots_dobot.errors import ConfigurationError, SafetyRejected
from inspect_robots_dobot.safety import (
    DobotApprover,
    WaypointPreCheck,
    action_target,
    validate_target,
)
from inspect_robots_dobot.types import RobotMode


@pytest.fixture
def sample(driver):
    driver.connect()
    return driver.snapshot()


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"x": 0.6}, "outside"),
        ({"z": 0.041}, "minimum z=0.08"),
        ({"x": 0.34}, "translation delta"),
        ({"x": 0.315, "y": 0.015}, "translation delta"),
        ({"rz": 0.5}, "orientation delta"),
        ({"x": float("nan")}, "NaN/Inf"),
        ({"ry": float("inf")}, "NaN/Inf"),
        ({"z": -float("inf")}, "NaN/Inf"),
    ],
)
def test_target_rejections(profile, sample, pose, clock, change, reason):
    with pytest.raises(SafetyRejected, match=reason):
        validate_target(replace(pose, **change), sample, profile, clock.monotonic())


@pytest.mark.parametrize(
    "mode",
    [
        RobotMode.INIT,
        RobotMode.BRAKE_OPEN,
        RobotMode.POWER_OFF,
        RobotMode.DISABLED,
        RobotMode.DRAG,
        RobotMode.RUNNING,
        RobotMode.SINGLE_MOVE,
        RobotMode.ERROR,
        RobotMode.PAUSED,
        RobotMode.COLLISION,
    ],
)
def test_mode_must_be_idle(profile, sample, pose, clock, mode):
    with pytest.raises(SafetyRejected, match="mode"):
        validate_target(pose, replace(sample, mode=mode), profile, clock.monotonic())


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"errors": (123,)}, "alarms"),
        ({"observed_at": -1}, "age"),
        ({"observed_at": float("nan")}, "timestamp"),
        ({"observed_at": 10}, "age"),
        ({"tool_frame": 1}, "frames"),
        ({"user_frame": 2}, "frames"),
        ({"joints": (float("nan"), 0, 0, 0, 0, 0)}, "NaN/Inf"),
    ],
)
def test_measurement_rejections(profile, sample, pose, clock, change, reason):
    with pytest.raises(SafetyRejected, match=reason):
        validate_target(pose, replace(sample, **change), profile, clock.monotonic())


def test_boundary_target_and_units_remain_unchanged(profile, sample, pose, clock):
    target = replace(pose, x=0.31)
    assert validate_target(target, sample, profile, clock.monotonic()) == pytest.approx((0.01, 0))
    with pytest.raises(SafetyRejected):
        validate_target(replace(pose, x=310), sample, profile, clock.monotonic())


@pytest.mark.parametrize(
    "values",
    [
        [0.3, 0, 0.2],
        [[0.3, 0, 0.2, 0, 0, 0, 0]],
        [0.3, 0, 0.2, 0, 0, 0, -0.1],
        [0.3, 0, float("nan"), 0, 0, 0, 0],
        ["bad", 0, 0.2, 0, 0, 0, 0],
    ],
)
def test_action_shape_finiteness_gripper(profile, sample, pose, values):
    with pytest.raises(SafetyRejected):
        action_target(values, sample, pose, profile)


def test_drift_rejected_not_adopted(profile, sample, pose):
    moved = replace(sample, pose=replace(pose, rz=pose.rz + 0.02))
    with pytest.raises(SafetyRejected, match="drift"):
        action_target([0.3, 0, 0.2, 0, 0, 0, 0], moved, pose, profile)


def test_precheck_checks_first_and_every_waypoint(profile, sample, pose, clock):
    check = WaypointPreCheck(lambda: sample, lambda: pose, profile, clock.monotonic)
    valid = np.array([[0.305, 0, 0.2, 0, 0, 0, 0], [0.31, 0, 0.2, 0, 0, 0, 0]])
    valid.flags.writeable = False
    assert check(valid) is None
    assert "waypoint 1" in check(np.array([[0.4, 0, 0.2, 0, 0, 0, 0]]))
    assert "waypoint 2" in check(
        np.array([[0.305, 0, 0.2, 0, 0, 0, 0], [0.31, 0, 0.041, 0, 0, 0, 0]])
    )
    assert "nonempty" in check(np.zeros((0, 7)))
    clock.sleep(1)
    assert "age" in check(valid)


def test_rejecting_approver_never_silently_clamps():
    calls = []
    guard = DobotApprover(calls.append)
    action = Action(np.array([0.3, 0, 0.2, 0, 0, 0, 0]))
    assert guard.review(action, {}) is action
    assert calls == [action]
    for flag in ("clamped", "delta_clamped"):
        with pytest.raises(SafetyRejected, match="modified"):
            guard.review(Action(action.data, meta={flag: True}), {})


@pytest.mark.parametrize(
    "change",
    [
        {"workspace_low": (0.5, 0, 0)},
        {"workspace_high": (float("inf"), 1, 1)},
        {"minimum_tcp_z": 0.6},
        {"max_translation_step": 0},
        {"max_orientation_step": math.pi + 1},
        {"speed_percent": 0},
        {"speed_percent": True},
        {"acceleration_percent": 101},
        {"position_tolerance": 0.1},
        {"orientation_tolerance": 0.2},
        {"settle_timeout": -1},
        {"telemetry_max_age": float("nan")},
        {"poll_interval": 1},
        {"user_frame": 1},
        {"tool_frame": 51},
        {"tool_frame": 1.5},
    ],
)
def test_profile_rejects_invalid_config(profile, change):
    with pytest.raises(ConfigurationError):
        replace(profile, **change)


def test_config_load_is_explicit_and_cannot_enable_motion(profile, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"safety": asdict(profile), "control_hz": 10}))
    assert load_config(path) == DobotConfig(profile, 10)
    path.write_text(json.dumps({"safety": asdict(profile), "allow_motion": True}))
    with pytest.raises(ConfigurationError, match="runtime-only"):
        load_config(path)
    path.write_text("{bad")
    with pytest.raises(ConfigurationError):
        load_config(path)


@pytest.mark.parametrize("hz", [0, -1, float("inf"), True])
def test_invalid_rate(hz):
    with pytest.raises(ConfigurationError):
        DobotConfig(control_hz=hz)


def test_camera_config_requires_valid_dimensions_and_time():
    with pytest.raises(ConfigurationError):
        CameraConfig(0, 10, 1, 1)
    with pytest.raises(ConfigurationError):
        CameraConfig(10, 10, -1, 1)
