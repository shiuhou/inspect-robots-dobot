from dataclasses import replace

import pytest

from inspect_robots_dobot.driver import FakeDobotDriver
from inspect_robots_dobot.errors import (
    DriverFault,
    MotionNotAuthorized,
    PhaseUnavailable,
    SafetyRejected,
)
from inspect_robots_dobot.gripper import NoOpGripper
from inspect_robots_dobot.safety import MotionAuthority
from inspect_robots_dobot.types import RobotMode


def test_initial_state_is_explicit_deterministic_and_read_only(profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose, initial_joints=(0.0,) * 6, profile=profile, clock=clock
    )
    with pytest.raises(DriverFault, match="not connected"):
        driver.snapshot()
    driver.connect()
    assert driver.robot_mode() == RobotMode.ENABLED_IDLE
    assert driver.get_pose() == pose
    assert driver.get_joints() == (0,) * 6
    assert driver.get_errors() == ()
    assert driver.snapshot().joints_synthetic
    assert driver.commands == ()
    with pytest.raises(MotionNotAuthorized):
        driver.move_linear(replace(pose, x=0.301))
    with pytest.raises(MotionNotAuthorized):
        driver.set_gripper(1)
    assert driver.commands == ()
    driver.close()
    with pytest.raises(DriverFault):
        driver.get_pose()


def test_authorized_move_records_target_without_inventing_joint_ik(driver, pose):
    driver.connect()
    target = replace(pose, x=0.305)
    receipt = driver.move_linear(target)
    assert receipt.command_id == 1
    assert receipt.physical_sent is False
    assert driver.commands[0].target == target
    assert driver.get_pose() == target
    assert driver.get_joints() == (0,) * 6
    assert driver.robot_mode() == RobotMode.ENABLED_IDLE


def test_ack_does_not_mean_arrival(profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(True),
        convergence_delay=0.2,
    )
    driver.connect()
    driver.move_linear(replace(pose, x=0.31))
    assert driver.robot_mode() == RobotMode.RUNNING
    assert driver.get_pose() == pose
    clock.sleep(0.1)
    assert driver.get_pose().x == pytest.approx(0.305)
    driver.stop()
    stopped = driver.get_pose()
    clock.sleep(1)
    assert driver.get_pose() == stopped
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]


def test_driver_rechecks_limits_independent_of_embodiment(driver, pose):
    driver.connect()
    with pytest.raises(SafetyRejected, match="translation delta"):
        driver.move_linear(replace(pose, x=0.4))
    assert driver.commands == ()


def test_unreachable_injection_does_not_record_command(profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(True),
        reachable=lambda _: False,
    )
    driver.connect()
    with pytest.raises(SafetyRejected, match="reachability"):
        driver.move_linear(replace(pose, x=0.301))
    assert driver.commands == ()


@pytest.mark.parametrize("mode,errors", [(RobotMode.ERROR, (123,)), (RobotMode.POWER_OFF, ())])
def test_documented_state_restrictions(driver, mode, errors):
    driver.connect()
    driver.inject_fault(mode=mode, errors=errors)
    assert driver.robot_mode() == mode
    assert driver.get_errors() == errors
    with pytest.raises(DriverFault, match="unavailable"):
        driver.get_pose()
    with pytest.raises(DriverFault, match="unavailable"):
        driver.get_joints()


def test_physical_capability_cannot_be_granted_in_phase_one():
    with pytest.raises(MotionNotAuthorized, match="physical"):
        MotionAuthority(True).require(is_simulated=False)
    with pytest.raises(MotionNotAuthorized):
        MotionAuthority(True, simulation_only=False).require(is_simulated=True)
    with pytest.raises(MotionNotAuthorized, match="booleans"):
        MotionAuthority("false")


def test_gripper_inactive_and_no_electrical_polarity(driver):
    gripper = NoOpGripper()
    gripper.set(0)
    assert gripper.read() == 0
    for value in (1, -1, float("nan")):
        with pytest.raises(SafetyRejected):
            gripper.set(value)
    with pytest.raises(PhaseUnavailable):
        driver.set_gripper(0)
