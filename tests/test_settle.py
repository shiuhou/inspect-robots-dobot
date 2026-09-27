from dataclasses import replace

import pytest

from inspect_robots_dobot.driver import FakeDobotDriver, MotionReceipt
from inspect_robots_dobot.errors import DriverFault, SettleTimeout
from inspect_robots_dobot.safety import MotionAuthority
from inspect_robots_dobot.settle import wait_until_settled
from inspect_robots_dobot.types import RobotMode


def test_already_settled(driver, pose, profile, clock):
    driver.connect()
    result = wait_until_settled(driver, pose, MotionReceipt(0, clock.monotonic()), profile, clock)
    assert result.duration == 0
    assert result.position_residual == 0


def test_delayed_convergence(profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(True),
        convergence_delay=0.2,
    )
    driver.connect()
    target = replace(pose, x=0.31)
    receipt = driver.move_linear(target)
    result = wait_until_settled(driver, target, receipt, profile, clock)
    assert result.duration == pytest.approx(0.2)
    assert result.final.pose == target
    assert result.final.mode == RobotMode.ENABLED_IDLE


def test_timeout_for_never_converging_motion(profile, pose, clock):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(True),
        never_converge=True,
    )
    driver.connect()
    target = replace(pose, x=0.31)
    with pytest.raises(SettleTimeout, match="no retry"):
        wait_until_settled(driver, target, driver.move_linear(target), profile, clock)


def test_queue_id_must_match_even_when_pose_is_exact(driver, pose, profile, clock):
    driver.connect()
    with pytest.raises(SettleTimeout):
        wait_until_settled(driver, pose, MotionReceipt(99, clock.monotonic()), profile, clock)


def test_pose_must_match_even_when_idle_and_id_matches(driver, pose, profile, clock):
    driver.connect()
    with pytest.raises(SettleTimeout):
        wait_until_settled(
            driver, replace(pose, x=0.31), MotionReceipt(0, clock.monotonic()), profile, clock
        )


def test_controller_error_during_settle(driver, pose, profile, clock, monkeypatch):
    driver.connect()
    original = driver.snapshot
    calls = 0

    def faulting_snapshot():
        nonlocal calls
        calls += 1
        sample = original()
        return (
            replace(sample, mode=RobotMode.RUNNING)
            if calls == 1
            else replace(sample, mode=RobotMode.ERROR, errors=(123,))
        )

    monkeypatch.setattr(driver, "snapshot", faulting_snapshot)
    with pytest.raises(DriverFault, match="123"):
        wait_until_settled(driver, pose, MotionReceipt(0, clock.monotonic()), profile, clock)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"observed_at": 0}, "stale"),
        ({"tool_frame": 2}, "frame"),
        ({"mode": RobotMode.PAUSED}, "interrupted"),
    ],
)
def test_invalid_settle_measurement(driver, pose, profile, clock, monkeypatch, change, reason):
    driver.connect()
    sample = replace(driver.snapshot(), **change)
    monkeypatch.setattr(driver, "snapshot", lambda: sample)
    with pytest.raises(DriverFault, match=reason):
        wait_until_settled(driver, pose, MotionReceipt(0, clock.monotonic()), profile, clock)


def test_settle_rejects_old_sample_even_if_inside_freshness_window(
    driver, pose, profile, clock, monkeypatch
):
    driver.connect()
    sample = replace(driver.snapshot(), observed_at=clock.monotonic() - 0.01)
    monkeypatch.setattr(driver, "snapshot", lambda: sample)
    with pytest.raises(DriverFault, match="predates"):
        wait_until_settled(driver, pose, MotionReceipt(0, clock.monotonic()), profile, clock)
