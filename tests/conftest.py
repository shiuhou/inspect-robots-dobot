"""Synthetic test-only rig. Socket prohibition also applies to model-provider clients."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from inspect_robots_dobot.clock import FakeClock
from inspect_robots_dobot.config import DobotConfig, SafetyProfile
from inspect_robots_dobot.driver import FakeDobotDriver
from inspect_robots_dobot.embodiment import DobotEmbodiment
from inspect_robots_dobot.safety import MotionAuthority
from inspect_robots_dobot.types import PoseSI


def deny_device(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        path = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if path.startswith(("/dev/video", "/dev/media", "/dev/v4l", "/dev/bus/usb")):
            raise AssertionError("physical camera/device access is prohibited in offline tests")


sys.addaudithook(deny_device)


@pytest.fixture(autouse=True)
def forbid_hardware_backends(monkeypatch):
    original = builtins.__import__

    def offline_import(name, *args, **kwargs):
        if name.split(".")[0] in {"cv2", "pyrealsense2", "dobot_api", "DobotDllType"}:
            raise AssertionError(f"physical hardware backend import prohibited: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", offline_import)


@pytest.fixture
def profile():
    # FICTIONAL values for offline tests, NOT measured Nova/production limits.
    return SafetyProfile(
        workspace_low=(0.1, -0.2, 0.01),
        workspace_high=(0.5, 0.2, 0.5),
        minimum_tcp_z=0.08,
        user_frame=0,
        tool_frame=0,
        max_translation_step=0.02,
        max_orientation_step=0.1,
        speed_percent=5,
        acceleration_percent=5,
        position_tolerance=0.0001,
        orientation_tolerance=0.001,
        settle_timeout=0.5,
        telemetry_max_age=0.25,
        poll_interval=0.01,
    )


@pytest.fixture
def pose():
    return PoseSI(0.3, 0.0, 0.2, 0.1, -0.2, 0.3)


@pytest.fixture
def clock():
    return FakeClock(1.0)


@pytest.fixture
def driver(profile, pose, clock):
    instance = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(motion_enabled=True),
    )
    yield instance
    instance.close()


@pytest.fixture
def embodiment(profile, driver, clock):
    instance = DobotEmbodiment(DobotConfig(profile, control_hz=10.0), driver=driver, clock=clock)
    yield instance
    instance.close()


@pytest.fixture
def package_root():
    return Path(__file__).parents[1]
