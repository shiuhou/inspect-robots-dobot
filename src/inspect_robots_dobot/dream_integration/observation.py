"""Measured Nova observations; readers are supplied by the caller, never opened here."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..camera import CameraReader, Frame, validate_frame
from ..clock import Clock
from ..config import CameraConfig
from ..dashboard import DobotDashboardClient
from ..types import PoseSI, RobotMode, RobotSnapshot


@dataclass(frozen=True)
class GripperMeasurement:
    position: float
    observed_at: float
    source: str
    units: str
    calibration_evidence: str | None = None


class StateReader(Protocol):
    def snapshot(self) -> RobotSnapshot: ...


class DashboardMeasurements:
    """Uses an already connected query-only client; no connect or control request.

    GetAngle is measured joints converted by the existing decoder to radians.
    GetPose is metres/native extrinsic XYZ radians in configured user/tool frames.
    The interval includes all sequential queries; this is not hardware synchronization.
    """

    def __init__(self, client: DobotDashboardClient, clock: Clock) -> None:
        self.client, self.clock = client, clock

    def snapshot(self) -> RobotSnapshot:
        if self.client.config.user_frame is None or self.client.config.tool_frame is None:
            raise ValueError("explicit Dashboard user/tool frame configuration required")
        started = self.clock.monotonic()
        joints = self.client.get_joints()
        pose = self.client.get_pose()
        mode = self.client.robot_mode()
        errors = self.client.get_errors()
        return RobotSnapshot(
            pose,
            joints,
            mode,
            errors,
            0,
            started,
            self.client.config.user_frame,
            self.client.config.tool_frame,
        )


@dataclass(frozen=True)
class Nova2Observation:
    front_rgb: Frame
    wrist_rgb: Frame
    state: RobotSnapshot
    gripper_state: GripperMeasurement | None
    collected_at: float
    calibration_evidence: str | None = None
    fixture: bool = False

    @property
    def joint_positions(self) -> tuple[float, ...]:
        return self.state.joints

    @property
    def tcp_pose(self) -> PoseSI:
        return self.state.pose

    @property
    def robot_mode(self) -> RobotMode:
        return self.state.mode

    def validate(
        self, configs: dict[str, CameraConfig], *, now: float, max_age: float, max_skew: float
    ) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (max_age, max_skew)):
            raise ValueError("age/skew bounds must be finite and positive")
        if set(configs) != {"front_rgb", "wrist_rgb"}:
            raise ValueError("front and wrist camera configurations required")
        times = [self.state.observed_at, self.collected_at]
        for name in ("front_rgb", "wrist_rgb"):
            frame = getattr(self, name)
            validate_frame(frame, configs[name], now=now, after=None)
            times.append(frame.timestamp)
        s = self.state
        if len(s.joints) != 6 or not all(math.isfinite(v) for v in (*s.joints, *s.pose.values)):
            raise ValueError("six finite measured joints and TCP pose required")
        if s.joints_synthetic and not self.fixture:
            raise ValueError("synthetic joints cannot be marked measured")
        RobotMode(s.mode)
        if any(type(v) is not int or v < 0 for v in (s.user_frame, s.tool_frame)):
            raise ValueError("explicit nonnegative user/tool indices required")
        if self.gripper_state is not None:
            g = self.gripper_state
            if not math.isfinite(g.position) or not g.source or not g.units:
                raise ValueError("invalid measured gripper feedback")
            times.append(g.observed_at)
        if not math.isfinite(now) or any(
            not math.isfinite(t) or not 0 <= t <= now or now - t > max_age for t in times
        ):
            raise ValueError("stale or inconsistent observation timestamps")
        if max(times) - min(times) > max_skew:
            raise ValueError("observation timestamp skew exceeded")
        if any(t > self.collected_at for t in times):
            raise ValueError("collection time precedes measurements")


class ObservationAdapter:
    """No resource ownership: consumes running CameraReaders and a measurement source."""

    def __init__(
        self,
        state: StateReader,
        front: CameraReader,
        wrist: CameraReader,
        configs: dict[str, CameraConfig],
        clock: Clock,
        *,
        max_age: float,
        max_skew: float,
        calibration_evidence: str | None = None,
    ) -> None:
        self.state, self.front, self.wrist = state, front, wrist
        self.configs, self.clock = configs, clock
        self.max_age, self.max_skew = max_age, max_skew
        self.calibration_evidence = calibration_evidence

    def capture(self, gripper: GripperMeasurement | None = None) -> Nova2Observation:
        state = self.state.snapshot()
        front, wrist = self.front.latest(), self.wrist.latest()
        obs = Nova2Observation(
            front, wrist, state, gripper, self.clock.monotonic(), self.calibration_evidence
        )
        obs.validate(
            self.configs, now=self.clock.monotonic(), max_age=self.max_age, max_skew=self.max_skew
        )
        return obs


def fixture() -> tuple[Nova2Observation, dict[str, CameraConfig]]:
    """Explicit synthetic fixture, unknown gripper and unverified calibration."""
    rgb = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    rgb.flags.writeable = False
    frame = Frame(rgb, 10.0, 10.01, 1, 1)
    state = RobotSnapshot(
        PoseSI(0.3, 0, 0.2, 0, 0, 0),
        (0.0,) * 6,
        RobotMode.DISABLED,
        (),
        0,
        10.0,
        0,
        0,
        joints_synthetic=True,
    )
    config = CameraConfig(32, 24, 0.5, 0.1)
    return Nova2Observation(frame, frame, state, None, 10.02, fixture=True), {
        "front_rgb": config,
        "wrist_rgb": config,
    }
