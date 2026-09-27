"""Hardware-independent driver contract and deterministic, non-networked FakeDobot."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .clock import Clock
from .config import SafetyProfile
from .errors import DriverFault, PhaseUnavailable, SafetyRejected
from .safety import MotionAuthority, validate_position, validate_snapshot, validate_target
from .transforms import interpolate_rotation, native_rotation, rotation_to_dobot_native
from .types import JointPositions, PoseSI, RobotMode, RobotSnapshot


@dataclass(frozen=True)
class MotionReceipt:
    command_id: int
    accepted_at: float
    physical_sent: bool = False


@dataclass(frozen=True)
class FakeCommand:
    """Semantic fake operation, deliberately not an executable TCP command."""

    name: str
    timestamp: float
    command_id: int
    target: PoseSI | None = None


class DobotDriver(Protocol):
    is_simulated: bool

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def snapshot(self) -> RobotSnapshot: ...
    def robot_mode(self) -> RobotMode: ...
    def get_pose(self) -> PoseSI: ...
    def get_joints(self) -> JointPositions: ...
    def get_errors(self) -> tuple[int, ...]: ...
    def stop(self) -> None: ...
    def move_linear(self, target: PoseSI) -> MotionReceipt: ...


class FakeDobotDriver:
    """Explicit initial state/profile; joints remain synthetic, never fabricated IK."""

    is_simulated = True

    def __init__(
        self,
        *,
        initial_pose: PoseSI,
        initial_joints: JointPositions,
        profile: SafetyProfile,
        clock: Clock,
        authority: MotionAuthority | None = None,
        mode: RobotMode = RobotMode.ENABLED_IDLE,
        errors: tuple[int, ...] = (),
        convergence_delay: float = 0.0,
        rotational_convergence_delay: float | None = None,
        never_converge: bool = False,
        reachable: Callable[[PoseSI], bool] | None = None,
    ) -> None:
        if type(profile) is not SafetyProfile:
            raise ValueError("FakeDobot cannot use a session-local micro-move envelope")
        if not math.isfinite(convergence_delay) or convergence_delay < 0:
            raise ValueError("convergence_delay must be finite and nonnegative")
        rotation_delay = (
            convergence_delay
            if rotational_convergence_delay is None
            else rotational_convergence_delay
        )
        if not math.isfinite(rotation_delay) or rotation_delay < 0:
            raise ValueError("rotational_convergence_delay must be finite and nonnegative")
        if len(initial_joints) != 6:
            raise ValueError("six synthetic joint positions are required")
        self.profile, self.clock = profile, clock
        self.authority = authority or MotionAuthority()
        self._pose, self._joints = initial_pose, initial_joints
        self._orientation_reference = initial_pose
        self._rotation_delay = rotation_delay
        self._mode, self._errors = mode, errors
        self._delay, self._never, self._reachable = convergence_delay, never_converge, reachable
        self._connected = False
        self._pending: tuple[PoseSI, PoseSI, float] | None = None
        self._command_id = 0
        self._commands: list[FakeCommand] = []

    @property
    def commands(self) -> tuple[FakeCommand, ...]:
        return tuple(self._commands)

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._pending = None
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise DriverFault("fake driver is not connected")

    def capture_orientation_reference(self) -> RobotSnapshot:
        """Read measured reset pose; never adopt an action target as reference."""
        sample = self.snapshot()
        validate_snapshot(sample, self.profile, self.clock.monotonic())
        validate_position(sample.pose, self.profile, label="reset pose")
        self._orientation_reference = sample.pose
        return sample

    def inject_fault(self, *, mode: RobotMode, errors: tuple[int, ...] = ()) -> None:
        self._mode, self._errors = mode, errors
        self._pending = None

    def inject_pose(self, pose: PoseSI) -> None:
        """Explicit test hook for external disturbance, never a real operation."""
        self._pose = pose
        self._pending = None

    def _advance(self) -> None:
        if self._pending is None or self._never:
            return
        start, target, accepted = self._pending
        fraction = (
            1.0 if self._delay == 0 else min(1.0, (self.clock.monotonic() - accepted) / self._delay)
        )
        rotational_fraction = (
            1.0
            if self._rotation_delay == 0
            else min(1.0, (self.clock.monotonic() - accepted) / self._rotation_delay)
        )
        if rotational_fraction == 0 or start.values[3:] == target.values[3:]:
            angles = start.values[3:]
        elif rotational_fraction == 1:
            angles = target.values[3:]
        else:
            angles = rotation_to_dobot_native(
                interpolate_rotation(
                    native_rotation(start), native_rotation(target), rotational_fraction
                )
            )
        x, y, z = (a + (b - a) * fraction for a, b in zip(start.xyz, target.xyz, strict=True))
        self._pose = PoseSI(x, y, z, *angles)
        if fraction >= 1.0 and rotational_fraction >= 1.0:
            self._pose, self._mode, self._pending = target, RobotMode.ENABLED_IDLE, None

    def snapshot(self) -> RobotSnapshot:
        self._require_connection()
        self._advance()
        return RobotSnapshot(
            self._pose,
            self._joints,
            self.robot_mode(),
            self._errors,
            self._command_id,
            self.clock.monotonic(),
            self.profile.user_frame,
            self.profile.tool_frame,
            joints_synthetic=True,
        )

    def robot_mode(self) -> RobotMode:
        self._require_connection()
        self._advance()
        return RobotMode.ERROR if self._errors else self._mode

    def _require_pose_query(self) -> None:
        if self.robot_mode() in (RobotMode.ERROR, RobotMode.POWER_OFF):
            raise DriverFault("GetPose/GetAngle unavailable in error/power-off state (manual p160)")

    def get_pose(self) -> PoseSI:
        self._require_pose_query()
        return self._pose

    def get_joints(self) -> JointPositions:
        self._require_pose_query()
        return self._joints

    def get_errors(self) -> tuple[int, ...]:
        self._require_connection()
        return self._errors

    def move_linear(self, target: PoseSI) -> MotionReceipt:
        self.authority.require(is_simulated=self.is_simulated)
        current = self.snapshot()
        validate_target(
            target,
            current,
            self.profile,
            self.clock.monotonic(),
            reference=self._orientation_reference,
        )
        if self._reachable is not None and not self._reachable(target):
            raise SafetyRejected(
                "REJECT: fake reachability predicate rejected target (not collision checking)"
            )
        self._command_id += 1
        now = self.clock.monotonic()
        self._pending = (self._pose, target, now)
        self._mode = RobotMode.RUNNING
        self._commands.append(FakeCommand("move_linear", now, self._command_id, target))
        return MotionReceipt(self._command_id, now)

    def stop(self) -> None:
        self._require_connection()
        self._advance()
        self._pending = None
        if self._mode == RobotMode.RUNNING:
            self._mode = RobotMode.ENABLED_IDLE
        self._commands.append(FakeCommand("stop", self.clock.monotonic(), self._command_id))

    def set_gripper(self, value: float) -> None:
        self.authority.require(is_simulated=self.is_simulated)
        raise PhaseUnavailable("physical gripper outputs are not implemented; use NoOpGripper")
