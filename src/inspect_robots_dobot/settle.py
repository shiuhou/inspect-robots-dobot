"""Acknowledgement is not arrival. Bounded, visible, fail-closed settling."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .clock import Clock
from .config import SafetyProfile
from .driver import DobotDriver, MotionReceipt
from .errors import DriverFault, SafetyRejected, SettleTimeout
from .safety import validate_orientation, validate_position
from .types import PoseSI, RobotMode, RobotSnapshot
from .units import orientation_distance, translation_distance


@dataclass(frozen=True)
class SettleResult:
    final: RobotSnapshot
    duration: float
    position_residual: float
    orientation_residual: float
    settled_at: float


def wait_until_settled(
    driver: DobotDriver,
    target: PoseSI,
    receipt: MotionReceipt,
    profile: SafetyProfile,
    clock: Clock,
    *,
    reference: PoseSI | None = None,
) -> SettleResult:
    start = clock.monotonic()
    deadline = start + profile.settle_timeout
    max_polls = math.ceil(profile.settle_timeout / profile.poll_interval) + 2
    position = orientation = float("inf")
    for _ in range(max_polls):
        sample = driver.snapshot()
        now = clock.monotonic()
        if sample.errors or sample.mode not in (RobotMode.RUNNING, RobotMode.ENABLED_IDLE):
            raise DriverFault(
                f"settle interrupted: mode={sample.mode.name}, errors={sample.errors}"
            )
        age = now - sample.observed_at
        if not math.isfinite(age) or not 0 <= age <= profile.telemetry_max_age:
            raise DriverFault("stale telemetry during settle")
        if sample.observed_at < receipt.accepted_at:
            raise DriverFault("settle telemetry predates command acceptance")
        if (sample.user_frame, sample.tool_frame) != (profile.user_frame, profile.tool_frame):
            raise DriverFault("coordinate frame changed during settle")
        if not all(math.isfinite(v) for v in sample.pose.values):
            raise DriverFault("non-finite pose during settle")
        try:
            validate_position(sample.pose, profile, label="settle measurement")
            if reference is not None:
                validate_orientation(sample.pose, reference, profile, label="settle measurement")
        except SafetyRejected as exc:
            raise DriverFault(f"settle interrupted: {exc}") from exc
        position = translation_distance(sample.pose, target)
        orientation = orientation_distance(sample.pose, target)
        if (
            sample.command_id == receipt.command_id
            and sample.mode == RobotMode.ENABLED_IDLE
            and position <= profile.position_tolerance
            and orientation <= profile.orientation_tolerance
            and now <= deadline
        ):
            return SettleResult(sample, now - start, position, orientation, now)
        if now >= deadline:
            break
        clock.sleep(min(profile.poll_interval, deadline - now))
    raise SettleTimeout(
        f"settle timeout for command {receipt.command_id}: position residual={position:.6g} m, "
        f"orientation residual={orientation:.6g} rad; no retry or corrective move"
    )
