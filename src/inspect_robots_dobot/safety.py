"""Local rejection gates independent of prompts, connectivity and framework clamps."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.types import Action

from .config import SafetyProfile
from .errors import MotionNotAuthorized, SafetyRejected
from .gripper import validate_normalized
from .transforms import (
    agent_relative_to_rotation,
    native_rotation,
    rotation_to_agent_relative,
    rotation_to_dobot_native,
)
from .types import PoseSI, RobotMode, RobotSnapshot
from .units import orientation_distance, translation_distance


@dataclass(frozen=True)
class MotionAuthority:
    """Runtime-only capability: simulation can be granted, never physical motion."""

    motion_enabled: bool = False
    simulation_only: bool = True

    def __post_init__(self) -> None:
        if type(self.motion_enabled) is not bool or type(self.simulation_only) is not bool:
            raise MotionNotAuthorized("motion authority flags must be explicit booleans")

    def require(self, *, is_simulated: bool) -> None:
        if not self.motion_enabled:
            raise MotionNotAuthorized("REJECT: motion authority is disabled")
        if not is_simulated or not self.simulation_only:
            raise MotionNotAuthorized(
                "REJECT: simulation authority cannot authorize physical motion"
            )


def validate_snapshot(snapshot: RobotSnapshot, profile: SafetyProfile, now: float) -> None:
    if not math.isfinite(now) or not math.isfinite(snapshot.observed_at):
        raise SafetyRejected("REJECT: invalid telemetry timestamp")
    age = now - snapshot.observed_at
    if age < 0 or age > profile.telemetry_max_age:
        raise SafetyRejected(f"REJECT: telemetry age {age:.6g} s is outside freshness limits")
    if snapshot.errors:
        raise SafetyRejected(f"REJECT: active controller alarms {snapshot.errors}")
    if snapshot.mode != RobotMode.ENABLED_IDLE:
        raise SafetyRejected(f"REJECT: robot mode {snapshot.mode.name} is not enabled and idle")
    if (snapshot.user_frame, snapshot.tool_frame) != (profile.user_frame, profile.tool_frame):
        raise SafetyRejected("REJECT: measured user/tool frames differ from configured frames")
    if not all(math.isfinite(v) for v in (*snapshot.pose.values, *snapshot.joints)):
        raise SafetyRejected("REJECT: measured state contains NaN/Inf")


def validate_position(pose: PoseSI, profile: SafetyProfile, *, label: str) -> None:
    if not all(math.isfinite(v) for v in pose.values):
        raise SafetyRejected(f"REJECT: {label} contains NaN/Inf")
    for axis, v, lo, hi in zip(
        "xyz", pose.xyz, profile.workspace_low, profile.workspace_high, strict=True
    ):
        if not lo <= v <= hi:
            raise SafetyRejected(
                f"REJECT: {label} {axis}={v:.6g} m is outside [{lo:.6g},{hi:.6g}] m"
            )
    if pose.z < profile.minimum_tcp_z:
        raise SafetyRejected(
            f"REJECT: {label} z={pose.z:.6g} m is below configured minimum "
            f"z={profile.minimum_tcp_z:.6g} m"
        )


def validate_target(
    target: PoseSI,
    current: RobotSnapshot,
    profile: SafetyProfile,
    now: float,
    *,
    reference: PoseSI | None = None,
) -> tuple[float, float]:
    validate_snapshot(current, profile, now)
    validate_position(current.pose, profile, label="current pose")
    validate_position(target, profile, label="requested target")
    if reference is not None:
        validate_orientation(current.pose, reference, profile, label="measured orientation drift")
        validate_orientation(target, reference, profile, label="requested orientation")
    translation = translation_distance(current.pose, target)
    orientation = orientation_distance(current.pose, target)
    if translation > profile.max_translation_step:
        raise SafetyRejected(
            f"REJECT: translation delta={translation:.6g} m exceeds "
            f"{profile.max_translation_step:.6g} m"
        )
    if orientation > profile.max_orientation_step:
        raise SafetyRejected(
            f"REJECT: orientation delta={orientation:.6g} rad exceeds "
            f"{profile.max_orientation_step:.6g} rad"
        )
    return translation, orientation


def relative_orientation(
    pose: PoseSI, reference: PoseSI, profile: SafetyProfile
) -> tuple[float, float, float]:
    try:
        return rotation_to_agent_relative(
            native_rotation(pose),
            native_rotation(reference),
            singularity_margin=profile.orientation_singularity_margin,
        )
    except ValueError as exc:
        raise SafetyRejected(f"REJECT: {exc}") from exc


def _validate_angles(
    angles: tuple[float, float, float],
    profile: SafetyProfile,
    *,
    label: str,
    roundoff: float = 0.0,
) -> None:
    for axis, value, low, high in zip(
        ("yaw", "pitch", "roll"),
        angles,
        profile.orientation_low,
        profile.orientation_high,
        strict=True,
    ):
        if not low - roundoff <= value <= high + roundoff:
            constraint = (
                f"is pinned at {low:.6g}"
                if low == high
                else (
                    f"exceeds configured upper bound {high:.6g}"
                    if value > high
                    else f"is below configured lower bound {low:.6g}"
                )
            )
            raise SafetyRejected(f"REJECT: {label} {axis}={value:.6g} rad {constraint} rad")


def validate_orientation(
    pose: PoseSI, reference: PoseSI, profile: SafetyProfile, *, label: str
) -> tuple[float, float, float]:
    angles = relative_orientation(pose, reference, profile)
    # Matrix decode roundoff only. Authored action values use exact bounds below.
    _validate_angles(angles, profile, label=label, roundoff=1e-12)
    return angles


def action_target(
    data: npt.ArrayLike, current: RobotSnapshot, held: PoseSI, profile: SafetyProfile
) -> PoseSI:
    try:
        vector = np.asarray(data, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise SafetyRejected("REJECT: action must be a numeric SI vector") from exc
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        raise SafetyRejected(
            "REJECT: action must be finite [x,y,z,yaw,pitch,roll,gripper] with shape (7,)"
        )
    validate_normalized(float(vector[6]), label="gripper")
    validate_orientation(current.pose, held, profile, label="measured orientation drift")
    angles = (float(vector[3]), float(vector[4]), float(vector[5]))
    _validate_angles(angles, profile, label="requested")
    if angles == (0.0, 0.0, 0.0):
        return held.with_translation(float(vector[0]), float(vector[1]), float(vector[2]))
    try:
        rotation = agent_relative_to_rotation(
            *angles,
            native_rotation(held),
            singularity_margin=profile.orientation_singularity_margin,
        )
        native = rotation_to_dobot_native(rotation)
    except ValueError as exc:
        raise SafetyRejected(f"REJECT: {exc}") from exc
    return PoseSI(float(vector[0]), float(vector[1]), float(vector[2]), *native)


class WaypointPreCheck:
    """Validate the exact emitted waypoints against a captured measured reference."""

    def __init__(
        self,
        snapshot: Callable[[], RobotSnapshot],
        held: Callable[[], PoseSI],
        profile: SafetyProfile,
        now: Callable[[], float],
    ) -> None:
        self._snapshot, self._held, self._profile, self._now = snapshot, held, profile, now

    def __call__(self, waypoints: npt.NDArray[np.float64]) -> str | None:
        if waypoints.ndim != 2 or waypoints.shape[1:] != (7,) or len(waypoints) == 0:
            return "expected nonempty waypoints with shape (steps,7)"
        current = self._snapshot()
        held, now = self._held(), self._now()
        for i, row in enumerate(waypoints, 1):
            try:
                target = action_target(row, current, held, self._profile)
                validate_target(target, current, self._profile, now, reference=held)
            except SafetyRejected as exc:
                return f"waypoint {i}: {exc}; choose a legal intermediate/raised target"
            current = replace(current, pose=target)
        return None


class DobotApprover:
    """Contributed framework gate; reject prior clamps and preserve accepted identity."""

    def __init__(self, validate: Callable[[Action], None]) -> None:
        self._validate = validate

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        if action.meta.get("clamped") or action.meta.get("delta_clamped"):
            raise SafetyRejected("REJECT: framework modified the target; resubmit a valid target")
        self._validate(action)
        return action
