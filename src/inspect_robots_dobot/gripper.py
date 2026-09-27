"""Normalized gripper semantics and the Phase 6A.2 shadow boundary.

The high-level contract follows the pinned Inspect Robots convention: ``0`` is
closed and ``1`` is open. Servo registers and serial transport stay outside
this module's shadow path; the pure mapping helper is retained for a future
opt-in Feetech backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import SafetyRejected

GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 1.0
GRIPPER_SEMANTICS = ("CLOSED", "OPEN", "INTERMEDIATE")


def validate_normalized(value: Any, *, label: str = "gripper") -> float:
    """Validate one normalized policy value without clamping."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafetyRejected(f"REJECT: {label} must be a finite normalized number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SafetyRejected(f"REJECT: {label} must be a finite normalized number")
    if not 0.0 <= normalized <= 1.0:
        raise SafetyRejected(f"REJECT: {label}={normalized:.6g} is outside [0,1]")
    return normalized


def semantic_target(value: Any) -> str:
    """Return the high-level target state represented by a normalized value."""

    normalized = validate_normalized(value)
    if normalized == GRIPPER_CLOSED:
        return "CLOSED"
    if normalized == GRIPPER_OPEN:
        return "OPEN"
    return "INTERMEDIATE"


def normalized_to_servo_position(
    value: Any, *, open_position: int = 1470, closed_position: int = 2490
) -> int:
    """Map Robocurve polarity to the audited Feetech endpoints.

    The x-trainer trigger helper uses the opposite trigger polarity internally
    (0=open, 1=closed). This mapping is explicit at the future hardware
    boundary: Robocurve 0=closed and 1=open.
    """

    normalized = validate_normalized(value)
    if type(open_position) is not int or type(closed_position) is not int:
        raise ValueError("gripper servo endpoints must be integers")
    if open_position >= closed_position:
        raise ValueError("open_position must be below closed_position")
    return int(round(closed_position + normalized * (open_position - closed_position)))


@dataclass(frozen=True)
class GripperState:
    normalized: float
    semantic: str
    source: str

    def __post_init__(self) -> None:
        normalized = validate_normalized(self.normalized, label="gripper state")
        if self.semantic not in GRIPPER_SEMANTICS:
            raise ValueError(f"unsupported gripper semantic state {self.semantic!r}")
        if self.normalized != normalized:
            raise ValueError("gripper state must use a finite normalized value")
        if not self.source:
            raise ValueError("gripper state source must be explicit")


class ShadowGripperExecutor:
    """Prospective gripper executor that never opens serial or writes hardware."""

    backend = "shadow"
    execution_mode = "shadow"

    def record_target(self, value: Any) -> dict[str, Any]:
        normalized = validate_normalized(value, label="gripper target")
        semantic = semantic_target(normalized)
        event = {
            "CLOSED": "WOULD_GRIPPER_CLOSE",
            "OPEN": "WOULD_GRIPPER_OPEN",
            "INTERMEDIATE": "WOULD_GRIPPER_TARGET",
        }[semantic]
        return {
            "backend": self.backend,
            "execution_mode": self.execution_mode,
            "target_normalized": normalized,
            "semantic_target": semantic,
            "would_execute": f"{event} normalized={normalized:g}",
            "physical_gripper_connected": False,
            "physical_gripper_command_sent": False,
            "gripper_serial_connections": 0,
            "gripper_commands_sent": 0,
        }


class ShadowGripper:
    """Fixture/recorded gripper state holder; setting it has no hardware effect."""

    def __init__(self, normalized: float = GRIPPER_CLOSED, *, source: str = "fixture") -> None:
        self._state = GripperState(normalized, semantic_target(normalized), source)

    def read(self) -> float:
        return self._state.normalized

    @property
    def state(self) -> GripperState:
        return self._state

    def set(self, value: float) -> None:
        normalized = validate_normalized(value, label="shadow gripper")
        self._state = GripperState(normalized, semantic_target(normalized), "shadow_proposal")


class NoOpGripper:
    """Legacy inactive adapter retained for older direct-driver tests."""

    def set(self, value: float) -> None:
        if value != 0:
            raise SafetyRejected("REJECT: legacy NoOpGripper only accepts placeholder 0")

    def read(self) -> float:
        return 0.0
