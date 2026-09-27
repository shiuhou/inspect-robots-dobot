"""Explicit production monitoring limits; no rig geometry or authority defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import LocalMicroMoveSettings, SafetyProfile, positive
from .errors import ConfigurationError
from .motion import KeepoutBox, require_translation_profile


@dataclass(frozen=True)
class LiveMotionProfile:
    safety: SafetyProfile | LocalMicroMoveSettings
    start_position_tolerance: float
    start_orientation_tolerance: float
    max_measurement_age: float
    consecutive_samples: int
    io_timeout: float
    acknowledgement_timeout: float
    stop_timeout: float
    standstill_position_tolerance: float
    standstill_orientation_tolerance: float
    authority_lifetime: float
    model: str
    firmware: str | None
    rig_verification_reference: str
    interruption_verification_reference: str
    keepouts: tuple[KeepoutBox, ...] = ()
    tool_tcp_description: str | None = None
    payload_description: str | None = None
    threshold_class: str | None = None

    def require_live_descriptions(self) -> None:
        for name in ("tool_tcp_description", "payload_description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"live CLI requires operator-supplied {name}")

    def __post_init__(self) -> None:
        if isinstance(self.safety, SafetyProfile):
            require_translation_profile(self.safety)
        if self.safety.max_translation_step > 0.020 or self.safety.acceleration_percent > 10:
            raise ConfigurationError("first live profile requires displacement <=20mm and a<=10%")
        # Engineering upper bounds, not validated rig values or safety-rated guarantees.
        caps = {
            "start_position_tolerance": 0.001,
            "start_orientation_tolerance": 0.01,
            "max_measurement_age": 0.5,
            "io_timeout": 0.25,
            "acknowledgement_timeout": 1.0,
            "stop_timeout": 5.0,
            "standstill_position_tolerance": 0.001,
            "standstill_orientation_tolerance": 0.01,
            "authority_lifetime": 30.0,
        }
        for name, cap in caps.items():
            value = getattr(self, name)
            positive(name, value)
            if value > cap:
                raise ConfigurationError(f"{name} exceeds first-profile engineering cap {cap}")
        if type(self.consecutive_samples) is not int or not 2 <= self.consecutive_samples <= 100:
            raise ConfigurationError("consecutive_samples must be an integer in [2,100]")
        if self.io_timeout > self.acknowledgement_timeout:
            raise ConfigurationError("io_timeout must not exceed acknowledgement_timeout")
        if self.safety.poll_interval > 0.1:
            raise ConfigurationError("live poll_interval must not exceed 0.1 seconds")
        if self.safety.position_tolerance > 0.001 or self.safety.orientation_tolerance > 0.01:
            raise ConfigurationError("live convergence tolerances exceed 1mm/0.01rad caps")
        if self.start_position_tolerance >= self.safety.max_translation_step:
            raise ConfigurationError("start tolerance must be below the displacement budget")
        for name in (
            "model",
            "rig_verification_reference",
            "interruption_verification_reference",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ConfigurationError(f"explicit {name} is required")
        if self.firmware is not None and (
            not isinstance(self.firmware, str) or not self.firmware.strip()
        ):
            raise ConfigurationError("firmware must be a recorded version or null (unknown)")
        if isinstance(self.safety, LocalMicroMoveSettings):
            if self.start_position_tolerance > self.safety.measurement_margin_m:
                raise ConfigurationError("local margin must cover start measurement tolerance")
            if self.standstill_position_tolerance > self.safety.measurement_margin_m:
                raise ConfigurationError("local margin must cover standstill tolerance")
            if self.threshold_class != "ENGINEERING_THRESHOLD_FOR_FIRST_MICROMOVE":
                raise ConfigurationError("local monitor requires explicit engineering provenance")
        if not isinstance(self.keepouts, tuple) or not all(
            isinstance(k, KeepoutBox) for k in self.keepouts
        ):
            raise ConfigurationError("keepouts must be an immutable tuple of KeepoutBox")


@dataclass(frozen=True)
class OperatorReadiness:
    """Current-session declarations from the operator, never inferred from a query.

    No ownership getter is source-verified. tcp_control_owned therefore records
    a controller-UI/operator check and must never be labelled automatic detection.
    """

    operator: str
    tcp_control_owned: bool
    frames_verified: bool
    production_profile_verified: bool
    estop_tested: bool
    operator_present: bool
    workspace_clear: bool
    gripper_disabled: bool

    def require(self) -> None:
        if not self.operator.strip():
            raise ConfigurationError("operator identity is required")
        for name in (
            "tcp_control_owned",
            "frames_verified",
            "production_profile_verified",
            "estop_tested",
            "operator_present",
            "workspace_clear",
            "gripper_disabled",
        ):
            if getattr(self, name) is not True:
                raise ConfigurationError(f"operator must verify {name} in this session")


def load_live_profile(
    path: Path, safety: SafetyProfile | LocalMicroMoveSettings
) -> LiveMotionProfile:
    """Load thresholds/evidence references only, never runtime permission or confirmation."""
    try:
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict) or "safety" in obj:
            raise ValueError("expected monitoring fields; safety comes from reviewed rig config")
        if "keepouts" in obj:
            obj["keepouts"] = tuple(
                KeepoutBox(tuple(item["low"]), tuple(item["high"])) for item in obj["keepouts"]
            )
        return LiveMotionProfile(safety=safety, **obj)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ConfigurationError(f"invalid live monitoring profile {path}: {exc}") from exc
