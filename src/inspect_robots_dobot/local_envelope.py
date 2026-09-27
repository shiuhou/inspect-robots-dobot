"""Measured-start-only geometry for one standalone +Z 10mm experiment.

The lower Z bound is a local observation threshold, NOT a measured table height.
An envelope is prospective data, never authority. Its session/expiry must also
match the live driver's one-use ledger. No persistence can restore that ledger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import LocalMicroMoveSettings, SafetyProfile
from .errors import ConfigurationError, SafetyRejected
from .types import NativeDecimalPose, RobotMode, RobotSnapshot
from .units import from_native


@dataclass(frozen=True)
class LocalMicroMoveEnvelope:
    settings: LocalMicroMoveSettings
    start: RobotSnapshot
    target_native: NativeDecimalPose
    session_nonce: str
    chunk_id: str
    bound_at: float
    expires_at: float

    def __post_init__(self) -> None:
        s = self.start
        if (
            not self.session_nonce
            or not self.chunk_id
            or not math.isfinite(self.bound_at)
            or not math.isfinite(self.expires_at)
            or not 0 < self.expires_at - self.bound_at <= 30
            or not math.isfinite(s.observed_at)
            or not 0 <= self.bound_at - s.observed_at <= self.settings.telemetry_max_age
        ):
            raise SafetyRejected("local envelope requires a fresh session measurement and expiry")
        if (
            s.joints_synthetic
            or s.native_decimal is None
            or from_native(s.native_decimal.native) != s.pose
            or s.mode is not RobotMode.ENABLED_IDLE
            or s.errors
            or (s.user_frame, s.tool_frame) != (0, 0)
            or not all(math.isfinite(v) for v in (*s.pose.values, *s.joints))
        ):
            raise SafetyRejected(
                "local envelope requires measured idle/error-free user0/tool0 pose"
            )
        if self.target_native != s.native_decimal.translated_10mm("+Z"):
            raise SafetyRejected("local envelope target must preserve native pose except +10mm Z")

    @property
    def low(self) -> tuple[float, float, float]:
        x, y, z = self.start.pose.xyz
        m = self.settings.measurement_margin_m
        return (x - m, y - m, z - m)

    @property
    def high(self) -> tuple[float, float, float]:
        x, y, _ = self.start.pose.xyz
        m = self.settings.measurement_margin_m
        return (x + m, y + m, from_native(self.target_native.native).z + m)


@dataclass(frozen=True, kw_only=True)
class LocalEnvelopeSafety(SafetyProfile):
    """Internal validator projection; forbidden as a general embodiment profile."""

    envelope: LocalMicroMoveEnvelope

    def __post_init__(self) -> None:
        super().__post_init__()
        e, s = self.envelope, self.envelope.settings
        if (
            self.workspace_low != e.low
            or self.workspace_high != e.high
            or self.minimum_tcp_z != e.low[2]
            or self.user_frame != 0
            or self.tool_frame != 0
            or self.orientation_low != (0, 0, 0)
            or self.orientation_high != (0, 0, 0)
            # decimal_micro_profile adds only its existing round-off allowance.
            or self.max_translation_step not in (0.010, 0.010 + 1e-12)
        ):
            raise ConfigurationError("local envelope geometry/direction budget cannot be widened")
        for name in (
            "speed_percent",
            "acceleration_percent",
            "position_tolerance",
            "orientation_tolerance",
            "settle_timeout",
            "telemetry_max_age",
            "poll_interval",
            "max_orientation_step",
        ):
            if getattr(self, name) != getattr(s, name):
                raise ConfigurationError(f"local envelope settings changed: {name}")


def envelope_safety(envelope: LocalMicroMoveEnvelope) -> LocalEnvelopeSafety:
    s = envelope.settings
    return LocalEnvelopeSafety(
        workspace_low=envelope.low,
        workspace_high=envelope.high,
        minimum_tcp_z=envelope.low[2],
        user_frame=0,
        tool_frame=0,
        max_translation_step=0.010,
        max_orientation_step=s.max_orientation_step,
        speed_percent=s.speed_percent,
        acceleration_percent=s.acceleration_percent,
        position_tolerance=s.position_tolerance,
        orientation_tolerance=s.orientation_tolerance,
        settle_timeout=s.settle_timeout,
        telemetry_max_age=s.telemetry_max_age,
        poll_interval=s.poll_interval,
        envelope=envelope,
    )
