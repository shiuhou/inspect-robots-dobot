"""Pure prospective motion models. No transport, socket or live authority exists here."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from decimal import Decimal

import numpy as np

from .config import SafetyProfile
from .errors import ConfigurationError, ProtocolError, SafetyRejected
from .local_envelope import LocalEnvelopeSafety, LocalMicroMoveEnvelope
from .protocol import decode_command_id, parse_response
from .safety import action_target, validate_snapshot, validate_target
from .transforms import native_rotation
from .types import NativeDecimalPose, NativePose, PoseSI, RobotSnapshot
from .units import from_native, to_native, translation_distance

AgentPose = tuple[float, ...]
Waypoints = tuple[AgentPose, ...]


def decimal_micro_profile(
    start: NativeDecimalPose, target: NativeDecimalPose, profile: SafetyProfile
) -> SafetyProfile:
    """Exact <=10mm budget first; SI checks then allow only arithmetic roundoff.

    The 1e-12m cushion is never a physical allowance: the decimal comparison
    independently rejects ANY excess, including shifts much smaller than an ULP.
    """
    budget_mm = min(Decimal("10"), Decimal(str(profile.max_translation_step)) * 1000)
    if start.squared_distance_mm(target) > budget_mm * budget_mm:
        raise SafetyRejected(
            "actual start-to-fixed-target displacement exceeds 10mm/profile budget"
        )
    return replace(profile, max_translation_step=profile.max_translation_step + 1e-12)


@dataclass(frozen=True)
class KeepoutBox:
    """Explicit TCP point keepout in metres; does not model robot/link geometry."""

    low: tuple[float, float, float]
    high: tuple[float, float, float]

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, tuple)
            or len(v) != 3
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in v)
            for v in (self.low, self.high)
        ) or any(a >= b for a, b in zip(self.low, self.high, strict=True)):
            raise ConfigurationError("keepout requires finite low < high XYZ metre tuples")

    def intersects(self, start: PoseSI, end: PoseSI) -> bool:
        """Continuous closed-segment/AABB slab test, including boundary contact."""
        entry, leave = 0.0, 1.0
        for a, b, low, high in zip(start.xyz, end.xyz, self.low, self.high, strict=True):
            delta = b - a
            if delta == 0:
                if a < low or a > high:
                    return False
                continue
            near, far = sorted(((low - a) / delta, (high - a) / delta))
            entry, leave = max(entry, near), min(leave, far)
            if entry > leave:
                return False
        return True


def require_translation_profile(profile: SafetyProfile) -> None:
    if profile.orientation_low != (0.0, 0.0, 0.0) or profile.orientation_high != (0.0, 0.0, 0.0):
        raise ConfigurationError("staged MovL requires yaw/pitch/roll pinned at zero")
    if profile.speed_percent > 10:
        raise ConfigurationError("first staged MovL profile requires speed <= 10 percent")


@dataclass(frozen=True)
class PathValidation:
    waypoint_count: int
    total_displacement_m: float
    displacement_limit_m: float
    keepout_count: int
    method: str = "straight monotonic XYZ; convex bounds; continuous TCP keepout segments"
    collision_free_claim: bool = False


def validate_cartesian_path(
    waypoints: Waypoints,
    start: RobotSnapshot,
    reference: PoseSI,
    profile: SafetyProfile,
    now: float,
    keepouts: tuple[KeepoutBox, ...] = (),
) -> PathValidation:
    """Recheck every waypoint, continuous segments and aggregate measured-start budget."""
    require_translation_profile(profile)
    validate_snapshot(start, profile, now)
    if not waypoints:
        raise SafetyRejected("REJECT: staged path is empty")
    targets = []
    for index, row in enumerate(waypoints, 1):
        try:
            targets.append(action_target(row, start, reference, profile))
        except SafetyRejected as exc:
            raise SafetyRejected(f"REJECT: waypoint {index}: {exc}") from exc
    previous = start
    displacement = translation_distance(start.pose, targets[-1])
    limit = min(profile.max_translation_step, 0.020)
    direction = np.asarray(targets[-1].xyz) - np.asarray(start.pose.xyz)
    length2 = float(direction @ direction)
    fraction_before = 0.0
    for index, target in enumerate(targets, 1):
        try:
            validate_target(target, previous, profile, now, reference=reference)
            validate_target(target, start, profile, now, reference=reference)
            if translation_distance(start.pose, target) > limit:
                raise SafetyRejected(f"aggregate displacement exceeds {limit:.6g} m")
            offset = np.asarray(target.xyz) - np.asarray(start.pose.xyz)
            fraction = float(offset @ direction) / length2 if length2 else 0.0
            if (
                np.linalg.norm(offset - fraction * direction) > 1e-12
                or fraction < fraction_before - 1e-12
                or not -1e-12 <= fraction <= 1 + 1e-12
            ):
                raise SafetyRejected("path must be straight monotonic XYZ for one MovL")
            for box in keepouts:
                if box.intersects(previous.pose, target):
                    raise SafetyRejected("TCP segment intersects configured keepout")
            previous = replace(previous, pose=target)
            fraction_before = fraction
        except SafetyRejected as exc:
            raise SafetyRejected(f"REJECT: waypoint {index}: {exc}") from exc
    # Also check the actual controller-planned segment, independent of intermediates.
    if any(box.intersects(start.pose, targets[-1]) for box in keepouts):
        raise SafetyRejected("REJECT: final native target path intersects configured keepout")
    return PathValidation(len(waypoints), displacement, limit, len(keepouts))


@dataclass(frozen=True)
class DobotMovLRequest:
    """V4.6.5 pp87–88 prospective request; intentionally has no send method."""

    pose: NativePose
    user: int
    tool: int
    speed_percent: int
    acceleration_percent: int
    native_decimal: NativeDecimalPose | None = None

    def __post_init__(self) -> None:
        if self.native_decimal is not None and self.native_decimal.native != self.pose:
            raise ConfigurationError("native decimal request disagrees with native float pose")
        if not all(type(v) in (float, int) and math.isfinite(v) for v in self.pose.values):
            raise ConfigurationError("MovL native pose must be finite mm/degree values")
        for name in ("user", "tool"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 50:
                raise ConfigurationError(f"MovL {name} must be an integer in [0,50]")
        for name, maximum in (("speed_percent", 10), ("acceleration_percent", 100)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ConfigurationError(f"MovL {name} must be an integer in [1,{maximum}]")

    def serialize(self) -> str:
        # Python's shortest round-trip decimal avoids unvalidated rounding/clamping.
        # Expand exponents: the manual demonstrates decimal literals, not scientific notation.
        values = ",".join(
            format(Decimal(repr(float(v))), "f") if v else "0.0" for v in self.pose.values
        )
        if self.native_decimal is not None:
            values = ",".join(format(Decimal(v), "f") for v in self.native_decimal.values)
        return (
            f"MovL(pose={{{values}}},user={self.user},tool={self.tool},"
            f"a={self.acceleration_percent},v={self.speed_percent},cp=0)"
        )


def parse_movl_acceptance(raw: bytes, request: DobotMovLRequest) -> int:
    """Offline acknowledgement decoder; queue acceptance is never arrival."""
    return decode_command_id(
        parse_response(raw, expected_command=request.serialize().encode("ascii"))
    )


def serialize_stop() -> str:
    """V4.6.5 p13, pure serialization only; no live Stop backend in Phase 3."""
    return "Stop()"


def parse_stop_acknowledgement(raw: bytes) -> None:
    response = parse_response(raw, expected_command=serialize_stop().encode("ascii"))
    if response.payload.strip():
        raise ProtocolError("Stop acknowledgement must have an empty payload")


@dataclass(frozen=True)
class CartesianMotionPlan:
    chunk_id: str
    staged_waypoints: Waypoints
    starting_measured_state: RobotSnapshot
    source_agent_target: AgentPose
    final_agent_pose: AgentPose
    final_pose_si: PoseSI
    final_rotation: tuple[tuple[float, ...], ...]
    request: DobotMovLRequest
    path_validation: PathValidation
    micro_move_direction: str | None = None
    local_envelope: LocalMicroMoveEnvelope | None = None
    dry_run: bool = field(default=True, init=False)
    physical_authorized: bool = field(default=False, init=False)
    physical_sent: bool = field(default=False, init=False)


def build_motion_plan(
    chunk_id: str,
    waypoints: Waypoints,
    start: RobotSnapshot,
    reference: PoseSI,
    profile: SafetyProfile,
    now: float,
    keepouts: tuple[KeepoutBox, ...] = (),
    *,
    micro_move_direction: str | None = None,
) -> CartesianMotionPlan:
    envelope = profile.envelope if isinstance(profile, LocalEnvelopeSafety) else None
    if envelope is not None and (
        micro_move_direction != "+Z"
        or start != envelope.start
        or chunk_id != envelope.chunk_id
        or not envelope.bound_at <= now < envelope.expires_at
    ):
        raise SafetyRejected("local envelope is bound to one start, +Z target, chunk and expiry")
    original_profile = profile
    if micro_move_direction is not None:
        if start.native_decimal is None:
            raise SafetyRejected("micro-move requires native decimal measurement")
        profile = decimal_micro_profile(
            start.native_decimal,
            start.native_decimal.translated_10mm(micro_move_direction),
            profile,
        )
    validation = validate_cartesian_path(waypoints, start, reference, profile, now, keepouts)
    if micro_move_direction is not None:
        validation = replace(
            validation,
            total_displacement_m=0.010,
            displacement_limit_m=min(0.010, original_profile.max_translation_step),
        )
    target = action_target(waypoints[-1], start, reference, profile)
    native_decimal = None
    if micro_move_direction is not None:
        native_start = start.native_decimal
        if native_start is None or from_native(native_start.native) != start.pose:
            raise SafetyRejected(
                "micro-move requires original native decimals matching measured SI"
            )
        native_decimal = native_start.translated_10mm(micro_move_direction)
        exact_target = from_native(native_decimal.native)
        expected = (*exact_target.xyz, 0.0, 0.0, 0.0, 0.0)
        if reference != start.pose or waypoints != (expected,) or target != exact_target:
            raise SafetyRejected(
                "micro-move must be exactly one 10mm target preserving measured orientation"
            )
    request = DobotMovLRequest(
        native_decimal.native if native_decimal is not None else to_native(target),
        profile.user_frame,
        profile.tool_frame,
        profile.speed_percent,
        profile.acceleration_percent,
        native_decimal,
    )
    # Protocol conversion must preserve the validated endpoint, including all bounds.
    validate_target(from_native(request.pose), start, profile, now, reference=reference)
    if translation_distance(
        start.pose, from_native(request.pose)
    ) > validation.displacement_limit_m + (1e-12 if micro_move_direction is not None else 0):
        raise SafetyRejected("REJECT: converted native endpoint exceeds aggregate displacement")
    if any(box.intersects(start.pose, from_native(request.pose)) for box in keepouts):
        raise SafetyRejected("REJECT: converted native endpoint path intersects keepout")
    return CartesianMotionPlan(
        chunk_id,
        waypoints,
        start,
        waypoints[-1],
        waypoints[-1],
        target,
        tuple(tuple(float(v) for v in row) for row in native_rotation(target)),
        request,
        validation,
        micro_move_direction,
        envelope,
    )


def build_micro_move_plan(
    chunk_id: str,
    start: RobotSnapshot,
    direction: str,
    profile: SafetyProfile,
    now: float,
    keepouts: tuple[KeepoutBox, ...] = (),
) -> CartesianMotionPlan:
    """One operator-selected native decimal 10mm translation; never regenerate at send."""
    if start.native_decimal is None or start.joints_synthetic:
        raise SafetyRejected("micro-move requires non-synthetic native measured pose")
    target = from_native(start.native_decimal.translated_10mm(direction).native)
    return build_motion_plan(
        chunk_id,
        ((*target.xyz, 0.0, 0.0, 0.0, 0.0),),
        start,
        start.pose,
        profile,
        now,
        keepouts,
        micro_move_direction=direction,
    )


@dataclass(frozen=True)
class MotionExecutionResult:
    chunk_id: str
    prospective_command: str
    fake_command_id: int
    final_measured_pose: PoseSI
    position_residual: float
    orientation_residual: float
    settle_duration: float
    status: str = "fake_settled"
    physical_sent: bool = field(default=False, init=False)
