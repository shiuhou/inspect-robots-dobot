"""Explicit SI/native coordinate types; no ambiguous roll/pitch/yaw aliases."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import IntEnum

from .errors import ProtocolError


class RobotMode(IntEnum):
    """V4.6.5 TCP manual pp35-36."""

    INIT = 1
    BRAKE_OPEN = 2
    POWER_OFF = 3
    DISABLED = 4
    ENABLED_IDLE = 5
    DRAG = 6
    RUNNING = 7
    SINGLE_MOVE = 8
    ERROR = 9
    PAUSED = 10
    COLLISION = 11


@dataclass(frozen=True)
class PoseSI:
    """Metres/radians; native Dobot extrinsic XYZ angles in selected user frame."""

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    @property
    def values(self) -> tuple[float, float, float, float, float, float]:
        return self.x, self.y, self.z, self.rx, self.ry, self.rz

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def with_translation(self, x: float, y: float, z: float) -> PoseSI:
        return PoseSI(x, y, z, self.rx, self.ry, self.rz)


@dataclass(frozen=True)
class NativePose:
    """Protocol-only representation: XYZ millimetres, rx/ry/rz degrees."""

    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float

    @property
    def values(self) -> tuple[float, float, float, float, float, float]:
        return self.x_mm, self.y_mm, self.z_mm, self.rx_deg, self.ry_deg, self.rz_deg


JointPositions = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class NativeDecimalPose:
    """Exact Dashboard decimal values; not angles reconstructed from SI floats.

    Bounded decimal syntax is a parser/resource restriction, not a robot limit.
    Tokens are retained for evidence and unchanged protocol serialization.
    """

    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple) or len(self.values) != 6:
            raise ProtocolError("native decimal pose requires six immutable tokens")
        for token in self.values:
            if (
                not isinstance(token, str)
                or len(token) > 128
                or re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token) is None
            ):
                raise ProtocolError("invalid native decimal pose token")
            value = Decimal(token)
            exponent = value.as_tuple().exponent
            if (
                not math.isfinite(float(value))
                or not isinstance(exponent, int)
                or abs(exponent) > 100
            ):
                raise ProtocolError("native decimal pose exceeds supported numeric range")

    @property
    def native(self) -> NativePose:
        return NativePose(*(float(v) for v in self.values))

    def translated_10mm(self, direction: str) -> NativeDecimalPose:
        if direction not in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
            raise ProtocolError("choose exactly +X/-X/+Y/-Y/+Z/-Z; no default direction")
        index = "XYZ".index(direction[1])
        tokens = list(self.values)
        with localcontext() as context:
            context.prec = 512
            tokens[index] = format(Decimal(tokens[index]) + Decimal(direction[0] + "10"), "f")
        return NativeDecimalPose(tuple(tokens))

    def squared_distance_mm(self, other: NativeDecimalPose) -> Decimal:
        with localcontext() as context:
            context.prec = 512
            return sum(
                (
                    (Decimal(a) - Decimal(b)) ** 2
                    for a, b in zip(self.values[:3], other.values[:3], strict=True)
                ),
                Decimal(0),
            )


@dataclass(frozen=True)
class RobotSnapshot:
    """One host-timestamped sample; fake joints are not an IK-derived estimate."""

    pose: PoseSI
    joints: JointPositions
    mode: RobotMode
    errors: tuple[int, ...]
    command_id: int
    observed_at: float
    user_frame: int
    tool_frame: int
    joints_synthetic: bool = False
    native_decimal: NativeDecimalPose | None = None
