"""Pure V4.6.5 Dashboard queries and bounded framing; no network or motion sender."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum

from .errors import CommandRejected, ProtocolError
from .types import JointPositions, NativePose, PoseSI, RobotMode
from .units import deg_to_rad, from_native


class Query(Enum):
    MODE = "RobotMode"
    POSE = "GetPose"
    JOINTS = "GetAngle"
    ERRORS = "GetErrorID"
    COMMAND_ID = "GetCurrentCommandID"


def serialize_query(query: Query, *, user: int | None = None, tool: int | None = None) -> bytes:
    """Only enumerated read queries; no arbitrary command escape hatch."""
    if not isinstance(query, Query):
        raise ProtocolError("only an enumerated query can be serialized")
    if query is not Query.POSE and (user is not None or tool is not None):
        raise ProtocolError("frames only apply to GetPose")
    if (user is None) != (tool is None):
        raise ProtocolError("GetPose requires both user and tool or neither")
    if user is not None:
        if any(type(v) is not int or not 0 <= v <= 50 for v in (user, tool)):
            raise ProtocolError("frame indices must be integers in [0,50]")
        return f"{query.value}(user={user},tool={tool})".encode("ascii")
    return f"{query.value}()".encode("ascii")


class ResponseFramer:
    """Incremental TCP byte stream framing; semicolon terminates query replies."""

    def __init__(self, max_bytes: int = 65536) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._buffer = bytearray()
        self._limit = max_bytes

    def feed(self, data: bytes) -> list[bytes]:
        output: list[bytes] = []
        for part in data.split(b";")[:-1]:
            self._buffer.extend(part)
            if len(self._buffer) + 1 > self._limit:
                self._buffer.clear()
                raise ProtocolError("Dashboard response exceeded framing limit")
            output.append(bytes(self._buffer) + b";")
            self._buffer.clear()
        self._buffer.extend(data.split(b";")[-1])
        if len(self._buffer) > self._limit:
            self._buffer.clear()
            raise ProtocolError("Dashboard response exceeded framing limit")
        return output

    def finish(self) -> None:
        if self._buffer:
            raise ProtocolError("Dashboard connection ended with a truncated response")

    @property
    def pending_bytes(self) -> bytes:
        return bytes(self._buffer)


@dataclass(frozen=True)
class DashboardResponse:
    error_id: int
    payload: str
    echo: str

    def require_success(self) -> DashboardResponse:
        if self.error_id != 0:
            raise CommandRejected(self.error_id, self.echo)
        return self


def parse_response(data: bytes, *, expected_command: bytes) -> DashboardResponse:
    try:
        text, expected = data.decode("ascii").strip(), expected_command.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ProtocolError("Dashboard query response is not ASCII") from exc
    match = re.fullmatch(r"(-?\d+),\{([^{}]*)\},([A-Za-z][A-Za-z0-9]*\([^;]*\));", text)
    if match is None:
        raise ProtocolError("malformed Dashboard ErrorID/value/echo response")
    error, payload, echo = match.groups()

    def normalize(s: str) -> str:
        return re.sub(r"\s+", "", s).casefold()

    if normalize(echo) != normalize(expected):
        raise ProtocolError(f"response echo {echo!r} does not match requested query")
    return DashboardResponse(int(error), payload, echo).require_success()


def _numbers(response: DashboardResponse, count: int) -> tuple[float, ...]:
    response.require_success()
    try:
        values = tuple(float(v.strip()) for v in response.payload.split(","))
    except ValueError as exc:
        raise ProtocolError("expected numeric response values") from exc
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise ProtocolError(f"expected exactly {count} finite response values")
    return values


def decode_pose(response: DashboardResponse) -> PoseSI:
    return from_native(NativePose(*_numbers(response, 6)))


def decode_joints(response: DashboardResponse) -> JointPositions:
    a, b, c, d, e, f = _numbers(response, 6)
    return tuple_six(
        deg_to_rad(a), deg_to_rad(b), deg_to_rad(c), deg_to_rad(d), deg_to_rad(e), deg_to_rad(f)
    )


def tuple_six(a: float, b: float, c: float, d: float, e: float, f: float) -> JointPositions:
    return a, b, c, d, e, f


def decode_command_id(response: DashboardResponse) -> int:
    response.require_success()
    if re.fullmatch(r"\d+", response.payload.strip()) is None:
        raise ProtocolError("expected nonnegative integer command ID")
    return int(response.payload)


def decode_mode(response: DashboardResponse) -> RobotMode:
    try:
        return RobotMode(decode_command_id(response))
    except ValueError as exc:
        raise ProtocolError("unknown RobotMode value") from exc


def decode_errors(response: DashboardResponse) -> tuple[int, ...]:
    response.require_success()
    try:
        values = json.loads(response.payload)
    except ValueError as exc:
        raise ProtocolError("invalid alarm array") from exc
    if not isinstance(values, list) or any(type(v) is not int for v in values):
        raise ProtocolError("V4.6.5 alarm values must be a flat integer array")
    return tuple(values)
