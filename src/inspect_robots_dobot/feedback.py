"""Offline binary subset from V4.6.5 pp151-154; unverified fields stay raw."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .errors import ProtocolError, UnverifiedField
from .types import JointPositions, PoseSI, RobotMode

PACKET_SIZE = 1440
TEST_VALUE = 0x0123456789ABCDEF


@dataclass(frozen=True)
class RawFeedback:
    """Pose/joints intentionally have no inferred units or coordinate frame."""

    robot_mode: int
    controller_unix_ms: int
    runtime_ms: int
    joints_raw: tuple[float, ...]
    tcp_pose_raw: tuple[float, ...]
    user_index: int
    tool_index: int
    enable_status: int
    error_status: int
    robot_type: int
    collision_state: int
    command_id: int
    received_at: float

    def age(self, now: float) -> float:
        if not math.isfinite(now) or now < self.received_at:
            raise ProtocolError("feedback age requires consistent host monotonic time")
        return now - self.received_at

    def pose_si(self) -> PoseSI:
        raise UnverifiedField("ToolVectorActual units/frame unverified; use Dashboard GetPose")

    def joints_si(self) -> JointPositions:
        raise UnverifiedField("QActual units unverified; use Dashboard GetAngle")

    def unverified_field(self, name: str) -> None:
        raise UnverifiedField(f"{name} is not in the source-verified feedback semantic subset")

    def is_fresh(self, now: float, max_age: float) -> bool:
        return (
            math.isfinite(now)
            and math.isfinite(max_age)
            and max_age > 0
            and 0 <= now - self.received_at <= max_age
        )


def parse_feedback(packet: bytes, *, received_at: float) -> RawFeedback:
    if len(packet) != PACKET_SIZE:
        raise ProtocolError(f"feedback frame must be exactly {PACKET_SIZE} bytes")
    if not math.isfinite(received_at) or received_at < 0:
        raise ProtocolError("invalid feedback receive timestamp")
    if struct.unpack_from("<H", packet, 0)[0] != PACKET_SIZE:
        raise ProtocolError("invalid little-endian feedback MessageSize")
    if struct.unpack_from("<Q", packet, 48)[0] != TEST_VALUE:
        raise ProtocolError("invalid feedback TestValue sentinel")
    mode = int(struct.unpack_from("<Q", packet, 24)[0])
    try:
        RobotMode(mode)
    except ValueError as exc:
        raise ProtocolError("unknown feedback RobotMode; document/firmware may differ") from exc
    joints = tuple(float(v) for v in struct.unpack_from("<6d", packet, 432))
    pose = tuple(float(v) for v in struct.unpack_from("<6d", packet, 624))
    if not all(math.isfinite(v) for v in (*joints, *pose)):
        raise ProtocolError("non-finite raw feedback coordinates")
    return RawFeedback(
        mode,
        int(struct.unpack_from("<Q", packet, 32)[0]),
        int(struct.unpack_from("<Q", packet, 40)[0]),
        joints,
        pose,
        packet[1012],
        packet[1013],
        packet[1026],
        packet[1029],
        packet[1031],
        packet[1038],
        int(struct.unpack_from("<Q", packet, 1112)[0]),
        received_at,
    )


class FeedbackFramer:
    """Fixed-size stream framing from connection start; invalid alignment fails closed."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes, *, received_at: float) -> list[RawFeedback]:
        result: list[RawFeedback] = []
        # Buffer at most one frame even if the caller supplies a large coalesced read.
        offset = 0
        while offset < len(data):
            count = min(PACKET_SIZE - len(self._buffer), len(data) - offset)
            self._buffer.extend(data[offset : offset + count])
            offset += count
            if (
                len(self._buffer) >= 2
                and struct.unpack_from("<H", self._buffer, 0)[0] != PACKET_SIZE
            ):
                self._buffer.clear()
                raise ProtocolError("invalid feedback MessageSize/alignment; no resynchronization")
            if len(self._buffer) == PACKET_SIZE:
                packet = bytes(self._buffer)
                self._buffer.clear()
                result.append(parse_feedback(packet, received_at=received_at))
        return result

    def finish(self) -> None:
        if self._buffer:
            raise ProtocolError("truncated feedback packet at end of stream")
