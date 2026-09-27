"""In-memory Dashboard demo peer. This module cannot construct a network socket."""

from __future__ import annotations

from .types import PoseSI
from .units import to_native


class MockMotionSocket:
    """One predeclared command/reply, then synthetic measured convergence.

    This is a byte-transport demonstration, not a controller or safety simulator.
    The tests use a separate fault-injecting peer.
    """

    def __init__(self, initial: PoseSI) -> None:
        self.measured = initial
        self.target: PoseSI | None = None
        self.expected_motion: bytes | None = None
        self.sent: list[bytes] = []
        self.closed = False
        self._buffer = bytearray()
        self._command_id = 42

    def settimeout(self, value: float) -> None:
        if value <= 0:
            raise ValueError("positive timeout required")

    def sendall(self, data: bytes) -> None:
        if self.closed:
            raise OSError("mock stream closed")
        self.sent.append(data)
        if data == b"RobotMode()":
            payload = "5"
        elif data == b"GetErrorID()":
            payload = "[]"
        elif data.startswith(b"GetPose(user=0,tool="):
            payload = ",".join(str(v) for v in to_native(self.measured).values)
        elif data == b"GetAngle()":
            payload = "0,0,0,0,0,0"
        elif data == b"GetCurrentCommandID()":
            payload = str(self._command_id)
        elif data == b"Stop()":
            payload = ""
        elif data == self.expected_motion and self.target is not None and self._command_id == 42:
            self.measured = self.target
            self._command_id = 43
            payload = "43"
        else:
            raise AssertionError(f"unplanned mock command {data!r}")
        self._buffer.extend(b"0,{" + payload.encode() + b"}," + data + b";")

    def recv(self, size: int) -> bytes:
        if not self._buffer:
            raise TimeoutError("mock stream has no response")
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def close(self) -> None:
        self.closed = True
