"""In-memory byte streams. Never wrap or open a real socket."""

from collections import deque


class ScriptedSocket:
    def __init__(self, chunks=(), *, responses=None, send_error=None, on_recv=None):
        self.chunks = deque(chunks)
        self.responses = responses or {}
        self.send_error = send_error
        self.on_recv = on_recv
        self.sent = []
        self.timeouts = []
        self.closed = False

    def settimeout(self, value):
        assert value > 0
        self.timeouts.append(value)

    def sendall(self, data):
        assert not self.closed
        self.sent.append(data)
        if self.send_error:
            raise self.send_error
        if data in self.responses:
            self.chunks.extend(self.responses[data])

    def recv(self, size):
        assert not self.closed
        if self.on_recv:
            self.on_recv()
        if not self.chunks:
            raise TimeoutError("scripted timeout")
        chunk = self.chunks.popleft()
        if isinstance(chunk, BaseException):
            raise chunk
        if len(chunk) > size:
            self.chunks.appendleft(chunk[size:])
        return chunk[:size]

    def close(self):
        self.closed = True


class ScriptedFactory:
    def __init__(self, dashboard=None, feedback=None):
        self.streams = {29999: dashboard, 30004: feedback}
        self.calls = []

    def __call__(self, host, port, timeout):
        self.calls.append((host, port, timeout))
        stream = self.streams[port]
        if isinstance(stream, BaseException):
            raise stream
        if stream is None:
            raise ConnectionRefusedError("scripted unreachable port")
        return stream


def feedback_packet(*, mode=5, command_id=42, timestamp=100000):
    import struct

    # Independently encoded from V4.6.5 pp151-154; no production parser constants.
    packet = bytearray(1440)
    struct.pack_into("<H", packet, 0, 1440)
    struct.pack_into("<Q", packet, 24, mode)
    struct.pack_into("<Q", packet, 32, timestamp)
    struct.pack_into("<Q", packet, 40, 2000)
    struct.pack_into("<Q", packet, 48, 0x0123456789ABCDEF)
    struct.pack_into("<6d", packet, 432, 0, 90, -180, 45, 0, 0)
    struct.pack_into("<6d", packet, 624, 300, -100, 200, 180, 0, 90)
    packet[1012], packet[1013] = 0, 2
    packet[1026], packet[1029], packet[1031], packet[1038] = 1, 0, 160, 0
    struct.pack_into("<Q", packet, 1112, command_id)
    return bytes(packet)


def dashboard_socket(*, mode=5, errors=b"[]"):
    return ScriptedSocket(
        responses={
            b"RobotMode()": [f"0,{{{mode}}},RobotMode();".encode()],
            b"GetPose()": [b"0,{300,-100,200,180,0,90},GetPose();"],
            b"GetPose(user=0,tool=2)": [b"0,{300,-100,200,180,0,90},GetPose(user=0,tool=2);"],
            b"GetAngle()": [b"0,{0,90,-180,45,0,0},GetAngle();"],
            b"GetErrorID()": [b"0,{" + errors + b"},GetErrorID();"],
            b"GetCurrentCommandID()": [b"0,{42},GetCurrentCommandID();"],
        }
    )
