"""V4.6.5 query-only Dashboard client. No generic command or ownership sender."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from typing import Any, TypeVar

from .clock import Clock, SystemClock
from .config import ConnectionConfig
from .errors import (
    CommandRejected,
    ControlNotOwned,
    ProtocolError,
    TransportError,
    TransportTimeout,
    UnsupportedProtocol,
)
from .protocol import (
    DashboardResponse,
    Query,
    ResponseFramer,
    decode_command_id,
    decode_errors,
    decode_joints,
    decode_mode,
    decode_pose,
    parse_response,
    serialize_query,
)
from .transport import Deadline, SocketFactory, SocketStream, connect_stream, open_socket
from .types import JointPositions, PoseSI, RobotMode

T = TypeVar("T")
SUPPORTED_PROTOCOL = "4.6.5"


class DobotDashboardClient:
    """Serial request/reply, one outstanding query, no automatic reconnect or retry.

    Extra coalesced replies are rejected, never reused for a future query. An I/O
    or protocol failure closes the stream so a late reply cannot be misattributed.
    A well-formed controller rejection retains the stream for independent queries.
    """

    def __init__(
        self,
        config: ConnectionConfig,
        *,
        socket_factory: SocketFactory = open_socket,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self._factory = socket_factory
        self.clock = clock or SystemClock()
        self._stream: SocketStream | None = None
        self._lock = threading.RLock()
        self._queries_sent = 0
        self._exchanges: deque[dict[str, Any]] = deque(maxlen=128)

    @property
    def exchanges(self) -> tuple[dict[str, Any], ...]:
        """Bounded, exact wire evidence retained after close; no extra query is issued."""
        with self._lock:
            return tuple(dict(record) for record in self._exchanges)

    @property
    def connected(self) -> bool:
        return self._stream is not None

    @property
    def queries_sent(self) -> int:
        return self._queries_sent

    def connect(self) -> None:
        with self._lock:
            if self.config.expected_protocol_version != SUPPORTED_PROTOCOL:
                raise UnsupportedProtocol("Dashboard implementation only verifies document 4.6.5")
            if self._stream is None:
                self._stream = connect_stream(
                    self._factory, self.config.host, self.config.dashboard_port, self.config.timeout
                )

    def close(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.close()
                except OSError as exc:
                    raise TransportError(f"Dashboard close failed: {exc}") from exc

    def _discard(self) -> None:
        # Preserve the original query failure; the stream is detached.
        with suppress(TransportError):
            self.close()

    def _query(
        self,
        query: Query,
        decode: Callable[[DashboardResponse], T],
        *,
        user: int | None = None,
        tool: int | None = None,
    ) -> T:
        # The enum serializer is the sole source of all outbound bytes.
        command = serialize_query(query, user=user, tool=tool)
        with self._lock:
            if self.config.tcp_control_owned is False:
                raise ControlNotOwned(
                    "TCP ownership declared unowned; manual p8 requires TCP mode. "
                    "No RequestControl or recovery command was sent"
                )
            if self._stream is None:
                raise TransportError("Dashboard is not connected; no automatic connect")
            stream = self._stream
            deadline = Deadline(self.clock, self.config.timeout)
            framer = ResponseFramer()
            received = bytearray()
            record: dict[str, Any] = {
                "command": command.decode("ascii"),
                "started_at": self.clock.monotonic(),
                "send_attempted": False,
                "send_completed": False,
                "error_id": None,
                "result": "INCOMPLETE",
            }
            try:
                stream.settimeout(deadline.remaining())
                self._queries_sent += 1
                record["send_attempted"] = True
                stream.sendall(command)
                record["send_completed"] = True
                while True:
                    stream.settimeout(deadline.remaining())
                    data = stream.recv(4096)
                    received.extend(data)
                    deadline.remaining()
                    if not data:
                        framer.finish()
                        raise TransportError("Dashboard closed before a complete response")
                    if len(data) > 4096:
                        raise ProtocolError("socket returned more than requested receive size")
                    responses = framer.feed(data)
                    if responses:
                        if len(responses) != 1 or framer.pending_bytes.strip():
                            raise ProtocolError(
                                "unsolicited/coalesced Dashboard reply; no pipelining"
                            )
                        response = parse_response(responses[0], expected_command=command)
                        record["error_id"] = response.error_id
                        decoded = decode(response)
                        record["result"] = "OK"
                        return decoded
            except CommandRejected as exc:
                record.update(error_id=exc.error_id, result="CONTROLLER_ERROR", reason=str(exc))
                raise
            except TimeoutError as exc:
                record.update(result="TIMEOUT", reason=str(exc))
                self._discard()
                raise TransportTimeout("Dashboard query timed out; no retry attempted") from exc
            except OSError as exc:
                record.update(result="IO_ERROR", reason=str(exc))
                self._discard()
                raise TransportError(f"Dashboard query I/O failed: {exc}") from exc
            except BaseException as exc:
                record.update(result=type(exc).__name__, reason=str(exc))
                self._discard()
                raise
            finally:
                record["received_at"] = self.clock.monotonic()
                record["response_hex"] = received.hex()
                try:
                    record["response_ascii"] = received.decode("ascii")
                except UnicodeDecodeError:
                    record["response_ascii"] = None
                self._exchanges.append(record)

    def robot_mode(self) -> RobotMode:
        return self._query(Query.MODE, decode_mode)

    def get_pose(self) -> PoseSI:
        return self._query(
            Query.POSE, decode_pose, user=self.config.user_frame, tool=self.config.tool_frame
        )

    def get_joints(self) -> JointPositions:
        return self._query(Query.JOINTS, decode_joints)

    def get_errors(self) -> tuple[int, ...]:
        return self._query(Query.ERRORS, decode_errors)

    def current_command_id(self) -> int:
        # Source: TCP manual pp117-118. Read-only, never used as motion authority.
        return self._query(Query.COMMAND_ID, decode_command_id)
