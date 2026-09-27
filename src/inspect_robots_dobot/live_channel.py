"""Exclusive single-Dashboard channel for typed one-shot motion and interruption.

No public raw command API and no second connection/reconnect. Phase 4A uses injected
streams only. The read-only Dashboard client is separate and unchanged.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .clock import Clock
from .config import ConnectionConfig
from .errors import CommandRejected, DriverFault, ProtocolError, TransportError
from .live_profile import LiveMotionProfile
from .motion import (
    DobotMovLRequest,
    parse_movl_acceptance,
    parse_stop_acknowledgement,
    serialize_stop,
)
from .protocol import DashboardResponse, Query, ResponseFramer, parse_response, serialize_query
from .transport import Deadline, SocketFactory, SocketStream, connect_stream


class MotionCancelled(DriverFault):
    """Worker observed an operator cancellation; no retry permitted."""


class _LiveDashboardChannel:
    def __init__(
        self,
        connection: ConnectionConfig,
        profile: LiveMotionProfile,
        clock: Clock,
        factory: SocketFactory,
        cancellation: threading.Event,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.connection, self.profile, self.clock = connection, profile, clock
        self._factory, self.cancel = factory, cancellation
        self._stream: SocketStream | None = None
        self._framer = ResponseFramer()
        self._pending: bytes | None = None
        self._writable = True
        self.motion_attempts = 0
        self.stop_attempts = 0
        self.records: list[dict[str, Any]] = []
        self._event_sink = event_sink

    def _emit(self, record: dict[str, Any], stage: str) -> None:
        if self._event_sink is not None:
            self._event_sink(
                {"kind": "wire", **record, "stage": stage, "timestamp": self.clock.monotonic()}
            )

    def connect(self) -> None:
        if self._stream is not None:
            raise TransportError("live channel already connected")
        self._stream = connect_stream(
            self._factory, self.connection.host, 29999, self.profile.io_timeout
        )

    def close(self) -> None:
        stream, self._stream = self._stream, None
        self._writable = False
        if stream is not None:
            stream.close()

    @property
    def connected(self) -> bool:
        return self._stream is not None

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise MotionCancelled("operator cancellation requested")

    def query(
        self, query: Query, *, deadline: Deadline, stop_monitor: bool = False
    ) -> DashboardResponse:
        frames = (
            {"user": self.connection.user_frame, "tool": self.connection.tool_frame}
            if query is Query.POSE
            else {}
        )
        command = serialize_query(query, **frames)
        return parse_response(
            self._exchange(command, deadline=deadline, stopping=stop_monitor),
            expected_command=command,
        )

    def _move_once(self, request: DobotMovLRequest, before_send: Callable[[], None]) -> int:
        if type(request) is not DobotMovLRequest:
            raise ProtocolError("only the immutable DobotMovLRequest is accepted")
        if self.motion_attempts:
            raise DriverFault("one motion attempt per channel; resend is forbidden")

        def consume() -> None:
            before_send()
            self.motion_attempts += 1

        raw = self._exchange(request.serialize().encode("ascii"), before_send=consume)
        return parse_movl_acceptance(raw, request)

    def stop_once(self) -> None:
        if self.stop_attempts:
            raise DriverFault("Stop already attempted for this abort episode")
        self.stop_attempts += 1
        if not self._writable:
            raise TransportError("Stop unavailable: partial/failed write or disconnected stream")
        # Exception-only best effort after a complete earlier write: match late echo
        # independently. Actual firmware reliability/latency remains unverified.
        parse_stop_acknowledgement(self._exchange(serialize_stop().encode(), stopping=True))

    def _exchange(
        self,
        command: bytes,
        *,
        deadline: Deadline | None = None,
        before_send: Callable[[], None] | None = None,
        stopping: bool = False,
    ) -> bytes:
        allowed = {
            serialize_query(
                q,
                **(
                    {"user": self.connection.user_frame, "tool": self.connection.tool_frame}
                    if q is Query.POSE
                    else {}
                ),
            )
            for q in Query
        }
        if (
            command not in allowed
            and command != b"Stop()"
            and not (before_send is not None and command.startswith(b"MovL(pose={"))
        ):
            raise ProtocolError("live channel only supports typed queries, reviewed MovL and Stop")
        stream = self._stream
        if stream is None or not self._writable:
            raise TransportError("live channel unavailable; no reconnect")
        if self._pending is not None and not stopping:
            raise ProtocolError("unresolved Dashboard response; only best-effort Stop is permitted")
        local = Deadline(self.clock, self.profile.acknowledgement_timeout)

        def remaining() -> float:
            return min(local.remaining(), deadline.remaining() if deadline else float("inf"))

        prior = self._pending if stopping else None
        received = bytearray()
        record: dict[str, Any] = {
            "command": command.decode(),
            "started_at": self.clock.monotonic(),
            "write_attempted": False,
            "write_completed": False,
        }
        self.records.append(record)
        try:
            self._emit(record, "write_intent")
            if not stopping:
                self._check_cancel()
            stream.settimeout(min(self.profile.io_timeout, remaining()))
            if before_send is not None:
                before_send()
            record["write_attempted"] = True
            # Any exception from sendall can mean a prefix reached the controller.
            try:
                stream.sendall(command)
            except BaseException:
                self._writable = False
                raise
            record["write_completed"] = True
            self._pending = command
            self._emit(record, "write_completed")
            while True:
                if not stopping:
                    self._check_cancel()
                stream.settimeout(min(self.profile.io_timeout, remaining()))
                try:
                    data = stream.recv(4096)
                except TimeoutError:
                    remaining()
                    if not stopping:
                        self._check_cancel()
                    # Retry receiving, never sending. A real recv waited its timeout;
                    # this tick also bounds fake/nonconforming immediate timeout streams.
                    self.clock.sleep(min(0.001, remaining()))
                    continue
                received.extend(data)
                record.update(
                    response_hex=received.hex(),
                    response_ascii=received.decode("ascii", errors="backslashreplace"),
                )
                self._emit(record, "received_bytes")
                remaining()
                if not data:
                    self._writable = False
                    raise TransportError("Dashboard disconnected before response")
                if len(data) > 4096:
                    raise ProtocolError("invalid recv length")
                frames = self._framer.feed(data)
                matched: bytes | None = None
                rejection: CommandRejected | None = None
                for frame in frames:
                    if stopping and prior is not None:
                        try:
                            parse_response(frame, expected_command=prior)
                        except CommandRejected:
                            prior = None
                            continue
                        except ProtocolError:
                            pass
                        else:
                            prior = None
                            continue
                    if matched is not None:
                        raise ProtocolError("extra unsolicited response")
                    # ErrorID rejection still proves framing/echo, not motion acceptance.
                    try:
                        parse_response(frame, expected_command=command)
                    except CommandRejected as exc:
                        rejection = exc
                    matched = frame
                if matched is not None:
                    if self._framer.pending_bytes.strip():
                        raise ProtocolError("trailing partial unsolicited response")
                    if prior is not None:
                        raise ProtocolError(
                            "Stop reply received but prior reply remains unresolved; "
                            "standstill queries cannot safely reuse this stream"
                        )
                    self._pending = None
                    if rejection is not None:
                        raise rejection
                    record["result"] = "ACKNOWLEDGED"
                    record["error_id"] = 0
                    if command.startswith(b"MovL("):
                        # Preserve parsed acceptance before handing control to monitoring.
                        from .protocol import decode_command_id

                        record["result_id"] = decode_command_id(
                            parse_response(matched, expected_command=command)
                        )
                    return matched
        except BaseException as exc:
            record.update(result=type(exc).__name__, reason=str(exc))
            if isinstance(exc, CommandRejected):
                record["error_id"] = exc.error_id
            if isinstance(exc, OSError) and not isinstance(exc, TimeoutError):
                self._writable = False
            raise
        finally:
            record.update(
                received_at=self.clock.monotonic(),
                response_hex=received.hex(),
                response_ascii=received.decode("ascii", errors="backslashreplace"),
            )
            self._emit(record, "exchange_finished")
