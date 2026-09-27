"""Receive-only 30004 client. A host receive timestamp is not controller freshness."""

from __future__ import annotations

import threading
from contextlib import suppress

from .clock import Clock, SystemClock
from .config import ConnectionConfig
from .errors import (
    ProtocolError,
    StaleFeedback,
    TransportError,
    TransportTimeout,
    UnsupportedProtocol,
)
from .feedback import FeedbackFramer, RawFeedback
from .transport import Deadline, SocketFactory, SocketStream, connect_stream, open_socket


class DobotFeedbackClient:
    """Bounded synchronous receiver for diagnostic sessions; no sends or background IO.

    read_sample consumes until at least one complete packet, choosing the last packet
    in that recv batch. Subsequent calls continue the stream. It does not claim to
    drain the OS buffer or provide a servo-quality latest sample. close invalidates
    cached data. latest never opens a socket or conceals a stale cached sample.
    """

    def __init__(
        self,
        config: ConnectionConfig,
        *,
        socket_factory: SocketFactory = open_socket,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.clock = clock or SystemClock()
        self._factory = socket_factory
        self._stream: SocketStream | None = None
        self._framer = FeedbackFramer()
        self._sample: RawFeedback | None = None
        self._lock = threading.RLock()
        self._last_read_wire = bytearray()

    @property
    def last_read_wire(self) -> bytes:
        """Exact bytes from the most recent read attempt, also retained after failure/close."""
        with self._lock:
            return bytes(self._last_read_wire)

    @property
    def connected(self) -> bool:
        return self._stream is not None

    def connect(self) -> None:
        with self._lock:
            if self.config.expected_protocol_version != "4.6.5":
                raise UnsupportedProtocol("only the V4.6.5 feedback layout is implemented")
            if self._stream is None:
                self._stream = connect_stream(
                    self._factory, self.config.host, self.config.feedback_port, self.config.timeout
                )
                self._framer = FeedbackFramer()
                self._sample = None
                self._last_read_wire.clear()

    def close(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            self._sample = None
            self._framer = FeedbackFramer()
            if stream is not None:
                try:
                    stream.close()
                except OSError as exc:
                    raise TransportError(f"feedback close failed: {exc}") from exc

    def read_sample(self) -> RawFeedback:
        with self._lock:
            if self._stream is None:
                raise TransportError("feedback is not connected; no automatic connect")
            stream = self._stream
            deadline = Deadline(self.clock, self.config.timeout)
            self._last_read_wire.clear()
            try:
                while True:
                    stream.settimeout(deadline.remaining())
                    data = stream.recv(16384)
                    self._last_read_wire.extend(data)
                    now = self.clock.monotonic()
                    deadline.remaining()
                    if not data:
                        self._framer.finish()
                        raise TransportError("feedback stream closed")
                    if len(data) > 16384:
                        raise ProtocolError("socket returned more than requested receive size")
                    packets = self._framer.feed(data, received_at=now)
                    if packets:
                        self._sample = packets[-1]
                        return self.latest()
            except TimeoutError as exc:
                self._discard()
                raise TransportTimeout("feedback packet timed out; no retry attempted") from exc
            except OSError as exc:
                self._discard()
                raise TransportError(f"feedback read failed: {exc}") from exc
            except BaseException:
                self._discard()
                raise

    def _discard(self) -> None:
        with suppress(TransportError):
            self.close()

    def latest(self) -> RawFeedback:
        with self._lock:
            if self._stream is None or self._sample is None:
                raise StaleFeedback("no feedback sample on an open connection")
            if not self._sample.is_fresh(self.clock.monotonic(), self.config.feedback_max_age):
                raise StaleFeedback("cached feedback exceeds host receive-age limit")
            return self._sample

    def age(self) -> float | None:
        with self._lock:
            return None if self._sample is None else self._sample.age(self.clock.monotonic())
