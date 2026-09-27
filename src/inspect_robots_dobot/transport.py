"""Lazy socket construction for explicitly invoked read-only diagnostics only."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from ipaddress import ip_address
from typing import Protocol

from .clock import Clock
from .errors import PhaseUnavailable, TransportError, TransportTimeout


class SocketStream(Protocol):
    """Small injected socket interface. Implementations must honor timeout/recv size."""

    def settimeout(self, value: float) -> None: ...
    def sendall(self, data: bytes) -> None: ...
    def recv(self, size: int) -> bytes: ...
    def close(self) -> None: ...


SocketFactory = Callable[[str, int, float], SocketStream]


def open_socket(host: str, port: int, timeout: float) -> SocketStream:
    """One literal-IP connect, no DNS, retries, scans or bytes sent on connection."""
    address = ip_address(host)
    stream = socket.socket(socket.AF_INET if address.version == 4 else socket.AF_INET6)
    try:
        stream.settimeout(timeout)
        stream.connect((host, port))
    except BaseException:
        stream.close()
        raise
    return stream


class Deadline:
    """Bound a whole operation, not each fragment; wall time also bounds fake clocks."""

    def __init__(self, clock: Clock, timeout: float) -> None:
        self._clock = clock
        self._end = clock.monotonic() + timeout
        self._wall_end = time.monotonic() + timeout

    def remaining(self) -> float:
        left = min(self._end - self._clock.monotonic(), self._wall_end - time.monotonic())
        if left <= 0:
            raise TransportTimeout("read-only operation deadline expired; no retry attempted")
        return left


def connect_stream(
    factory: SocketFactory, host: str | None, port: int, timeout: float
) -> SocketStream:
    if host is None:
        raise TransportError("an explicit host is required; no default robot is contacted")
    try:
        return factory(host, port, timeout)
    except TimeoutError as exc:
        raise TransportTimeout(f"connection to {host}:{port} timed out") from exc
    except OSError as exc:
        raise TransportError(f"connection to {host}:{port} failed: {exc}") from exc


def real_transport_unavailable() -> None:
    """Legacy fail-closed seam; never upgrades older callers into a live connection."""
    raise PhaseUnavailable(
        "Phase 2 read-only access requires explicit construction/connection of the new clients; "
        "the legacy Phase 1 transport entry point remains disabled"
    )
