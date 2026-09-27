"""Latest-frame capture with explicit host timestamp provenance; no hardware backend."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .clock import Clock, SystemClock
from .config import CameraConfig
from .errors import CameraFault


@dataclass(frozen=True)
class Frame:
    rgb: npt.NDArray[np.uint8]
    acquisition_started_at: float
    published_at: float
    sequence: int
    generation: int
    timestamp_source: str = "host_monotonic_acquisition_start_and_publication"
    host_receive_time_monotonic: float | None = None

    @property
    def timestamp(self) -> float:
        return (
            self.host_receive_time_monotonic
            if self.host_receive_time_monotonic is not None
            else self.acquisition_started_at
        )


@dataclass(frozen=True)
class ReceivedRgb:
    rgb: npt.NDArray[np.uint8]
    host_receive_time_monotonic: float


def validate_frame(frame: Frame, config: CameraConfig, *, now: float, after: float | None) -> None:
    if after is not None and not math.isfinite(after):
        raise CameraFault("invalid post-settle camera time constraint")
    if frame.rgb.shape != (config.height, config.width, 3) or frame.rgb.dtype != np.uint8:
        raise CameraFault("camera must supply declared HWC uint8 RGB dimensions")
    if not all(
        math.isfinite(v)
        for v in (now, frame.acquisition_started_at, frame.published_at, frame.timestamp)
    ):
        raise CameraFault("non-finite camera timestamp")
    if not 0 <= frame.acquisition_started_at <= frame.timestamp <= frame.published_at <= now:
        raise CameraFault("camera timestamps are inconsistent with host monotonic time")
    if now - frame.timestamp > config.max_age:
        raise CameraFault("camera frame is stale")
    if after is not None and (frame.acquisition_started_at <= after or frame.timestamp <= after):
        raise CameraFault("camera frame host read/receive did not start after settling")


class CameraReader(Protocol):
    def start(self) -> None: ...
    def latest(self, *, after: float | None = None) -> Frame: ...
    def close(self) -> None: ...


class FrameSource(Protocol):
    """read(timeout) and close must be bounded; close must unblock a pending read.

    Physical implementations must document/flush device buffering separately.
    A host read timestamp alone cannot establish sensor exposure time.
    """

    def read(self, timeout: float) -> npt.NDArray[np.uint8] | ReceivedRgb: ...
    def close(self) -> None: ...


class LatestFrameReader:
    """One continuously draining thread; at most one stored frame, never request-triggered read."""

    def __init__(
        self,
        source: FrameSource,
        config: CameraConfig,
        *,
        clock: Clock | None = None,
        startup_timeout: float | None = None,
    ) -> None:
        if startup_timeout is not None and (
            not math.isfinite(startup_timeout) or startup_timeout <= 0
        ):
            raise CameraFault("camera startup timeout must be finite and positive")
        self._source, self.config = source, config
        self._startup_timeout = startup_timeout or config.wait_timeout
        self._clock = clock or SystemClock()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: Frame | None = None
        self._failure: str | None = None
        self._generation = 0
        self._sequence = 0
        self._closed = False

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise CameraFault(
                    "closed camera reader cannot restart; construct a new source/reader"
                )
            if self._thread is not None:
                return
            self._generation += 1
            self._frame, self._failure = None, None
            start_source = getattr(self._source, "start", None)
            if start_source is not None:
                start_source()
            self._thread = threading.Thread(
                target=self._drain, args=(self._generation,), daemon=True
            )
            self._thread.start()

    def _drain(self, generation: int) -> None:
        try:
            first = True
            while not self._stop.is_set():
                started = self._clock.monotonic()
                received = self._source.read(
                    self._startup_timeout if first else self.config.wait_timeout
                )
                first = False
                published = self._clock.monotonic()
                rgb = received.rgb if isinstance(received, ReceivedRgb) else received
                copy = np.array(rgb, copy=True)
                copy.flags.writeable = False
                with self._condition:
                    if self._stop.is_set() or generation != self._generation:
                        return
                    self._sequence += 1
                    frame = Frame(
                        copy,
                        started,
                        published,
                        self._sequence,
                        generation,
                        "host_monotonic_jpeg_complete"
                        if isinstance(received, ReceivedRgb)
                        else "host_monotonic_acquisition_start_and_publication",
                        received.host_receive_time_monotonic
                        if isinstance(received, ReceivedRgb)
                        else None,
                    )
                    validate_frame(frame, self.config, now=published, after=None)
                    self._frame = frame
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                if not self._stop.is_set():
                    self._failure = str(exc)
                self._condition.notify_all()

    def latest(self, *, after: float | None = None) -> Frame:
        # Wall deadline also bounds a broken or manually frozen injected clock.
        deadline = time.monotonic() + (
            self._startup_timeout if self._frame is None else self.config.wait_timeout
        )
        reason = "no camera frame available"
        with self._condition:
            while True:
                if self._closed or self._thread is None:
                    raise CameraFault("camera reader is not running")
                if self._failure is not None:
                    raise CameraFault(f"camera capture failed: {self._failure}")
                if self._frame is not None and self._frame.generation == self._generation:
                    try:
                        validate_frame(
                            self._frame, self.config, now=self._clock.monotonic(), after=after
                        )
                    except CameraFault as exc:
                        reason = str(exc)
                    else:
                        return self._frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraFault(f"camera wait timeout: {reason}")
                self._condition.wait(min(remaining, 0.02))

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._frame = None
            self._stop.set()
            self._condition.notify_all()
        self._source.close()
        if self._thread is not None:
            self._thread.join(timeout=self.config.wait_timeout)
            if self._thread.is_alive():
                raise CameraFault("camera source violated bounded close/read contract")


class FakeCamera:
    """Clock-driven latest-frame simulator, refreshed on simulated acquisition ticks."""

    def __init__(self, config: CameraConfig, clock: Clock, *, period: float = 0.01) -> None:
        if not math.isfinite(period) or period <= 0:
            raise ValueError("fake camera period must be positive and finite")
        self.config, self.clock, self.period = config, clock, period
        self.freeze = False
        self.failure: str | None = None
        self._running = False
        self._generation = 0
        self._frame: Frame | None = None

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._generation += 1
            self._frame = None

    def _refresh(self) -> None:
        now = self.clock.monotonic()
        tick = int(now / self.period)
        if self._frame is None or (not self.freeze and tick > self._frame.sequence):
            rgb = np.full((self.config.height, self.config.width, 3), tick % 256, dtype=np.uint8)
            rgb.flags.writeable = False
            self._frame = Frame(rgb, tick * self.period, now, tick, self._generation)

    def latest(self, *, after: float | None = None) -> Frame:
        if not self._running:
            raise CameraFault("fake camera is not running")
        deadline = self.clock.monotonic() + self.config.wait_timeout
        max_polls = math.ceil(self.config.wait_timeout / self.period) + 3
        reason = "no frame"
        for _ in range(max_polls):
            if self.failure:
                raise CameraFault(self.failure)
            self._refresh()
            assert self._frame is not None
            try:
                validate_frame(self._frame, self.config, now=self.clock.monotonic(), after=after)
                return self._frame
            except CameraFault as exc:
                reason = str(exc)
            if self.clock.monotonic() >= deadline:
                break
            self.clock.sleep(min(self.period, deadline - self.clock.monotonic()))
        raise CameraFault(f"fake camera timeout: {reason}")

    def close(self) -> None:
        self._running = False
        self._frame = None
