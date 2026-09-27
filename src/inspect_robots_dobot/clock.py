"""Injectable monotonic time for bounded polling and deterministic tests."""

from __future__ import annotations

import math
import time
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeClock:
    """Single-threaded simulation clock; advancing never sleeps on wall time."""

    def __init__(self, start: float = 0.0) -> None:
        if not math.isfinite(start) or start < 0:
            raise ValueError("clock start must be finite and nonnegative")
        self._now = start

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("clock advance must be finite and nonnegative")
        self._now += seconds
