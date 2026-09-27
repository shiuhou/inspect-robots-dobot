"""One-attempt durable evidence. Failure latches; cleanup must remain possible.

Disk writes are synchronous and fsynced, but not a real-time safety channel.
Physical E-stop remains independent of this process and its filesystem.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any


class EvidenceError(RuntimeError):
    """Required evidence could not be persisted; never continue normal execution."""


class AttemptEvidence:
    FILES = (
        "preflight.json",
        "reviewed_plan.json",
        "raw_dashboard.log",
        "execution.json",
        "settle_samples.json",
        "final_report.md",
    )

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.failed: str | None = None
        self.events: list[dict[str, Any]] = []
        # Require a NEW directory. Atomic mkdir also arbitrates concurrent invocations.
        # Never truncate existing evidence, including partially completed attempts.
        directory.mkdir(parents=True, exist_ok=False)
        for name in self.FILES:
            with (directory / name).open("x") as handle:
                handle.write(
                    "NOT RUN\n" if name.endswith(".md") else "" if name.endswith(".log") else "{}\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        self.write(
            "final_report.md",
            "# Phase 4B — BLOCKED / NOT RUN\n\nAttempt incomplete. No PASS evidence.\n",
        )

    def _fail(self, exc: OSError) -> None:
        first = self.failed is None
        self.failed = self.failed or str(exc)
        if first:
            raise EvidenceError(f"required evidence persistence failed: {exc}") from exc

    def write(self, name: str, value: Any) -> None:
        if name not in self.FILES or name == "raw_dashboard.log":
            raise ValueError("unknown evidence artifact")
        data = (
            value if name.endswith(".md") else json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
        temporary = self.directory / (name + ".pending")
        try:
            with temporary.open("w") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.directory / name)
            fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            self._fail(exc)

    def record(self, event: dict[str, Any]) -> None:
        entry = copy.deepcopy(event)
        entry.update(sequence=len(self.events), host_wall_time_ns=time.time_ns())
        self.events.append(entry)
        try:
            # Includes transitions/measurements as well as incremental wire stages,
            # making actual cross-layer order recoverable even after a process crash.
            with (self.directory / "raw_dashboard.log").open("a") as handle:
                handle.write(json.dumps(entry, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            self._fail(exc)

    def require_healthy(self) -> None:
        if self.failed is not None:
            raise EvidenceError(self.failed)
