"""Persist adapter attempts through Inspect Robots trial metadata, not parallel logs."""

from __future__ import annotations

from inspect_robots.logging.sink import NullSink
from inspect_robots.rollout import TrialRecord

from .embodiment import DobotEmbodiment


class DobotAuditSink(NullSink):
    """Pair with JsonLogSink; includes failed attempts that have no StepResult.

    Core passes the same metadata mapping into SceneResult after on_trial_end,
    so these records are persisted by the standard aggregate JSON sink.
    """

    def __init__(self, embodiment: DobotEmbodiment) -> None:
        self._embodiment = embodiment
        self._start = 0

    def on_trial_start(self, scene_id: str, epoch: int) -> None:
        self._start = len(self._embodiment.audit_records)

    def on_trial_end(self, record: TrialRecord) -> None:
        record.metadata["dobot_audit"] = list(self._embodiment.audit_records[self._start :])
        record.metadata["dobot_framework_approvals"] = [
            {"t": event.t, "data": dict(event.data)}
            for event in record.events
            if event.kind == "approval"
        ]
