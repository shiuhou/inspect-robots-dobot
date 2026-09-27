import json

import numpy as np
import pytest
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_dobot.audit import DobotAuditSink
from inspect_robots_dobot.errors import SafetyRejected


def test_sink_keeps_failed_attempts_and_scopes_each_trial(embodiment):
    sink = DobotAuditSink(embodiment)
    scene = Scene(id="one", instruction="offline rejection")
    sink.on_trial_start("one", 0)
    embodiment.reset(scene)
    with pytest.raises(SafetyRejected):
        embodiment.step(Action(np.array([float("inf"), 0, 0.2, 0, 0, 0, 0])))
    record = TrialRecord(scene_id="one", epoch=0, seed=0)
    sink.on_trial_end(record)
    attempts = record.metadata["dobot_audit"]
    assert len(attempts) == 1 and attempts[0]["status"] == "failed"
    json.dumps(record.metadata, allow_nan=False)
    sink.on_trial_start("two", 0)
    second = TrialRecord(scene_id="two", epoch=0, seed=0)
    sink.on_trial_end(second)
    assert second.metadata["dobot_audit"] == []
