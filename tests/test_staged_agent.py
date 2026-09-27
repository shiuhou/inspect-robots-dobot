import json
import re
from dataclasses import replace

import httpx
import numpy as np
import pytest
from inspect_robots.logging.json_log import JsonLogSink
from inspect_robots.logging.sink import NullSink
from inspect_robots.scene import Scene
from inspect_robots.task import Task
from inspect_robots_agent import LLMAgentPolicy
from test_agent_integration import response_for

from inspect_robots_dobot.camera import FakeCamera
from inspect_robots_dobot.chunks import DobotExecutionSession, StagedDobotEmbodiment
from inspect_robots_dobot.config import CameraConfig, DobotConfig
from inspect_robots_dobot.errors import SafetyRejected


class Recorder(NullSink):
    def __init__(self):
        self.steps, self.records, self.messages = [], [], []

    def log_step(self, t, observation, action, result):
        self.steps.append((t, observation, action, result))

    def on_trial_end(self, record):
        self.records.append(record)

    def log_policy_messages(self, t, messages):
        self.messages.extend(messages)


def test_mocked_public_agent_real_rollout_one_prospective_movl(profile, driver, clock, tmp_path):
    camera_config = CameraConfig(8, 6, 0.2, 0.1)
    camera = FakeCamera(camera_config, clock)
    emb = StagedDobotEmbodiment(
        DobotConfig(profile, 10, camera_config), driver=driver, camera=camera
    )
    payloads, checked, approved = [], [], []

    def scripted(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            response = response_for({"x": 0.318}, 1).json()
            response["choices"][0]["message"]["tool_calls"].append(
                {
                    "id": "post-move-picture",
                    "type": "function",
                    "function": {
                        "name": "take_pic",
                        "arguments": json.dumps({"cameras": ["table"], "note": "Check arrival"}),
                    },
                }
            )
            return httpx.Response(200, json=response)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "done",
                                    "type": "function",
                                    "function": {
                                        "name": "done",
                                        "arguments": json.dumps({"summary": "Offline complete"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    def pre_check(rows):
        assert rows.shape[1] == 7 and not rows.flags.writeable
        checked.append(rows.copy())
        return emb.pre_check(rows)

    class CountApprovals:
        def review(self, action, store):
            approved.append(action.data.copy())
            return action

    policy = LLMAgentPolicy(
        model="offline",
        base_url="http://offline.invalid/v1",
        env={},
        transport=httpx.MockTransport(scripted),
        pre_check=pre_check,
        images="on_demand",
    )
    recorder = Recorder()
    with DobotExecutionSession(emb, extra_approvers=(CountApprovals(),)) as session:
        logs = session.evaluate(
            Task(
                name="staged-agent",
                scenes=[Scene(id="one", instruction="Translate 18mm in fake")],
                scorer=[],
                max_steps=10,
            ),
            policy=policy,
            log_dir=str(tmp_path),
            sinks=[recorder, JsonLogSink(str(tmp_path))],
        )
    assert logs[0].status == "success"
    assert len(payloads) == 2 and len(checked) == 1
    rows = checked[0]
    assert rows.shape == (2, 7)
    assert len(driver.commands) == len(emb.plans) == len(emb.execution_results) == 1
    assert emb.plans[0].path_validation.waypoint_count == 2
    np.testing.assert_array_equal(approved[:2], rows)
    assert len(approved) == 3  # final done action is also reviewed
    assert recorder.steps[0][3].observation.state["eef_state"][0] == 0.3
    final = recorder.steps[1][3]
    assert final.observation.state["eef_state"][0] == 0.318
    assert final.observation.image_times["table"] > final.info["dobot_motion"]["settled_at"]
    np.testing.assert_allclose(rows[:, 1:], [[0, 0.2, 0, 0, 0, 0]] * 2)
    schema = next(t["function"] for t in payloads[0]["tools"] if t["function"]["name"] == "move_to")
    assert "x, y, z, yaw, pitch, roll, gripper" in json.dumps(schema)
    assert {t["function"]["name"] for t in payloads[0]["tools"]} == {
        "move_to",
        "done",
        "give_up",
        "take_pic",
    }
    assert recorder.messages and recorder.records[0].policy_transcript
    inferences = [event for event in recorder.records[0].events if event.kind == "inference"]
    assert [e.data["chunk_len"] for e in inferences] == [2, 1]
    assert recorder.records[0].inference_latencies == []  # agent emits no chunk latency
    text = json.dumps(payloads[1])
    residual = re.search(r"Largest remaining offset.*?is ([\deE.+-]+) on (\w+)", text)
    assert residual and float(residual.group(1)) < 1e-12
    assert "motion finished playing" in text
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    audit = saved["samples"][0]["trial_metadata"][0]["dobot_audit"]
    assert len([a for a in audit if a["kind"] == "prospective_movl"]) == 1
    assert len([a for a in audit if a["kind"] == "waypoint_staged"]) == 2
    assert all(not a.get("physical_sent", False) for a in audit)


@pytest.mark.parametrize("mode", ["max_steps", "reject", "rewrite", "cancel", "sink_failure"])
def test_actual_rollout_partial_chunk_never_commits(profile, driver, tmp_path, mode):
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    recorder = Recorder()

    class Guard:
        calls = 0

        def review(self, action, store):
            self.calls += 1
            if self.calls == 2 and mode == "reject":
                raise SafetyRejected("operator refused")
            if self.calls == 2 and mode == "rewrite":
                return replace(action, data=action.data - np.array([0.001, 0, 0, 0, 0, 0, 0]))
            return action

    class FailingSink(NullSink):
        def log_step(self, t, observation, action, result):
            if mode == "cancel":
                raise KeyboardInterrupt
            if mode == "sink_failure":
                raise RuntimeError("logging failed")

    policy = LLMAgentPolicy(
        model="offline",
        base_url="http://offline.invalid/v1",
        env={},
        transport=httpx.MockTransport(lambda _: response_for({"x": 0.318}, 1)),
        pre_check=emb.pre_check,
    )
    with DobotExecutionSession(emb, extra_approvers=(Guard(),)) as session:
        try:
            logs = session.evaluate(
                Task(
                    name="failure",
                    scenes=[Scene(id="one", instruction="Offline")],
                    scorer=[],
                    max_steps=1 if mode == "max_steps" else 2,
                ),
                policy=policy,
                log_dir=str(tmp_path),
                sinks=[recorder, FailingSink()],
            )
            assert mode != "cancel"
            if mode in ("reject", "rewrite"):
                assert logs[0].status == "error"
        except KeyboardInterrupt:
            assert mode == "cancel"
        except RuntimeError:
            assert mode == "sink_failure"
        assert emb.pending_chunk_id is None
    assert not driver.commands and not emb.plans
    assert any(a["kind"] == "chunk_aborted" for a in emb.audit_records)
