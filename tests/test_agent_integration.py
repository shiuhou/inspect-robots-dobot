import json
import re
from dataclasses import replace

import httpx
import numpy as np
import pytest
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.compat import check_compatibility
from inspect_robots.scene import Scene
from inspect_robots_agent import LLMAgentPolicy

from inspect_robots_dobot.errors import SafetyRejected


def response_for(targets, sequence):
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
                                "id": f"move-{sequence}",
                                "type": "function",
                                "function": {
                                    "name": "move_to",
                                    "arguments": json.dumps(
                                        {
                                            "targets": targets,
                                            "note": "Small synthetic translation.",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )


def test_public_agent_move_to_chunk_approvers_and_fake_execution(embodiment, driver):
    captured = []
    checked = []

    def scripted(request):
        captured.append(json.loads(request.content))
        return response_for({"x": 0.32}, 1)

    def pre_check(waypoints):
        assert waypoints.dtype == np.float64
        assert not waypoints.flags.writeable
        checked.append(waypoints.copy())
        return embodiment.pre_check(waypoints)

    policy = LLMAgentPolicy(
        model="offline-test",
        base_url="http://agent.test/v1",
        env={},
        transport=httpx.MockTransport(scripted),
        pre_check=pre_check,
    )
    scene = Scene(id="fake-agent", instruction="Translate slightly in x")
    observation = embodiment.reset(scene)
    policy.bind(embodiment.info)
    assert check_compatibility(policy, embodiment).ok
    policy.reset(scene)
    chunk = policy.act(observation)
    schema = next(t for t in captured[0]["tools"] if t["function"]["name"] == "move_to")
    assert "targets" in schema["function"]["parameters"]["properties"]
    assert "x, y, z, yaw, pitch, roll, gripper" in json.dumps(schema)
    assert len(chunk.actions) > 1
    assert len(checked) == 1
    np.testing.assert_allclose(checked[0][-1], [0.32, 0, 0.2, 0, 0, 0, 0])
    chain = ChainApprover(
        ClampApprover(embodiment.info.action_space),
        DeltaLimitApprover(embodiment.info.action_space),
        embodiment.contribute_guardrails(embodiment.info.action_space).approvers[0][1],
    )
    store = {}
    for action in chunk.actions:
        assert chain.review(action, store) is action
        result = embodiment.step(action)
        assert result.info["dobot_motion"]["physical_sent"] is False
    assert len(driver.commands) == len(chunk.actions)
    np.testing.assert_allclose(result.observation.state["eef_state"], [0.32, 0, 0.2, 0, 0, 0, 0])


def test_model_can_repair_precheck_rejection_before_any_emission(embodiment, driver):
    payloads = []
    rejections = []

    def check(waypoints):
        # A local fixture obstacle-like constraint exercises the public correction channel.
        # This is a synthetic rejection, NOT a claimed physical collision model.
        if np.any(waypoints[:, 0] > 0.305):
            message = (
                "waypoint target x exceeds synthetic test zone 0.305 m; choose a nearer target"
            )
            rejections.append(message)
            return message
        return embodiment.pre_check(waypoints)

    def scripted(request):
        payloads.append(json.loads(request.content))
        return response_for({"x": 0.31 if len(payloads) == 1 else 0.302}, len(payloads))

    scene = Scene(id="repair", instruction="Translate within test zone")
    observation = embodiment.reset(scene)
    policy = LLMAgentPolicy(
        model="offline-test",
        base_url="http://agent.test/v1",
        env={},
        transport=httpx.MockTransport(scripted),
        pre_check=check,
    )
    policy.bind(embodiment.info)
    policy.reset(scene)
    chunk = policy.act(observation)
    assert len(payloads) == 2
    assert rejections[0] in json.dumps(payloads[1])
    assert driver.commands == ()
    assert chunk.actions[-1].data[0] == 0.302
    for action in chunk.actions:
        embodiment.step(action)
    assert driver.get_pose().x == 0.302


def test_standard_clamp_cannot_silently_rewrite_unsafe_motion(embodiment, driver):
    from inspect_robots.types import Action

    embodiment.reset(Scene(id="clamp", instruction="Reject unsafe target"))
    chain = ChainApprover(
        ClampApprover(embodiment.info.action_space),
        DeltaLimitApprover(embodiment.info.action_space),
        embodiment.contribute_guardrails(embodiment.info.action_space).approvers[0][1],
    )
    with pytest.raises(SafetyRejected, match="modified"):
        chain.review(Action(np.array([0.3, 0, 0.041, 0, 0, 0, 0])), {})
    assert driver.commands == ()


def test_real_framework_rollout_with_fake_embodiment_logs_steps(embodiment, driver, tmp_path):
    from inspect_robots import eval as robot_eval
    from inspect_robots.controller import DefaultController
    from inspect_robots.logging.json_log import JsonLogSink
    from inspect_robots.task import Task

    from inspect_robots_dobot.audit import DobotAuditSink

    # Scripted model still goes through the real agent and complete framework rollout.
    policy = LLMAgentPolicy(
        model="offline-test",
        base_url="http://agent.test/v1",
        env={},
        transport=httpx.MockTransport(lambda _: response_for({"x": 0.305}, 1)),
        pre_check=embodiment.pre_check,
    )
    task = Task(
        name="fake",
        scenes=[Scene(id="one", instruction="Small fake translation")],
        scorer=[],
        max_steps=1,
    )
    chain = ChainApprover(
        ClampApprover(embodiment.info.action_space),
        DeltaLimitApprover(embodiment.info.action_space),
        embodiment.contribute_guardrails(embodiment.info.action_space).approvers[0][1],
    )
    log = robot_eval(
        task,
        policy=policy,
        embodiment=embodiment,
        controller=DefaultController(),
        approver=chain,
        log_dir=str(tmp_path),
        sinks=[DobotAuditSink(embodiment), JsonLogSink(str(tmp_path))],
    )
    assert driver.commands
    assert driver.commands[0].name == "move_linear"
    assert len(log) == 1
    assert log[0].status == "success"
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    audit = saved["samples"][0]["trial_metadata"][0]["dobot_audit"]
    assert audit[0]["kind"] == "pre_check"
    assert audit[-1]["kind"] == "motion_attempt"
    assert audit[-1]["target_native"][0] == 305
    assert audit[-1]["physical_sent"] is False
    assert audit[-1]["status"] == "settled"


@pytest.mark.parametrize("initial_angles", [(0.0, 0.0, 0.0), (0.02, -0.03, 0.04)])
@pytest.mark.parametrize("partial_playout", [False, True])
def test_public_7d_move_to_holds_state_runs_guards_and_reports_labeled_residual(
    profile, pose, clock, initial_angles, partial_playout
):
    """Only public policy API, scripted LLM, fake camera and real framework guards."""
    from inspect_robots.types import Action

    from inspect_robots_dobot.camera import FakeCamera
    from inspect_robots_dobot.config import CameraConfig, DobotConfig
    from inspect_robots_dobot.driver import FakeDobotDriver
    from inspect_robots_dobot.embodiment import EEF_DIM_LABELS, DobotEmbodiment
    from inspect_robots_dobot.safety import MotionAuthority

    profile = replace(profile, orientation_low=(-0.5, -0.3, -0.4), orientation_high=(0.5, 0.3, 0.4))
    camera_config = CameraConfig(8, 6, 0.2, 0.1)
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(True),
        convergence_delay=0.02,
        rotational_convergence_delay=0.03,
    )
    emb = DobotEmbodiment(
        DobotConfig(profile, 10, camera_config),
        driver=driver,
        camera=FakeCamera(camera_config, clock),
    )
    payloads, checked, approvals = [], [], []

    def scripted(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            response = response_for({"x": 0.32, "yaw": 0.10}, 1).json()
            response["choices"][0]["message"]["tool_calls"].append(
                {
                    "id": "capture-after-move",
                    "type": "function",
                    "function": {
                        "name": "take_pic",
                        "arguments": json.dumps(
                            {
                                "cameras": ["table"],
                                "note": "Check measured arrival after fake move.",
                            }
                        ),
                    },
                }
            )
            return httpx.Response(200, json=response)
        return response_for({"x": 0.32, "yaw": 0.10}, 2)

    def pre_check(waypoints):
        assert waypoints.ndim == 2 and waypoints.shape[1] == 7
        assert waypoints.dtype == np.float64 and not waypoints.flags.writeable
        checked.append(waypoints.copy())
        return emb.pre_check(waypoints)

    class RecordedGuard:
        def __init__(self, name, guard):
            self.name, self.guard = name, guard

        def review(self, action, store):
            result = self.guard.review(action, store)
            approvals.append((self.name, action.data.copy()))
            assert result is action
            return result

    try:
        scene = Scene(id="public-6dof", instruction="Simulate an x and relative yaw move")
        observed = emb.reset(scene)
        if any(initial_angles):
            observed = emb.step(Action(np.array([*pose.xyz, *initial_angles, 0]))).observation
        initial = observed.state["eef_state"].copy()
        policy = LLMAgentPolicy(
            model="offline-test",
            base_url="http://agent.test/v1",
            env={},
            transport=httpx.MockTransport(scripted),
            pre_check=pre_check,
            images="on_demand",
        )
        policy.bind(emb.info)
        assert check_compatibility(policy, emb).ok
        policy.reset(scene)
        chunk = policy.act(replace(observed, extra={**observed.extra, "env_step": 0}))
        schema = next(
            t["function"] for t in payloads[0]["tools"] if t["function"]["name"] == "move_to"
        )
        assert set(t["function"]["name"] for t in payloads[0]["tools"]) == {
            "move_to",
            "done",
            "give_up",
            "take_pic",
        }
        assert (
            ", ".join(EEF_DIM_LABELS)
            in schema["parameters"]["properties"]["targets"]["description"]
        )
        assert "gripper: [0, 1]" in schema["description"]
        assert "yaw=" in json.dumps(payloads[0]) and "pitch=" in json.dumps(payloads[0])
        target = initial.copy()
        target[[0, 3]] = [0.32, 0.10]
        waypoints = np.stack([a.data for a in chunk.actions])
        assert len(chunk.actions) > 1 and chunk.control_hz == 10
        np.testing.assert_array_equal(checked[0], waypoints)
        np.testing.assert_allclose(waypoints[-1], target, atol=1e-14)
        for row in waypoints:
            np.testing.assert_array_equal(row[[1, 2, 4, 5, 6]], initial[[1, 2, 4, 5, 6]])
        assert np.all(np.diff(waypoints[:, 0]) > 0)
        assert np.all(np.diff(waypoints[:, 3]) > 0)
        space = emb.info.action_space
        chain = ChainApprover(
            RecordedGuard("clamp", ClampApprover(space)),
            RecordedGuard("delta", DeltaLimitApprover(space)),
            RecordedGuard("dobot", emb.contribute_guardrails(space).approvers[0][1]),
        )
        store = {}
        steps = 1 if partial_playout else len(chunk.actions)
        for action in chunk.actions[:steps]:
            assert chain.review(action, store) is action
            result = emb.step(action)
            observed = result.observation
            assert result.info["dobot_motion"]["physical_sent"] is False
            assert observed.image_times["table"] > result.info["dobot_motion"]["settled_at"]
        assert [name for name, _ in approvals] == ["clamp", "delta", "dobot"] * steps
        np.testing.assert_allclose(observed.state["eef_state"], waypoints[steps - 1], atol=1e-14)
        # on_demand's queued capture is the current public residual-reporting path.
        policy.act(replace(observed, extra={**observed.extra, "env_step": steps}))
        delivery = next(
            part["text"]
            for message in reversed(payloads[1]["messages"])
            if message["role"] == "user" and isinstance(message["content"], list)
            for part in message["content"]
            if part.get("type") == "text" and "Largest remaining offset" in part["text"]
        )
        match = re.search(r"Largest remaining offset.*?is ([\deE.+-]+) on (\w+)\.", delivery)
        assert match, delivery
        magnitude, label = float(match.group(1)), match.group(2)
        if partial_playout:
            assert label == "yaw"
            assert magnitude == pytest.approx(target[3] - observed.state["eef_state"][3], rel=1e-3)
            assert "did not run to the end" in delivery
        else:
            assert label in EEF_DIM_LABELS and magnitude < 1e-12
            assert "motion finished playing" in delivery
        # Finish the already approved original chunk, no second plan is executed.
        for action in chunk.actions[steps:]:
            observed = emb.step(chain.review(action, store)).observation
        np.testing.assert_allclose(observed.state["eef_state"], target, atol=1e-14)
        assert all(not record.get("physical_sent", False) for record in emb.audit_records)
    finally:
        emb.close()


def test_full_framework_rollout_executes_the_entire_7d_chunk(profile, pose, clock, tmp_path):
    from inspect_robots import eval as robot_eval
    from inspect_robots.controller import DefaultController
    from inspect_robots.logging.json_log import JsonLogSink
    from inspect_robots.task import Task

    from inspect_robots_dobot.audit import DobotAuditSink
    from inspect_robots_dobot.config import DobotConfig
    from inspect_robots_dobot.driver import FakeDobotDriver
    from inspect_robots_dobot.embodiment import DobotEmbodiment
    from inspect_robots_dobot.safety import MotionAuthority
    from inspect_robots_dobot.transforms import (
        agent_relative_to_rotation,
        native_rotation,
        rotation_distance,
    )

    profile = replace(profile, orientation_low=(-0.5, -0.3, -0.4), orientation_high=(0.5, 0.3, 0.4))
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(True),
    )
    emb = DobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    proposed = []

    def pre_check(waypoints):
        proposed.append(waypoints.copy())
        return emb.pre_check(waypoints)

    policy = LLMAgentPolicy(
        model="offline-test",
        base_url="http://agent.test/v1",
        env={},
        transport=httpx.MockTransport(lambda _: response_for({"x": 0.32, "yaw": 0.1}, 1)),
        pre_check=pre_check,
    )
    space = emb.info.action_space
    chain = ChainApprover(
        ClampApprover(space),
        DeltaLimitApprover(space),
        emb.contribute_guardrails(space).approvers[0][1],
    )
    logs = robot_eval(
        Task(
            name="6dof", scenes=[Scene(id="one", instruction="Fake x+yaw")], scorer=[], max_steps=4
        ),
        policy=policy,
        embodiment=emb,
        controller=DefaultController(),
        approver=chain,
        log_dir=str(tmp_path),
        sinks=[DobotAuditSink(emb), JsonLogSink(str(tmp_path))],
    )
    assert logs[0].status == "success"
    assert len(proposed) == 1 and proposed[0].shape == (4, 7)
    assert len(driver.commands) == 4
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    audit = saved["samples"][0]["trial_metadata"][0]["dobot_audit"]
    moves = [a for a in audit if a["kind"] == "motion_attempt"]
    assert len(moves) == 4
    np.testing.assert_allclose(moves[-1]["requested_action"], [0.32, 0, 0.2, 0.1, 0, 0, 0])
    assert all(m["status"] == "settled" and not m["physical_sent"] for m in moves)
    assert driver.commands[-1].target.x == pytest.approx(0.32)
    expected = agent_relative_to_rotation(0.1, 0, 0, native_rotation(pose))
    assert rotation_distance(native_rotation(driver.commands[-1].target), expected) == 0
