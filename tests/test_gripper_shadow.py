"""Hardware-independent Phase 6A.2 normalized gripper shadow checks."""

import json

import pytest

from inspect_robots_dobot.astra_shadow import build_fixture_observation
from inspect_robots_dobot.errors import SafetyRejected
from inspect_robots_dobot.gripper import (
    ShadowGripperExecutor,
    normalized_to_servo_position,
    validate_normalized,
)
from inspect_robots_dobot.official_agent_shadow import run_official_shadow


def _move_response(target: float) -> dict[str, object]:
    return {
        "output": [
            {
                "type": "function_call",
                "call_id": "phase6a2-gripper",
                "name": "move_to",
                "arguments": json.dumps(
                    {"targets": {"gripper": target}, "note": "Use the gripper."}
                ),
            }
        ]
    }


def test_normalized_contract_and_audited_mapping():
    assert validate_normalized(0.0) == 0.0
    assert validate_normalized(1.0) == 1.0
    assert normalized_to_servo_position(0.0) == 2490
    assert normalized_to_servo_position(1.0) == 1470
    with pytest.raises(SafetyRejected):
        validate_normalized(-0.01)
    with pytest.raises(SafetyRejected):
        validate_normalized(1.01)
    with pytest.raises(SafetyRejected):
        validate_normalized(float("nan"))


def test_shadow_executor_records_without_hardware():
    result = ShadowGripperExecutor().record_target(0.0)
    assert result["would_execute"] == "WOULD_GRIPPER_CLOSE normalized=0"
    assert result["gripper_serial_connections"] == 0
    assert result["gripper_commands_sent"] == 0


def test_pure_official_gripper_tool_reaches_shadow_boundary():
    record = run_official_shadow(response_json=_move_response(1.0), env={})
    assert record.approver_result == "approved"
    assert record.gripper_action_available is True
    assert record.gripper_target_normalized == 1.0
    assert record.gripper_semantic_target == "OPEN"
    assert record.hypothetical_gripper == "WOULD_GRIPPER_OPEN normalized=1"
    assert record.hypothetical_movl is None
    assert record.execution_performed is False
    assert record.gripper_serial_connections == 0
    assert record.gripper_commands_sent == 0
    assert record.nova_connections == 0


def test_arm_only_tool_does_not_claim_a_gripper_command():
    response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "phase6a2-arm",
                "name": "move_to",
                "arguments": json.dumps({"targets": {"x": 0.305}, "note": "Approach the block."}),
            }
        ]
    }
    record = run_official_shadow(response_json=response, env={})
    assert record.hypothetical_movl is not None
    assert record.hypothetical_gripper is None
    assert record.gripper_target_normalized is None


def test_fixture_observation_has_explicit_gripper_provenance():
    embodiment, observation = build_fixture_observation()
    try:
        assert float(observation.state["eef_state"][6]) == 0.0
        assert observation.extra["dobot"]["gripper"] == {
            "normalized": 0.0,
            "semantic": "CLOSED",
            "source": "fixture",
            "execution_mode": "shadow",
        }
    finally:
        embodiment.close()


def test_model_notes_advertise_shadow_gripper_capability():
    from inspect_robots_dobot.config import DobotConfig
    from inspect_robots_dobot.embodiment import build_info

    docs = build_info(DobotConfig()).docs or ""
    assert "0 means closed and 1 means open" in docs
    assert "inactive placeholder" not in docs
