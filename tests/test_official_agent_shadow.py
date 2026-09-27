import json
from dataclasses import asdict
from types import SimpleNamespace

import httpx
import pytest

from inspect_robots_dobot import astra_shadow_cli, official_agent_shadow
from inspect_robots_dobot.astra_shadow_cli import main
from inspect_robots_dobot.errors import ConfigurationError
from inspect_robots_dobot.official_agent_shadow import (
    OFFICIAL_EFFORT,
    OFFICIAL_MODEL,
    OFFICIAL_WIRE,
    MissingOpenAIKey,
    OfficialAgentConfig,
    require_openai_key,
    run_official_shadow,
)


def move_response(x: float = 0.305) -> dict[str, object]:
    return {
        "output": [
            {
                "type": "function_call",
                "call_id": "offline-call-1",
                "name": "move_to",
                "arguments": json.dumps(
                    {"targets": {"x": x}, "note": "Move a small bounded distance toward the block."}
                ),
            }
        ]
    }


def give_up_response() -> dict[str, object]:
    return {
        "output": [
            {
                "type": "function_call",
                "call_id": "offline-stop-1",
                "name": "give_up",
                "arguments": json.dumps(
                    {
                        "reason": "The gripper is inactive in this shadow phase.",
                        "hindsight": "Enable a reviewed gripper action before grasping.",
                    }
                ),
            }
        ]
    }


def test_official_config_matches_pinned_recipe():
    config = OfficialAgentConfig()
    assert (config.model, config.wire, config.effort) == (
        OFFICIAL_MODEL,
        OFFICIAL_WIRE,
        OFFICIAL_EFFORT,
    )
    assert config.images == "always"
    assert config.image_horizon == 2
    assert config.base_url is None
    assert config.api_key_env is None


def test_missing_key_fails_closed_before_provider_setup():
    with pytest.raises(MissingOpenAIKey, match="OPENAI_API_KEY is not configured"):
        require_openai_key({})


def test_cli_missing_key_returns_blocked_without_evidence(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    replay = tmp_path / "replay.json"
    replay.write_text("{}", encoding="utf-8")
    evidence = tmp_path / "evidence"
    status = main(
        [
            "--provider",
            "official-agent",
            "--replay",
            str(replay),
            "--json",
            "--evidence-dir",
            str(evidence),
        ]
    )
    assert status == 1
    output = capsys.readouterr().out
    assert "BLOCKED_MISSING_API_KEY" in output
    assert "OPENAI_API_KEY" in output
    assert not evidence.exists()


def test_official_agent_mocked_responses_path_reaches_shadow_boundary():
    record = run_official_shadow(response_json=move_response(), env={})
    assert record.provider == "official-inspect-robots-agent"
    assert record.model == OFFICIAL_MODEL
    assert record.request_metadata["upstream_policy"] == "inspect_robots_agent.LLMAgentPolicy"
    assert record.request_metadata["tool_names"] == ["move_to", "done", "give_up"]
    assert record.image_roles == ("front_rgb", "right_rgb", "wrist_rgb")
    assert record.number_of_waypoints == 1
    assert record.pre_check_result is None
    assert record.approver_result == "approved"
    assert record.hypothetical_movl is not None
    assert record.hypothetical_movl.startswith("WOULD_SEND: MovL(")
    assert record.execution_performed is False
    assert record.nova_dashboard_connections == 0
    assert record.nova_feedback_connections == 0
    assert record.motion_commands_sent == 0
    assert record.gripper_serial_connections == 0


def test_official_stop_chunk_does_not_claim_move_to():
    record = run_official_shadow(response_json=give_up_response(), env={})
    assert record.proposed_high_level_action == "give_up"
    assert record.move_to_parameters is None
    assert record.hypothetical_movl is None
    assert record.execution_performed is False
    assert record.nova_connections == record.motion_commands_sent == 0


def test_official_agent_rejects_non_official_configuration():
    with pytest.raises(ConfigurationError, match="requires model=openai/gpt-6-astra"):
        run_official_shadow(
            response_json=move_response(),
            env={},
            config=OfficialAgentConfig(model="openai/other"),
        )


@pytest.fixture
def captured_policy(monkeypatch):
    """Run the real upstream client through MockTransport, with hardware traps."""
    original = official_agent_shadow.LLMAgentPolicy
    captured = {"config": [], "requests": [], "status": 200, "response": move_response()}

    def respond(request):
        captured["requests"].append((request.method, str(request.url), json.loads(request.content)))
        return httpx.Response(captured["status"], json=captured["response"])

    def construct(**kwargs):
        captured["config"].append(
            {key: kwargs[key] for key in ("base_url", "api_key_env", "model", "wire")}
        )
        kwargs["transport"] = httpx.MockTransport(respond)
        return original(**kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("gateway shadow reached a hardware or motion boundary")

    monkeypatch.setattr(official_agent_shadow, "LLMAgentPolicy", construct)
    for target in (
        "socket.socket",
        "inspect_robots_dobot.live_driver.LiveDobotMotionDriver.__init__",
        "inspect_robots_dobot.live_authority.LiveMotionAuthority.__init__",
        "inspect_robots_dobot.driver.FakeDobotDriver.move_linear",
        "inspect_robots_dobot.chunks.StagedDobotEmbodiment.step",
        "inspect_robots_dobot.gripper.NoOpGripper.set",
    ):
        monkeypatch.setattr(target, forbidden)
    return captured


@pytest.mark.parametrize("model", ["gpt-6-astra", "openai/gpt-6-astra", "relay-model-id"])
def test_gateway_forwards_config_preserves_model_and_excludes_secret(
    captured_policy, model, capsys, caplog, tmp_path
):
    # Inert test marker; never a real credential and never sent over a network.
    marker = "inert-gateway-credential-marker"
    config = OfficialAgentConfig(
        base_url="https://relay.example/v1", api_key_env="RELAY_API_KEY", model=model
    )
    record = run_official_shadow(
        config=config, env={"RELAY_API_KEY": marker}, require_live_key=True
    )
    assert captured_policy["config"] == [
        {
            "base_url": config.base_url,
            "api_key_env": "RELAY_API_KEY",
            "model": model,
            "wire": "responses",
        }
    ]
    method, url, body = captured_policy["requests"][0]
    assert (method, url, body["model"]) == ("POST", "https://relay.example/v1/responses", model)
    assert record.request_metadata["base_url"] == config.base_url
    assert record.request_metadata["api_key_env"] == "RELAY_API_KEY"
    assert record.approver_result == "approved"
    assert record.hypothetical_movl.startswith("WOULD_SEND: MovL(")
    assert record.execution_performed is False
    assert record.nova_connections == record.motion_commands_sent == 0
    assert record.gripper_serial_connections == record.live_authorities_created == 0
    path = official_agent_shadow.ShadowExecutor.write(record, tmp_path)
    output = capsys.readouterr()
    serialized = (
        path.read_text() + json.dumps(asdict(record)) + output.out + output.err + caplog.text
    )
    assert marker not in serialized
    assert "Authorization" not in serialized


def test_default_official_route_is_unchanged(captured_policy):
    record = run_official_shadow(
        env={"OPENAI_API_KEY": "inert-official-credential-marker"}, require_live_key=True
    )
    assert captured_policy["config"] == [
        {"base_url": None, "api_key_env": None, "model": OFFICIAL_MODEL, "wire": "responses"}
    ]
    method, url, body = captured_policy["requests"][0]
    assert (method, url, body["model"]) == (
        "POST",
        "https://api.openai.com/v1/responses",
        "gpt-6-astra",
    )
    assert record.approver_result == "approved"


def test_gateway_uses_upstream_key_default_and_redacts_transcript(captured_policy):
    marker = "inert-default-gateway-marker"
    response = move_response()
    response["output"][0]["arguments"] = json.dumps(
        {"targets": {"x": 0.305}, "note": f"Provider echoed {marker}"}
    )
    captured_policy["response"] = response
    record = run_official_shadow(
        config=OfficialAgentConfig(base_url="https://relay.example/v1"),
        env={"OPENROUTER_API_KEY": marker},
        require_live_key=True,
    )
    assert captured_policy["config"][0]["api_key_env"] is None
    assert record.approver_result == "approved"
    serialized = json.dumps(asdict(record))
    assert marker not in serialized
    assert "[REDACTED]" in serialized


@pytest.mark.parametrize("key_name", ["RELAY_API_KEY", None])
def test_missing_gateway_key_blocks_before_policy_construction(captured_policy, key_name):
    expected = key_name or "OPENROUTER_API_KEY"
    with pytest.raises(MissingOpenAIKey, match=f"{expected} is not configured"):
        run_official_shadow(
            config=OfficialAgentConfig(base_url="https://relay.example/v1", api_key_env=key_name),
            env={"OPENAI_API_KEY": "inert-wrong-provider-marker"},
            require_live_key=True,
        )
    assert captured_policy["config"] == captured_policy["requests"] == []


@pytest.mark.parametrize("configured", [False, True])
def test_gateway_cli_key_selection_and_evidence(
    configured, captured_policy, tmp_path, monkeypatch, capsys
):
    marker = "inert-cli-gateway-marker"
    env = {"RELAY_API_KEY": marker} if configured else {}
    monkeypatch.setattr(astra_shadow_cli, "os", SimpleNamespace(environ=env))
    replay = tmp_path / "replay.json"
    replay.write_text("{}", encoding="utf-8")
    evidence = tmp_path / "evidence"
    status = main(
        [
            "--provider",
            "official-agent",
            "--base-url",
            "https://relay.example/v1",
            "--api-key-env",
            "RELAY_API_KEY",
            "--model",
            "gpt-6-astra",
            "--wire",
            "responses",
            "--replay",
            str(replay),
            "--json",
            "--evidence-dir",
            str(evidence),
        ]
    )
    output = capsys.readouterr()
    data = json.loads(output.out)
    assert marker not in output.out + output.err
    if configured:
        assert status == 0
        assert captured_policy["config"][0]["api_key_env"] == "RELAY_API_KEY"
        assert data["records"][0]["request_metadata"]["base_url"] == "https://relay.example/v1"
        assert marker not in next(evidence.glob("*.json")).read_text()
    else:
        assert status == 1
        assert data["status"] == "BLOCKED_MISSING_API_KEY"
        assert "RELAY_API_KEY is not configured" in data["message"]
        assert not evidence.exists()
        assert captured_policy["config"] == captured_policy["requests"] == []


def test_gateway_responses_rejection_has_no_wire_fallback_and_redacts_echo(
    captured_policy, capsys, caplog
):
    marker = "inert-rejected-gateway-marker"
    captured_policy.update(status=404, response={"error": f"Responses unavailable; {marker}"})
    record = run_official_shadow(
        config=OfficialAgentConfig(
            base_url="https://relay.example/v1", api_key_env="RELAY_API_KEY"
        ),
        env={"RELAY_API_KEY": marker},
        require_live_key=True,
    )
    assert len(captured_policy["requests"]) == 1
    assert captured_policy["requests"][0][1].endswith("/responses")
    assert "HTTP 404" in record.rejection_reason
    assert "[REDACTED]" in record.rejection_reason
    assert record.hypothetical_movl is None
    output = capsys.readouterr()
    assert marker not in json.dumps(asdict(record)) + output.out + output.err + caplog.text


def test_gateway_does_not_accept_chat_wire(captured_policy):
    with pytest.raises(ConfigurationError, match="Chat Completions-only gateway is incompatible"):
        run_official_shadow(
            config=OfficialAgentConfig(base_url="https://relay.example/v1", wire="chat"), env={}
        )
    assert captured_policy["config"] == captured_policy["requests"] == []
