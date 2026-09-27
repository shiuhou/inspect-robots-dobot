"""Terminal confirmation and offline-only CLI cannot promote config to authority."""

import json
from dataclasses import asdict

import pytest
from live_fakes import production_profile

from inspect_robots_dobot.config import DobotConfig, load_config
from inspect_robots_dobot.errors import ConfigurationError, MotionNotAuthorized
from inspect_robots_dobot.live_profile import load_live_profile
from inspect_robots_dobot.micro_move import main, rehearse_micro_move, terminal_confirmation


def rehearsal(profile, pose, clock, confirm, **kwargs):
    return rehearse_micro_move(
        DobotConfig(safety=profile),
        production_profile(profile),
        pose,
        (0.01, 0.0, 0.0),
        host_label="192.0.2.1",
        allow_motion=kwargs.get("allow_motion", True),
        confirm=confirm,
        clock=clock,
    )


def test_complete_rehearsal_prints_review_before_authority(profile, pose, clock):
    seen = []

    def confirm(review):
        seen.append(review)
        assert not review["authority_issued"]
        assert not review["PHYSICAL_SEND"]
        assert review["PHYSICAL COMMAND COUNT"] == 1
        assert review["translation_delta_m"] == pytest.approx(0.01)
        assert review["angular_delta_rad"] == pytest.approx(0)
        assert review["exact_agent_target"] == pytest.approx((0.31, 0, 0.2, 0, 0, 0, 0))
        assert review["robot_mode"] == 5
        assert review["speed_percent"] <= 10 and review["cp"] == 0
        return "MOVE ONCE"

    report = rehearsal(profile, pose, clock, confirm)
    assert report["status"] == "DRY_RUN_OK"
    assert len(seen) == 1
    assert report["mock_motion_writes"] == 1
    assert report["robot_connections_opened"] == report["motion_commands_sent"] == 0
    assert not report["PHYSICAL_SEND"] and not report["motion_ready"]
    assert report["connections_closed"]
    assert report["authority_details"]["plan_digest"] == seen[0]["plan_digest"]
    assert report["authority_details"]["expires_at"] == seen[0]["authority_expires_at_monotonic"]
    assert "BLOCKED" in report["phase4b_blocker"]


@pytest.mark.parametrize("phrase", ["y", "yes", "move once", " MOVE ONCE", "MOVE ONCE ", ""])
def test_deliberate_confirmation_phrase_required(profile, pose, clock, phrase):
    report = rehearsal(profile, pose, clock, lambda _: phrase)
    assert report["status"] == "BLOCKED"
    assert report["mock_motion_writes"] == 0
    assert report["connections_closed"]


def test_confirmation_does_not_override_runtime_gate(profile, pose, clock):
    report = rehearsal(profile, pose, clock, lambda _: "MOVE ONCE", allow_motion=False)
    assert report["status"] == "BLOCKED"
    assert report["mock_motion_writes"] == 0


def test_reviewed_expiry_does_not_extend_after_operator_delay(profile, pose, clock):
    def slow_confirmation(_):
        clock.sleep(2)
        return "MOVE ONCE"

    report = rehearsal(profile, pose, clock, slow_confirmation)
    assert report["status"] == "BLOCKED"
    assert "expiry" in report["reason"]
    assert report["mock_motion_writes"] == 0


def test_terminal_rejects_piped_confirmation(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("must not read pipe"))
    with pytest.raises(MotionNotAuthorized, match="terminal"):
        terminal_confirmation({"PHYSICAL_SEND": False})
    assert "PHYSICAL COMMAND COUNT = 1" in capsys.readouterr().out


def test_terminal_prints_review_before_reading(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def phrase(prompt):
        assert "plan_digest" in capsys.readouterr().out
        assert "MOVE ONCE" in prompt
        return "MOVE ONCE"

    monkeypatch.setattr("builtins.input", phrase)
    assert terminal_confirmation({"plan_digest": "mock"}) == "MOVE ONCE"


@pytest.mark.parametrize(
    "args", [[], ["--allow-motion"], ["--host", "192.0.2.1", "--allow-motion"]]
)
def test_live_cli_is_blocked_before_any_factory(args, capsys, monkeypatch):
    monkeypatch.setattr(
        "inspect_robots_dobot.live_driver.open_socket",
        lambda *args: pytest.fail("socket factory reached"),
    )
    assert main(args) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "BLOCKED" and not report["PHYSICAL_SEND"]
    assert "interruption" in report["reason"]


def test_cli_full_mock_flow(profile, pose, tmp_path, monkeypatch, capsys):
    rig, monitor, output = (tmp_path / name for name in ("rig.json", "monitor.json", "report.json"))
    rig.write_text(json.dumps({"safety": asdict(profile)}))
    data = asdict(production_profile(profile))
    del data["safety"]
    monitor.write_text(json.dumps(data))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "MOVE ONCE")
    assert (
        main(
            [
                "--dry-run",
                "--allow-motion",
                "--config",
                str(rig),
                "--profile",
                str(monitor),
                "--initial-native-si",
                *map(str, pose.values),
                "--delta-xyz",
                ".01",
                "0",
                "0",
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["status"] == "DRY_RUN_OK"
    assert report["mock_motion_writes"] == 1 and report["connections_closed"]
    assert not report["PHYSICAL_SEND"]
    assert "authority_expires_at_monotonic" in capsys.readouterr().out


@pytest.mark.parametrize("key", ["confirmation", "allow_motion", "authority", "expires_at"])
def test_config_cannot_grant_runtime_authority(profile, tmp_path, key):
    path = tmp_path / "monitor.json"
    data = asdict(production_profile(profile))
    del data["safety"]
    data[key] = "MOVE ONCE"
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError):
        load_live_profile(path, profile)
    path.write_text(json.dumps({"safety": asdict(profile), key: "MOVE ONCE"}))
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_env_cannot_confirm(profile, pose, clock, monkeypatch):
    monkeypatch.setenv("DOBOT_CONFIRMATION", "MOVE ONCE")
    monkeypatch.setenv("ALLOW_MOTION", "true")
    report = rehearsal(profile, pose, clock, lambda _: "", allow_motion=False)
    assert report["status"] == "BLOCKED" and report["mock_motion_writes"] == 0


@pytest.mark.parametrize("delta", [(0.021, 0.0, 0.0), (0.02, 0.01, 0.0), (float("nan"), 0.0, 0.0)])
def test_mock_cli_rejects_invalid_displacement_before_confirm(profile, pose, clock, delta):
    report = rehearse_micro_move(
        DobotConfig(safety=profile),
        production_profile(profile),
        pose,
        delta,
        host_label="192.0.2.1",
        allow_motion=True,
        confirm=lambda _: pytest.fail("invalid plan should not reach review"),
        clock=clock,
    )
    assert report["status"] == "BLOCKED" and report["mock_motion_writes"] == 0


def test_cli_refuses_connection_config(profile, pose, clock):
    from live_fakes import connection

    with pytest.raises(ConfigurationError):
        rehearse_micro_move(
            DobotConfig(safety=profile, connection=connection()),
            production_profile(profile),
            pose,
            (0.01, 0.0, 0.0),
            host_label="192.0.2.1",
            allow_motion=True,
            confirm=lambda _: "MOVE ONCE",
            clock=clock,
        )


def test_ctrl_c_at_prompt_never_sends(profile, pose, clock):
    def interrupt(_):
        raise KeyboardInterrupt

    report = rehearsal(profile, pose, clock, interrupt)
    assert report["status"] == "BLOCKED"
    assert report["mock_motion_writes"] == 0 and report["connections_closed"]
