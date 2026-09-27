import json
from dataclasses import asdict
from importlib.metadata import entry_points

import pytest

from inspect_robots_dobot.config import DobotConfig
from inspect_robots_dobot.embodiment import DobotEmbodiment
from inspect_robots_dobot.health import main as health_main
from inspect_robots_dobot.preflight import main, run_preflight
from inspect_robots_dobot.transport import real_transport_unavailable


def test_missing_profile_preflight_fails_without_connections():
    report = run_preflight()
    assert not report["ok"]
    assert report["errors"]
    assert report["hardware_connected"] is False
    assert report["motion_commands_sent"] == 0


def test_explicit_metadata_compatibility_does_not_connect(profile, monkeypatch):
    def forbid_reset(*args, **kwargs):
        raise AssertionError("preflight may not reset/connect")

    monkeypatch.setattr(DobotEmbodiment, "reset", forbid_reset)
    report = run_preflight(DobotConfig(profile, 10))
    assert report["ok"]
    assert report["agent_compatible"]
    assert report["phase"] == "2C" and report["action_dim"] == 7
    assert report["action_labels"] == ("x", "y", "z", "yaw", "pitch", "roll", "gripper")
    assert report["warnings"]  # inactive pinned gripper is explicitly reported
    assert not report["hardware_readiness_verified"]


def test_preflight_json_exit_codes(profile, tmp_path, capsys):
    assert main(["--json"]) == 1
    assert json.loads(capsys.readouterr().out)["motion_enabled"] is False
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"safety": asdict(profile), "control_hz": 10}))
    assert main(["--config", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    path.write_text("bad")
    assert main(["--config", str(path), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["errors"]


def test_health_default_remains_blocked_and_never_connects(capsys):
    assert health_main(["--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["hardware_connected"] is False
    assert report["motion_commands_sent"] == 0
    assert report["status"] == "BLOCKED"
    assert report["request_control_attempted"] is False
    # A host alone never grants read-only access.
    assert health_main(["--host", "192.0.2.1", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["hardware_connected"] is False


def test_registered_entrypoint_constructs_without_connecting():
    entries = list(entry_points(group="inspect_robots.embodiments", name="dobot_nova"))
    assert len(entries) == 1
    assert isinstance(entries[0].load()(), DobotEmbodiment)


def test_real_transport_seam_fails_explicitly():
    from inspect_robots_dobot.errors import PhaseUnavailable

    with pytest.raises(PhaseUnavailable, match="Phase 2"):
        real_transport_unavailable()
