"""Hardware-free regressions for prerequisites exposed by the first bring-up workflow."""

import json

import pytest
from fake_sockets import ScriptedFactory, ScriptedSocket, dashboard_socket, feedback_packet

from inspect_robots_dobot.config import ConnectionConfig, DobotConfig
from inspect_robots_dobot.dashboard import DobotDashboardClient
from inspect_robots_dobot.errors import CommandRejected, ProtocolError, TransportTimeout
from inspect_robots_dobot.feedback_client import DobotFeedbackClient
from inspect_robots_dobot.health import run_health
from inspect_robots_dobot.preflight import main as preflight_main
from inspect_robots_dobot.preflight import run_preflight, run_readonly_preflight


def test_readonly_preflight_does_not_need_motion_geometry_or_construct_clients(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not construct transport clients")

    monkeypatch.setattr(DobotDashboardClient, "__init__", forbidden)
    monkeypatch.setattr(DobotFeedbackClient, "__init__", forbidden)
    config = DobotConfig(connection=ConnectionConfig(host="192.0.2.1"))
    report = run_readonly_preflight(config)
    assert report["ok"] and report["warnings"]
    assert not report["hardware_connected"] and not report["motion_ready"]
    assert not report["request_control_attempted"] and report["motion_commands_sent"] == 0
    assert not run_preflight(config)["ok"]  # Original action preflight stays strict.


@pytest.mark.parametrize(
    "config",
    [
        DobotConfig(),
        DobotConfig(connection=ConnectionConfig()),
        DobotConfig(connection=ConnectionConfig(host="192.0.2.1", expected_protocol_version="4.5")),
    ],
)
def test_readonly_preflight_rejects_missing_host_and_unsupported_version(config):
    report = run_readonly_preflight(config)
    assert not report["ok"] and report["errors"]


def test_readonly_preflight_cli_uses_actual_minimal_config(tmp_path, capsys):
    path = tmp_path / "readonly.json"
    path.write_text('{"connection":{"host":"192.0.2.1"}}')
    assert preflight_main(["--config", str(path), "--read-only", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "OFFLINE READ-ONLY CONFIGURATION"
    assert not report["hardware_readiness_verified"]
    assert preflight_main(["--config", str(path), "--json"]) == 1


@pytest.mark.parametrize(
    "wire,error,error_id",
    [
        (b"-1,{},GetPose();", CommandRejected, -1),
        (b"abc,{},GetPose();", ProtocolError, None),
        (b"\xff;", ProtocolError, None),
    ],
)
def test_exact_dashboard_error_evidence_survives_close(clock, wire, error, error_id):
    stream = ScriptedSocket([wire[:2], wire[2:]])
    client = DobotDashboardClient(
        ConnectionConfig(host="192.0.2.1"),
        socket_factory=ScriptedFactory(dashboard=stream),
        clock=clock,
    )
    client.connect()
    with pytest.raises(error):
        client.get_pose()
    client.close()
    record = client.exchanges[0]
    assert record["command"] == "GetPose()" and record["error_id"] == error_id
    assert bytes.fromhex(record["response_hex"]) == wire
    assert record["send_completed"] and record["send_attempted"]
    if wire.isascii():
        assert record["response_ascii"] == wire.decode("ascii")
    else:
        assert record["response_ascii"] is None
    record["command"] = "mutation"
    assert client.exchanges[0]["command"] == "GetPose()"
    assert stream.sent == [b"GetPose()"]


def test_partial_dashboard_response_preserved_on_timeout(clock):
    stream = ScriptedSocket([b"0,{300,"])
    client = DobotDashboardClient(
        ConnectionConfig(host="192.0.2.1"),
        socket_factory=ScriptedFactory(dashboard=stream),
        clock=clock,
    )
    client.connect()
    with pytest.raises(TransportTimeout):
        client.get_pose()
    assert client.exchanges[0]["response_ascii"] == "0,{300,"
    assert stream.closed


@pytest.mark.parametrize("wire", [b"xx", feedback_packet(mode=99)])
def test_invalid_feedback_bytes_preserved_without_changing_parser(clock, wire):
    stream = ScriptedSocket([wire])
    client = DobotFeedbackClient(
        ConnectionConfig(host="192.0.2.1"),
        socket_factory=ScriptedFactory(feedback=stream),
        clock=clock,
    )
    client.connect()
    with pytest.raises(ProtocolError):
        client.read_sample()
    assert client.last_read_wire == wire and stream.closed and stream.sent == []


def test_health_captures_all_five_queries_and_wire_evidence(clock):
    stream = dashboard_socket()
    packet = feedback_packet()
    factory = ScriptedFactory(stream, ScriptedSocket([packet[:12], packet[12:]]))
    report = run_health(
        DobotConfig(connection=ConnectionConfig(host="192.0.2.1")),
        allow_read_only=True,
        socket_factory=factory,
        clock=clock,
    )
    assert report["command_id"] == report["feedback_raw"]["command_id"] == 42
    assert report["queries"]["command_id"]["status"] == "OK"
    assert bytes.fromhex(report["feedback_wire_hex"]) == packet
    assert report["feedback_message_size_valid"] and report["feedback_test_value_valid"]
    assert report["feedback_packet_size"] == 1440
    assert report["feedback_test_value"] == "0x0123456789ABCDEF"
    assert [e["command"] for e in report["dashboard_exchanges"]] == [
        c.decode() for c in stream.sent
    ]
    assert len(report["dashboard_exchanges"]) == 7
    assert all(e["error_id"] == 0 for e in report["dashboard_exchanges"])
    assert report["connections_closed"] and not report["motion_ready"]
    assert report["request_control_attempted"] is False and report["motion_commands_sent"] == 0
    assert report["status"] == "PARTIAL"  # Unknown firmware remains unknown.
    json.dumps(report, allow_nan=False)
