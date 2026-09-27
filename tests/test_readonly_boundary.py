"""Tests of the outbound boundary, not just of the health report's boolean flags."""

import pytest
from fake_sockets import ScriptedFactory, ScriptedSocket, dashboard_socket

from inspect_robots_dobot.config import ConnectionConfig, DobotConfig
from inspect_robots_dobot.dashboard import DobotDashboardClient
from inspect_robots_dobot.feedback_client import DobotFeedbackClient
from inspect_robots_dobot.preflight import run_preflight
from inspect_robots_dobot.protocol import Query, decode_mode
from inspect_robots_dobot.transport import open_socket


@pytest.mark.parametrize(
    "command",
    [
        "RequestControl()",
        "PowerOn()",
        "EnableRobot()",
        "ClearError()",
        "MovL()",
        "MovJ()",
        "ServoP()",
        "ToolDO(1,1)",
        "ToolDOInstant(1,0)",
        "Stop()",
    ],
)
def test_even_private_query_entry_cannot_send_arbitrary_commands(clock, command):
    from inspect_robots_dobot.errors import ProtocolError

    stream = dashboard_socket()
    client = DobotDashboardClient(
        ConnectionConfig(host="192.0.2.1"),
        socket_factory=ScriptedFactory(dashboard=stream),
        clock=clock,
    )
    client.connect()
    with pytest.raises(ProtocolError, match="enumerated query"):
        client._query(command, decode_mode)
    assert stream.sent == []
    client.close()


def test_query_enum_contains_exact_verified_read_allowlist():
    assert {q.value for q in Query} == {
        "RobotMode",
        "GetPose",
        "GetAngle",
        "GetErrorID",
        "GetCurrentCommandID",
    }


def test_valid_preflight_with_connection_config_cannot_connect(profile, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("preflight attempted hardware access")

    monkeypatch.setattr(DobotDashboardClient, "connect", fail)
    monkeypatch.setattr(DobotFeedbackClient, "connect", fail)
    config = DobotConfig(profile, 10, connection=ConnectionConfig(host="192.0.2.1"))
    report = run_preflight(config)
    assert report["ok"] and not report["hardware_connected"]
    assert report["motion_commands_sent"] == 0


def test_socket_constructor_uses_literal_ip_and_closes_on_failed_connect(monkeypatch):
    # Replace socket construction itself. Never open even a localhost socket.
    from inspect_robots_dobot import transport

    events = []

    class FakeConnectSocket(ScriptedSocket):
        def connect(self, address):
            events.append(address)
            raise TimeoutError("injected connect timeout")

    stream = FakeConnectSocket()
    monkeypatch.setattr(transport.socket, "socket", lambda family: stream)
    with pytest.raises(TimeoutError):
        open_socket("192.0.2.1", 29999, 1.0)
    assert stream.closed and stream.sent == []
    assert events == [("192.0.2.1", 29999)]
    assert stream.timeouts == [1.0]


def test_socket_constructor_ipv6_sends_no_bytes(monkeypatch):
    from inspect_robots_dobot import transport

    events = []
    stream = ScriptedSocket()
    stream.connect = lambda address: events.append(address)
    monkeypatch.setattr(transport.socket, "socket", lambda family: stream)
    assert open_socket("2001:db8::1", 30004, 1.0) is stream
    assert events == [("2001:db8::1", 30004)] and stream.sent == []
    stream.close()


def test_hostless_clients_do_not_default_to_nova(clock):
    from inspect_robots_dobot.errors import TransportError

    factory = ScriptedFactory()
    for cls in (DobotDashboardClient, DobotFeedbackClient):
        client = cls(ConnectionConfig(), socket_factory=factory, clock=clock)
        with pytest.raises(TransportError, match="explicit host"):
            client.connect()
    assert factory.calls == []


def test_valid_motion_profile_does_not_grant_any_real_authority(profile, clock):
    from fake_sockets import feedback_packet

    from inspect_robots_dobot.health import run_health

    factory = ScriptedFactory(dashboard_socket(), ScriptedSocket([feedback_packet()]))
    config = DobotConfig(
        profile,
        10,
        connection=ConnectionConfig(
            host="192.0.2.1", controller_firmware="fixture", protocol_compatibility_confirmed=True
        ),
    )
    result = run_health(config, allow_read_only=True, socket_factory=factory, clock=clock)
    assert result["motion_profile_configured"]
    assert result["status"] == "READ_ONLY_OK" and not result["motion_ready"]
    assert not result["motion_enabled"] and not result["request_control_attempted"]
    assert result["motion_commands_sent"] == 0
