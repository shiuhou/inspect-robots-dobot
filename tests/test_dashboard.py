import math
from dataclasses import replace

import pytest
from fake_sockets import ScriptedFactory, ScriptedSocket, dashboard_socket

from inspect_robots_dobot.config import ConnectionConfig
from inspect_robots_dobot.dashboard import DobotDashboardClient
from inspect_robots_dobot.errors import (
    CommandRejected,
    ConfigurationError,
    ControlNotOwned,
    ProtocolError,
    TransportError,
    TransportTimeout,
    UnsupportedProtocol,
)


def client_for(stream, clock, **kwargs):
    factory = ScriptedFactory(dashboard=stream)
    client = DobotDashboardClient(
        ConnectionConfig(host="192.0.2.1", **kwargs), socket_factory=factory, clock=clock
    )
    return client, factory


def test_connection_is_lazy_idempotent_and_sends_nothing(clock):
    stream = dashboard_socket()
    client, factory = client_for(stream, clock)
    assert factory.calls == []
    with pytest.raises(TransportError, match="not connected"):
        client.robot_mode()
    client.connect()
    client.connect()
    assert len(factory.calls) == 1
    assert stream.sent == []
    client.close()
    client.close()
    assert stream.closed and not client.connected


def test_all_verified_queries_and_boundary_units(clock):
    stream = dashboard_socket()
    client, _ = client_for(stream, clock, user_frame=0, tool_frame=2)
    client.connect()
    assert client.robot_mode() == 5
    assert client.get_pose().values == pytest.approx((0.3, -0.1, 0.2, math.pi, 0, math.pi / 2))
    assert client.get_joints() == pytest.approx((0, math.pi / 2, -math.pi, math.pi / 4, 0, 0))
    assert client.get_errors() == ()
    assert client.current_command_id() == 42
    assert stream.sent == [
        b"RobotMode()",
        b"GetPose(user=0,tool=2)",
        b"GetAngle()",
        b"GetErrorID()",
        b"GetCurrentCommandID()",
    ]
    assert client.queries_sent == 5
    client.close()


def test_fragmented_reply_and_terminator_whitespace(clock):
    client, _ = client_for(ScriptedSocket([b"0,{5}", b",Robot", b"Mode();\r\n"]), clock)
    client.connect()
    assert client.robot_mode() == 5


@pytest.mark.parametrize(
    "chunks,error",
    [
        ([b"abc,{5},RobotMode();"], ProtocolError),
        ([b"0,{5},GetAngle();"], ProtocolError),
        ([b"0,{5},RobotMode();0,{5},RobotMode();"], ProtocolError),
        ([b"0,{5},RobotMode();0,"], ProtocolError),
        ([b"0,{5}", b""], ProtocolError),
        ([b""], TransportError),
        ([TimeoutError()], TransportTimeout),
        ([ConnectionResetError()], TransportError),
        ([b"0,{12},RobotMode();"], ProtocolError),
        ([b"x" * 65537], ProtocolError),
    ],
)
def test_bad_reply_closes_without_retry(clock, chunks, error):
    stream = ScriptedSocket(chunks)
    client, factory = client_for(stream, clock)
    client.connect()
    with pytest.raises(error):
        client.robot_mode()
    assert not client.connected and stream.closed
    assert stream.sent == [b"RobotMode()"]
    with pytest.raises(TransportError, match="not connected"):
        client.robot_mode()
    assert len(factory.calls) == 1


def test_rejected_query_preserves_error_and_does_not_retry_or_request_control(clock):
    stream = ScriptedSocket([b"-1,{},GetPose();", b"0,{5},RobotMode();"])
    client, _ = client_for(stream, clock)
    client.connect()
    with pytest.raises(CommandRejected) as exc:
        client.get_pose()
    assert exc.value.error_id == -1  # Generic failure; NOT an invented ownership code.
    assert client.connected
    assert client.robot_mode() == 5
    assert stream.sent == [b"GetPose()", b"RobotMode()"]


@pytest.mark.parametrize(
    "failure,expected", [(TimeoutError(), TransportTimeout), (OSError(), TransportError)]
)
def test_connect_failure_is_typed_once(clock, failure, expected):
    client, factory = client_for(failure, clock)
    with pytest.raises(expected):
        client.connect()
    assert not client.connected and len(factory.calls) == 1


def test_send_failure_never_replays(clock):
    stream = ScriptedSocket(send_error=TimeoutError())
    client, _ = client_for(stream, clock)
    client.connect()
    with pytest.raises(TransportTimeout):
        client.robot_mode()
    assert stream.sent == [b"RobotMode()"] and stream.closed


def test_whole_query_deadline_bounds_slow_fragments(clock):
    stream = ScriptedSocket([b"0,", b"{5},", b"RobotMode();"], on_recv=lambda: clock.sleep(0.4))
    client, _ = client_for(stream, clock, timeout=1.0)
    client.connect()
    with pytest.raises(TransportTimeout):
        client.robot_mode()
    assert stream.closed
    assert stream.timeouts[-1] <= 0.21


def test_declared_unowned_blocks_queries_without_ownership_command(clock):
    stream = dashboard_socket()
    client, _ = client_for(stream, clock, tcp_control_owned=False)
    client.connect()
    for query in (client.robot_mode, client.get_pose, client.get_joints, client.get_errors):
        with pytest.raises(ControlNotOwned):
            query()
    assert stream.sent == []
    assert not hasattr(client, "request_control")


def test_unsupported_document_version_blocks_before_socket(clock):
    client, factory = client_for(dashboard_socket(), clock, expected_protocol_version="4.5")
    with pytest.raises(UnsupportedProtocol):
        client.connect()
    assert factory.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"host": "robot.local"},
        {"host": "bad"},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"dashboard_port": 30004},
        {"feedback_port": 29999},
        {"user_frame": 0},
        {"user_frame": True, "tool_frame": 0},
        {"tcp_control_owned": "yes"},
        {"protocol_compatibility_confirmed": True},
        {"controller_firmware": ""},
        {"feedback_max_age": -1},
        {"expected_protocol_version": ""},
    ],
)
def test_invalid_connection_configuration(changes):
    with pytest.raises(ConfigurationError):
        replace(ConnectionConfig(), **changes)
