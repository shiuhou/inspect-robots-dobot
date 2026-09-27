import struct

import pytest
from fake_sockets import ScriptedFactory, ScriptedSocket, feedback_packet

from inspect_robots_dobot.config import ConnectionConfig
from inspect_robots_dobot.errors import (
    ProtocolError,
    StaleFeedback,
    TransportError,
    TransportTimeout,
    UnsupportedProtocol,
    UnverifiedField,
)
from inspect_robots_dobot.feedback import parse_feedback
from inspect_robots_dobot.feedback_client import DobotFeedbackClient


def make_client(stream, clock, **kwargs):
    factory = ScriptedFactory(feedback=stream)
    client = DobotFeedbackClient(
        ConnectionConfig(host="192.0.2.1", **kwargs), socket_factory=factory, clock=clock
    )
    return client, factory


def test_fragmented_feedback_lifecycle_and_freshness(clock):
    wire = feedback_packet()
    stream = ScriptedSocket([wire[:1], wire[1:711], wire[711:]])
    client, factory = make_client(stream, clock)
    assert factory.calls == [] and client.age() is None
    with pytest.raises(TransportError):
        client.read_sample()
    with pytest.raises(StaleFeedback):
        client.latest()
    client.connect()
    client.connect()
    assert len(factory.calls) == 1
    sample = client.read_sample()
    assert sample.robot_mode == 5
    assert sample.controller_unix_ms == 100000
    assert sample.received_at == 1
    assert sample.command_id == 42
    clock.sleep(0.1)
    assert client.age() == pytest.approx(0.1)
    assert client.latest() is sample
    clock.sleep(0.2)
    with pytest.raises(StaleFeedback):
        client.latest()
    client.close()
    client.close()
    assert stream.closed and not client.connected and client.age() is None
    assert stream.sent == []


def test_coalesced_packets_choose_last_complete_and_preserve_partial(clock):
    first, second, third = (feedback_packet(command_id=i) for i in (41, 42, 43))
    stream = ScriptedSocket([first + second + third[:100], third[100:]])
    client, _ = make_client(stream, clock)
    client.connect()
    assert client.read_sample().command_id == 42
    clock.sleep(0.1)
    sample = client.read_sample()
    assert sample.command_id == 43 and sample.received_at == pytest.approx(1.1)
    assert stream.sent == []


@pytest.mark.parametrize(
    "chunks,error",
    [
        ([b""], TransportError),
        ([feedback_packet()[:50], b""], ProtocolError),
        ([b"xx"], ProtocolError),
        ([TimeoutError()], TransportTimeout),
        ([OSError("lost")], TransportError),
        ([feedback_packet(mode=99)], ProtocolError),
        ([feedback_packet() + b"xx"], ProtocolError),
    ],
)
def test_invalid_feedback_closes_without_send_or_reconnect(clock, chunks, error):
    stream = ScriptedSocket(chunks)
    client, factory = make_client(stream, clock)
    client.connect()
    with pytest.raises(error):
        client.read_sample()
    assert stream.closed and not client.connected
    assert stream.sent == [] and len(factory.calls) == 1
    with pytest.raises(StaleFeedback):
        client.latest()


def test_total_deadline_applies_to_fragmented_stream(clock):
    stream = ScriptedSocket([feedback_packet()[:10]], on_recv=lambda: clock.sleep(2))
    client, _ = make_client(stream, clock, timeout=1)
    client.connect()
    with pytest.raises(TransportTimeout):
        client.read_sample()
    assert stream.closed


def test_unverified_semantics_fail_explicitly():
    sample = parse_feedback(feedback_packet(), received_at=1)
    for operation in (sample.pose_si, sample.joints_si):
        with pytest.raises(UnverifiedField):
            operation()
    for field in ("SpeedScaling", "AutoManualMode", "acceleration_scaling"):
        with pytest.raises(UnverifiedField):
            sample.unverified_field(field)
    assert sample.tcp_pose_raw == (300, -100, 200, 180, 0, 90)
    assert sample.joints_raw == (0, 90, -180, 45, 0, 0)


@pytest.mark.parametrize("offset,fmt,value", [(48, ">Q", 0x0123456789ABCDEF), (24, ">Q", 5)])
def test_endianness_rejects_swapped_verified_fields(offset, fmt, value):
    packet = bytearray(feedback_packet())
    struct.pack_into(fmt, packet, offset, value)
    with pytest.raises(ProtocolError):
        parse_feedback(bytes(packet), received_at=1)


def test_unknown_protocol_never_opens_stream(clock):
    client, factory = make_client(ScriptedSocket(), clock, expected_protocol_version="4.6.4")
    with pytest.raises(UnsupportedProtocol):
        client.connect()
    assert factory.calls == []


def test_controller_epoch_never_drives_host_age(clock):
    client, _ = make_client(ScriptedSocket([feedback_packet(timestamp=2**63)]), clock)
    client.connect()
    sample = client.read_sample()
    assert sample.controller_unix_ms == 2**63 and client.age() == 0
    with pytest.raises(ProtocolError):
        sample.age(0)
