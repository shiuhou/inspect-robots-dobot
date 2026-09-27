import math

import pytest

from inspect_robots_dobot.errors import CommandRejected, ProtocolError
from inspect_robots_dobot.protocol import (
    Query,
    ResponseFramer,
    decode_command_id,
    decode_errors,
    decode_joints,
    decode_mode,
    decode_pose,
    parse_response,
    serialize_query,
)
from inspect_robots_dobot.types import RobotMode


@pytest.mark.parametrize(
    "query,wire",
    [
        (Query.MODE, b"RobotMode()"),
        (Query.JOINTS, b"GetAngle()"),
        (Query.ERRORS, b"GetErrorID()"),
        (Query.COMMAND_ID, b"GetCurrentCommandID()"),
        (Query.POSE, b"GetPose()"),
    ],
)
def test_exact_read_query_serialization(query, wire):
    assert serialize_query(query) == wire


def test_explicit_frame_query():
    assert serialize_query(Query.POSE, user=0, tool=3) == b"GetPose(user=0,tool=3)"


@pytest.mark.parametrize(
    "query,kwargs",
    [
        (Query.POSE, {"user": 0}),
        (Query.POSE, {"tool": 1}),
        (Query.POSE, {"user": -1, "tool": 0}),
        (Query.POSE, {"user": 0, "tool": 51}),
        (Query.POSE, {"user": True, "tool": 0}),
        (Query.MODE, {"user": 0, "tool": 0}),
        ("MovL", {}),
    ],
)
def test_query_serialization_rejects_invalid_or_actuating_commands(query, kwargs):
    with pytest.raises(ProtocolError):
        serialize_query(query, **kwargs)


def test_parses_pose_and_joints_into_si():
    pose = parse_response(
        b"0,{300,-100,200,180,-90,45},GetPose(user=0,tool=1);",
        expected_command=b"GetPose(user=0,tool=1)",
    )
    assert decode_pose(pose).values == pytest.approx(
        (0.3, -0.1, 0.2, math.pi, -math.pi / 2, math.pi / 4)
    )
    joints = parse_response(b"0,{0,90,-180,45,360,1e1},GetAngle();", expected_command=b"GetAngle()")
    assert decode_joints(joints) == pytest.approx(
        (0, math.pi / 2, -math.pi, math.pi / 4, 2 * math.pi, math.pi / 18)
    )


@pytest.mark.parametrize("payload,expected", [(b"[]", ()), (b"[123,456]", (123, 456))])
def test_v465_alarm_array(payload, expected):
    response = parse_response(
        b"0,{" + payload + b"},GetErrorID();", expected_command=b"GetErrorID()"
    )
    assert decode_errors(response) == expected


@pytest.mark.parametrize("payload", [b"[[123],[],[]]", b"[true]", b"[1.1]", b"{}", b"nope"])
def test_rejects_old_nested_or_invalid_alarm_schema(payload):
    with pytest.raises(ProtocolError):
        decode_errors(
            parse_response(b"0,{" + payload + b"},GetErrorID();", expected_command=b"GetErrorID()")
        )


def test_mode_queue_id_and_empty_payload():
    assert (
        decode_mode(parse_response(b"0,{5},robotmode();", expected_command=b"RobotMode()"))
        == RobotMode.ENABLED_IDLE
    )
    assert (
        decode_command_id(
            parse_response(
                b"0,{9007199254740993},GetCurrentCommandID();",
                expected_command=b"GetCurrentCommandID()",
            )
        )
        == 9007199254740993
    )
    assert parse_response(b"0,{},Stop();", expected_command=b"Stop()").payload == ""


def test_error_is_typed_and_retains_id():
    with pytest.raises(CommandRejected) as caught:
        parse_response(b"-2,{},GetPose();", expected_command=b"GetPose()")
    assert caught.value.error_id == -2


@pytest.mark.parametrize(
    "wire",
    [
        b"0,{5},RobotMode()",
        b"0,{5},GetAngle();",
        b"abc,{5},RobotMode();",
        b"0,{5},RobotMode();junk",
        b"\xff;",
        b"0,{5},RobotMode(;",
    ],
)
def test_bad_frame_or_echo_rejected(wire):
    with pytest.raises(ProtocolError):
        parse_response(wire, expected_command=b"RobotMode()")


@pytest.mark.parametrize("payload", ["1,2", "0,0,nan,0,0,0", "0,0,inf,0,0,0", "a,0,0,0,0,0"])
def test_bad_coordinate_values_rejected(payload):
    with pytest.raises(ProtocolError):
        decode_pose(
            parse_response(f"0,{{{payload}}},GetPose();".encode(), expected_command=b"GetPose()")
        )


def test_fragmented_and_coalesced_responses():
    framer = ResponseFramer()
    assert framer.feed(b"0,{5},Robot") == []
    assert framer.feed(b"Mode();0,{[]},GetErrorID();0,") == [
        b"0,{5},RobotMode();",
        b"0,{[]},GetErrorID();",
    ]
    with pytest.raises(ProtocolError, match="truncated"):
        framer.finish()
    assert framer.feed(b"{},Stop();") == [b"0,{},Stop();"]
    framer.finish()


def test_framing_limit():
    with pytest.raises(ProtocolError, match="limit"):
        ResponseFramer(max_bytes=8).feed(b"0123456789")
    with pytest.raises(ProtocolError, match="limit"):
        ResponseFramer(max_bytes=8).feed(b"012345678;")


@pytest.mark.parametrize("payload", ["0", "12", "5.0", "-1"])
def test_unknown_or_noninteger_mode_is_rejected(payload):
    with pytest.raises(ProtocolError):
        decode_mode(
            parse_response(
                f"0,{{{payload}}},RobotMode();".encode(), expected_command=b"RobotMode()"
            )
        )
