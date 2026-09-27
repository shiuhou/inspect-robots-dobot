import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from inspect_robots_dobot.errors import (
    CommandRejected,
    ConfigurationError,
    ProtocolError,
    SafetyRejected,
)
from inspect_robots_dobot.motion import (
    DobotMovLRequest,
    KeepoutBox,
    build_motion_plan,
    parse_movl_acceptance,
    parse_stop_acknowledgement,
    serialize_stop,
    validate_cartesian_path,
)
from inspect_robots_dobot.protocol import Query, serialize_query
from inspect_robots_dobot.transforms import native_rotation
from inspect_robots_dobot.types import NativePose, RobotMode
from inspect_robots_dobot.units import from_native


@pytest.fixture
def movl_request():
    return DobotMovLRequest(NativePose(315.0, -1.0, 200.0, 90.0, 0.0, -180.0), 0, 2, 5, 8)


def test_exact_documented_movl_serialization(movl_request):
    assert movl_request.serialize() == (
        "MovL(pose={315.0,-1.0,200.0,90.0,0.0,-180.0},user=0,tool=2,a=8,v=5,cp=0)"
    )
    assert not hasattr(movl_request, "send")
    with pytest.raises(FrozenInstanceError):
        movl_request.speed_percent = 100


def test_decimal_serialization_no_scientific_notation_or_silent_rounding(movl_request):
    values = (1e-12, -1e-10, 200.12345678912345, -0.0, 1e-8, 90.0)
    command = replace(movl_request, pose=NativePose(*values)).serialize()
    text = command.split("{")[1].split("}")[0]
    assert "e" not in text and "-0.0" not in text.split(",")
    assert tuple(map(float, text.split(","))) == values


@pytest.mark.parametrize(
    "changes",
    [
        {"speed_percent": 0},
        {"speed_percent": 11},
        {"speed_percent": True},
        {"acceleration_percent": 0},
        {"acceleration_percent": 101},
        {"user": -1},
        {"tool": 51},
        {"tool": 1.0},
        {"pose": NativePose(float("nan"), 0, 200, 0, 0, 0)},
        {"pose": NativePose(300, 0, 200, float("inf"), 0, 0)},
    ],
)
def test_invalid_serialization_inputs_rejected(movl_request, changes):
    with pytest.raises(ConfigurationError):
        replace(movl_request, **changes)


def test_acceptance_id_is_decoded_but_never_execution(movl_request):
    reply = f"0,{{42}},{movl_request.serialize()};".encode()
    assert parse_movl_acceptance(reply, movl_request) == 42


@pytest.mark.parametrize("kind", ["error", "echo", "malformed", "bad_id"])
def test_movl_ack_rejects_bad_response(movl_request, kind):
    replies = {
        "error": f"-1,{{}},{movl_request.serialize()};",
        "echo": "0,{3},MovL(pose={1,2,3,4,5,6});",
        "malformed": "not a response",
        "bad_id": f"0,{{NaN}},{movl_request.serialize()};",
    }
    with pytest.raises(ProtocolError):
        parse_movl_acceptance(replies[kind].encode(), movl_request)


def test_stop_serializer_and_ack_are_offline_only():
    assert serialize_stop() == "Stop()"
    assert parse_stop_acknowledgement(b"0,{},Stop();") is None
    with pytest.raises(CommandRejected):
        parse_stop_acknowledgement(b"-1,{},Stop();")
    with pytest.raises(ProtocolError):
        parse_stop_acknowledgement(b"0,{123},Stop();")
    with pytest.raises(ProtocolError):
        parse_stop_acknowledgement(b"0,{},Pause();")


def test_no_new_wire_query_or_generic_sender(movl_request):
    assert len(Query) == 5
    for command in (movl_request.serialize(), serialize_stop()):
        with pytest.raises(ProtocolError, match="enumerated"):
            serialize_query(command)


def test_plan_preserves_nontrivial_reference_and_native_branch(driver, profile, pose, clock):
    driver.connect()
    rows = ((0.305, 0, 0.2, 0, 0, 0, 0), (0.31, 0, 0.2, 0, 0, 0, 0))
    plan = build_motion_plan("chunk", rows, driver.snapshot(), pose, profile, clock.monotonic())
    assert plan.final_agent_pose == rows[-1]
    assert plan.request.pose.values[:3] == (310, 0, 200)
    np.testing.assert_allclose(plan.request.pose.values[3:], np.degrees(pose.values[3:]))
    np.testing.assert_allclose(plan.final_rotation, native_rotation(pose))
    np.testing.assert_allclose(
        native_rotation(from_native(plan.request.pose)), native_rotation(pose)
    )
    assert plan.dry_run and not plan.physical_authorized and not plan.physical_sent
    assert driver.commands == ()


@pytest.mark.parametrize(
    "rows,reason",
    [
        ((), "empty"),
        (((0.31, 0, 0.2, 0, 0, 0),), "shape"),
        (((float("nan"), 0, 0.2, 0, 0, 0, 0),), "finite"),
        (((0.31, 0, 0.2, 0.001, 0, 0, 0),), "pinned"),
        (((0.31, 0, 0.2, 0, 0, 0, -0.1),), "outside"),
        (((0.51, 0, 0.2, 0, 0, 0, 0),), "outside"),
        (((0.3, 0, 0.04, 0, 0, 0, 0),), "minimum"),
        (((0.315, 0, 0.2, 0, 0, 0, 0), (0.33, 0, 0.2, 0, 0, 0, 0)), "delta"),
        (((0.305, 0.001, 0.2, 0, 0, 0, 0), (0.31, 0, 0.2, 0, 0, 0, 0)), "straight"),
        (((0.308, 0, 0.2, 0, 0, 0, 0), (0.305, 0, 0.2, 0, 0, 0, 0)), "straight"),
        (((0.301, 0, 0.2, 0, 0, 0, 0), (0.3, 0, 0.2, 0, 0, 0, 0)), "straight"),
    ],
)
def test_full_path_rejections(driver, pose, profile, clock, rows, reason):
    driver.connect()
    with pytest.raises(SafetyRejected, match=reason):
        validate_cartesian_path(rows, driver.snapshot(), pose, profile, clock.monotonic())


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_open_orientation_profile_incompatible(driver, pose, profile, clock, axis):
    driver.connect()
    high = [0.0] * 3
    high[axis] = 0.1
    with pytest.raises(ConfigurationError, match="pinned"):
        validate_cartesian_path(
            ((0.31, 0, 0.2, 0, 0, 0, 0),),
            driver.snapshot(),
            pose,
            replace(profile, orientation_high=tuple(high)),
            clock.monotonic(),
        )


def test_hard_aggregate_cap_cannot_be_opened_by_profile(driver, pose, profile, clock):
    driver.connect()
    with pytest.raises(SafetyRejected, match="aggregate"):
        validate_cartesian_path(
            ((0.33, 0, 0.2, 0, 0, 0, 0),),
            driver.snapshot(),
            pose,
            replace(profile, max_translation_step=0.1),
            clock.monotonic(),
        )
    with pytest.raises(ConfigurationError, match="speed"):
        validate_cartesian_path(
            ((0.31, 0, 0.2, 0, 0, 0, 0),),
            driver.snapshot(),
            pose,
            replace(profile, speed_percent=11),
            clock.monotonic(),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"mode": RobotMode.DISABLED},
        {"errors": (123,)},
        {"user_frame": 1},
        {"observed_at": 0},
    ],
)
def test_path_checks_measured_state(driver, pose, profile, clock, change):
    driver.connect()
    with pytest.raises(SafetyRejected):
        validate_cartesian_path(
            ((0.31, 0, 0.2, 0, 0, 0, 0),),
            replace(driver.snapshot(), **change),
            pose,
            profile,
            clock.monotonic(),
        )


@pytest.mark.parametrize(
    "start,end,hit",
    [
        ((0, 0, 0), (3, 0, 0), True),
        ((3, 0, 0), (0, 0, 0), True),
        ((0, 2, 0), (3, 2, 0), False),
        ((1.5, 0, 0), (1.5, 0, 0), True),
        ((0, 0, 0), (0, 0, 0), False),
        ((0, 1, 1), (1, 1, 1), True),
        ((3, 0, 0), (4, 0, 0), False),
    ],
)
def test_continuous_segment_keepout_edges(pose, start, end, hit):
    box = KeepoutBox((1, -1, -1), (2, 1, 1))
    assert box.intersects(pose.with_translation(*start), pose.with_translation(*end)) is hit


@pytest.mark.parametrize(
    "low,high", [((0, 0, 0), (0, 1, 1)), ((0, 0), (1, 1, 1)), ((0, math.inf, 0), (1, 1, 1))]
)
def test_invalid_keepout_configuration(low, high):
    with pytest.raises(ConfigurationError):
        KeepoutBox(low, high)
