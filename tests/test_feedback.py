import struct

import pytest

from inspect_robots_dobot.errors import ProtocolError
from inspect_robots_dobot.feedback import FeedbackFramer, parse_feedback


@pytest.fixture
def wire():
    # Explicit independent values at source-verified offsets (manual pp151-154).
    buf = bytearray(1440)
    struct.pack_into("<H", buf, 0, 1440)
    struct.pack_into("<Q", buf, 24, 5)
    struct.pack_into("<Q", buf, 32, 100000)
    struct.pack_into("<Q", buf, 40, 2000)
    struct.pack_into("<Q", buf, 48, 0x0123456789ABCDEF)
    struct.pack_into("<6d", buf, 432, 1, 2, 3, 4, 5, 6)
    struct.pack_into("<6d", buf, 624, 300, -100, 200, 180, 0, 90)
    buf[1012], buf[1013], buf[1026], buf[1029], buf[1031], buf[1038] = 0, 2, 1, 0, 160, 0
    struct.pack_into("<Q", buf, 1112, 42)
    return bytes(buf)


def test_verified_feedback_subset_stays_raw(wire):
    sample = parse_feedback(wire, received_at=10)
    assert sample.robot_mode == 5
    assert sample.controller_unix_ms == 100000
    assert sample.runtime_ms == 2000
    assert sample.command_id == 42
    assert sample.joints_raw == (1, 2, 3, 4, 5, 6)
    assert sample.tcp_pose_raw == (300, -100, 200, 180, 0, 90)
    assert (sample.user_index, sample.tool_index, sample.robot_type) == (0, 2, 160)
    assert sample.is_fresh(10.05, 0.1)
    assert not sample.is_fresh(10.2, 0.1)
    assert not sample.is_fresh(9, 0.1)


@pytest.mark.parametrize("mutation", ["short", "long", "endian", "sentinel", "nan"])
def test_reject_corrupt_feedback(wire, mutation):
    buf = bytearray(wire)
    if mutation == "short":
        buf.pop()
    elif mutation == "long":
        buf.append(0)
    elif mutation == "endian":
        struct.pack_into(">H", buf, 0, 1440)
    elif mutation == "sentinel":
        buf[48] = 0
    else:
        struct.pack_into("<d", buf, 624, float("nan"))
    with pytest.raises(ProtocolError):
        parse_feedback(bytes(buf), received_at=10)


def test_feedback_stream_framing(wire):
    framer = FeedbackFramer()
    assert framer.feed(wire[:20], received_at=1) == []
    frames = framer.feed(wire[20:] + wire + wire[:10], received_at=2)
    assert len(frames) == 2
    assert all(s.received_at == 2 for s in frames)
    with pytest.raises(ProtocolError, match="truncated"):
        framer.finish()
    assert len(framer.feed(wire[10:], received_at=3)) == 1
    framer.finish()
