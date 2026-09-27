"""Deterministic Dashboard peer; bytes only, no socket or hardware imports."""

from collections import deque
from dataclasses import replace

from fake_sockets import ScriptedFactory, ScriptedSocket

from inspect_robots_dobot.config import ConnectionConfig
from inspect_robots_dobot.live_driver import LiveDobotMotionDriver
from inspect_robots_dobot.live_profile import LiveMotionProfile, OperatorReadiness
from inspect_robots_dobot.motion import build_motion_plan
from inspect_robots_dobot.types import RobotMode, RobotSnapshot
from inspect_robots_dobot.units import to_native


def production_profile(safety, **changes):
    """FICTIONAL test profile. Strings do not establish real verification."""
    return replace(
        LiveMotionProfile(
            safety=safety,
            start_position_tolerance=0.0005,
            start_orientation_tolerance=0.005,
            max_measurement_age=0.2,
            consecutive_samples=3,
            io_timeout=0.01,
            acknowledgement_timeout=0.04,
            stop_timeout=0.1,
            standstill_position_tolerance=0.0002,
            standstill_orientation_tolerance=0.001,
            authority_lifetime=1.0,
            model="MOCK_MODEL",
            firmware="MOCK_FIRMWARE",
            rig_verification_reference="MOCK_ONLY",
            interruption_verification_reference="MOCK_ONLY",
        ),
        **changes,
    )


def connection(**changes):
    return replace(
        ConnectionConfig(
            host="192.0.2.1",
            controller_firmware="MOCK_FIRMWARE",
            protocol_compatibility_confirmed=True,
            tcp_control_owned=True,
            user_frame=0,
            tool_frame=0,
        ),
        **changes,
    )


def readiness(**changes):
    return replace(
        OperatorReadiness("test-operator", True, True, True, True, True, True, True), **changes
    )


def response(command, payload=b"", error=0):
    return str(error).encode() + b",{" + payload + b"}," + command + b";"


class MotionPeer(ScriptedSocket):
    def __init__(self, start, clock):
        super().__init__()
        self.start, self.clock = start, clock
        self.current = start
        self.moving = False
        self.stopped = False
        self.execution = deque()
        self.after_stop = deque()
        self.mode_queries = 0
        self.samples = 0
        self.move_reply = "ok"
        self.stop_reply = "ok"
        self.pending_move = None
        self.on_send = None
        self.query_delay = 0.0
        self.close_error = None

    def sendall(self, data):
        self.sent.append(data)
        if self.on_send:
            self.on_send(data)
        if data.startswith(b"MovL("):
            self.moving = True
            if self.move_reply == "write_failure":
                raise OSError("partial write possible")
            if self.move_reply in ("lost", "late"):
                self.pending_move = data
                return
            payload = b"43"
            if self.move_reply == "malformed":
                self.chunks.append(b"not a response;")
            elif self.move_reply == "bad_id":
                self.chunks.append(response(data, b"not-an-id"))
            elif self.move_reply == "wrong_echo":
                self.chunks.append(response(b"MovL(pose={1,2,3,4,5,6})", payload))
            elif self.move_reply == "eof":
                self.chunks.append(b"")
            elif self.move_reply == "keyboard":
                self.chunks.append(KeyboardInterrupt())
            elif self.move_reply == "rejected":
                self.chunks.append(response(data, b"", -2))
            elif self.move_reply == "fragmented":
                reply = response(data, payload)
                self.chunks.extend(reply[i : i + 3] for i in range(0, len(reply), 3))
            elif self.move_reply == "duplicate":
                self.chunks.append(response(data, payload) * 2)
            elif self.move_reply == "trailing":
                self.chunks.append(response(data, payload) + b"0,{5}")
            else:
                self.chunks.append(response(data, payload))
            return
        if data == b"Stop()":
            self.stopped = True
            self.current = replace(self.start, command_id=43)
            self.mode_queries = 0
            if self.stop_reply == "write_failure":
                raise OSError("Stop write failed")
            if self.stop_reply == "timeout":
                return
            if self.move_reply == "late" and self.pending_move:
                self.chunks.append(response(self.pending_move, b"43") + response(data))
                return
            if self.stop_reply == "rejected":
                self.chunks.append(response(data, error=-1))
            elif self.stop_reply == "bad_payload":
                self.chunks.append(response(data, b"43"))
            else:
                self.chunks.append(response(data))
            return
        self.clock.sleep(self.query_delay)
        if data == b"RobotMode()":
            if self.mode_queries % 2 == 0:
                self.samples += 1
                sequence = self.after_stop if self.stopped else self.execution
                if (self.moving or self.stopped) and sequence:
                    self.current = sequence.popleft()
                elif not self.moving and not self.stopped:
                    self.current = self.start
            self.mode_queries += 1
            payload = str(int(self.current.mode)).encode()
        elif data == b"GetErrorID()":
            payload = str(list(self.current.errors)).encode()
        elif data == b"GetCurrentCommandID()":
            payload = str(self.current.command_id).encode()
        elif data.startswith(b"GetPose("):
            payload = ",".join(str(v) for v in to_native(self.current.pose).values).encode()
        elif data == b"GetAngle()":
            payload = b"0,0,0,0,0,0"
        else:
            raise AssertionError(f"forbidden outbound command {data!r}")
        self.chunks.append(response(data, payload))

    def close(self):
        super().close()
        if self.close_error:
            raise self.close_error


def setup(safety, pose, clock, *, live_profile=None, config=None, allow_motion=True):
    profile = live_profile or production_profile(safety)
    start = RobotSnapshot(pose, (0.0,) * 6, RobotMode.ENABLED_IDLE, (), 42, clock.monotonic(), 0, 0)
    plan = build_motion_plan(
        "one-reviewed-plan",
        ((0.31, 0.0, 0.2, 0, 0, 0, 0),),
        start,
        pose,
        profile.safety,
        clock.monotonic(),
        profile.keepouts,
    )
    peer = MotionPeer(start, clock)
    peer.execution.append(replace(start, pose=plan.final_pose_si, command_id=43))
    factory = ScriptedFactory(dashboard=peer)
    driver = LiveDobotMotionDriver(
        config or connection(),
        profile,
        allow_motion=allow_motion,
        socket_factory=factory,
        clock=clock,
        transport_is_mock=True,
    )
    return driver, peer, factory, plan


def arm(driver, plan):
    driver.connect(allow_connection=True)
    driver.confirm_readiness(readiness())
    return driver.arm(plan, confirmation="MOVE ONCE")


def movements(peer):
    return [x for x in peer.sent if x.startswith(b"MovL(")]
