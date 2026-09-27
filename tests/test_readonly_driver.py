import pytest
from fake_sockets import ScriptedFactory, dashboard_socket
from inspect_robots.scene import Scene

from inspect_robots_dobot.config import ConnectionConfig, DobotConfig
from inspect_robots_dobot.dashboard import DobotDashboardClient
from inspect_robots_dobot.driver import DobotDriver
from inspect_robots_dobot.embodiment import DobotEmbodiment
from inspect_robots_dobot.errors import MotionNotAuthorized, PhaseUnavailable, QueryUnavailable
from inspect_robots_dobot.readonly_driver import ReadOnlyDobotDriver
from inspect_robots_dobot.types import RobotMode


def make_driver(clock, mode=5, *, frames=True):
    socket = dashboard_socket(mode=mode)
    config = (
        ConnectionConfig(host="192.0.2.1", user_frame=0, tool_frame=2)
        if frames
        else ConnectionConfig(host="192.0.2.1")
    )
    factory = ScriptedFactory(dashboard=socket)
    dashboard = DobotDashboardClient(config, socket_factory=factory, clock=clock)
    return ReadOnlyDobotDriver(dashboard), socket, factory


def test_driver_contract_sequential_snapshot_is_measured_readonly(clock):
    concrete, stream, factory = make_driver(clock)
    driver: DobotDriver = concrete
    assert factory.calls == []
    driver.connect()
    assert stream.sent == []
    sample = driver.snapshot()
    assert sample.pose.x == 0.3
    assert not sample.joints_synthetic
    assert sample.command_id == 42
    assert sample.mode == RobotMode.ENABLED_IDLE
    assert (sample.user_frame, sample.tool_frame) == (0, 2)
    assert sample.observed_at == 1
    assert set(stream.sent) == {
        b"RobotMode()",
        b"GetPose(user=0,tool=2)",
        b"GetAngle()",
        b"GetErrorID()",
        b"GetCurrentCommandID()",
    }
    driver.close()
    assert stream.closed


@pytest.mark.parametrize("mode", [3, 9])
def test_state_restriction_never_recovers_or_sends_pose_query(clock, mode):
    driver, stream, _ = make_driver(clock, mode)
    driver.connect()
    for query in (driver.get_pose, driver.get_joints):
        with pytest.raises(QueryUnavailable, match="p160"):
            query()
    assert stream.sent == [b"RobotMode()", b"RobotMode()"]
    assert driver.get_errors() == ()
    driver.close()


def test_missing_snapshot_frames_does_not_invent_identity(clock):
    driver, stream, _ = make_driver(clock, frames=False)
    driver.connect()
    with pytest.raises(QueryUnavailable, match="explicit"):
        driver.snapshot()
    assert stream.sent == []
    assert driver.get_pose().x == 0.3
    assert stream.sent[-1] == b"GetPose()"


def test_snapshot_mode_change_is_visible(clock):
    driver, stream, _ = make_driver(clock)
    count = 0
    send = stream.sendall

    def scripted(data):
        nonlocal count
        if data == b"RobotMode()":
            count += 1
            if count == 4:
                stream.responses[data] = [b"0,{7},RobotMode();"]
        send(data)

    stream.sendall = scripted
    driver.connect()
    with pytest.raises(QueryUnavailable, match="changed"):
        driver.snapshot()


def test_no_real_driver_can_enter_action_embodiment(clock, profile, pose):
    driver, stream, factory = make_driver(clock)
    for method, arg in ((driver.move_linear, pose), (driver.set_gripper, 1)):
        with pytest.raises(MotionNotAuthorized):
            method(arg)
    with pytest.raises(PhaseUnavailable):
        driver.stop()
    with pytest.raises(PhaseUnavailable):
        DobotEmbodiment(DobotConfig(profile, 10), driver=driver).reset(Scene(id="no-real"))
    assert stream.sent == [] and factory.calls == []
    with pytest.raises(QueryUnavailable):
        driver.read_feedback()


@pytest.mark.parametrize("mode", range(1, 12))
def test_requestcontrol_is_absent_for_every_mode_including_eligible_states(clock, mode):
    # No partial prerequisite implementation: even modes 3/4 cannot request ownership.
    driver, stream, _ = make_driver(clock, mode)
    driver.connect()
    assert driver.robot_mode() == mode
    for obj in (driver, driver.dashboard):
        assert not hasattr(obj, "request_control")
        for name in ("power_on", "enable_robot", "clear_error", "move_joint", "servo_p", "tool_do"):
            assert not hasattr(obj, name)
    driver.close()
    assert stream.sent == [b"RobotMode()"]
