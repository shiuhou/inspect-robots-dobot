import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from inspect_robots_dobot.chunks import StagedDobotEmbodiment
from inspect_robots_dobot.config import ConnectionConfig, DobotConfig
from inspect_robots_dobot.dashboard import DobotDashboardClient
from inspect_robots_dobot.errors import ConfigurationError, MotionNotAuthorized, PhaseUnavailable
from inspect_robots_dobot.motion_dry_run import run_dry_run
from inspect_robots_dobot.readonly_driver import ReadOnlyDobotDriver


def test_dry_run_has_no_socket_path(profile, pose, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("socket construction or Dashboard connection attempted")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(DobotDashboardClient, "connect", forbidden)
    report = run_dry_run(
        DobotConfig(profile, 10), pose, {"x": 0.315}, allow_fake_motion=True, log_dir=tmp_path
    )
    assert report["status"] == "DRY_RUN_OK"
    assert report["prospective_movl_count"] == 1
    assert not report["PHYSICAL_SEND"] and not report["motion_ready"]
    assert report["motion_commands_sent"] == report["robot_connections_opened"] == 0
    assert not report["request_control_attempted"] and report["connections_closed"]


@pytest.mark.parametrize("targets", [{"x": 0.4}, {"yaw": 0.1}, {"gripper": 1}, {"bad_axis": 0.1}])
def test_rejected_scripted_move_stops_without_retry(profile, pose, tmp_path, targets):
    report = run_dry_run(
        DobotConfig(profile, 10), pose, targets, allow_fake_motion=True, log_dir=tmp_path
    )
    assert report["status"] == "BLOCKED" and report["prospective_movl_count"] == 0
    assert report["scripted_model_responses"] == 2


def test_fake_authority_and_connection_config_separated(profile, pose, tmp_path):
    with pytest.raises(MotionNotAuthorized):
        run_dry_run(
            DobotConfig(profile, 10), pose, {"x": 0.31}, allow_fake_motion=False, log_dir=tmp_path
        )
    with pytest.raises(ConfigurationError, match="connection"):
        run_dry_run(
            DobotConfig(profile, 10, connection=ConnectionConfig(host="192.0.2.1")),
            pose,
            {"x": 0.31},
            allow_fake_motion=True,
            log_dir=tmp_path,
        )


def test_real_driver_remains_ineligible_and_cannot_actuate(profile, pose):
    class NoIO:
        def __getattr__(self, name):
            raise AssertionError(f"real operation {name} attempted")

    readonly = ReadOnlyDobotDriver(NoIO())
    with pytest.raises(PhaseUnavailable, match="FakeDobot"):
        StagedDobotEmbodiment(DobotConfig(profile, 10), driver=readonly)
    with pytest.raises(MotionNotAuthorized):
        readonly.move_linear(pose)
    with pytest.raises(MotionNotAuthorized):
        readonly.set_gripper(1)
    with pytest.raises(PhaseUnavailable):
        readonly.stop()


def test_console_entry_point_has_hard_socket_guard(package_root, tmp_path):
    # Child CLI has its own permanent audit guard; inherits no pytest monkeypatches.
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "inspect-robots-dobot-motion-dry-run"),
            "--config",
            str(package_root / "examples/fake_motion.json"),
            "--initial-native-si",
            "0.3",
            "0",
            "0.2",
            "0",
            "0",
            "0",
            "--targets",
            '{"x":0.315}',
            "--allow-fake-motion",
            "--json",
            "--log-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "DRY_RUN_OK" and report["PHYSICAL_SEND"] is False
    assert report["commands_would_send"] == [
        "MovL(pose={315.0,0.0,200.0,0.0,0.0,0.0},user=0,tool=0,a=5,v=5,cp=0)"
    ]


@pytest.mark.parametrize("operation", ["socket", "camera"])
def test_cli_audit_hook_blocks_hardware_before_device_creation(tmp_path, operation):
    # Catch inside child so the only attempted operations are denied before creation.
    code = """
import sys, socket
from inspect_robots_dobot.motion_dry_run import _deny_hardware
sys.addaudithook(_deny_hardware)
try:
    OPERATION
except RuntimeError as exc:
    print(exc)
else:
    raise AssertionError('hardware guard did not reject')
""".replace(
        "OPERATION", "socket.socket()" if operation == "socket" else "open('/dev/video0', 'rb')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert "dry-run prohibits" in result.stdout


def test_default_driver_authority_remains_disabled_in_staged_mode(profile, pose, clock):
    from inspect_robots.scene import Scene
    from test_chunks import ChunkPolicy, advance, chunk_to

    from inspect_robots_dobot.chunks import DobotExecutionSession
    from inspect_robots_dobot.driver import FakeDobotDriver

    driver = FakeDobotDriver(
        initial_pose=pose, initial_joints=(0.0,) * 6, profile=profile, clock=clock
    )
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    with DobotExecutionSession(emb) as s:
        obs = emb.reset(Scene(id="authority", instruction="Offline"))
        with pytest.raises(MotionNotAuthorized):
            advance((emb, s, obs, {}), ChunkPolicy(chunk_to(steps=1)), 0)
    assert not emb.plans and not driver.commands
