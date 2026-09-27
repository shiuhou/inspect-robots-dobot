from dataclasses import replace

import pytest
from inspect_robots.scene import Scene
from test_chunks import ChunkPolicy, advance, chunk_to

from inspect_robots_dobot.camera import FakeCamera
from inspect_robots_dobot.chunks import DobotExecutionSession, StagedDobotEmbodiment
from inspect_robots_dobot.config import CameraConfig, DobotConfig
from inspect_robots_dobot.driver import FakeDobotDriver
from inspect_robots_dobot.errors import CameraFault, DriverFault, SettleTimeout
from inspect_robots_dobot.safety import MotionAuthority
from inspect_robots_dobot.types import RobotMode


@pytest.mark.parametrize("delay", [0.0, 0.05, 0.2])
def test_staged_fake_immediate_and_delayed_settle(profile, pose, clock, delay):
    driver = FakeDobotDriver(
        initial_pose=pose,
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(True),
        convergence_delay=delay,
    )
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    with DobotExecutionSession(emb) as session:
        obs = emb.reset(Scene(id="settle", instruction="Offline"))
        result = advance((emb, session, obs, {}), ChunkPolicy(chunk_to(steps=1)), 0)
        assert result.observation.state["eef_state"][0] == 0.31
        assert emb.execution_results[0].settle_duration == pytest.approx(delay)
    assert len(driver.commands) == 1


@pytest.mark.parametrize(
    "fault,exception",
    [
        ("no_motion", SettleTimeout),
        ("wrong_id", SettleTimeout),
        ("position_residual", SettleTimeout),
        ("orientation_residual", DriverFault),
        ("alarm", DriverFault),
        ("collision", DriverFault),
        ("stale", DriverFault),
        ("cancel", KeyboardInterrupt),
    ],
)
def test_failure_after_one_fake_command_never_retries(
    profile, driver, pose, clock, monkeypatch, fault, exception
):
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10), driver=driver)
    original = driver.snapshot

    def failing_snapshot():
        sample = original()
        if not driver.commands:
            return sample
        if fault == "cancel":
            raise KeyboardInterrupt
        changes = {
            "no_motion": {"pose": pose, "mode": RobotMode.RUNNING},
            "wrong_id": {"command_id": sample.command_id + 1},
            "position_residual": {"pose": replace(sample.pose, x=sample.pose.x - 0.002)},
            "orientation_residual": {"pose": replace(sample.pose, rx=sample.pose.rx + 0.01)},
            "alarm": {"mode": RobotMode.ERROR, "errors": (42,)},
            "collision": {"mode": RobotMode.COLLISION},
            "stale": {"observed_at": clock.monotonic() - 1},
        }[fault]
        return replace(sample, **changes)

    with DobotExecutionSession(emb) as s:
        obs = emb.reset(Scene(id="fault", instruction="Offline"))
        monkeypatch.setattr(driver, "snapshot", failing_snapshot)
        with pytest.raises(exception):
            advance((emb, s, obs, {}), ChunkPolicy(chunk_to(steps=1)), 0)
        assert emb.pending_chunk_id is None
        assert len(emb.plans) == 1 and not emb.execution_results
        with pytest.raises(DriverFault, match="fault-latched"):
            emb.reset(Scene(id="retry", instruction="Must refuse"))
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]


def test_post_settle_stale_camera_visible_without_second_move(profile, driver, clock):
    cfg = CameraConfig(8, 6, 0.2, 0.1)
    camera = FakeCamera(cfg, clock)
    emb = StagedDobotEmbodiment(DobotConfig(profile, 10, cfg), driver=driver, camera=camera)
    with DobotExecutionSession(emb) as s:
        obs = emb.reset(Scene(id="camera", instruction="Offline"))
        camera.freeze = True
        with pytest.raises(CameraFault, match="after settling|stale"):
            advance((emb, s, obs, {}), ChunkPolicy(chunk_to(steps=1)), 0)
        assert not emb.execution_results and len(emb.plans) == 1
    assert [c.name for c in driver.commands] == ["move_linear", "stop"]
