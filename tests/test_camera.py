import queue
import threading
from dataclasses import replace

import numpy as np
import pytest

from inspect_robots_dobot.camera import FakeCamera, Frame, LatestFrameReader, validate_frame
from inspect_robots_dobot.config import CameraConfig
from inspect_robots_dobot.errors import CameraFault


@pytest.fixture
def camera_config():
    return CameraConfig(8, 6, max_age=0.2, wait_timeout=0.1)


def test_fake_clock_camera_freshness_and_postsettle(camera_config, clock):
    camera = FakeCamera(camera_config, clock)
    camera.start()
    first = camera.latest()
    after = clock.monotonic()
    newer = camera.latest(after=after)
    assert newer.sequence > first.sequence
    assert newer.acquisition_started_at > after
    assert newer.rgb.dtype == np.uint8
    assert newer.rgb.shape == (6, 8, 3)
    assert not newer.rgb.flags.writeable
    assert "host" in newer.timestamp_source
    camera.close()


def test_freeze_and_restart_cannot_reuse_old_frames(camera_config, clock):
    camera = FakeCamera(camera_config, clock)
    camera.start()
    first = camera.latest()
    camera.freeze = True
    with pytest.raises(CameraFault, match="after settling"):
        camera.latest(after=clock.monotonic())
    clock.sleep(0.3)
    with pytest.raises(CameraFault, match="stale"):
        camera.latest()
    camera.close()
    camera.start()
    assert camera.latest().generation > first.generation
    camera.close()


def test_fake_camera_fault_and_not_running(camera_config, clock):
    camera = FakeCamera(camera_config, clock)
    with pytest.raises(CameraFault, match="not running"):
        camera.latest()
    camera.start()
    camera.failure = "disconnected source"
    with pytest.raises(CameraFault, match="disconnected"):
        camera.latest()


@pytest.mark.parametrize(
    "change",
    [
        {"published_at": 3.0},
        {"acquisition_started_at": float("nan")},
        {"acquisition_started_at": -1.0},
        {"acquisition_started_at": 1.5},
        {"rgb": np.zeros((6, 8, 3), dtype=np.float64)},
        {"rgb": np.zeros((8, 6, 3), dtype=np.uint8)},
    ],
)
def test_malformed_frames_fail(camera_config, change):
    frame = Frame(np.zeros((6, 8, 3), dtype=np.uint8), 1, 1, 1, 1)
    with pytest.raises(CameraFault):
        validate_frame(replace(frame, **change), camera_config, now=1.1, after=None)


class PushSource:
    def __init__(self):
        self.items = queue.Queue()
        self.drained = threading.Event()
        self.count = 0

    def read(self, timeout):
        item = self.items.get(timeout=timeout)
        if item is None:
            raise RuntimeError("closed")
        self.count += 1
        if self.count >= 3:
            self.drained.set()
        return item

    def close(self):
        self.items.put(None)


def test_background_reader_drains_and_copies_latest_frame():
    source = PushSource()
    originals = [np.full((6, 8, 3), n, dtype=np.uint8) for n in (1, 2, 3)]
    for rgb in originals:
        source.items.put(rgb)
    reader = LatestFrameReader(source, CameraConfig(8, 6, 1, 0.5))
    reader.start()
    try:
        assert source.drained.wait(0.5)
        # Synchronize publication separately from the source's read notification.
        with reader._condition:
            assert reader._condition.wait_for(lambda: reader._sequence >= 3, timeout=0.5)
        frame = reader.latest()
        assert frame.sequence == 3
        originals[-1][:] = 99
        assert np.all(frame.rgb == 3)
        assert not frame.rgb.flags.writeable
    finally:
        reader.close()
    with pytest.raises(CameraFault, match="not running"):
        reader.latest()
    with pytest.raises(CameraFault, match="cannot restart"):
        reader.start()


def test_background_reader_reports_source_failure():
    class BrokenSource:
        def read(self, timeout):
            raise RuntimeError("capture unplugged")

        def close(self):
            pass

    reader = LatestFrameReader(BrokenSource(), CameraConfig(8, 6, 1, 0.1))
    reader.start()
    try:
        with pytest.raises(CameraFault, match="capture unplugged"):
            reader.latest()
    finally:
        reader.close()


def test_background_reader_timeout_for_frame_started_before_settle():
    source = PushSource()
    source.items.put(np.zeros((6, 8, 3), dtype=np.uint8))
    reader = LatestFrameReader(source, CameraConfig(8, 6, 1, 0.03))
    reader.start()
    try:
        first = reader.latest()
        with pytest.raises(CameraFault, match="after settling|capture failed"):
            reader.latest(after=first.published_at)
    finally:
        reader.close()
