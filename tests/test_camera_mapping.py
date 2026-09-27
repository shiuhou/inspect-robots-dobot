"""Three-camera contract and physical backend tests without opening a device."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from inspect_robots.scene import Scene

from inspect_robots_dobot.camera import FakeCamera, LatestFrameReader
from inspect_robots_dobot.camera_smoke import acquire_after
from inspect_robots_dobot.camera_v4l2 import V4L2MjpegFrameSource, decode_jpeg_ffmpeg
from inspect_robots_dobot.config import DobotConfig, PhysicalCameraConfig, load_config
from inspect_robots_dobot.embodiment import DobotEmbodiment, build_info
from inspect_robots_dobot.errors import CameraFault, ConfigurationError
from inspect_robots_dobot.mjpeg import JpegFramer, ffmpeg_capture_command


@pytest.fixture
def specs():
    return tuple(
        PhysicalCameraConfig(
            name,
            f"/dev/v4l/by-id/{name}",
            width,
            height,
            30,
            "mjpeg",
            True,
            1.0,
            0.1,
            0.2,
        )
        for name, width, height in (
            ("front_rgb", 8, 6),
            ("right_rgb", 8, 6),
            ("wrist_rgb", 10, 8),
        )
    )


def test_production_mapping_and_legacy_single_camera_are_explicit(package_root, specs):
    config = load_config(package_root / "examples/cameras.json")
    assert {spec.name for spec in config.cameras} == {"front_rgb", "right_rgb", "wrist_rgb"}
    assert config.cameras[0].device_path.endswith("1080P_USB_Camera-video-index0")
    assert [(s.width, s.height) for s in config.cameras] == [(1280, 720), (640, 480), (640, 480)]
    assert tuple(
        spec.name for spec in build_info(DobotConfig(cameras=specs)).observation_space.cameras
    ) == ("front_rgb", "right_rgb", "wrist_rgb")
    with pytest.raises(ConfigurationError, match="front_rgb/right_rgb/wrist_rgb"):
        DobotConfig(cameras=specs[:2])
    with pytest.raises(ConfigurationError, match="no top_rgb"):
        replace(specs[0], name="top_rgb")
    with pytest.raises(ConfigurationError, match="stable"):
        replace(specs[0], device_path="/dev/video10")


def test_three_camera_observation_and_post_settle_barrier(profile, driver, clock, specs):
    cameras = {spec.name: FakeCamera(spec.frame_config, clock) for spec in specs}
    emb = DobotEmbodiment(
        DobotConfig(profile, 10, cameras=specs), driver=driver, camera=cameras, clock=clock
    )
    try:
        initial = emb.reset(Scene(id="three", instruction="offline"))
        assert set(initial.images) == set(initial.image_times) == set(cameras)
        assert initial.images["wrist_rgb"].shape == (8, 10, 3)
        assert initial.extra["dobot"]["camera"]["host_receive_skew_s"] >= 0
        before = initial.extra["dobot"]["camera"]["frames"]
        clock.sleep(0.02)
        barrier = clock.monotonic()
        after = emb._observe(after=barrier)
        assert all(after.image_times[name] > barrier for name in cameras)
        assert all(
            after.extra["dobot"]["camera"]["frames"][name]["sequence"] > before[name]["sequence"]
            for name in cameras
        )
        assert not any(name == "top_rgb" for name in after.images)
        cameras["wrist_rgb"].freeze = True
        clock.sleep(0.02)
        with pytest.raises(CameraFault, match="wrist_rgb|after settling"):
            emb._observe(after=clock.monotonic())
    finally:
        emb.close()
    assert all(not camera._running for camera in cameras.values())


def test_static_barrier_checks_generation_and_timestamp(specs, clock):
    cameras = {spec.name: FakeCamera(spec.frame_config, clock) for spec in specs}
    for camera in cameras.values():
        camera.start()
    try:
        before = {
            name: (frame.generation, frame.sequence)
            for name, camera in cameras.items()
            if (frame := camera.latest())
        }
        clock.sleep(0.02)
        barrier = clock.monotonic()
        fresh = acquire_after(cameras, before, barrier)
        assert all(frame.timestamp > barrier for frame in fresh.values())
        assert all(
            (frame.generation, frame.sequence) > before[name] for name, frame in fresh.items()
        )
    finally:
        for camera in cameras.values():
            camera.close()


def test_jpeg_framer_split_burst_junk_and_incomplete():
    parser = JpegFramer(max_frame_bytes=20)
    assert parser.feed(b"junk\xff") == []
    assert parser.feed(b"\xd8abc") == []
    assert parser.incomplete
    assert parser.feed(b"\xff\xd9\xff\xd8x\xff\xd9") == [
        b"\xff\xd8abc\xff\xd9",
        b"\xff\xd8x\xff\xd9",
    ]
    assert not parser.incomplete
    with pytest.raises(ValueError, match="limit"):
        parser.feed(b"\xff\xd8" + b"x" * 30)


def test_capture_command_exactly_preserves_working_preview_pipeline():
    command = ffmpeg_capture_command("/dev/v4l/by-id/camera", 640, 480, 30)
    assert command[-6:] == ["-an", "-c:v", "copy", "-f", "image2pipe", "pipe:1"]
    assert "mjpeg" in command and "640x480" in command


def test_jpeg_decoder_rgb_shape_and_corruption():
    raw = bytes([255, 0, 0] * 16)
    encode = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            "4x4",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ],
        input=raw,
        capture_output=True,
        timeout=5,
        check=True,
    )
    image = decode_jpeg_ffmpeg(encode.stdout, 4, 4, 5)
    assert image.shape == (4, 4, 3) and image.dtype == np.uint8
    assert image[:, :, 0].mean() > 200 and image[:, :, 1].mean() < 60
    with pytest.raises(CameraFault, match="corrupted JPEG"):
        decode_jpeg_ffmpeg(b"\xff\xd8bad\xff\xd9", 4, 4, 5)


def _fake_process(code: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
    )


def test_physical_source_missing_path_and_eof(monkeypatch, specs):
    source = V4L2MjpegFrameSource(specs[0])
    with pytest.raises(CameraFault, match="missing"):
        source.start()
    source.close()
    monkeypatch.setattr(Path, "exists", lambda self: True)
    source = V4L2MjpegFrameSource(
        specs[0],
        process_factory=lambda cmd: _fake_process(
            "import sys; sys.stdout.buffer.write(b'\\xff\\xd8partial')"
        ),
    )
    try:
        with pytest.raises(CameraFault, match="incomplete JPEG"):
            source.read(1)
    finally:
        source.close()


def test_physical_source_fake_process_rgb_and_bounded_close(monkeypatch, specs):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    source = V4L2MjpegFrameSource(
        specs[0],
        process_factory=lambda cmd: _fake_process(
            "import sys,time; sys.stdout.buffer.write(b'junk\\xff\\xd8a\\xff\\xd9');"
            "sys.stdout.flush(); time.sleep(10)"
        ),
        decoder=lambda jpeg, width, height, timeout: np.full((height, width, 3), 7, np.uint8),
    )
    reader = LatestFrameReader(source, specs[0].frame_config)
    reader.start()
    try:
        frame = reader.latest()
        assert frame.rgb.shape == (6, 8, 3)
        assert frame.rgb[0, 0, 0] == 7
        assert frame.host_receive_time_monotonic is not None
        assert frame.timestamp_source == "host_monotonic_jpeg_complete"
        assert source.raw_frame_count == 1
    finally:
        reader.close()
    assert source._process is not None and source._process.poll() is not None
    with pytest.raises(CameraFault, match="cannot restart"):
        reader.start()


def test_physical_source_crash_and_repeated_open_close(monkeypatch, specs):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    for _ in range(2):
        source = V4L2MjpegFrameSource(
            specs[0],
            process_factory=lambda cmd: _fake_process(
                "import sys; sys.stderr.write('VIDIOC_STREAMON: No space left on device');"
                "sys.exit(3)"
            ),
        )
        try:
            with pytest.raises(CameraFault, match="EOF|exited"):
                source.read(1)
        finally:
            source.close()
        assert source._process is not None and source._process.poll() is not None


def test_failed_reset_closes_all_three_readers(profile, driver, clock, specs):
    cameras = {spec.name: FakeCamera(spec.frame_config, clock) for spec in specs}
    cameras["right_rgb"].failure = "camera unplugged"
    emb = DobotEmbodiment(
        DobotConfig(profile, 10, cameras=specs), driver=driver, camera=cameras, clock=clock
    )
    with pytest.raises(CameraFault, match="right_rgb: camera unplugged"):
        emb.reset(Scene(id="bad", instruction="offline"))
    assert all(not camera._running for camera in cameras.values())
