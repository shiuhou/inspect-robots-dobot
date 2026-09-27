"""Explicit physical V4L2 MJPEG source; no device enumeration or robot access."""

from __future__ import annotations

import os
import selectors
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .camera import ReceivedRgb
from .config import PhysicalCameraConfig
from .errors import CameraFault
from .mjpeg import JpegFramer, ffmpeg_capture_command

Decoder = Callable[[bytes, int, int, float], npt.NDArray[np.uint8]]
ProcessFactory = Callable[[list[str]], subprocess.Popen[bytes]]


def decode_jpeg_ffmpeg(
    jpeg: bytes, width: int, height: int, timeout: float
) -> npt.NDArray[np.uint8]:
    """Decode one complete JPEG using ffmpeg, rejecting corruption or wrong dimensions."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mjpeg",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command, input=jpeg, capture_output=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CameraFault(f"JPEG decoder unavailable or timed out: {exc}") from exc
    expected = width * height * 3
    if result.returncode != 0 or len(result.stdout) != expected:
        detail = result.stderr.decode(errors="replace")[-1000:].strip()
        raise CameraFault(
            f"corrupted JPEG or wrong dimensions: decoded {len(result.stdout)} bytes, "
            f"expected {expected}; ffmpeg: {detail}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def _spawn_capture(command: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)


class V4L2MjpegFrameSource:
    """One ffmpeg capture process; read and close are bounded and independently testable."""

    def __init__(
        self,
        config: PhysicalCameraConfig,
        *,
        process_factory: ProcessFactory = _spawn_capture,
        decoder: Decoder = decode_jpeg_ffmpeg,
    ) -> None:
        self.config = config
        self._process_factory = process_factory
        self._decoder = decoder
        self._process: subprocess.Popen[bytes] | None = None
        self._framer = JpegFramer()
        self._pending: deque[tuple[bytes, float]] = deque()
        self._stderr = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._closed = False
        self.raw_frame_count = 0
        self.latest_jpeg: bytes | None = None

    def start(self) -> None:
        if self._closed:
            raise CameraFault("closed physical camera source cannot restart")
        if self._process is not None:
            return
        if not Path(self.config.device_path).exists():
            raise CameraFault(
                f"configured camera {self.config.name} missing: {self.config.device_path}"
            )
        command = ffmpeg_capture_command(
            self.config.device_path, self.config.width, self.config.height, self.config.fps
        )
        try:
            process = self._process_factory(command)
        except OSError as exc:
            raise CameraFault(f"cannot start ffmpeg for {self.config.name}: {exc}") from exc
        if process.stdout is None or process.stderr is None:
            process.terminate()
            process.wait(timeout=2)
            raise CameraFault("ffmpeg capture requires stdout and stderr pipes")
        self._process = process
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while not self._closed:
            chunk = process.stderr.read(4096)
            if not chunk:
                return
            self._stderr.extend(chunk)
            if len(self._stderr) > 8192:
                del self._stderr[:-8192]

    def _failure(self, reason: str) -> CameraFault:
        detail = self._stderr.decode(errors="replace").strip()
        return CameraFault(f"{self.config.name} ffmpeg {reason}: {detail or 'no stderr detail'}")

    def read(self, timeout: float) -> ReceivedRgb:
        if self._process is None:
            self.start()
        process = self._process
        assert process is not None and process.stdout is not None
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while not self._pending:
                if self._closed:
                    raise CameraFault("camera source closed during read")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._failure("frame timeout")
                if not selector.select(min(remaining, 0.1)):
                    if process.poll() is not None:
                        raise self._failure(f"exited with code {process.returncode}")
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    reason = "EOF with incomplete JPEG" if self._framer.incomplete else "EOF"
                    raise self._failure(reason)
                try:
                    frames = self._framer.feed(chunk)
                except ValueError as exc:
                    raise CameraFault(f"{self.config.name}: {exc}") from exc
                received_at = time.monotonic()
                self.raw_frame_count += len(frames)
                self._pending.extend((jpeg, received_at) for jpeg in frames)
        # A burst can contain several frames; take the newest complete host frame.
        jpeg, received_at = self._pending.pop()
        self._pending.clear()
        self.latest_jpeg = jpeg
        rgb = self._decoder(
            jpeg, self.config.width, self.config.height, max(0.1, deadline - time.monotonic())
        )
        if rgb.shape != (self.config.height, self.config.width, 3) or rgb.dtype != np.uint8:
            raise CameraFault(f"{self.config.name} decoder returned non-RGB or wrong shape")
        return ReceivedRgb(rgb, received_at)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)
            if self._stderr_thread.is_alive():
                raise CameraFault(f"{self.config.name} stderr reader did not stop")
