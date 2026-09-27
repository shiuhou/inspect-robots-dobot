"""Shared V4L2/MJPEG framing used by the adapter and local preview."""

from __future__ import annotations


def ffmpeg_capture_command(device: str, width: int, height: int, fps: int) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-framerate",
        str(fps),
        "-video_size",
        f"{width}x{height}",
        "-i",
        device,
        "-an",
        "-c:v",
        "copy",
        "-f",
        "image2pipe",
        "pipe:1",
    ]


class JpegFramer:
    """Bounded SOI/EOI framing; decoding separately verifies JPEG integrity."""

    def __init__(self, max_frame_bytes: int = 8_000_000) -> None:
        self._buffer = bytearray()
        self.max_frame_bytes = max_frame_bytes

    @property
    def incomplete(self) -> bool:
        return bool(self._buffer)

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer.extend(chunk)
        frames: list[bytes] = []
        while True:
            start = self._buffer.find(b"\xff\xd8")
            if start < 0:
                if len(self._buffer) > 1:
                    del self._buffer[:-1]
                return frames
            if start:
                del self._buffer[:start]
            end = self._buffer.find(b"\xff\xd9", 2)
            if end < 0:
                if len(self._buffer) > self.max_frame_bytes:
                    raise ValueError("MJPEG frame exceeds configured byte limit")
                return frames
            if end + 2 > self.max_frame_bytes:
                raise ValueError("MJPEG frame exceeds configured byte limit")
            frames.append(bytes(self._buffer[: end + 2]))
            del self._buffer[: end + 2]
