"""No rig-specific defaults. A safety profile must be explicitly supplied."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .transforms import DEFAULT_SINGULARITY_MARGIN


def positive(name: str, value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be finite and > 0")


@dataclass(frozen=True)
class SafetyProfile:
    workspace_low: tuple[float, float, float]
    workspace_high: tuple[float, float, float]
    minimum_tcp_z: float
    user_frame: int
    tool_frame: int
    max_translation_step: float
    max_orientation_step: float
    speed_percent: int
    acceleration_percent: int
    position_tolerance: float
    orientation_tolerance: float
    settle_timeout: float
    telemetry_max_age: float
    poll_interval: float
    # Agent-relative yaw,pitch,roll in radians. Zero-width defaults grant no rotation.
    orientation_low: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_high: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_singularity_margin: float = DEFAULT_SINGULARITY_MARGIN

    def __post_init__(self) -> None:
        for name, vector in (
            ("workspace_low", self.workspace_low),
            ("workspace_high", self.workspace_high),
        ):
            if (
                not isinstance(vector, tuple)
                or len(vector) != 3
                or not all(type(v) in (int, float) and math.isfinite(v) for v in vector)
            ):
                raise ConfigurationError(f"{name} must contain three finite metre values")
        if any(lo >= hi for lo, hi in zip(self.workspace_low, self.workspace_high, strict=True)):
            raise ConfigurationError("workspace_low must be strictly below workspace_high")
        if (
            type(self.minimum_tcp_z) not in (int, float)
            or not math.isfinite(self.minimum_tcp_z)
            or not (self.workspace_low[2] <= self.minimum_tcp_z < self.workspace_high[2])
        ):
            raise ConfigurationError("minimum_tcp_z must be within the configured workspace")
        for name in ("user_frame", "tool_frame"):
            v = getattr(self, name)
            if type(v) is not int or not 0 <= v <= 50:
                raise ConfigurationError(f"{name} must be an integer in [0,50]")
        if self.user_frame != 0:
            raise ConfigurationError("base-frame actions require explicit user_frame=0")
        for name in ("speed_percent", "acceleration_percent"):
            v = getattr(self, name)
            if type(v) is not int or not 1 <= v <= 100:
                raise ConfigurationError(f"{name} must be an integer in [1,100]")
        for name in (
            "max_translation_step",
            "max_orientation_step",
            "position_tolerance",
            "orientation_tolerance",
            "settle_timeout",
            "telemetry_max_age",
            "poll_interval",
        ):
            positive(name, getattr(self, name))
        if self.max_orientation_step > math.pi:
            raise ConfigurationError("max_orientation_step must not exceed pi radians")
        if self.position_tolerance >= self.max_translation_step:
            raise ConfigurationError("position_tolerance must be smaller than max_translation_step")
        if self.orientation_tolerance >= self.max_orientation_step:
            raise ConfigurationError(
                "orientation_tolerance must be smaller than max_orientation_step"
            )
        if self.poll_interval > self.settle_timeout:
            raise ConfigurationError("poll_interval must not exceed settle_timeout")
        margin = self.orientation_singularity_margin
        positive("orientation_singularity_margin", margin)
        if margin >= math.pi / 2:
            raise ConfigurationError("orientation_singularity_margin must be below pi/2")
        for name in ("orientation_low", "orientation_high"):
            vector = getattr(self, name)
            if (
                not isinstance(vector, tuple)
                or len(vector) != 3
                or not all(type(v) in (int, float) and math.isfinite(v) for v in vector)
            ):
                raise ConfigurationError(f"{name} must contain three finite radian values")
        for axis, lo, hi in zip(
            ("yaw", "pitch", "roll"), self.orientation_low, self.orientation_high, strict=True
        ):
            if not -math.pi <= lo <= 0 <= hi <= math.pi:
                raise ConfigurationError(f"{axis} bounds must contain reset zero within [-pi,pi]")
            if axis == "pitch" and max(abs(lo), abs(hi)) >= math.pi / 2 - margin:
                raise ConfigurationError("pitch bounds enter the configured singularity exclusion")


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    max_age: float
    wait_timeout: float

    def __post_init__(self) -> None:
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or min(self.width, self.height) <= 0
        ):
            raise ConfigurationError("camera dimensions must be positive integers")
        positive("camera.max_age", self.max_age)
        positive("camera.wait_timeout", self.wait_timeout)


CAMERA_NAMES = frozenset({"front_rgb", "right_rgb", "wrist_rgb"})
REVIEWED_CAMERA_HARDWARE = {
    "wrist_rgb": (
        "/dev/v4l/by-id/usb-1080P_USB_Camera-video-index0",
        1280,
        720,
        30,
    ),
    "front_rgb": (
        "/dev/v4l/by-path/pci-0000:0e:00.0-usb-0:3.1:1.0-video-index0",
        640,
        480,
        30,
    ),
    "right_rgb": (
        "/dev/v4l/by-path/pci-0000:0e:00.0-usb-0:3.2:1.0-video-index0",
        640,
        480,
        30,
    ),
}


@dataclass(frozen=True)
class PhysicalCameraConfig:
    name: str
    device_path: str
    width: int
    height: int
    fps: int
    pixel_format: str
    required: bool
    startup_timeout: float
    fresh_timeout: float
    max_age: float

    def __post_init__(self) -> None:
        if self.name not in CAMERA_NAMES:
            raise ConfigurationError(f"unknown camera name {self.name!r}; no top_rgb/table device")
        if not self.device_path.startswith(("/dev/v4l/by-id/", "/dev/v4l/by-path/")):
            raise ConfigurationError("physical camera requires a stable /dev/v4l/by-id or by-path")
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or min(self.width, self.height) <= 0
        ):
            raise ConfigurationError("camera dimensions must be positive integers")
        if type(self.fps) is not int or self.fps <= 0:
            raise ConfigurationError("camera fps must be a positive integer")
        if self.pixel_format != "mjpeg":
            raise ConfigurationError("physical camera capture format must be mjpeg")
        if type(self.required) is not bool:
            raise ConfigurationError("camera required must be boolean")
        for field in ("startup_timeout", "fresh_timeout", "max_age"):
            positive(f"camera.{field}", getattr(self, field))

    @property
    def frame_config(self) -> CameraConfig:
        return CameraConfig(self.width, self.height, self.max_age, self.fresh_timeout)


def validate_camera_hardware_mapping(cameras: tuple[PhysicalCameraConfig, ...]) -> None:
    for camera in cameras:
        expected = REVIEWED_CAMERA_HARDWARE[camera.name]
        actual = (camera.device_path, camera.width, camera.height, camera.fps)
        if actual != expected:
            raise ConfigurationError(
                f"{camera.name} hardware mapping differs from the reviewed camera profile"
            )


@dataclass(frozen=True)
class LocalMicroMoveSettings:
    """Unbound standalone +Z experiment limits, NEVER a production workspace.

    No geometry exists until one live session measures its start. These settings
    cannot be supplied as SafetyProfile to the generic embodiment or fake driver.
    """

    measurement_margin_m: float
    position_tolerance: float
    orientation_tolerance: float
    settle_timeout: float
    telemetry_max_age: float
    poll_interval: float
    max_orientation_step: float
    direction: str = "+Z"
    distance_m: float = 0.010
    user_frame: int = 0
    tool_frame: int = 0
    speed_percent: int = 5
    acceleration_percent: int = 5
    threshold_class: str = "ENGINEERING_THRESHOLD_FOR_FIRST_MICROMOVE"

    @property
    def max_translation_step(self) -> float:
        return self.distance_m

    @property
    def orientation_low(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    @property
    def orientation_high(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.direction != "+Z" or self.distance_m != 0.010:
            raise ConfigurationError("local micro-move is exclusively +Z exactly 0.010 m")
        for name, required in (
            ("user_frame", 0),
            ("tool_frame", 0),
            ("speed_percent", 5),
            ("acceleration_percent", 5),
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) != required:
                raise ConfigurationError(f"reviewed local micro-move requires {name}={required}")
        for name in (
            "measurement_margin_m",
            "position_tolerance",
            "orientation_tolerance",
            "settle_timeout",
            "telemetry_max_age",
            "poll_interval",
            "max_orientation_step",
        ):
            positive(name, getattr(self, name))
        if not self.position_tolerance <= self.measurement_margin_m <= 0.001:
            raise ConfigurationError("local margin must cover settle tolerance and be <=1mm")
        if not self.orientation_tolerance < self.max_orientation_step <= math.pi:
            raise ConfigurationError("invalid local angular residual/step thresholds")
        if self.poll_interval > self.settle_timeout:
            raise ConfigurationError("poll interval must not exceed settle timeout")
        if self.threshold_class != "ENGINEERING_THRESHOLD_FOR_FIRST_MICROMOVE":
            raise ConfigurationError("local thresholds are engineering choices, not certification")


@dataclass(frozen=True)
class ConnectionConfig:
    """Read-only connection settings, independent of motion geometry/authority.

    Firmware and ownership fields are operator declarations, not detected facts.
    Literal IPs avoid an unbounded DNS lookup during a bounded diagnostic operation.
    """

    host: str | None = None
    dashboard_port: int = 29999
    feedback_port: int = 30004
    timeout: float = 2.0
    feedback_max_age: float = 0.25
    expected_protocol_version: str = "4.6.5"
    controller_firmware: str | None = None
    protocol_compatibility_confirmed: bool = False
    tcp_control_owned: bool | None = None
    user_frame: int | None = None
    tool_frame: int | None = None

    def __post_init__(self) -> None:
        if self.host is not None:
            try:
                if not isinstance(self.host, str):
                    raise ValueError("host must be a literal IP address")
                ip_address(self.host)
            except ValueError as exc:
                raise ConfigurationError("host must be an explicit literal IP address") from exc
        for name, expected in (("dashboard_port", 29999), ("feedback_port", 30004)):
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ConfigurationError(f"read-only query {name} must be {expected}")
        positive("connection.timeout", self.timeout)
        positive("connection.feedback_max_age", self.feedback_max_age)
        if (
            not isinstance(self.expected_protocol_version, str)
            or not self.expected_protocol_version
        ):
            raise ConfigurationError("expected_protocol_version must be a nonempty string")
        if self.controller_firmware is not None and (
            not isinstance(self.controller_firmware, str) or not self.controller_firmware.strip()
        ):
            raise ConfigurationError("controller_firmware must be a nonempty string or null")
        if type(self.protocol_compatibility_confirmed) is not bool:
            raise ConfigurationError("protocol_compatibility_confirmed must be boolean")
        if self.protocol_compatibility_confirmed and self.controller_firmware is None:
            raise ConfigurationError("confirmed compatibility requires operator-recorded firmware")
        if self.tcp_control_owned is not None and type(self.tcp_control_owned) is not bool:
            raise ConfigurationError("tcp_control_owned must be boolean or null (unknown)")
        if (self.user_frame is None) != (self.tool_frame is None):
            raise ConfigurationError("read-only GetPose requires both frames or neither")
        if self.user_frame is not None:
            for value in (self.user_frame, self.tool_frame):
                if type(value) is not int or not 0 <= value <= 50:
                    raise ConfigurationError("read-only frame indices must be integers in [0,50]")


@dataclass(frozen=True)
class DobotConfig:
    safety: SafetyProfile | None = None
    control_hz: float | None = None
    camera: CameraConfig | None = None
    connection: ConnectionConfig | None = None
    local_micro_move: LocalMicroMoveSettings | None = None
    cameras: tuple[PhysicalCameraConfig, ...] = ()

    def __post_init__(self) -> None:
        if self.safety is not None and type(self.safety) is not SafetyProfile:
            raise ConfigurationError("general config cannot accept a session-local envelope")
        if self.local_micro_move is not None and (
            self.safety is not None
            or self.control_hz is not None
            or self.camera is not None
            or self.cameras
        ):
            raise ConfigurationError("local micro-move cannot define general safety/rate/camera")
        if self.camera is not None and self.cameras:
            raise ConfigurationError("legacy table camera and named cameras cannot be mixed")
        if self.cameras:
            if len(self.cameras) != 3 or {camera.name for camera in self.cameras} != CAMERA_NAMES:
                raise ConfigurationError(
                    "camera mode requires front_rgb/right_rgb/wrist_rgb exactly"
                )
            if not all(camera.required for camera in self.cameras):
                raise ConfigurationError("camera benchmark requires all three cameras")
        if self.control_hz is not None:
            positive("control_hz", self.control_hz)


def load_config(path: Path) -> DobotConfig:
    """Load explicit metadata/safety configuration, never runtime motion authority."""
    try:
        obj: Any = json.loads(path.read_text())
        if not isinstance(obj, dict) or set(obj) - {
            "safety",
            "control_hz",
            "camera",
            "connection",
            "local_micro_move",
            "cameras",
        }:
            raise ValueError(
                "expected safety/control_hz/camera/connection keys only; authority is runtime-only"
            )
        safety_data = obj.get("safety")
        safety = None
        if safety_data is not None:
            data = dict(safety_data)
            for key in ("workspace_low", "workspace_high", "orientation_low", "orientation_high"):
                if key in data:
                    data[key] = tuple(data[key])
            safety = SafetyProfile(**data)
        camera_data = obj.get("camera")
        cameras_data = obj.get("cameras", [])
        connection_data = obj.get("connection")
        return DobotConfig(
            safety=safety,
            control_hz=obj.get("control_hz"),
            camera=CameraConfig(**camera_data) if camera_data is not None else None,
            connection=ConnectionConfig(**connection_data) if connection_data is not None else None,
            local_micro_move=LocalMicroMoveSettings(**obj["local_micro_move"])
            if obj.get("local_micro_move") is not None
            else None,
            cameras=tuple(PhysicalCameraConfig(**item) for item in cameras_data),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ConfigurationError(f"invalid configuration {path}: {exc}") from exc
