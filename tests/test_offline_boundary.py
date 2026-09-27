import importlib
import socket

import pytest
from pytest_socket import SocketBlockedError


def test_sockets_are_actually_blocked():
    with pytest.warns(UserWarning, match="socket.socket"), pytest.raises(SocketBlockedError):
        socket.socket()


def test_physical_camera_device_access_is_blocked():
    with pytest.raises(AssertionError, match="prohibited"), open("/dev/video0", "rb"):
        pass


def test_hardware_backend_import_is_blocked():
    with pytest.raises(AssertionError, match="prohibited"):
        __import__("cv2")


def test_package_imports_do_not_connect_or_require_devices():
    for name in (
        "protocol",
        "transport",
        "driver",
        "feedback",
        "camera",
        "embodiment",
        "preflight",
        "health",
    ):
        importlib.import_module(f"inspect_robots_dobot.{name}")
