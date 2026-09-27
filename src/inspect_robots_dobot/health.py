"""Explicit read-only diagnostics. No ownership changes, motion, or implicit recovery."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .camera import CameraReader, validate_frame
from .clock import Clock, SystemClock
from .config import ConnectionConfig, DobotConfig, load_config
from .dashboard import DobotDashboardClient
from .errors import (
    CameraFault,
    CommandRejected,
    ConfigurationError,
    ControlNotOwned,
    DriverFault,
    ProtocolError,
    QueryUnavailable,
    TransportError,
    TransportTimeout,
)
from .feedback import PACKET_SIZE, TEST_VALUE, RawFeedback
from .feedback_client import DobotFeedbackClient
from .readonly_driver import ReadOnlyDobotDriver
from .transport import SocketFactory, open_socket
from .types import RobotMode


def _failure(exc: DriverFault) -> dict[str, Any]:
    if isinstance(exc, ControlNotOwned):
        kind = "CONTROL_NOT_OWNED"
    elif isinstance(exc, CommandRejected):
        return {"status": "CONTROLLER_ERROR", "reason": str(exc), "error_id": exc.error_id}
    elif isinstance(exc, ProtocolError):
        kind = "PROTOCOL_ERROR"
    elif isinstance(exc, TransportTimeout):
        kind = "TIMEOUT"
    elif isinstance(exc, TransportError):
        kind = "TRANSPORT_ERROR"
    elif isinstance(exc, QueryUnavailable):
        kind = "QUERY_UNAVAILABLE"
    else:
        kind = "ERROR"
    return {"status": kind, "reason": str(exc)}


def _base_report(config: DobotConfig) -> dict[str, Any]:
    c = config.connection or ConnectionConfig()
    return {
        "phase": "2A",
        "mode": "READ ONLY — NO MOTION COMMANDS SENT",
        "status": "BLOCKED",
        "ok": False,
        "host": c.host,
        "expected_protocol_version": c.expected_protocol_version,
        "controller_firmware": c.controller_firmware,
        "firmware_source": "operator_configuration" if c.controller_firmware else "unknown",
        "firmware_detected": False,
        "protocol_compatibility": (
            "OPERATOR_CONFIRMED" if c.protocol_compatibility_confirmed else "UNVERIFIED"
        ),
        "protocol_compatibility_detected": False,
        "dashboard_port": c.dashboard_port,
        "feedback_port": c.feedback_port,
        "tcp_dashboard_reachable": False,
        "tcp_feedback_reachable": False,
        "dashboard_connection": {"status": "NOT_ATTEMPTED"},
        "feedback_status": {"status": "NOT_ATTEMPTED"},
        "dashboard_feedback_mode_consistent": None,
        "network_reachable": False,
        "hardware_connected": False,
        "connections_closed": True,
        "robot_mode": None,
        "pose": None,
        "joints": None,
        "active_errors": None,
        "command_id": None,
        "pose_units": "m,rad (native Dobot rx/ry/rz)",
        "joint_units": "rad",
        "pose_frames": {
            "user": c.user_frame,
            "tool": c.tool_frame,
            "source": "explicit_query" if c.user_frame is not None else "controller_global_unknown",
        },
        "tcp_control_owned_declared": c.tcp_control_owned,
        "tcp_control_ownership_verified": False,
        "request_control_attempted": False,
        "motion_commands_sent": 0,
        "motion_enabled": False,
        "motion_ready": False,
        "motion_profile_configured": config.safety is not None,
        "hardware_readiness_verified": False,
        "dashboard_queries_sent": 0,
        "dashboard_exchanges": [],
        "feedback_age": None,
        "feedback_packet_valid": None,
        "feedback_fresh": False,
        "feedback_raw": None,
        "feedback_wire_hex": "",
        "feedback_packet_size": None,
        "feedback_message_size_valid": None,
        "feedback_test_value_valid": None,
        "feedback_test_value": None,
        "feedback_coordinate_semantics": "RAW_ONLY; units/frame unverified; no SI conversion",
        "feedback_age_source": "host_monotonic_receive; controller generation age is unknown",
        "camera_status": "NOT_CONFIGURED" if config.camera is None else "BACKEND_UNAVAILABLE",
        "camera_age": None,
        "queries": {},
        "warnings": [
            "Read-only diagnostics are not motion authorization or safety certification.",
            "TCP mode is required by manual p8. No ownership-specific ErrorID is verified; "
            "generic rejection must not be interpreted as confirmed ownership loss.",
            "Host feedback receive age cannot prove controller generation freshness "
            "or empty buffers.",
        ],
        "errors": [],
    }


def run_health(
    config: DobotConfig | None = None,
    *,
    allow_read_only: bool = False,
    socket_factory: SocketFactory = open_socket,
    clock: Clock | None = None,
    camera: CameraReader | None = None,
) -> dict[str, Any]:
    """Only explicit allow_read_only=True plus a configured host can open sockets.

    Injected cameras are diagnostic sources; there is no physical camera backend.
    Every resource opened here is closed, including on interruption. No driver or
    socket is constructed by preflight. No command is retried.
    """
    config = config or DobotConfig()
    report = _base_report(config)
    connection = config.connection
    if allow_read_only is not True or connection is None or connection.host is None:
        report["errors"].append("explicit --read-only and a configured/CLI host are required")
        return report
    if connection.expected_protocol_version != "4.6.5":
        report["protocol_compatibility"] = "UNSUPPORTED_DOCUMENT"
        report["errors"].append(
            "only protocol document 4.6.5 is implemented; no connection attempted"
        )
        return report
    if not connection.protocol_compatibility_confirmed:
        report["warnings"].append(
            "Controller firmware/layout compatibility is unverified; "
            "matching packet shape is not proof."
        )
    clock = clock or SystemClock()
    dashboard = DobotDashboardClient(connection, socket_factory=socket_factory, clock=clock)
    feedback = DobotFeedbackClient(connection, socket_factory=socket_factory, clock=clock)
    driver = ReadOnlyDobotDriver(dashboard, feedback)
    sample: RawFeedback | None = None
    camera_started = False
    try:
        try:
            driver.connect()
            report["tcp_dashboard_reachable"] = True
            report["dashboard_connection"] = {"status": "CONNECTED"}
        except DriverFault as exc:
            report["dashboard_connection"] = _failure(exc)
        operations: tuple[tuple[str, Callable[[], Any]], ...] = (
            ("robot_mode", driver.robot_mode),
            ("pose", lambda: list(driver.get_pose().values)),
            ("joints", lambda: list(driver.get_joints())),
            ("active_errors", lambda: list(driver.get_errors())),
            ("command_id", dashboard.current_command_id),
        )
        for name, query in operations:
            if not report["tcp_dashboard_reachable"]:
                report["queries"][name] = {"status": "NOT_CONNECTED"}
                continue
            try:
                started = clock.monotonic()
                value = query()
                report[name] = int(value) if isinstance(value, RobotMode) else value
                report["queries"][name] = {
                    "status": "OK",
                    "started_at": started,
                    "received_at": clock.monotonic(),
                }
            except DriverFault as exc:
                report["queries"][name] = _failure(exc)
        try:
            feedback.connect()
            report["tcp_feedback_reachable"] = True
            sample = driver.read_feedback()
            report["feedback_packet_valid"] = True
            report["feedback_packet_size"] = PACKET_SIZE
            report["feedback_message_size_valid"] = True
            report["feedback_test_value_valid"] = True
            report["feedback_test_value"] = f"0x{TEST_VALUE:016X}"
            report["feedback_raw"] = asdict(sample)
            report["feedback_status"] = {"status": "OK"}
        except DriverFault as exc:
            report["feedback_status"] = _failure(exc)
            if isinstance(exc, ProtocolError):
                report["feedback_packet_valid"] = False
        if camera is not None and config.camera is not None:
            try:
                camera_started = True
                camera.start()
                frame = camera.latest()
                validate_frame(frame, config.camera, now=clock.monotonic(), after=None)
                report["camera_age"] = clock.monotonic() - frame.acquisition_started_at
                report["camera_status"] = "FRESH"
                report["camera_timestamp_source"] = frame.timestamp_source
                report["camera_exposure_time_verified"] = False
            except (CameraFault, OSError) as exc:
                report["camera_status"] = "ERROR"
                report["errors"].append(f"camera: {exc}")
        elif camera is not None:
            report["camera_status"] = "CONFIGURATION_MISSING"
            report["errors"].append("injected camera needs explicit camera configuration")
        if sample is not None:
            report["feedback_age"] = sample.age(clock.monotonic())
            report["feedback_fresh"] = sample.is_fresh(
                clock.monotonic(), connection.feedback_max_age
            )
            if not report["feedback_fresh"]:
                report["feedback_status"] = {
                    "status": "STALE",
                    "reason": "host receive age exceeded",
                }
    finally:
        report["dashboard_queries_sent"] = dashboard.queries_sent
        report["dashboard_exchanges"] = list(dashboard.exchanges)
        report["feedback_wire_hex"] = feedback.last_read_wire.hex()
        for client in (dashboard, feedback):
            try:
                client.close()
            except TransportError as exc:
                report["connections_closed"] = False
                report["errors"].append(str(exc))
        if camera_started and camera is not None:
            try:
                camera.close()
            except (CameraFault, OSError) as exc:
                report["errors"].append(f"camera close: {exc}")
    report["network_reachable"] = (
        report["tcp_dashboard_reachable"] or report["tcp_feedback_reachable"]
    )
    report["hardware_connected"] = report["network_reachable"]
    queries = report["queries"]
    all_queries_ok = all(q["status"] == "OK" for q in queries.values())
    modes_agree = sample is None or report["robot_mode"] in (None, sample.robot_mode)
    report["dashboard_feedback_mode_consistent"] = (
        None if sample is None or report["robot_mode"] is None else modes_agree
    )
    if not modes_agree:
        report["warnings"].append(
            "Dashboard and feedback RobotMode differ; samples are sequential, not atomic."
        )
    controller_fault = bool(report["active_errors"]) or report["robot_mode"] in (9, 11)
    if sample is not None:
        controller_fault = (
            controller_fault
            or sample.robot_mode in (9, 11)
            or bool(sample.error_status or sample.collision_state)
        )
    protocol_error = any(q["status"] == "PROTOCOL_ERROR" for q in queries.values()) or (
        report["feedback_packet_valid"] is False
    )
    if protocol_error or report["errors"]:
        report["status"] = "ERROR"
    elif controller_fault or connection.tcp_control_owned is False:
        report["status"] = "BLOCKED"
    elif (
        all_queries_ok
        and report["feedback_fresh"]
        and modes_agree
        and connection.protocol_compatibility_confirmed
        and report["camera_status"] in ("NOT_CONFIGURED", "FRESH")
    ):
        report["status"] = "READ_ONLY_OK"
    elif sample is not None or any(q["status"] == "OK" for q in queries.values()):
        report["status"] = "PARTIAL"
    else:
        report["status"] = "ERROR"
    report["ok"] = report["status"] == "READ_ONLY_OK"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host", help="explicit literal IP; never inferred from a default")
    parser.add_argument(
        "--read-only", action="store_true", help="explicitly allow read-only connection"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config) if args.config is not None else DobotConfig()
        if args.host is not None:
            config = replace(
                config, connection=replace(config.connection or ConnectionConfig(), host=args.host)
            )
        report = run_health(config, allow_read_only=args.read_only)
    except ConfigurationError as exc:
        report = _base_report(DobotConfig())
        report["errors"].append(str(exc))
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(report["mode"])
        print(report["status"])
        print(json.dumps(report, indent=2, allow_nan=False))
    return {"READ_ONLY_OK": 0, "PARTIAL": 1, "BLOCKED": 2, "ERROR": 3}[report["status"]]
