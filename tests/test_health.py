import json
from dataclasses import asdict, replace

import pytest
from fake_sockets import ScriptedFactory, ScriptedSocket, dashboard_socket, feedback_packet

from inspect_robots_dobot import health
from inspect_robots_dobot.camera import FakeCamera
from inspect_robots_dobot.config import CameraConfig, ConnectionConfig, DobotConfig, load_config
from inspect_robots_dobot.dashboard import DobotDashboardClient
from inspect_robots_dobot.feedback_client import DobotFeedbackClient
from inspect_robots_dobot.health import run_health
from inspect_robots_dobot.preflight import run_preflight


def config_for(*, confirmed=True, **kwargs):
    return DobotConfig(
        connection=ConnectionConfig(
            host="192.0.2.1",
            controller_firmware="synthetic-fixture" if confirmed else None,
            protocol_compatibility_confirmed=confirmed,
            **kwargs,
        )
    )


def full_factory(**kwargs):
    return ScriptedFactory(dashboard_socket(**kwargs), ScriptedSocket([feedback_packet()]))


def assert_query_only(factory):
    for port, stream in factory.streams.items():
        if stream is None or isinstance(stream, BaseException):
            continue
        if port == 30004:
            assert stream.sent == []
        else:
            assert all(
                command
                in {
                    b"RobotMode()",
                    b"GetPose()",
                    b"GetPose(user=0,tool=2)",
                    b"GetAngle()",
                    b"GetErrorID()",
                    b"GetCurrentCommandID()",
                }
                for command in stream.sent
            )
        assert stream.closed


def test_full_mocked_success_does_not_certify_motion(clock):
    factory = full_factory()
    report = run_health(config_for(), allow_read_only=True, socket_factory=factory, clock=clock)
    assert report["status"] == "READ_ONLY_OK"
    assert report["ok"]
    assert report["tcp_dashboard_reachable"] and report["tcp_feedback_reachable"]
    assert report["network_reachable"] and report["connections_closed"]
    assert report["robot_mode"] == 5 and report["pose"][:3] == [0.3, -0.1, 0.2]
    assert report["joints"][0] == 0 and report["active_errors"] == []
    assert report["feedback_packet_valid"] and report["feedback_age"] == 0
    assert report["feedback_raw"]["controller_unix_ms"] == 100000
    assert report["camera_status"] == "NOT_CONFIGURED"
    assert report["pose_frames"]["source"] == "controller_global_unknown"
    assert not report["motion_ready"] and not report["motion_profile_configured"]
    assert not report["request_control_attempted"] and report["motion_commands_sent"] == 0
    assert report["firmware_source"] == "operator_configuration"
    assert not report["firmware_detected"] and not report["protocol_compatibility_detected"]
    assert report["dashboard_queries_sent"] == 7
    assert len(factory.calls) == 2
    json.dumps(report, allow_nan=False)
    assert_query_only(factory)


@pytest.mark.parametrize("available", ["dashboard", "feedback"])
def test_independent_port_diagnostics(clock, available):
    factory = ScriptedFactory(
        dashboard=dashboard_socket() if available == "dashboard" else ConnectionRefusedError(),
        feedback=ScriptedSocket([feedback_packet()]) if available == "feedback" else TimeoutError(),
    )
    report = run_health(config_for(), allow_read_only=True, socket_factory=factory, clock=clock)
    assert report["status"] == "PARTIAL"
    assert report["tcp_dashboard_reachable"] == (available == "dashboard")
    assert report["tcp_feedback_reachable"] == (available == "feedback")
    assert len(factory.calls) == 2
    assert_query_only(factory)


def test_unknown_firmware_is_visible_and_cannot_become_compatible_from_shape(clock):
    factory = full_factory()
    report = run_health(
        config_for(confirmed=False), allow_read_only=True, socket_factory=factory, clock=clock
    )
    assert report["status"] == "PARTIAL" and report["feedback_packet_valid"]
    assert report["controller_firmware"] is None
    assert report["protocol_compatibility"] == "UNVERIFIED"
    assert not report["motion_ready"]
    assert_query_only(factory)


def test_declared_unowned_reports_limitation_and_still_reads_feedback(clock):
    factory = full_factory()
    report = run_health(
        config_for(tcp_control_owned=False),
        allow_read_only=True,
        socket_factory=factory,
        clock=clock,
    )
    assert report["status"] == "BLOCKED" and report["tcp_dashboard_reachable"]
    assert all(q["status"] == "CONTROL_NOT_OWNED" for q in report["queries"].values())
    assert report["feedback_packet_valid"]
    assert factory.streams[29999].sent == []
    assert not report["request_control_attempted"]
    assert_query_only(factory)


def test_generic_controller_error_is_not_fabricated_ownership_detection(clock):
    factory = full_factory()
    factory.streams[29999].responses[b"GetPose()"] = [b"-1,{},GetPose();"]
    report = run_health(config_for(), allow_read_only=True, socket_factory=factory, clock=clock)
    assert report["status"] == "PARTIAL"
    assert report["queries"]["pose"]["status"] == "CONTROLLER_ERROR"
    assert report["queries"]["pose"]["error_id"] == -1
    assert report["queries"]["joints"]["status"] == "OK"
    assert report["tcp_control_owned_declared"] is None
    assert not report["tcp_control_ownership_verified"]
    assert factory.streams[29999].sent.count(b"GetPose()") == 1
    assert_query_only(factory)


@pytest.mark.parametrize(
    "mode,errors,status",
    [(9, b"[123]", "BLOCKED"), (3, b"[]", "PARTIAL"), (5, b"[123]", "BLOCKED")],
)
def test_controller_state_and_alarms_never_trigger_recovery(clock, mode, errors, status):
    factory = full_factory(mode=mode, errors=errors)
    report = run_health(config_for(), allow_read_only=True, socket_factory=factory, clock=clock)
    assert report["status"] == status
    if mode in (3, 9):
        assert report["queries"]["pose"]["status"] == "QUERY_UNAVAILABLE"
        assert b"GetPose()" not in factory.streams[29999].sent
        assert b"GetAngle()" not in factory.streams[29999].sent
    assert_query_only(factory)


@pytest.mark.parametrize("side", ["dashboard", "feedback"])
def test_protocol_error_is_distinct_from_reachability(clock, side):
    factory = full_factory()
    if side == "dashboard":
        factory.streams[29999].responses[b"RobotMode()"] = [b"bad;"]
    else:
        factory.streams[30004] = ScriptedSocket([b"xx"])
    report = run_health(config_for(), allow_read_only=True, socket_factory=factory, clock=clock)
    assert report["status"] == "ERROR"
    assert report["tcp_dashboard_reachable"] and report["tcp_feedback_reachable"]
    assert_query_only(factory)


def test_both_ports_unreachable(clock):
    factory = ScriptedFactory()
    report = run_health(config_for(), allow_read_only=True, socket_factory=factory, clock=clock)
    assert report["status"] == "ERROR" and not report["network_reachable"]
    assert report["pose"] is None and report["feedback_packet_valid"] is None


@pytest.mark.parametrize(
    "config,allowed",
    [
        (None, False),
        (config_for(), False),
        (DobotConfig(connection=ConnectionConfig()), True),
        (config_for(expected_protocol_version="4.5"), True),
    ],
)
def test_missing_authority_host_or_supported_version_prevents_all_io(clock, config, allowed):
    factory = full_factory()
    report = run_health(config, allow_read_only=allowed, socket_factory=factory, clock=clock)
    assert report["status"] == "BLOCKED"
    assert factory.calls == []
    assert not report["hardware_connected"]


def test_camera_reports_freshness_independently(clock):
    settings = CameraConfig(8, 6, 0.2, 0.1)
    config = replace(config_for(), camera=settings)
    camera = FakeCamera(settings, clock)
    report = run_health(
        config, allow_read_only=True, socket_factory=full_factory(), clock=clock, camera=camera
    )
    assert report["status"] == "READ_ONLY_OK"
    assert report["camera_status"] == "FRESH" and report["camera_age"] <= settings.max_age
    assert not report["camera_exposure_time_verified"]


def test_configured_camera_without_backend_is_partial(clock):
    config = replace(config_for(), camera=CameraConfig(8, 6, 0.2, 0.1))
    report = run_health(config, allow_read_only=True, socket_factory=full_factory(), clock=clock)
    assert report["status"] == "PARTIAL" and report["camera_status"] == "BACKEND_UNAVAILABLE"


def test_camera_wait_can_make_feedback_stale(clock):
    settings = CameraConfig(8, 6, 0.2, 0.1)
    config = replace(config_for(), camera=settings)

    class SlowCamera(FakeCamera):
        def latest(self, *, after=None):
            clock.sleep(0.3)
            return super().latest(after=after)

    report = run_health(
        config,
        allow_read_only=True,
        socket_factory=full_factory(),
        clock=clock,
        camera=SlowCamera(settings, clock),
    )
    assert report["camera_status"] == "FRESH"
    assert report["feedback_packet_valid"] and not report["feedback_fresh"]
    assert report["feedback_age"] == pytest.approx(0.3) and report["status"] == "PARTIAL"


def test_camera_error_closes_resources(clock):
    settings = CameraConfig(8, 6, 0.2, 0.1)
    camera = FakeCamera(settings, clock)
    camera.failure = "injected camera error"
    factory = full_factory()
    report = run_health(
        replace(config_for(), camera=settings),
        allow_read_only=True,
        socket_factory=factory,
        clock=clock,
        camera=camera,
    )
    assert report["status"] == "ERROR" and report["camera_status"] == "ERROR"
    assert_query_only(factory)


def test_interrupt_closes_sockets_without_control_command(clock):
    factory = full_factory()
    factory.streams[30004] = ScriptedSocket([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        run_health(config_for(), allow_read_only=True, socket_factory=factory, clock=clock)
    assert_query_only(factory)


def test_readonly_json_needs_no_motion_profile_and_preflight_stays_offline(tmp_path, monkeypatch):
    path = tmp_path / "readonly.json"
    path.write_text(json.dumps({"connection": asdict(config_for().connection)}))
    config = load_config(path)
    assert config.safety is None and config.control_hz is None

    def forbid(*args, **kwargs):
        raise AssertionError("preflight cannot connect")

    monkeypatch.setattr(DobotDashboardClient, "connect", forbid)
    monkeypatch.setattr(DobotFeedbackClient, "connect", forbid)
    report = run_preflight(config)
    assert not report["ok"] and not report["hardware_connected"]


def test_health_cli_runs_with_injected_streams_only(tmp_path, monkeypatch, capsys, clock):
    path = tmp_path / "readonly.json"
    path.write_text(json.dumps({"connection": asdict(config_for().connection)}))
    factory = full_factory()
    original = health.run_health

    def injected(config, *, allow_read_only):
        return original(
            config, allow_read_only=allow_read_only, socket_factory=factory, clock=clock
        )

    monkeypatch.setattr(health, "run_health", injected)
    assert health.main(["--config", str(path), "--host", "192.0.2.2", "--read-only", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["host"] == "192.0.2.2" and report["status"] == "READ_ONLY_OK"
    assert_query_only(factory)


def test_cli_invalid_config_and_motion_flags_do_not_connect(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"connection": {"motion_enabled": true}}')
    assert health.main(["--config", str(path), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["errors"]
    with pytest.raises(SystemExit) as caught:
        health.main(["--allow-motion"])
    assert caught.value.code == 2


def test_dashboard_feedback_mode_difference_is_partial(clock):
    factory = full_factory(mode=4)
    report = run_health(config_for(), allow_read_only=True, socket_factory=factory, clock=clock)
    assert report["robot_mode"] == 4 and report["feedback_raw"]["robot_mode"] == 5
    assert not report["dashboard_feedback_mode_consistent"]
    assert report["status"] == "PARTIAL"


def test_close_errors_are_visible_and_other_socket_still_closes(clock):
    factory = full_factory()

    def failed_close():
        raise OSError("injected close failure")

    factory.streams[29999].close = failed_close
    report = run_health(config_for(), allow_read_only=True, socket_factory=factory, clock=clock)
    assert report["status"] == "ERROR"
    assert not report["connections_closed"]
    assert factory.streams[30004].closed
    assert any("close" in message for message in report["errors"])
