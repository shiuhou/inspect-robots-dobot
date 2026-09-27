"""Local envelope/terminal tests use bytes in memory; robot access is globally denied."""

import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest
from fake_sockets import ScriptedFactory
from live_fakes import MotionPeer, connection, movements, readiness

from inspect_robots_dobot.config import DobotConfig, load_config
from inspect_robots_dobot.driver import FakeDobotDriver
from inspect_robots_dobot.embodiment import DobotEmbodiment
from inspect_robots_dobot.errors import ConfigurationError, MotionNotAuthorized, SafetyRejected
from inspect_robots_dobot.live_authority import plan_digest
from inspect_robots_dobot.live_driver import LiveDobotMotionDriver, LiveMotionState
from inspect_robots_dobot.live_profile import load_live_profile
from inspect_robots_dobot.live_trial import run_terminal_trial, validate_live_inputs
from inspect_robots_dobot.motion import build_micro_move_plan, build_motion_plan
from inspect_robots_dobot.preflight import run_readonly_preflight
from inspect_robots_dobot.types import NativeDecimalPose, RobotMode, RobotSnapshot
from inspect_robots_dobot.units import from_native

ROOT = Path(__file__).parents[1]


@pytest.fixture
def local(clock, monkeypatch, tmp_path):
    config = load_config(ROOT / "examples/live_profile.json")
    profile = load_live_profile(ROOT / "examples/live_monitor.json", config.local_micro_move)
    # EXCLUSIVELY mock gates. Never persist this invented firmware in production files.
    profile = replace(profile, firmware="fixture-fw")
    config = replace(config, connection=connection(controller_firmware="fixture-fw"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    # Deliberately outside old fixture workspace; not inferred from neutral joint constants.
    raw = NativeDecimalPose(
        ("-190.1234", "-100.25", "375.55", "166.123456789123", "-27.2", "-171.4")
    )
    start = RobotSnapshot(
        from_native(raw.native),
        (0.0,) * 6,
        RobotMode.ENABLED_IDLE,
        (),
        42,
        clock.monotonic(),
        0,
        0,
        native_decimal=raw,
    )

    class ExactPeer(MotionPeer):
        def sendall(self, data):
            super().sendall(data)
            if data.startswith(b"GetPose(") and self.current.native_decimal:
                self.chunks.pop()
                self.chunks.append(
                    b"0,{"
                    + ",".join(self.current.native_decimal.values).encode()
                    + b"},"
                    + data
                    + b";"
                )

    peer = ExactPeer(start, clock)
    native_target = raw.translated_10mm("+Z")
    peer.execution.append(
        replace(
            start,
            pose=from_native(native_target.native),
            native_decimal=native_target,
            command_id=43,
        )
    )
    factory = ScriptedFactory(dashboard=peer)
    answers = iter(["operator", "CONFIRM", "+Z", "MOVE ONCE"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    return config, profile, peer, factory, tmp_path / "attempt", start


def prepared(local, clock):
    config, profile, peer, factory, _, _ = local
    driver = LiveDobotMotionDriver(
        config.connection,
        profile,
        allow_motion=True,
        socket_factory=factory,
        clock=clock,
        transport_is_mock=True,
    )
    driver.connect(allow_connection=True)
    driver.confirm_readiness(readiness())
    start = driver.measure_start()
    envelope = driver.validation_safety.envelope
    plan = build_micro_move_plan(
        envelope.chunk_id,
        start,
        "+Z",
        driver.validation_safety,
        clock.monotonic(),
    )
    return driver, peer, plan


def run(local, clock):
    config, profile, _, factory, directory, _ = local
    return run_terminal_trial(
        config,
        profile,
        directory,
        allow_motion=True,
        _socket_factory=factory,
        _clock=clock,
    )


def test_actual_files_keep_unknown_firmware_truthful_and_allow_local_exception(local, clock):
    config = load_config(ROOT / "examples/live_profile.json")
    profile = load_live_profile(ROOT / "examples/live_monitor.json", config.local_micro_move)
    assert config.safety is None
    assert config.connection.controller_firmware is None and profile.firmware is None
    assert not config.connection.protocol_compatibility_confirmed
    assert config.connection.tcp_control_owned is None
    assert "PROVISIONAL" in profile.payload_description
    assert "NOT calibrated" in profile.tool_tcp_description
    assert run_readonly_preflight(config)["motion_ready"] is False
    # Unknown firmware remains truthful. The explicit exception is limited to
    # this measured, standalone local micro-move profile.
    validate_live_inputs(config, profile, allow=True)


def test_local_terminal_e2e_review_then_exact_one_move(local, clock, monkeypatch):
    directory = local[4]
    original = __import__("builtins").input

    def answer(prompt):
        if "exactly +Z" in prompt:
            preflight = json.loads((directory / "preflight.json").read_text())
            assert preflight["initial_measurement"]
            assert preflight["bound_profile"]["safety"]["envelope"]
        if "MOVE ONCE" in prompt:
            review = json.loads((directory / "reviewed_plan.json").read_text())
            assert review["local_envelope"]["session_nonce"]
            assert not review["authority_issued"] and not movements(local[2])
        return original(prompt)

    monkeypatch.setattr("builtins.input", answer)
    report = run(local, clock)
    assert report["status"] == "MOCK_OK", report
    assert report["hardware_validation"] == "NOT RUN" and not report["PHYSICAL_SEND"]
    assert report["connections_closed"] and report["evidence_complete"]
    review = json.loads((directory / "reviewed_plan.json").read_text())
    assert review["native_target"]["native_decimal"]["values"] == [
        "-190.1234",
        "-100.25",
        "385.55",
        "166.123456789123",
        "-27.2",
        "-171.4",
    ]
    assert movements(local[2]) == [review["MovL"].encode()]
    assert len(local[3].calls) == 1
    assert all(x.startswith((b"Get", b"RobotMode(", b"MovL(")) for x in local[2].sent)
    assert review["local_envelope"]["expires_at"] == review["authority_expires_at_monotonic"]
    residuals = json.loads((directory / "settle_samples.json").read_text())
    assert [x for x in residuals if x["kind"] == "residual"][-1]["consecutive"] == 5


def test_geometry_is_local_not_table_or_reusable_workspace(local, clock):
    driver, _, plan = prepared(local, clock)
    e = plan.local_envelope
    s = driver.validation_safety
    x, y, z = plan.starting_measured_state.pose.xyz
    assert e.low == (x - 0.0005, y - 0.0005, z - 0.0005)
    assert e.high == (x + 0.0005, y + 0.0005, plan.final_pose_si.z + 0.0005)
    assert s.minimum_tcp_z == e.low[2]
    assert plan.final_agent_pose[3:] == (0, 0, 0, 0)
    record = next(x for x in driver.audit_records if x["kind"] == "local_envelope_bound")
    assert record["table_z_known"] is False and record["general_workspace"] is False
    with pytest.raises(FrozenInstanceError):
        e.chunk_id = "changed"
    with pytest.raises(MotionNotAuthorized, match="regenerated"):
        driver.measure_start()
    driver.close()


@pytest.mark.parametrize("direction", ["+X", "-X", "+Y", "-Y", "-Z", ""])
def test_other_direction_cannot_produce_plan_or_command(local, clock, monkeypatch, direction):
    answers = iter(["operator", "CONFIRM", direction, "MOVE ONCE"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    result = run(local, clock)
    assert result["status"] == "BLOCKED" and not movements(local[2])
    assert local[2].closed


@pytest.mark.parametrize(
    "field,value",
    [
        ("direction", "-Z"),
        ("distance_m", 0.02),
        ("distance_m", 0.009),
        ("measurement_margin_m", 0.002),
        ("measurement_margin_m", 0.0001),
        ("speed_percent", 6),
        ("acceleration_percent", 6),
        ("tool_frame", 1),
        ("user_frame", 1),
        ("threshold_class", "CERTIFIED"),
    ],
)
def test_settings_cannot_widen_reviewed_scope(local, field, value):
    with pytest.raises(ConfigurationError):
        replace(local[0].local_micro_move, **{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "measurement_margin_m",
        "position_tolerance",
        "orientation_tolerance",
        "settle_timeout",
        "telemetry_max_age",
        "poll_interval",
        "max_orientation_step",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1, True])
def test_local_numeric_validation(local, field, value):
    with pytest.raises(ConfigurationError):
        replace(local[0].local_micro_move, **{field: value})


def test_envelope_is_unusable_by_normal_framework_and_fake(local, clock):
    with pytest.raises(ConfigurationError, match="not a Robocurve"):
        DobotEmbodiment(local[0])
    driver, _, _ = prepared(local, clock)
    with pytest.raises(ConfigurationError, match="general config"):
        DobotConfig(safety=driver.validation_safety, control_hz=10)
    with pytest.raises(ValueError, match="session-local"):
        FakeDobotDriver(
            initial_pose=local[5].pose,
            initial_joints=(0.0,) * 6,
            profile=driver.validation_safety,
            clock=clock,
        )
    driver.close()


@pytest.mark.parametrize("change", ["target", "start", "chunk", "expiry", "removed"])
def test_plan_tampering_rejected_without_send(local, clock, change):
    driver, peer, plan = prepared(local, clock)
    auth = driver.arm(plan, confirmation="MOVE ONCE")
    e = plan.local_envelope
    modified = {
        "target": lambda: replace(plan, final_agent_pose=(*plan.final_pose_si.xyz, 0, 0, 0, 1)),
        "start": lambda: replace(plan, starting_measured_state=replace(e.start, command_id=99)),
        "chunk": lambda: replace(plan, chunk_id="another"),
        "expiry": lambda: replace(plan, local_envelope=replace(e, expires_at=e.expires_at - 1)),
        "removed": lambda: replace(plan, local_envelope=None),
    }[change]()
    assert plan_digest(plan, driver.profile, driver.connection) != plan_digest(
        modified,
        driver.profile,
        driver.connection,
    )
    result = driver.execute(modified, auth)
    assert result.state is LiveMotionState.REJECTED_BEFORE_SEND
    assert not movements(peer)


def test_bound_envelope_cannot_import_or_rebind_other_session(local, clock):
    driver, _, plan = prepared(local, clock)
    with pytest.raises(ConfigurationError, match="another live session"):
        LiveDobotMotionDriver(local[0].connection, driver.profile, socket_factory=local[3])
    second = LiveDobotMotionDriver(
        local[0].connection,
        local[1],
        allow_motion=True,
        socket_factory=local[3],
        clock=clock,
    )
    second.connect(allow_connection=True)
    second.confirm_readiness(readiness())
    second.measure_start()
    with pytest.raises(SafetyRejected, match="does not belong"):
        second.arm(plan, confirmation="MOVE ONCE")
    second.close()
    driver.close()


@pytest.mark.parametrize("stage", ["build", "arm", "send"])
def test_expiry_cannot_extend_or_regenerate(local, clock, stage):
    driver, peer, plan = prepared(local, clock)
    auth = driver.arm(plan, confirmation="MOVE ONCE") if stage == "send" else None
    clock.sleep(31)
    if stage == "build":
        with pytest.raises(SafetyRejected):
            build_micro_move_plan(
                plan.chunk_id,
                plan.starting_measured_state,
                "+Z",
                driver.validation_safety,
                clock.monotonic(),
            )
    elif stage == "arm":
        with pytest.raises(SafetyRejected, match="expiry"):
            driver.arm(plan, confirmation="MOVE ONCE")
    else:
        assert driver.execute(plan, auth).state is LiveMotionState.REJECTED_BEFORE_SEND
    assert not movements(peer)
    driver.close()


def test_authority_cannot_extend_envelope_expiry(local, clock):
    driver, peer, plan = prepared(local, clock)
    clock.sleep(0.1)
    with pytest.raises(MotionNotAuthorized, match="expiry cannot"):
        driver.arm(plan, confirmation="MOVE ONCE", expires_at=clock.monotonic() + 30)
    assert not movements(peer)
    driver.close()


@pytest.mark.parametrize("change", ["geometry", "budget", "speed", "tolerance"])
def test_projection_cannot_widen(local, clock, change):
    driver, _, _ = prepared(local, clock)
    kwargs = {
        "geometry": {"workspace_low": (-10, -10, -10), "minimum_tcp_z": -10},
        "budget": {"max_translation_step": 0.02},
        "speed": {"speed_percent": 6},
        "tolerance": {"position_tolerance": 0.0003},
    }[change]
    with pytest.raises(ConfigurationError):
        replace(driver.validation_safety, **kwargs)
    driver.close()


def test_step_plan_cannot_reuse_local_projection(local, clock):
    driver, _, plan = prepared(local, clock)
    with pytest.raises(SafetyRejected, match="bound to one start"):
        build_motion_plan(
            plan.chunk_id,
            plan.staged_waypoints,
            plan.starting_measured_state,
            plan.starting_measured_state.pose,
            driver.validation_safety,
            clock.monotonic(),
        )
    driver.close()


def test_lateral_excursion_during_settle_aborts(local, clock):
    config, profile, peer, _, _, start = local
    outside = replace(start.native_decimal, values=("-189.1234", *start.native_decimal.values[1:]))
    peer.execution.clear()
    peer.execution.append(replace(start, pose=from_native(outside.native), native_decimal=outside))
    result = run(local, clock)
    assert result["status"] == "FAILED"
    assert len(movements(peer)) == 1 and peer.sent.count(b"Stop()") == 1 and peer.closed


@pytest.mark.parametrize("fault", ["lost", "malformed", "write_failure", "rejected"])
def test_local_faults_never_resend(local, clock, fault):
    local[2].move_reply = fault
    report = run(local, clock)
    assert report["status"] == "FAILED" and len(movements(local[2])) == 1
    assert local[2].sent.count(b"Stop()") <= 1 and local[2].closed
    if fault == "write_failure":
        assert b"Stop()" not in local[2].sent


@pytest.mark.parametrize("gate", ["allow", "stdin", "stdout"])
def test_local_existing_preconnection_gates(local, clock, monkeypatch, gate):
    config, profile, peer, factory, directory, _ = local
    if gate in ("stdin", "stdout"):
        monkeypatch.setattr(f"sys.{gate}.isatty", lambda: False)
    if gate == "ownership":
        config = replace(config, connection=replace(config.connection, tcp_control_owned=None))
    if gate == "compatibility":
        config = replace(
            config, connection=replace(config.connection, protocol_compatibility_confirmed=False)
        )
    with pytest.raises((ConfigurationError, MotionNotAuthorized)):
        run_terminal_trial(
            config,
            profile,
            directory,
            allow_motion=gate != "allow",
            _socket_factory=factory,
            _clock=clock,
        )
    assert not factory.calls and not peer.sent


def test_local_config_cannot_mix_full_workspace_or_camera(local, profile):
    with pytest.raises(ConfigurationError):
        replace(local[0], safety=profile)
    with pytest.raises(ConfigurationError):
        replace(local[0], control_hz=10)
    assert isinstance(asdict(local[0])["local_micro_move"], dict)
    assert isinstance(local[0].local_micro_move.measurement_margin_m, float)


def test_honest_files_load_via_actual_cli_without_network(local, monkeypatch, capsys):
    from inspect_robots_dobot.micro_move import main

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("test stops before any live input or socket")

    monkeypatch.setattr("builtins.input", forbidden)
    assert (
        main(
            [
                "--live",
                "--allow-motion",
                "--config",
                str(ROOT / "examples/live_profile.json"),
                "--profile",
                str(ROOT / "examples/live_monitor.json"),
                "--evidence-dir",
                str(local[4]),
            ]
        )
        == 2
    )
    output = capsys.readouterr().out
    assert "firmware" in output
    assert "PHYSICAL_SEND" in output


def test_human_review_can_take_seconds_but_target_never_regenerates(local, clock, monkeypatch):
    original = __import__("builtins").input

    def answer(prompt):
        if "exactly +Z" in prompt or "MOVE ONCE" in prompt:
            clock.sleep(3.0)  # Longer than telemetry age; fresh start recheck at arm/send.
        return original(prompt)

    monkeypatch.setattr("builtins.input", answer)
    report = run(local, clock)
    assert report["status"] == "MOCK_OK", report
    events = report["audit"]
    assert sum(e["kind"] == "local_envelope_bound" for e in events) == 1
    assert len(movements(local[2])) == 1


@pytest.mark.parametrize("prompt_fragment", ["exactly +Z", "MOVE ONCE"])
def test_expired_human_review_stops_attempt(local, clock, monkeypatch, prompt_fragment):
    original = __import__("builtins").input

    def answer(prompt):
        if prompt_fragment in prompt:
            clock.sleep(31)
        return original(prompt)

    monkeypatch.setattr("builtins.input", answer)
    report = run(local, clock)
    assert report["status"] == "BLOCKED"
    assert not movements(local[2]) and local[2].closed
    assert sum(e["kind"] == "local_envelope_bound" for e in report["audit"]) == 1


@pytest.mark.parametrize("axis,delta", [(0, 0.00001), (2, -0.00001), (2, 0.001)])
def test_start_drift_cannot_expand_10mm_budget_or_retarget(local, clock, axis, delta):
    from decimal import Decimal

    driver, peer, plan = prepared(local, clock)
    native = plan.starting_measured_state.native_decimal
    values = list(native.values)
    values[axis] = str(Decimal(values[axis]) + Decimal(str(delta)) * 1000)
    drifted = NativeDecimalPose(tuple(values))
    peer.start = replace(peer.start, pose=from_native(drifted.native), native_decimal=drifted)
    with pytest.raises(SafetyRejected):
        driver.arm(plan, confirmation="MOVE ONCE")
    assert not movements(peer)
    assert plan.request.native_decimal == native.translated_10mm("+Z")
    driver.close()


@pytest.mark.parametrize("method", ["reset", "close"])
def test_local_authority_cannot_survive_close_or_reset(local, clock, method):
    driver, peer, plan = prepared(local, clock)
    auth = driver.arm(plan, confirmation="MOVE ONCE")
    if method == "reset":
        with pytest.raises(MotionNotAuthorized):
            driver.reset()
    else:
        driver.close()
    with pytest.raises(MotionNotAuthorized):
        driver.execute(plan, auth)
    assert not movements(peer) and peer.closed


def test_local_success_cannot_repeat_authority_or_connect(local, clock):
    driver, peer, plan = prepared(local, clock)
    auth = driver.arm(plan, confirmation="MOVE ONCE")
    assert driver.execute(plan, auth).state is LiveMotionState.COMPLETED
    with pytest.raises(MotionNotAuthorized):
        driver.execute(plan, auth)
    with pytest.raises(MotionNotAuthorized):
        driver.connect(allow_connection=True)
    assert len(movements(peer)) == 1


@pytest.mark.parametrize("failure", ["running", "alarm", "slow", "nan"])
def test_first_measurement_must_be_valid_before_binding(local, clock, failure):
    peer = local[2]
    if failure == "running":
        peer.start = replace(peer.start, mode=RobotMode.RUNNING)
    elif failure == "alarm":
        peer.start = replace(peer.start, errors=(1,))
    elif failure == "slow":
        peer.query_delay = 0.1
    else:
        peer.start = replace(
            peer.start, pose=replace(peer.start.pose, x=float("nan")), native_decimal=None
        )
    result = run(local, clock)
    assert result["status"] == "BLOCKED"
    assert not movements(peer) and peer.closed
    assert not any(e["kind"] == "local_envelope_bound" for e in result["audit"])


def test_existing_keepout_validation_remains_active(local, clock):
    from inspect_robots_dobot.motion import KeepoutBox

    x, y, z = local[5].pose.xyz
    box = KeepoutBox((x - 0.01, y - 0.01, z + 0.004), (x + 0.01, y + 0.01, z + 0.006))
    changed = (*local[:1], replace(local[1], keepouts=(box,)), *local[2:])
    result = run(changed, clock)
    assert result["status"] == "BLOCKED" and not movements(local[2])
    assert "keepout" in result["reason"]
