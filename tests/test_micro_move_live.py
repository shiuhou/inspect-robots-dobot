"""Future live CLI exercised exclusively using an injected in-memory Dashboard peer."""

import json
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

import pytest
from fake_sockets import ScriptedFactory
from live_fakes import MotionPeer, connection, movements, production_profile

from inspect_robots_dobot.config import DobotConfig
from inspect_robots_dobot.errors import ConfigurationError, MotionNotAuthorized
from inspect_robots_dobot.evidence import AttemptEvidence
from inspect_robots_dobot.live_trial import run_terminal_trial
from inspect_robots_dobot.motion import build_micro_move_plan
from inspect_robots_dobot.types import NativeDecimalPose, RobotMode, RobotSnapshot
from inspect_robots_dobot.units import from_native


@pytest.fixture
def trial(profile, clock, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    # Fictional labels exclusively for injected tests, never used in a rig file.
    p = production_profile(
        profile,
        model="fixture-Nova",
        firmware="fixture-fw",
        rig_verification_reference="fixture-review",
        interruption_verification_reference="fixture-best-effort-model",
        tool_tcp_description="fixture-TCP",
        payload_description="fixture-payload",
    )
    c = connection(controller_firmware=p.firmware)
    raw = NativeDecimalPose(
        (
            "300.000",
            "0.000",
            "200.000",
            "12.123456789123456",
            "-23.098765432123456",
            "179.9999999999999",
        )
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
    target = build_micro_move_plan("fixture", start, "+X", profile, clock.monotonic())
    peer.execution.append(
        replace(
            start,
            pose=target.final_pose_si,
            native_decimal=target.request.native_decimal,
            command_id=43,
        )
    )
    factory = ScriptedFactory(dashboard=peer)
    answers = iter(["operator", "CONFIRM", "+X", "MOVE ONCE"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    return DobotConfig(safety=profile, connection=c), p, peer, factory, tmp_path / "attempt", start


def run(trial, clock, **kwargs):
    config, profile, _, factory, directory, _ = trial
    return run_terminal_trial(
        config,
        profile,
        directory,
        allow_motion=True,
        _socket_factory=factory,
        _clock=clock,
        **kwargs,
    )


def test_complete_terminal_trial_one_write_and_six_evidence_files(
    trial, clock, monkeypatch, capsys
):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    _, _, peer, _, directory, start = trial
    original_input = __import__("builtins").input

    def inspect_order(prompt):
        if "choose exactly" in prompt:
            assert "measured_start" in capsys.readouterr().out
            assert peer.sent and not movements(peer)
            assert json.loads((directory / "preflight.json").read_text())["initial_measurement"]
            assert (
                json.loads((directory / "preflight.json").read_text())["attestations"][
                    "aggregate_confirmation"
                ]
                is True
            )
        if "MOVE ONCE" in prompt:
            assert "PHYSICAL COMMAND COUNT = 1" in capsys.readouterr().out
            reviewed = json.loads((directory / "reviewed_plan.json").read_text())
            assert reviewed["authority_issued"] is False
            assert reviewed["direction"] == "+X"
            assert not movements(peer)
            assert "authority_issued" not in (directory / "raw_dashboard.log").read_text()
        return original_input(prompt)

    monkeypatch.setattr("builtins.input", inspect_order)
    report = run(trial, clock)
    assert report["status"] == "MOCK_OK", report
    assert report["hardware_validation"] == "NOT RUN" and not report["PHYSICAL_SEND"]
    assert report["evidence_complete"] and report["connections_closed"]
    assert len(movements(peer)) == 1
    assert set(p.name for p in directory.iterdir()) == set(AttemptEvidence.FILES)
    review = json.loads((directory / "reviewed_plan.json").read_text())
    assert movements(peer)[0].decode() == review["MovL"]
    assert review["native_target"]["native_decimal"]["values"][3:] == list(
        start.native_decimal.values[3:]
    )
    events = [
        json.loads(line) for line in (directory / "raw_dashboard.log").read_text().splitlines()
    ]
    assert [e["sequence"] for e in events] == list(range(len(events)))
    assert any(e.get("result_id") == 43 and e.get("error_id") == 0 for e in events)
    residuals = [
        e
        for e in json.loads((directory / "settle_samples.json").read_text())
        if e["kind"] == "residual"
    ]
    assert residuals[-1]["consecutive"] == 3
    assert all(
        data.startswith(
            (
                b"RobotMode(",
                b"GetPose(",
                b"GetAngle(",
                b"GetErrorID(",
                b"GetCurrentCommandID(",
                b"MovL(",
            )
        )
        for data in peer.sent
    )


@pytest.mark.parametrize("field", ["tool_tcp_description", "payload_description"])
@pytest.mark.parametrize("value", [None, "", " ", "UNKNOWN", "MOCK_ONLY"])
def test_missing_real_descriptions_block_before_connection(trial, clock, field, value):
    config, profile, peer, factory, directory, _ = trial
    with pytest.raises(ConfigurationError):
        run_terminal_trial(
            config,
            replace(profile, **{field: value}),
            directory,
            allow_motion=True,
            _socket_factory=factory,
            _clock=clock,
        )
    assert not peer.sent and not factory.calls


@pytest.mark.parametrize(
    "change",
    [
        dict(controller_firmware="wrong"),
        dict(protocol_compatibility_confirmed=False),
        dict(tcp_control_owned=None),
        dict(tool_frame=1),
        dict(expected_protocol_version="unknown"),
        dict(host=None),
    ],
)
def test_connection_inconsistency_blocks_before_factory(trial, clock, change):
    config, profile, peer, factory, directory, _ = trial
    with pytest.raises(ConfigurationError):
        run_terminal_trial(
            replace(config, connection=replace(config.connection, **change)),
            profile,
            directory,
            allow_motion=True,
            _socket_factory=factory,
            _clock=clock,
        )
    assert not peer.sent and not factory.calls


@pytest.mark.parametrize("gate", ["stdin", "stdout", "flag"])
def test_each_runtime_terminal_gate_blocks_before_factory(trial, clock, monkeypatch, gate):
    config, profile, peer, factory, directory, _ = trial
    if gate != "flag":
        monkeypatch.setattr(f"sys.{gate}.isatty", lambda: False)
    with pytest.raises(MotionNotAuthorized):
        run_terminal_trial(
            config,
            profile,
            directory,
            allow_motion=gate != "flag",
            _socket_factory=factory,
            _clock=clock,
        )
    assert not peer.sent and not factory.calls


@pytest.mark.parametrize("failed_answer", ["identity", "aggregate"])
def test_operator_identity_and_aggregate_attestation_required_before_connect(
    trial, clock, monkeypatch, failed_answer
):
    answers = ["operator", "CONFIRM", "+X", "MOVE ONCE"]
    answers[0 if failed_answer == "identity" else 1] = ""
    iterator = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _: next(iterator))
    report = run(trial, clock)
    assert report["status"] == "BLOCKED" and report["connections_closed"]
    assert not trial[3].calls and not trial[2].sent


@pytest.mark.parametrize(
    "which,value",
    [("direction", ""), ("direction", "up"), ("phrase", "yes"), ("phrase", "MOVE ONCE ")],
)
def test_wrong_direction_or_confirmation_ends_attempt_without_retry(
    trial, clock, monkeypatch, which, value
):
    answers = ["operator", "CONFIRM", "+X", "MOVE ONCE"]
    answers[2 if which == "direction" else 3] = value
    iterator = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _: next(iterator))
    report = run(trial, clock)
    assert report["status"] == "BLOCKED" and not movements(trial[2])
    assert len(trial[3].calls) == 1 and trial[2].closed


@pytest.mark.parametrize(
    "fault", ["lost", "malformed", "bad_id", "wrong_echo", "write_failure", "rejected", "keyboard"]
)
def test_one_attempt_failures_are_evidenced_and_never_retried(trial, clock, fault):
    trial[2].move_reply = fault
    report = run(trial, clock)
    assert report["status"] == "FAILED", report
    assert len(movements(trial[2])) == 1 and trial[2].closed
    assert report["result"]["authority_consumed"]
    assert report["motion_acknowledged"] == 0
    assert trial[2].sent.count(b"Stop()") <= 1
    if fault == "write_failure":
        assert b"Stop()" not in trial[2].sent
        assert report["motion_writes_completed"] == 0
    if fault in ("lost", "write_failure"):
        assert report["standstill"] == "UNKNOWN"
        assert report["result"]["acceptance_ambiguous"]


@pytest.mark.parametrize("shift", ["away", "sideways", "beyond_tolerance", "expiry"])
def test_post_review_shift_or_expiry_never_regenerates_plan(trial, clock, monkeypatch, shift):
    peer = trial[2]
    original = __import__("builtins").input

    def answer(prompt):
        if "MOVE ONCE" in prompt:
            if shift == "expiry":
                clock.sleep(2)
            else:
                xyz = {
                    "away": ("299.99", "0", "200"),
                    "sideways": ("300", "0.01", "200"),
                    "beyond_tolerance": ("301", "0", "200"),
                }[shift]
                raw = NativeDecimalPose((*xyz, *peer.start.native_decimal.values[3:]))
                peer.start = replace(peer.start, pose=from_native(raw.native), native_decimal=raw)
        return original(prompt)

    monkeypatch.setattr("builtins.input", answer)
    report = run(trial, clock)
    assert report["status"] == "BLOCKED" and not movements(peer)
    review = json.loads((trial[4] / "reviewed_plan.json").read_text())
    assert Decimal(review["native_target"]["native_decimal"]["values"][0]) == 310


def test_existing_attempt_is_never_overwritten(trial, clock):
    directory = trial[4]
    directory.mkdir()
    (directory / "final_report.md").write_text("historical evidence")
    with pytest.raises(ConfigurationError, match="before connection"):
        run(trial, clock)
    assert (directory / "final_report.md").read_text() == "historical evidence"
    assert not trial[3].calls


@pytest.mark.parametrize(
    "stage", ["review", "authority", "write_completed", "received_bytes", "residual", "final"]
)
def test_persistence_failure_never_yields_pass_and_does_not_disable_abort(
    trial, clock, monkeypatch, stage
):
    original_open = Path.open
    armed_failure = False
    original_record = AttemptEvidence.record

    def record(self, event):
        nonlocal armed_failure
        if (
            (stage == "authority" and event["kind"] == "authority_consumed")
            or (
                stage in ("write_completed", "received_bytes")
                and event.get("command", "").startswith("MovL(")
                and event.get("stage") == stage
            )
            or (stage == "residual" and event["kind"] == "residual")
        ):
            armed_failure = True
        original_record(self, event)

    def fail_open(path, *args, **kwargs):
        nonlocal armed_failure
        if stage == "review" and path.name == "reviewed_plan.json.pending":
            armed_failure = True
        if stage == "final" and path.name == "settle_samples.json.pending":
            armed_failure = True
        if armed_failure and path.parent == trial[4]:
            raise OSError("injected persistent disk failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(AttemptEvidence, "record", record)
    monkeypatch.setattr(Path, "open", fail_open)
    report = run(trial, clock)
    assert report["status"] in ("BLOCKED", "FAILED")
    assert not report["evidence_complete"]
    assert len(movements(trial[2])) <= 1 and trial[2].closed
    if stage in ("review", "authority"):
        assert not movements(trial[2]) and b"Stop()" not in trial[2].sent
    if stage in ("write_completed", "received_bytes", "residual"):
        assert trial[2].sent.count(b"Stop()") == 1


def test_terminal_cli_main_uses_mocked_live_factory(trial, clock, tmp_path, monkeypatch):
    from inspect_robots_dobot.micro_move import main

    config, profile, peer, factory, directory, _ = trial
    rig, monitor = tmp_path / "rig.json", tmp_path / "monitor.json"
    rig.write_text(json.dumps(asdict(config)))
    data = asdict(profile)
    del data["safety"]
    monitor.write_text(json.dumps(data))

    def injected(*args, **kwargs):
        return run_terminal_trial(*args, **kwargs, _socket_factory=factory, _clock=clock)

    monkeypatch.setattr("inspect_robots_dobot.live_trial.run_terminal_trial", injected)
    assert (
        main(
            [
                "--live",
                "--allow-motion",
                "--config",
                str(rig),
                "--profile",
                str(monitor),
                "--evidence-dir",
                str(directory),
            ]
        )
        == 0
    )
    assert len(movements(peer)) == 1 and peer.closed


def test_env_confirmation_cannot_replace_terminal_response(trial, clock, monkeypatch):
    monkeypatch.setenv("DOBOT_CONFIRMATION", "MOVE ONCE")
    monkeypatch.setenv("ALLOW_MOTION", "true")
    monkeypatch.setattr("builtins.input", lambda _: "")
    report = run(trial, clock)
    assert report["status"] == "BLOCKED" and not trial[3].calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", RobotMode.RUNNING),
        ("mode", RobotMode.ERROR),
        ("mode", RobotMode.COLLISION),
        ("errors", (123,)),
    ],
)
def test_precheck_fault_blocks_before_direction_prompt(trial, clock, monkeypatch, field, value):
    peer = trial[2]
    peer.start = replace(peer.start, **{field: value})
    original = __import__("builtins").input

    def answer(prompt):
        assert "choose exactly" not in prompt and "MOVE ONCE" not in prompt
        return original(prompt)

    monkeypatch.setattr("builtins.input", answer)
    report = run(trial, clock)
    assert report["status"] == "BLOCKED" and not movements(peer) and peer.closed


@pytest.mark.parametrize(
    "fault",
    [
        "no_motion",
        "collision",
        "controller_error",
        "wrong_id",
        "orientation_drift",
        "noisy_then_settle",
    ],
)
def test_live_trial_monitoring_cases(trial, clock, fault):
    peer = trial[2]
    target = peer.execution[-1]
    if fault == "no_motion":
        peer.execution.clear()
    elif fault == "collision":
        peer.execution[0] = replace(target, mode=RobotMode.COLLISION)
    elif fault == "controller_error":
        peer.execution[0] = replace(target, errors=(123,))
    elif fault == "wrong_id":
        peer.execution[0] = replace(target, command_id=99)
    elif fault == "orientation_drift":
        raw = NativeDecimalPose(
            (*target.native_decimal.values[:3], "20", *target.native_decimal.values[4:])
        )
        peer.execution[0] = replace(target, pose=from_native(raw.native), native_decimal=raw)
    else:
        peer.execution.extend(
            [
                replace(target, mode=RobotMode.RUNNING),
                target,
                replace(target, command_id=42),
                target,
                target,
                target,
            ]
        )
    report = run(trial, clock)
    assert len(movements(peer)) == 1 and peer.closed
    assert report["status"] == ("MOCK_OK" if fault == "noisy_then_settle" else "FAILED")
    if fault != "noisy_then_settle":
        assert peer.sent.count(b"Stop()") == 1


@pytest.mark.parametrize("reply", ["timeout", "rejected", "write_failure"])
def test_cli_stop_failure_is_unknown_standstill(trial, clock, reply):
    peer = trial[2]
    peer.execution.clear()
    peer.stop_reply = reply
    report = run(trial, clock)
    assert report["status"] == "FAILED" and report["standstill"] == "UNKNOWN"
    assert peer.sent.count(b"Stop()") == 1 and len(movements(peer)) == 1


@pytest.mark.parametrize("when", ["direction", "confirmation", "during_monitor"])
def test_ctrl_c_closes_and_never_retries(trial, clock, monkeypatch, when):
    peer = trial[2]
    original = __import__("builtins").input

    def answer(prompt):
        if (when == "direction" and "choose exactly" in prompt) or (
            when == "confirmation" and "MOVE ONCE" in prompt
        ):
            raise KeyboardInterrupt
        return original(prompt)

    monkeypatch.setattr("builtins.input", answer)
    if when == "during_monitor":

        def interrupt(data):
            if peer.moving and not peer.stopped and data == b"RobotMode()":
                raise KeyboardInterrupt

        peer.on_send = interrupt
    report = run(trial, clock)
    assert peer.closed and len(movements(peer)) == (1 if when == "during_monitor" else 0)
    assert report["status"] in ("FAILED", "BLOCKED")


def test_close_failure_cannot_be_hidden_by_finally_second_close(trial, clock):
    trial[2].close_error = OSError("close failure")
    report = run(trial, clock)
    assert report["status"] == "FAILED" and not report["connections_closed"]


def test_slow_evidence_io_cannot_send_with_expired_authority(trial, clock, monkeypatch):
    original = AttemptEvidence.record

    def slow(self, event):
        original(self, event)
        if event["kind"] == "authority_consumed":
            clock.sleep(2)

    monkeypatch.setattr(AttemptEvidence, "record", slow)
    report = run(trial, clock)
    assert report["status"] == "BLOCKED" and not movements(trial[2])
    assert report["result"]["authority_consumed"]
    assert not report["result"]["physical_send_attempted"]
    assert b"Stop()" not in trial[2].sent


@pytest.mark.parametrize(
    "extra",
    [
        ["--initial-native-si", "0", "0", "0", "0", "0", "0"],
        ["--delta-xyz", ".01", "0", "0"],
        ["--output", "forbidden.json"],
        ["--host", "198.51.100.2"],
    ],
)
def test_live_cli_rejects_synthetic_overrides_before_connection(trial, tmp_path, extra):
    from inspect_robots_dobot.micro_move import main

    rig, monitor = tmp_path / "rig.json", tmp_path / "profile.json"
    rig.write_text(json.dumps(asdict(trial[0])))
    data = asdict(trial[1])
    del data["safety"]
    monitor.write_text(json.dumps(data))
    assert (
        main(
            [
                "--live",
                "--allow-motion",
                "--config",
                str(rig),
                "--profile",
                str(monitor),
                "--evidence-dir",
                str(trial[4]),
                *extra,
            ]
        )
        == 2
    )
    assert not trial[4].exists()


def test_evidence_creation_failure_blocks_without_connection(trial, clock, monkeypatch):
    original = Path.open

    def fail(path, *args, **kwargs):
        if path.parent == trial[4]:
            raise OSError("no evidence storage")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(ConfigurationError, match="before connection"):
        run(trial, clock)
    assert not trial[3].calls


def test_initial_final_report_is_blocked_until_complete(trial, clock, monkeypatch):
    original = __import__("builtins").input

    def answer(prompt):
        if "MOVE ONCE" in prompt:
            assert "BLOCKED / NOT RUN" in (trial[4] / "final_report.md").read_text()
            assert not movements(trial[2])
        return original(prompt)

    monkeypatch.setattr("builtins.input", answer)
    assert run(trial, clock)["status"] == "MOCK_OK"


@pytest.mark.parametrize("file", ["execution.json.pending", "final_report.md.pending"])
def test_failure_publishing_success_cannot_leave_agreeing_pass_pair(
    trial, clock, monkeypatch, file
):
    original = Path.open

    def fail(path, *args, **kwargs):
        if path.parent == trial[4] and path.name == file and trial[2].closed:
            raise OSError("cannot publish complete result")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    report = run(trial, clock)
    assert report["status"] == "FAILED" and not report["evidence_complete"]
    assert len(movements(trial[2])) == 1
    execution = json.loads((trial[4] / "execution.json").read_text())
    final = (trial[4] / "final_report.md").read_text()
    assert execution.get("status") != "MOCK_OK" or "MOCK_OK" not in final
