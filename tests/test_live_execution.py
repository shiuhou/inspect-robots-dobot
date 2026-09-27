"""One motion write, independently measured completion, no corrective retry."""

import threading
from dataclasses import replace

import pytest
from live_fakes import arm, movements, response, setup

from inspect_robots_dobot.errors import MotionNotAuthorized
from inspect_robots_dobot.live_driver import LiveMotionState
from inspect_robots_dobot.types import RobotMode


def states(driver):
    return [e["state"] for e in driver.audit_records if e["kind"] == "transition"]


def residuals(driver):
    return [e for e in driver.audit_records if e["kind"] == "residual"]


def assert_only_allowed(peer):
    allowed = {
        b"RobotMode()",
        b"GetPose(user=0,tool=0)",
        b"GetAngle()",
        b"GetErrorID()",
        b"GetCurrentCommandID()",
        b"Stop()",
    }
    assert all(c in allowed or c.startswith(b"MovL(pose={") for c in peer.sent)
    assert len(movements(peer)) <= 1
    assert peer.sent.count(b"Stop()") <= 1
    assert peer.closed


@pytest.mark.parametrize("reply", ["ok", "fragmented"])
def test_exact_single_send_completion_audit_and_reuse(profile, pose, clock, reply):
    driver, peer, factory, plan = setup(profile, pose, clock)
    peer.move_reply = reply
    authority = arm(driver, plan)
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.COMPLETED
    assert result.failure_cause is None
    assert result.command_id == 43
    assert result.motion_write_attempts == 1 and result.authority_consumed
    assert result.stop_attempts == 0
    assert result.connections_closed
    assert result.final_sample.pose.values == pytest.approx(plan.final_pose_si.values)
    assert not result.physical_send_attempted and result.transport_is_mock
    assert movements(peer) == [plan.request.serialize().encode()]
    wire = [
        e for e in driver.audit_records if e["kind"] == "wire" and e["command"].startswith("MovL(")
    ]
    assert len(wire) == 1
    assert wire[0]["write_completed"]
    assert wire[0]["response_ascii"] == response(movements(peer)[0], b"43").decode()
    assert states(driver) == ["ARMED_FOR_ONE_PLAN", "SENDING", "ACCEPTED", "SETTLING", "COMPLETED"]
    assert [r["consecutive"] for r in residuals(driver)] == [1, 2, 3]
    with pytest.raises(MotionNotAuthorized):
        driver.execute(plan, authority)
    with pytest.raises(MotionNotAuthorized):
        driver.arm(plan, confirmation="MOVE ONCE")
    with pytest.raises(MotionNotAuthorized):
        driver.connect(allow_connection=True)
    driver.close()
    assert driver.result == result
    assert len(factory.calls) == 1
    assert_only_allowed(peer)


@pytest.mark.parametrize(
    "reply,ambiguous,standstill",
    [
        ("rejected", False, True),
        ("malformed", True, False),
        ("bad_id", True, True),
        ("wrong_echo", True, False),
        ("eof", True, False),
        ("write_failure", True, False),
        ("lost", True, False),
        ("late", True, True),
        ("duplicate", True, False),
        ("trailing", True, False),
        ("keyboard", True, False),
    ],
)
def test_ack_failures_never_resend(profile, pose, clock, reply, ambiguous, standstill):
    driver, peer, factory, plan = setup(profile, pose, clock)
    peer.move_reply = reply
    result = driver.execute(plan, arm(driver, plan))
    assert result.acceptance_ambiguous is ambiguous
    assert result.standstill_confirmed is standstill
    assert result.state == (LiveMotionState.ABORTED if standstill else LiveMotionState.FAULTED)
    assert result.failure_cause
    if reply == "rejected":
        wire = [
            e
            for e in driver.audit_records
            if e["kind"] == "wire" and e["command"].startswith("MovL(")
        ]
        assert wire[0]["error_id"] == -2
    assert result.authority_consumed and result.motion_write_attempts == 1
    assert result.stop_attempts == 1
    assert len(movements(peer)) == 1
    assert len(factory.calls) == 1
    if reply in ("eof", "write_failure"):
        assert b"Stop()" not in peer.sent
    else:
        assert peer.sent.count(b"Stop()") == 1
    if ambiguous:
        assert "AMBIGUOUS_ACCEPTANCE" in states(driver)
    with pytest.raises(MotionNotAuthorized):
        driver.execute(plan, None)
    assert_only_allowed(peer)


def test_late_reply_and_stop_reply_are_independently_matched(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    peer.move_reply = "late"
    result = driver.execute(plan, arm(driver, plan))
    assert result.acceptance_ambiguous
    assert result.standstill_confirmed
    wire = [e for e in driver.audit_records if e["kind"] == "wire" and e["command"] == "Stop()"]
    assert plan.request.serialize() in wire[0]["response_ascii"]
    assert "0,{},Stop();" in wire[0]["response_ascii"]
    assert result.command_id is None  # late acknowledgement does not revive the motion session


@pytest.mark.parametrize("condition", ["no_motion", "wrong_id", "running", "rotation_residual"])
def test_acceptance_is_not_arrival(profile, pose, clock, condition):
    driver, peer, _, plan = setup(profile, pose, clock)
    at_target = replace(peer.start, pose=plan.final_pose_si, command_id=43)
    sample = {
        "no_motion": replace(peer.start, command_id=43),
        "wrong_id": replace(at_target, command_id=42),
        "running": replace(at_target, mode=RobotMode.RUNNING),
        "rotation_residual": replace(at_target, pose=replace(at_target.pose, rx=pose.rx + 0.002)),
    }[condition]
    peer.execution.clear()
    peer.execution.append(sample)
    result = driver.execute(plan, arm(driver, plan))
    assert result.state == LiveMotionState.ABORTED
    assert result.command_id == 43
    assert "TIMEOUT" in states(driver)
    assert "COMPLETED" not in states(driver)
    assert all(r["consecutive"] == 0 for r in residuals(driver))
    assert result.standstill_confirmed
    assert_only_allowed(peer)


def test_id_progression_and_running_to_idle(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    at_target = replace(peer.start, pose=plan.final_pose_si, command_id=43)
    peer.execution.clear()
    peer.execution.extend(
        [
            replace(peer.start, mode=RobotMode.RUNNING),
            replace(at_target, mode=RobotMode.RUNNING),
            at_target,
        ]
    )
    result = driver.execute(plan, arm(driver, plan))
    assert result.state == LiveMotionState.COMPLETED
    assert [r["command_id"] for r in residuals(driver)] == [42, 43, 43, 43, 43]
    assert [r["consecutive"] for r in residuals(driver)] == [0, 0, 1, 2, 3]
    assert "EXECUTING" in states(driver)
    assert_only_allowed(peer)


def test_mode_transition_inside_acquisition_is_not_arrival(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    first = True

    def transition(data):
        nonlocal first
        if peer.moving and data == b"RobotMode()" and first:
            first = False
            peer.execution[0] = replace(peer.execution[0], mode=RobotMode.RUNNING)
        elif peer.moving and data == b"GetCurrentCommandID()":
            peer.current = replace(peer.current, mode=RobotMode.ENABLED_IDLE)

    peer.on_send = transition
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.COMPLETED
    assert [r["consecutive"] for r in residuals(driver)] == [0, 1, 2, 3]


def test_noise_resets_consecutive_convergence(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    end = replace(peer.start, pose=plan.final_pose_si, command_id=43)
    peer.execution.clear()
    peer.execution.extend(
        replace(end, pose=replace(end.pose, x=end.pose.x + noise))
        for noise in [0, 0.00005, 0.0002, -0.00004, 0.00003, -0.00002]
    )
    result = driver.execute(plan, arm(driver, plan))
    assert result.state == LiveMotionState.COMPLETED
    assert [r["consecutive"] for r in residuals(driver)] == [1, 2, 0, 1, 2, 3]
    assert_only_allowed(peer)


@pytest.mark.parametrize("fault", ["collision", "error_mode", "alarm", "drift", "bounds", "nan"])
def test_execution_faults_request_stop(profile, pose, clock, fault):
    driver, peer, _, plan = setup(profile, pose, clock)
    end = replace(peer.start, pose=plan.final_pose_si, command_id=43)
    faulty = {
        "collision": replace(end, mode=RobotMode.COLLISION),
        "error_mode": replace(end, mode=RobotMode.ERROR),
        "alarm": replace(end, errors=(123,)),
        "drift": replace(end, pose=replace(end.pose, rz=pose.rz + 0.1)),
        "bounds": replace(end, pose=replace(end.pose, z=0.01)),
        "nan": replace(end, pose=replace(end.pose, x=float("nan"))),
    }[fault]
    peer.execution.clear()
    peer.execution.append(faulty)
    result = driver.execute(plan, arm(driver, plan))
    assert "FAULTED" in states(driver)
    assert result.state == LiveMotionState.ABORTED
    assert result.stop_attempts == 1
    assert_only_allowed(peer)


@pytest.mark.parametrize("failure", ["timeout", "rejected", "bad_payload", "write_failure"])
def test_stop_failure_never_claims_standstill(profile, pose, clock, failure):
    driver, peer, _, plan = setup(profile, pose, clock)
    peer.execution.clear()
    peer.execution.append(replace(peer.start, command_id=43))
    peer.stop_reply = failure
    result = driver.execute(plan, arm(driver, plan))
    assert result.state == LiveMotionState.FAULTED
    assert not result.standstill_confirmed
    assert result.stop_attempts == 1
    assert_only_allowed(peer)


@pytest.mark.parametrize("condition", ["moving", "pose_drift", "fault", "changing_id"])
def test_stop_ack_requires_measured_standstill(profile, pose, clock, condition):
    driver, peer, _, plan = setup(profile, pose, clock)
    peer.execution.clear()
    peer.execution.append(replace(peer.start, command_id=43))
    for n in range(30):
        snapshot = replace(peer.start, command_id=43)
        if condition == "moving":
            snapshot = replace(snapshot, mode=RobotMode.RUNNING)
        elif condition == "pose_drift":
            snapshot = replace(snapshot, pose=replace(pose, x=pose.x + n * 0.0003))
        elif condition == "fault":
            snapshot = replace(snapshot, mode=RobotMode.ERROR)
        else:
            snapshot = replace(snapshot, command_id=43 + n)
        peer.after_stop.append(snapshot)
    result = driver.execute(plan, arm(driver, plan))
    assert result.state == LiveMotionState.FAULTED
    assert not result.standstill_confirmed
    assert_only_allowed(peer)


def test_cancel_before_send_invalidates_authority_without_stop(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    driver.request_stop()
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.REJECTED_BEFORE_SEND
    assert not movements(peer)
    assert result.stop_attempts == 0
    assert result.connections_closed
    assert not result.authority_consumed


@pytest.mark.parametrize("cancel", ["request_stop", "close", "keyboard"])
def test_active_cancellation_during_monitor(profile, pose, clock, cancel):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    cancelled = False

    def on_send(data):
        nonlocal cancelled
        if peer.moving and not peer.stopped and not cancelled and data == b"GetCurrentCommandID()":
            cancelled = True
            if cancel == "keyboard":
                # Interrupt receive, not send; preserve a complete outstanding reply.
                peer.chunks.append(KeyboardInterrupt())
            else:
                getattr(driver, cancel)()

    peer.on_send = on_send
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.ABORTED
    assert result.standstill_confirmed
    assert result.command_id == 43
    assert_only_allowed(peer)


def test_cancellation_from_another_thread(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    waiting, release = threading.Event(), threading.Event()
    result = []
    once = False

    def on_send(data):
        nonlocal once
        if peer.moving and not once and data == b"RobotMode()":
            once = True
            waiting.set()
            assert release.wait(1)

    peer.on_send = on_send
    worker = threading.Thread(target=lambda: result.append(driver.execute(plan, authority)))
    worker.start()
    try:
        assert waiting.wait(1)
        with pytest.raises(MotionNotAuthorized):
            driver.measure_start()
        with pytest.raises(MotionNotAuthorized):
            driver.execute(plan, authority)
        driver.request_stop()
    finally:
        release.set()
        worker.join(1)
    assert not worker.is_alive()
    assert result[0].state == LiveMotionState.ABORTED
    assert_only_allowed(peer)


def test_cancel_during_lost_ack_has_bounded_mock_stop_attempt(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    peer.move_reply = "late"
    triggered = False

    def on_recv():
        nonlocal triggered
        if peer.moving and not triggered:
            triggered = True
            driver.request_stop()

    peer.on_recv = on_recv
    started = clock.monotonic()
    result = driver.execute(plan, authority)
    assert clock.monotonic() - started < 0.1
    assert result.acceptance_ambiguous
    assert result.state == LiveMotionState.ABORTED
    assert_only_allowed(peer)


def test_close_failure_is_reported(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    peer.close_error = OSError("close failure")
    result = driver.execute(plan, arm(driver, plan))
    assert "close failed" in result.failure_cause
    assert not result.connections_closed
    assert any(e["kind"] == "close_failed" for e in driver.audit_records)


def test_collision_arriving_at_end_of_acquisition_aborts(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)

    def fault(data):
        if peer.moving and not peer.stopped and data == b"GetCurrentCommandID()":
            peer.current = replace(peer.current, mode=RobotMode.COLLISION)

    peer.on_send = fault
    result = driver.execute(plan, arm(driver, plan))
    assert result.state == LiveMotionState.ABORTED
    assert "controller state" in result.failure_cause
    assert_only_allowed(peer)


def test_measured_keepout_intrusion_aborts(profile, pose, clock):
    from live_fakes import production_profile

    from inspect_robots_dobot.motion import KeepoutBox

    box = KeepoutBox((0.29, 0.001, 0.19), (0.32, 0.01, 0.21))
    driver, peer, _, plan = setup(
        profile, pose, clock, live_profile=production_profile(profile, keepouts=(box,))
    )
    peer.execution.clear()
    peer.execution.append(replace(peer.start, pose=replace(pose, y=0.005), command_id=43))
    result = driver.execute(plan, arm(driver, plan))
    assert "keepout" in result.failure_cause
    assert result.state == LiveMotionState.ABORTED
    assert_only_allowed(peer)


def test_freezing_host_clock_does_not_create_consecutive_samples(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    clock.sleep = lambda _: None
    # Wall-clock Deadline still terminates this loop; repeated timestamp is not fresh evidence.
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.FAULTED
    assert "TIMEOUT" in states(driver)
    assert "COMPLETED" not in states(driver)
    assert not result.standstill_confirmed
    assert_only_allowed(peer)


def test_cancel_between_converging_samples_requests_stop(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    original_sleep = clock.sleep

    def cancel_on_pause(seconds):
        if seconds > 0 and not peer.stopped:
            driver.request_stop()
        original_sleep(seconds)

    clock.sleep = cancel_on_pause
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.ABORTED
    assert "SETTLING" in states(driver)
    assert_only_allowed(peer)


def test_non_ascii_ack_is_preserved_and_ambiguous(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)

    def corrupt(data):
        if data.startswith(b"MovL("):
            peer.chunks.append(b"\xff;")

    peer.on_send = corrupt
    result = driver.execute(plan, arm(driver, plan))
    assert result.acceptance_ambiguous
    wire = [
        e for e in driver.audit_records if e["kind"] == "wire" and e["command"].startswith("MovL(")
    ]
    assert wire[0]["response_hex"] == "ff3b"
    assert_only_allowed(peer)


def test_rejection_plus_unsolicited_reply_is_ambiguous(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)

    def corrupt(data):
        if data.startswith(b"MovL("):
            peer.chunks.append(response(data, error=-2) + response(b"RobotMode()", b"5"))

    peer.on_send = corrupt
    result = driver.execute(plan, arm(driver, plan))
    assert result.acceptance_ambiguous
    assert "extra unsolicited" in result.failure_cause
    assert_only_allowed(peer)
