"""Independent gate failures must precede any motion write."""

import copy
import pickle
from dataclasses import replace

import pytest
from live_fakes import arm, connection, movements, production_profile, readiness, setup

from inspect_robots_dobot.errors import ConfigurationError, MotionNotAuthorized, SafetyRejected
from inspect_robots_dobot.live_authority import LiveMotionAuthority, plan_digest
from inspect_robots_dobot.live_channel import MotionCancelled
from inspect_robots_dobot.live_driver import LiveDobotMotionDriver, LiveMotionState
from inspect_robots_dobot.types import RobotMode


def test_default_construction_and_connect_never_send(profile, pose, clock):
    driver, peer, factory, _ = setup(profile, pose, clock, allow_motion=False)
    assert factory.calls == []
    with pytest.raises(MotionNotAuthorized):
        driver.connect()
    driver.connect(allow_connection=True)
    assert peer.sent == []
    assert len(factory.calls) == 1
    driver.close()
    assert peer.closed


@pytest.mark.parametrize(
    "missing",
    [
        "allow_motion",
        "connection",
        "readiness",
        "ownership",
        "firmware",
        "frames",
        "confirmation",
        "synthetic_start",
    ],
)
def test_individual_gates(profile, pose, clock, missing):
    config = connection(
        **{
            "ownership": {"tcp_control_owned": False},
            "firmware": {"protocol_compatibility_confirmed": False},
            "frames": {"tool_frame": 1},
        }.get(missing, {})
    )
    driver, peer, _, plan = setup(
        profile, pose, clock, config=config, allow_motion=missing != "allow_motion"
    )
    if missing != "connection":
        driver.connect(allow_connection=True)
        if missing != "readiness":
            driver.confirm_readiness(readiness())
    if missing == "synthetic_start":
        plan = replace(
            plan,
            starting_measured_state=replace(plan.starting_measured_state, joints_synthetic=True),
        )
    with pytest.raises((MotionNotAuthorized, SafetyRejected)):
        driver.arm(plan, confirmation="y" if missing == "confirmation" else "MOVE ONCE")
    assert not movements(peer)
    assert driver.state == LiveMotionState.REJECTED_BEFORE_SEND
    driver.close()


@pytest.mark.parametrize(
    "field",
    [
        "tcp_control_owned",
        "frames_verified",
        "production_profile_verified",
        "estop_tested",
        "operator_present",
        "workspace_clear",
        "gripper_disabled",
        "operator",
    ],
)
def test_each_operator_attestation_required(profile, pose, clock, field):
    driver, peer, _, plan = setup(profile, pose, clock)
    driver.connect(allow_connection=True)
    with pytest.raises(ConfigurationError):
        driver.confirm_readiness(readiness(**{field: "" if field == "operator" else False}))
    with pytest.raises(MotionNotAuthorized):
        driver.arm(plan, confirmation="MOVE ONCE")
    assert not movements(peer)
    driver.close()


def test_complete_production_profile_required(profile):
    with pytest.raises(ConfigurationError):
        LiveDobotMotionDriver(connection(), profile, allow_motion=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"start_position_tolerance": 0},
        {"start_position_tolerance": 0.002},
        {"start_orientation_tolerance": 0.02},
        {"max_measurement_age": 0.6},
        {"consecutive_samples": 1},
        {"consecutive_samples": True},
        {"io_timeout": 0.3},
        {"acknowledgement_timeout": 2},
        {"stop_timeout": 6},
        {"standstill_position_tolerance": 0.002},
        {"standstill_orientation_tolerance": 0.02},
        {"authority_lifetime": 31},
        {"model": ""},
        {"firmware": ""},
        {"rig_verification_reference": ""},
        {"interruption_verification_reference": ""},
        {"keepouts": []},
    ],
)
def test_profile_requires_explicit_conservative_thresholds(profile, changes):
    with pytest.raises(ConfigurationError):
        production_profile(profile, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"speed_percent": 11},
        {"acceleration_percent": 11},
        {"max_translation_step": 0.021},
        {"orientation_high": (0.1, 0, 0)},
        {"position_tolerance": 0.002},
        {"orientation_tolerance": 0.02},
        {"poll_interval": 0.2},
    ],
)
def test_first_profile_cannot_be_widened(profile, changes):
    with pytest.raises(ConfigurationError):
        production_profile(replace(profile, **changes))


@pytest.mark.parametrize("change", ["target", "digest", "foreign", "expiry", "details"])
def test_authority_is_exact_session_plan_identity(profile, pose, clock, change):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    if change == "target":
        plan = replace(plan, final_agent_pose=(0.311, 0, 0.2, 0, 0, 0, 0))
    elif change == "digest":
        authority._details = replace(authority.details, plan_digest="bad")
    elif change == "foreign":
        authority = LiveMotionAuthority(authority.details)
    elif change == "expiry":
        clock.sleep(1.0)
    else:
        authority._details = replace(authority.details, expires_at=1000)
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.REJECTED_BEFORE_SEND
    assert not movements(peer)
    assert not result.authority_consumed
    assert peer.closed


@pytest.mark.parametrize("method", ["close", "reset"])
def test_close_reset_invalidate_authority_and_reconnect(profile, pose, clock, method):
    driver, peer, factory, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    if method == "reset":
        with pytest.raises(MotionNotAuthorized):
            driver.reset()
    else:
        driver.close()
    with pytest.raises(MotionNotAuthorized):
        driver.execute(plan, authority)
    with pytest.raises(MotionNotAuthorized):
        driver.connect(allow_connection=True)
    assert len(factory.calls) == 1
    assert not movements(peer)


@pytest.mark.parametrize("copy_method", [copy.copy, copy.deepcopy, pickle.dumps])
def test_authority_not_persistable(profile, pose, clock, copy_method):
    driver, _, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    with pytest.raises(TypeError):
        copy_method(authority)
    driver.close()


def test_digest_binds_host_profile_and_all_plan_fields(profile, pose, clock):
    driver, _, _, plan = setup(profile, pose, clock)
    digest = plan_digest(plan, driver.profile, driver.connection)
    assert digest != plan_digest(plan, driver.profile, replace(driver.connection, host="192.0.2.2"))
    assert digest != plan_digest(
        plan, replace(driver.profile, authority_lifetime=0.5), driver.connection
    )
    assert digest != plan_digest(replace(plan, chunk_id="other"), driver.profile, driver.connection)
    assert len(digest) == 64


@pytest.mark.parametrize("state", ["position", "orientation", "mode", "errors", "id", "stale"])
def test_changed_measured_start_rejects_before_send(profile, pose, clock, state):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    if state == "position":
        peer.start = replace(peer.start, pose=replace(pose, x=0.301))
    elif state == "orientation":
        peer.start = replace(peer.start, pose=replace(pose, rx=0.11))
    elif state == "mode":
        peer.start = replace(peer.start, mode=RobotMode.RUNNING)
    elif state == "errors":
        peer.start = replace(peer.start, errors=(123,))
    elif state == "id":
        peer.start = replace(peer.start, command_id=100)
    else:
        peer.query_delay = 0.05
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.REJECTED_BEFORE_SEND
    assert not movements(peer)
    assert not result.authority_consumed
    assert peer.closed


def test_start_tolerance_never_regenerates_absolute_target(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    peer.start = replace(peer.start, pose=replace(pose, x=pose.x + 0.0002, rx=pose.rx + 0.0001))
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.COMPLETED
    assert movements(peer) == [plan.request.serialize().encode()]


def test_default_fake_boolean_authority_cannot_arm_live(profile, pose, clock):
    from inspect_robots_dobot.safety import MotionAuthority

    driver, peer, _, plan = setup(profile, pose, clock)
    driver.connect(allow_connection=True)
    result = driver.execute(plan, MotionAuthority(True))
    assert result.state == LiveMotionState.REJECTED_BEFORE_SEND
    assert not movements(peer)


def test_runtime_fields_are_readonly(profile, pose, clock):
    driver, _, _, _ = setup(profile, pose, clock)
    for key in ("profile", "connection", "allow_motion", "transport_is_mock"):
        with pytest.raises(AttributeError):
            setattr(driver, key, None)


def test_tampered_plan_is_rebuilt_before_issuing_authority(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    driver.connect(allow_connection=True)
    driver.confirm_readiness(readiness())
    plan = replace(plan, request=replace(plan.request, speed_percent=6))
    with pytest.raises(SafetyRejected, match="inconsistent"):
        driver.arm(plan, confirmation="MOVE ONCE")
    assert not movements(peer)
    driver.close()


def test_profile_or_host_change_after_arm_invalidates_digest(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    # Deliberate internal tampering is not supported API, but must not bypass digest.
    driver._connection = replace(driver.connection, host="192.0.2.2")
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.REJECTED_BEFORE_SEND
    assert "digest" in result.failure_cause
    assert not movements(peer)


def test_foreign_session_capability_is_not_transferable(profile, pose, clock):
    first, _, _, plan = setup(profile, pose, clock)
    authority = arm(first, plan)
    second, peer, _, second_plan = setup(profile, pose, clock)
    arm(second, second_plan)
    result = second.execute(second_plan, authority)
    first.close()
    assert result.state == LiveMotionState.REJECTED_BEFORE_SEND
    assert not movements(peer)


def test_authority_expiry_rechecked_after_fresh_queries(profile, pose, clock):
    driver, peer, _, plan = setup(
        profile, pose, clock, live_profile=production_profile(profile, authority_lifetime=0.03)
    )
    authority = arm(driver, plan)
    peer.query_delay = 0.005
    result = driver.execute(plan, authority)
    assert "expired" in result.failure_cause
    assert not result.authority_consumed and not movements(peer)


def test_start_age_rechecked_immediately_before_write(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    authority = arm(driver, plan)
    original_timeout = peer.settimeout

    def delayed_settimeout(value):
        # 7 pre-send queries * 2 timeouts. The 15th is the motion-write setup.
        original_timeout(value)
        if len(peer.timeouts) == 29:  # arm queries (14) + execute queries (14) + 1
            clock.sleep(0.21)

    peer.settimeout = delayed_settimeout
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.REJECTED_BEFORE_SEND
    assert not result.authority_consumed and not movements(peer)
    assert "stale" in result.failure_cause


def test_start_jitter_cannot_cross_keepout(profile, pose, clock):
    from inspect_robots_dobot.motion import KeepoutBox

    box = KeepoutBox((0.299, 0.0001, 0.19), (0.302, 0.001, 0.21))
    driver, peer, _, plan = setup(
        profile, pose, clock, live_profile=production_profile(profile, keepouts=(box,))
    )
    authority = arm(driver, plan)
    peer.start = replace(peer.start, pose=replace(pose, y=0.0002))
    result = driver.execute(plan, authority)
    assert result.state == LiveMotionState.REJECTED_BEFORE_SEND
    assert "keepout" in result.failure_cause
    assert not movements(peer)


def test_close_during_arming_never_leaves_capability_or_connection(profile, pose, clock):
    driver, peer, _, plan = setup(profile, pose, clock)
    driver.connect(allow_connection=True)
    driver.confirm_readiness(readiness())
    peer.on_recv = driver.close
    with pytest.raises(MotionCancelled):
        driver.arm(plan, confirmation="MOVE ONCE")
    assert peer.closed
    assert driver.state == LiveMotionState.REJECTED_BEFORE_SEND
    assert not movements(peer)
    with pytest.raises(MotionNotAuthorized):
        driver.connect(allow_connection=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allow_motion": 1},
        {"transport_is_mock": True},
    ],
)
def test_no_boolean_coercion_or_real_socket_disguised_as_mock(profile, kwargs):
    with pytest.raises(ConfigurationError):
        LiveDobotMotionDriver(connection(), production_profile(profile), **kwargs)


@pytest.mark.parametrize(
    "command",
    [
        b"PowerOn()",
        b"EnableRobot()",
        b"ClearError()",
        b"RequestControl()",
        b"EmergencyStop(1)",
        b"MovJ(pose={1,2,3,4,5,6})",
        b"ServoP()",
        b"ServoJ()",
        b"ToolDO(1,1)",
        b"ToolDOInstant(1,1)",
    ],
)
def test_even_internal_channel_rejects_lifecycle_and_other_motion(profile, pose, clock, command):
    from inspect_robots_dobot.errors import ProtocolError

    driver, peer, _, _ = setup(profile, pose, clock)
    driver.connect(allow_connection=True)
    with pytest.raises(ProtocolError):
        driver._channel._exchange(command)
    assert peer.sent == []
    driver.close()
