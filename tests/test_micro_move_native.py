"""Independent decimal expectations; no robot geometry inferred from neutral joints."""

from dataclasses import asdict, replace
from decimal import Decimal

import pytest
from live_fakes import connection, production_profile

from inspect_robots_dobot.errors import ConfigurationError, ProtocolError, SafetyRejected
from inspect_robots_dobot.live_authority import plan_digest
from inspect_robots_dobot.motion import DobotMovLRequest, build_micro_move_plan, build_motion_plan
from inspect_robots_dobot.types import NativeDecimalPose, RobotMode, RobotSnapshot
from inspect_robots_dobot.units import from_native, to_native


@pytest.fixture
def measured(clock):
    # FICTIONAL TCP fixture, NOT derived from reset_nova_comf joint constants.
    raw = NativeDecimalPose(
        (
            "300.12345678912345",
            "-0.000100",
            "200.250",
            "12.123456789123456",
            "-23.098765432123456",
            "179.9999999999999",
        )
    )
    return RobotSnapshot(
        from_native(raw.native),
        (0.1, -0.6, 2.1, 0.02, -1.5, -1.6),
        RobotMode.ENABLED_IDLE,
        (),
        42,
        clock.monotonic(),
        0,
        0,
        native_decimal=raw,
    )


@pytest.mark.parametrize(
    "direction,index,delta",
    [("+X", 0, 10), ("-X", 0, -10), ("+Y", 1, 10), ("-Y", 1, -10), ("+Z", 2, 10), ("-Z", 2, -10)],
)
def test_exact_decimal_target_preserves_every_other_token(
    measured, profile, clock, direction, index, delta
):
    plan = build_micro_move_plan("native", measured, direction, profile, clock.monotonic())
    before = measured.native_decimal.values
    after = plan.request.native_decimal.values
    assert Decimal(after[index]) - Decimal(before[index]) == delta
    assert all(a == b for i, (a, b) in enumerate(zip(before, after, strict=True)) if i != index)
    assert plan.final_agent_pose[3:] == (0, 0, 0, 0)
    assert plan.final_pose_si.values[3:] == measured.pose.values[3:]
    assert (
        plan.request.serialize() == f"MovL(pose={{{','.join(after)}}},user=0,tool=0,a=5,v=5,cp=0)"
    )
    rebuilt = build_motion_plan(
        plan.chunk_id,
        plan.staged_waypoints,
        measured,
        measured.pose,
        profile,
        clock.monotonic(),
        micro_move_direction=direction,
    )
    assert asdict(rebuilt) == asdict(plan)


@pytest.mark.parametrize("direction", ["", "X", "x", "+x", " +X", "+X ", "up", "+X+Y", None])
def test_direction_has_no_default(measured, profile, clock, direction):
    with pytest.raises(ProtocolError):
        build_micro_move_plan("invalid", measured, direction, profile, clock.monotonic())


def test_degree_roundtrip_is_not_source_for_wire_angles(measured, profile, clock):
    plan = build_micro_move_plan("roundtrip", measured, "+X", profile, clock.monotonic())
    assert any(
        str(v) != raw
        for v, raw in zip(
            to_native(measured.pose).values[3:], measured.native_decimal.values[3:], strict=True
        )
    )
    assert plan.request.native_decimal.values[3:] == measured.native_decimal.values[3:]
    assert ",12.123456789123456,-23.098765432123456,179.9999999999999}" in plan.request.serialize()


def test_native_text_and_direction_are_digest_bound(measured, profile, clock):
    plan = build_micro_move_plan("digest", measured, "+X", profile, clock.monotonic())
    digest = plan_digest(plan, production_profile(profile), connection())
    for changed in (
        replace(plan, micro_move_direction="-X"),
        replace(plan, request=replace(plan.request, native_decimal=None)),
    ):
        assert plan_digest(changed, production_profile(profile), connection()) != digest


@pytest.mark.parametrize("variant", ["synthetic", "missing", "mismatch"])
def test_no_synthetic_or_reconstructed_native_start(measured, profile, clock, variant):
    changed = {
        "synthetic": replace(measured, joints_synthetic=True),
        "missing": replace(measured, native_decimal=None),
        "mismatch": replace(measured, pose=measured.pose.with_translation(0.3, 0, 0.2)),
    }[variant]
    with pytest.raises(SafetyRejected):
        build_micro_move_plan("invalid", changed, "+X", profile, clock.monotonic())


@pytest.mark.parametrize("token", ["NaN", "Inf", "1e1000", "1);Stop()", "", "1,2", " 3", "1e-101"])
def test_native_decimal_rejects_nonfinite_injection_or_unbounded_exponents(token):
    with pytest.raises(ProtocolError):
        NativeDecimalPose((token, "0", "200", "0", "0", "0"))


def test_mismatched_decimal_request_rejected(measured):
    with pytest.raises(ConfigurationError):
        DobotMovLRequest(
            measured.native_decimal.native,
            0,
            0,
            5,
            5,
            measured.native_decimal.translated_10mm("+X"),
        )


def test_exact_squared_distance_does_not_expand_10mm_budget():
    origin = NativeDecimalPose(("0", "0", "200", "0", "0", "0"))
    target = origin.translated_10mm("+X")
    assert target.squared_distance_mm(origin) == 100
    shifted = NativeDecimalPose(("-0.00000000000000001", "0", "200", "0", "0", "0"))
    assert target.squared_distance_mm(shifted) > 100


def test_exact_10mm_profile_allows_binary_roundoff_without_raising_budget(measured, profile, clock):
    strict = replace(profile, max_translation_step=0.010)
    plan = build_micro_move_plan("ten", measured, "+X", strict, clock.monotonic())
    assert plan.path_validation.displacement_limit_m == 0.010
    assert plan.path_validation.total_displacement_m == 0.010


@pytest.mark.parametrize("kind", ["workspace", "minimum_z", "keepout", "budget", "orientation"])
def test_micro_move_retains_existing_path_constraints(measured, profile, clock, kind):
    from inspect_robots_dobot.motion import KeepoutBox

    keepouts = ()
    if kind == "workspace":
        profile = replace(profile, workspace_high=(0.305, 0.2, 0.5))
    elif kind == "minimum_z":
        profile = replace(profile, minimum_tcp_z=0.3)
    elif kind == "keepout":
        keepouts = (KeepoutBox((0.304, -0.01, 0.19), (0.306, 0.01, 0.21)),)
    elif kind == "budget":
        profile = replace(profile, max_translation_step=0.009)
    else:
        profile = replace(profile, orientation_high=(0.1, 0.0, 0.0))
    with pytest.raises((SafetyRejected, ConfigurationError)):
        build_micro_move_plan("rejected", measured, "+X", profile, clock.monotonic(), keepouts)
