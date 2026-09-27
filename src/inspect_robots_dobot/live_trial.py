"""Supervised one-shot CLI workflow; all development/tests use injected streams.

No lifecycle commands, camera, model API, direction default or second attempt.
Only micro_move.main selects live mode, after explicit runtime/TTY/config gates.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .clock import Clock, SystemClock
from .config import DobotConfig, LocalMicroMoveSettings
from .errors import ConfigurationError, MotionNotAuthorized
from .evidence import AttemptEvidence, EvidenceError
from .live_authority import plan_digest
from .live_driver import LiveDobotMotionDriver, LiveMotionState
from .live_profile import LiveMotionProfile, OperatorReadiness
from .local_envelope import LocalEnvelopeSafety
from .motion import CartesianMotionPlan, build_micro_move_plan
from .transport import SocketFactory, open_socket
from .types import RobotMode, RobotSnapshot
from .units import orientation_distance, translation_distance

STOP_LIMITATION = (
    "Physical E-stop is the hard safety channel. Software Stop is BEST-EFFORT; "
    "broken TCP, partial writes or ambiguous replies may prevent interruption. "
    "If robot behavior is unexpected, use the physical E-stop immediately."
)


def terminal_input(prompt: str) -> str:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise MotionNotAuthorized("live operation requires interactive input AND review terminal")
    return input(prompt)


def validate_live_inputs(config: DobotConfig, profile: LiveMotionProfile, allow: bool) -> None:
    if allow is not True:
        raise MotionNotAuthorized("live mode requires explicit --allow-motion")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise MotionNotAuthorized("live operation requires an interactive terminal; no pipes")
    if (
        config.connection is None
        or not config.connection.host
        or (config.safety or config.local_micro_move) != profile.safety
    ):
        raise ConfigurationError("live requires explicit connection and matching complete safety")
    if config.camera is not None:
        raise ConfigurationError("standalone micro-move does not accept camera configuration")
    profile.require_live_descriptions()
    c = config.connection
    local_micro_move = isinstance(config.local_micro_move, LocalMicroMoveSettings)
    if not local_micro_move and (
        c.controller_firmware is None
        or profile.firmware is None
        or not c.protocol_compatibility_confirmed
    ):
        raise ConfigurationError(
            "LIVE RELEASE BLOCKED before connection: controller firmware is unknown or "
            "protocol compatibility is unverified/empirically partial; document 4.6.5 is "
            "not a firmware version. The standalone supervised micro-move uses the "
            "operator-approved unknown-firmware exception; this is not general release."
        )
    if (
        c.expected_protocol_version != "4.6.5"
        or (
            not local_micro_move
            and (
                not c.protocol_compatibility_confirmed
                or c.controller_firmware != profile.firmware
                or c.tcp_control_owned is not True
            )
        )
        or (c.user_frame, c.tool_frame) != (profile.safety.user_frame, profile.safety.tool_frame)
    ):
        raise ConfigurationError(
            "review firmware compatibility, ownership and explicit matching frames"
        )
    required_metadata = (
        "model",
        "tool_tcp_description",
        "payload_description",
        "rig_verification_reference",
        "interruption_verification_reference",
    )
    for name in required_metadata:
        value = str(getattr(profile, name)).strip()
        if value.upper() in {"UNKNOWN", "TODO", "TBD", "NONE", "N/A"} or value.upper().startswith(
            "MOCK"
        ):
            raise ConfigurationError(f"{name} is a placeholder, not reviewed rig data")
    if not local_micro_move:
        firmware = str(profile.firmware).strip()
        if firmware.upper() in {"UNKNOWN", "TODO", "TBD", "NONE", "N/A"}:
            raise ConfigurationError("firmware is a placeholder, not reviewed rig data")
    if profile.safety.max_translation_step < 0.010:
        raise ConfigurationError("reviewed displacement budget does not permit the 10mm experiment")


def collect_readiness(
    config: DobotConfig, profile: LiveMotionProfile
) -> tuple[OperatorReadiness, dict[str, bool]]:
    print(
        json.dumps(
            {
                "connection": asdict(config.connection) if config.connection else None,
                "profile": asdict(profile),
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    print(STOP_LIMITATION, flush=True)
    print(
        "Manual neutral reset must already be complete and settled. No automatic reset.", flush=True
    )
    operator = terminal_input("Operator name: ").strip()
    if not operator:
        raise MotionNotAuthorized("operator identity required")
    checks = {
        "tcp_control_owned": "TCP ownership is verified in the controller UI",
        "frames_verified": "base user=0, configured tool and physical TCP calibration match",
        "production_profile_verified": (
            "model, firmware compatibility, limits and review references are correct"
        ),
        "payload_verified": "the recorded tool/payload configuration matches the physical rig",
        "gripper_disabled": "gripper is inactive and will remain inactive",
        "workspace_clear": "workspace and the robot/link swept volume are clear",
        "operator_present": "you are physically beside the robot",
        "estop_tested": "physical E-stop is tested and reachable now",
        "best_effort_stop_accepted": (
            "you reviewed the referenced research safety model and best-effort Stop limits"
        ),
    }
    if config.local_micro_move is not None:
        checks["frames_verified"] = (
            "default base user0/tool0 flange TCP is used; gripper TCP is NOT calibrated"
        )
        checks["production_profile_verified"] = (
            "reviewed standalone +Z10mm envelope/engineering thresholds and the "
            "operator-approved unknown-firmware exception; this is NOT a general "
            "workspace or measured table Z"
        )
        checks["payload_verified"] = (
            "controller assumption 1.0kg/zero CoM is provisional, NOT measured calibration; "
            "gripper is empty and you accept this documented limitation"
        )
    print("Current-session readiness checklist (every item must be true):", flush=True)
    for number, (key, message) in enumerate(checks.items(), start=1):
        print(f"  {number}. {key}: {message}", flush=True)
    if terminal_input("If every item above is true, type CONFIRM once: ") != "CONFIRM":
        raise MotionNotAuthorized("operator did not confirm complete readiness; no retry")
    confirmed = {key: True for key in checks}
    confirmed["aggregate_confirmation"] = True
    readiness = OperatorReadiness(
        operator,
        confirmed["tcp_control_owned"],
        confirmed["frames_verified"],
        confirmed["production_profile_verified"],
        confirmed["estop_tested"],
        confirmed["operator_present"],
        confirmed["workspace_clear"],
        confirmed["gripper_disabled"],
    )
    readiness.require()
    return readiness, confirmed


def failure_alert(reason: str) -> None:
    print(
        f"ATTEMPT STOPPED: {reason}\n{STOP_LIMITATION}\nNo corrective move or retry.",
        file=sys.stderr,
        flush=True,
    )


def review_plan(
    plan: CartesianMotionPlan, driver: LiveDobotMotionDriver, expiry: float
) -> dict[str, Any]:
    return {
        "model": driver.profile.model,
        "firmware": driver.profile.firmware,
        "host": driver.connection.host,
        "robot_mode": int(plan.starting_measured_state.mode),
        "pose_before": asdict(plan.starting_measured_state),
        "direction": plan.micro_move_direction,
        "displacement_m": "0.010",
        "user": plan.request.user,
        "tool": plan.request.tool,
        "agent_target": plan.final_agent_pose,
        "native_target": asdict(plan.request),
        "translation_delta_m": translation_distance(
            plan.starting_measured_state.pose, plan.final_pose_si
        ),
        "angular_delta_rad": orientation_distance(
            plan.starting_measured_state.pose, plan.final_pose_si
        ),
        "MovL": plan.request.serialize(),
        "speed_percent": plan.request.speed_percent,
        "acceleration_percent": plan.request.acceleration_percent,
        "cp": 0,
        "plan_digest": plan_digest(plan, driver.profile, driver.connection),
        "authority_expires_at_monotonic": expiry,
        "authority_lifetime_seconds": driver.profile.authority_lifetime,
        "authority_issued": False,
        "PHYSICAL COMMAND COUNT": 1,
        "plan": asdict(plan),
        "local_envelope": asdict(plan.local_envelope) if plan.local_envelope else None,
        "protocol_compatibility_confirmed": driver.connection.protocol_compatibility_confirmed,
    }


def _summarize(driver: LiveDobotMotionDriver, plan: CartesianMotionPlan | None) -> dict[str, Any]:
    result = driver.result
    records = driver.audit_records
    wire = [e for e in records if e["kind"] == "wire"]
    moves = [e for e in wire if e["command"].startswith("MovL(")]
    after = result.final_sample if result else None
    summary: dict[str, Any] = {
        "result": asdict(result) if result else None,
        "state": driver.state.value,
        "motion_write_attempts": sum(bool(e.get("write_attempted")) for e in moves),
        "motion_writes_completed": sum(bool(e.get("write_completed")) for e in moves),
        "motion_acknowledged": sum("result_id" in e and e.get("error_id") == 0 for e in moves),
        "accepted_result_id": result.command_id if result else None,
        "connections_closed": driver.connections_closed,
        "standstill": "CONFIRMED" if result and result.standstill_confirmed else "UNKNOWN",
        "transport_is_mock": driver.transport_is_mock,
        "request_control_attempted": False,
        "pose_before": asdict(plan.starting_measured_state) if plan else None,
        "pose_target": asdict(plan.final_pose_si) if plan else None,
        "pose_after": asdict(after) if after else None,
        "audit": records,
    }
    if plan and after:
        summary.update(
            delta_commanded_m=[
                b - a
                for a, b in zip(
                    plan.starting_measured_state.pose.xyz, plan.final_pose_si.xyz, strict=True
                )
            ],
            delta_measured_m=[
                b - a
                for a, b in zip(plan.starting_measured_state.pose.xyz, after.pose.xyz, strict=True)
            ],
            position_residual_m=translation_distance(after.pose, plan.final_pose_si),
            orientation_residual_rad=orientation_distance(after.pose, plan.final_pose_si),
        )
    checks: dict[str, bool] = {
        "one_motion_write": summary["motion_write_attempts"]
        == summary["motion_writes_completed"]
        == 1,
        "one_valid_result_id": summary["motion_acknowledged"] == 1
        and result is not None
        and result.command_id is not None,
        "matching_current_command_id": bool(
            after and result and after.command_id == result.command_id
        ),
        "enabled_idle": bool(after and after.mode is RobotMode.ENABLED_IDLE),
        "position_residual": bool(
            after
            and plan
            and translation_distance(after.pose, plan.final_pose_si)
            <= driver.profile.safety.position_tolerance
        ),
        "orientation_residual": bool(
            after
            and plan
            and orientation_distance(after.pose, plan.final_pose_si)
            <= driver.profile.safety.orientation_tolerance
        ),
        "fresh_consecutive_settle": bool(
            result and result.state is LiveMotionState.COMPLETED and result.failure_cause is None
        ),
        "no_observed_error_collision": bool(
            after and not after.errors and after.mode is RobotMode.ENABLED_IDLE
        ),
        "exact_10mm_fixed_native_orientation": bool(
            plan
            and plan.micro_move_direction
            and plan.request.native_decimal
            and plan.starting_measured_state.native_decimal
            and plan.request.native_decimal
            == plan.starting_measured_state.native_decimal.translated_10mm(
                plan.micro_move_direction
            )
        ),
        "no_automatic_lifecycle": all(
            e["command"].startswith(
                (
                    "RobotMode(",
                    "GetPose(",
                    "GetAngle(",
                    "GetErrorID(",
                    "GetCurrentCommandID(",
                    "MovL(",
                    "Stop(",
                )
            )
            for e in wire
        ),
        "connections_closed": driver.connections_closed,
    }
    summary["checks"] = checks
    return summary


def run_terminal_trial(
    config: DobotConfig,
    profile: LiveMotionProfile,
    evidence_dir: Path,
    *,
    allow_motion: bool,
    _socket_factory: SocketFactory = open_socket,
    _clock: Clock | None = None,
) -> dict[str, Any]:
    """The console supplies no injection flags. Private injection is offline-test-only.

    Even injected tests traverse every input/readiness/review gate and are labelled
    MOCK_OK, never hardware PASS. Inputs always come from an interactive terminal.
    """
    validate_live_inputs(config, profile, allow_motion)
    assert config.connection is not None
    clock = _clock or SystemClock()
    try:
        evidence = AttemptEvidence(evidence_dir)
    except (OSError, EvidenceError) as exc:
        raise ConfigurationError(
            f"evidence destination unavailable before connection: {exc}"
        ) from exc
    driver = LiveDobotMotionDriver(
        config.connection,
        profile,
        allow_motion=allow_motion,
        socket_factory=_socket_factory,
        clock=clock,
        transport_is_mock=_socket_factory is not open_socket,
        event_sink=evidence.record,
        failure_alert=failure_alert,
    )
    plan = None
    start: RobotSnapshot | None = None
    error = None
    preflight: dict[str, Any] = {
        "config": asdict(config),
        "profile": asdict(profile),
        "status": "BLOCKED",
        "hardware_validation": "NOT RUN",
        "stop_policy": STOP_LIMITATION,
        "transport_is_mock": driver.transport_is_mock,
    }
    try:
        evidence.write("preflight.json", preflight)
        readiness, attestations = collect_readiness(config, profile)
        preflight.update(readiness=asdict(readiness), attestations=attestations)
        evidence.write("preflight.json", preflight)
        evidence.require_healthy()
        driver.connect(allow_connection=True)
        driver.confirm_readiness(readiness)
        start = driver.measure_start()
        profile = driver.profile
        envelope = (
            profile.safety.envelope if isinstance(profile.safety, LocalEnvelopeSafety) else None
        )
        preflight.update(
            initial_measurement=asdict(start),
            status="READ_ONLY_PRECHECK_OK",
            bound_profile=asdict(profile),
        )
        evidence.write("preflight.json", preflight)
        print(json.dumps({"measured_start": asdict(start)}, indent=2, allow_nan=False), flush=True)
        direction = terminal_input(
            "Visually confirm +Z is upward and the entire 10mm path is clear. Type exactly +Z: "
            if envelope is not None
            else (
                "After inspecting clearance, choose exactly +X/-X/+Y/-Y/+Z/-Z "
                "(10mm, user/base axes): "
            )
        )
        if envelope is not None and (
            not math.isfinite(clock.monotonic()) or clock.monotonic() >= envelope.expires_at
        ):
            raise MotionNotAuthorized(
                "local envelope expired during visual review; no regeneration"
            )
        plan = build_micro_move_plan(
            envelope.chunk_id if envelope else uuid4().hex,
            start,
            direction,
            driver.validation_safety,
            # Freshness was checked when bound. Human review may exceed one sample
            # age; arm/execute MUST remeasure without changing this fixed target.
            envelope.bound_at if envelope else clock.monotonic(),
            profile.keepouts,
        )
        expiry = envelope.expires_at if envelope else clock.monotonic() + profile.authority_lifetime
        review = review_plan(plan, driver, expiry)
        evidence.write("reviewed_plan.json", review)
        evidence.require_healthy()
        print(json.dumps({"review": review}, indent=2, allow_nan=False), flush=True)
        print("PHYSICAL COMMAND COUNT = 1", flush=True)
        phrase = terminal_input("Review every value above. Type exactly MOVE ONCE: ")
        evidence.require_healthy()
        authority = driver.arm(plan, confirmation=phrase, expires_at=expiry)
        evidence.write(
            "execution.json", {"status": "ARMED", "authority": asdict(authority.details)}
        )
        evidence.require_healthy()
        driver.execute(plan, authority)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        failure_alert(error)
    finally:
        try:
            driver.close()
        except Exception as exc:
            error = error or f"close failed: {exc}"
    report = _summarize(driver, plan)
    report["reason"] = error or (
        driver.result.failure_cause if driver.result else "attempt ended before send"
    )
    successful = bool(
        not error
        and driver.result
        and driver.result.state is LiveMotionState.COMPLETED
        and driver.result.failure_cause is None
        and report["motion_write_attempts"]
        == report["motion_writes_completed"]
        == report["motion_acknowledged"]
        == 1
        and report["connections_closed"]
        and all(report["checks"].values())
        and evidence.failed is None
    )
    report["status"] = (
        ("MOCK_OK" if driver.transport_is_mock else "PASS")
        if successful
        else ("FAILED" if report["motion_write_attempts"] else "BLOCKED")
    )
    report["hardware_validation"] = "NOT RUN" if driver.transport_is_mock else report["status"]
    report["PHYSICAL_SEND"] = bool(report["motion_write_attempts"] and not driver.transport_is_mock)
    report["physical_MovL_writes_completed"] = (
        0 if driver.transport_is_mock else report["motion_writes_completed"]
    )
    report["evidence_complete"] = False
    report["checks"]["complete_evidence"] = False
    try:
        evidence.write(
            "settle_samples.json",
            [
                e
                for e in evidence.events
                if e["kind"] in ("measurement", "residual", "standstill_sample", "controller_fault")
            ],
        )
        evidence.write("execution.json", report)
        evidence.require_healthy()
        report["evidence_complete"] = True
        report["checks"]["complete_evidence"] = True
        # A hardware PASS requires BOTH this report and execution.json to agree.
        # Failure/crash between writes leaves a visibly incomplete pair.
        evidence.write("final_report.md", _final_markdown(report))
        evidence.require_healthy()
        evidence.write("execution.json", report)
        evidence.require_healthy()
    except (OSError, ValueError, EvidenceError) as exc:
        report.update(
            status="FAILED" if report["motion_write_attempts"] else "BLOCKED",
            evidence_complete=False,
            evidence_error=str(exc),
        )
        report["hardware_validation"] = "NOT RUN" if driver.transport_is_mock else report["status"]
        report["checks"]["complete_evidence"] = False
        # Best effort record of persistence failure. No reconnect or compensating move.
        evidence.write("execution.json", report)
        evidence.write("final_report.md", _final_markdown(report))
        failure_alert(str(exc))
    report["evidence_dir"] = str(evidence_dir)
    return report


def _final_markdown(report: dict[str, Any]) -> str:
    return (
        f"# Phase 4B — {report['status']}\n\n"
        f"Hardware validation: {report['hardware_validation']}.\n\n"
        f"{STOP_LIMITATION}\n\n"
        "No automatic lifecycle, recovery, reconnect, correction or second motion.\n"
        "Dashboard mode/alarm samples cannot rule out unobserved transient collisions.\n\n"
        "```json\n"
        + json.dumps({k: v for k, v in report.items() if k != "audit"}, indent=2, allow_nan=False)
        + "\n```\n\nSTOP. Human inspection/review required before any later trial.\n"
    )
