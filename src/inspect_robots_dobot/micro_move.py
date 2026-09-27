"""Explicit mock rehearsal or supervised terminal-only one-shot live experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .clock import Clock, SystemClock
from .config import ConnectionConfig, DobotConfig, SafetyProfile, load_config
from .errors import ConfigurationError, MotionNotAuthorized
from .live_authority import plan_digest
from .live_driver import LiveDobotMotionDriver, LiveMotionState
from .live_mock import MockMotionSocket
from .live_profile import LiveMotionProfile, OperatorReadiness, load_live_profile
from .motion import build_motion_plan
from .motion_dry_run import _deny_hardware
from .types import PoseSI
from .units import orientation_distance, translation_distance

LIVE_BLOCKER = (
    "Phase 4B BLOCKED: choose explicit --dry-run or --live; no socket opened. "
    "Software interruption is best-effort; live additionally requires reviewed rig data, "
    "--allow-motion and current terminal operator confirmations."
)


def terminal_confirmation(review: dict[str, Any]) -> str:
    print(json.dumps({"review": review}, indent=2, allow_nan=False), flush=True)
    print("PHYSICAL COMMAND COUNT = 1 (future maximum; this mock sends zero physical commands)")
    if not sys.stdin.isatty():
        raise MotionNotAuthorized("confirmation requires an interactive terminal, not piped input")
    return input("Review the mock one-shot plan. Type MOVE ONCE: ")


def rehearse_micro_move(
    config: DobotConfig,
    profile: LiveMotionProfile,
    initial: PoseSI,
    delta: tuple[float, float, float],
    *,
    host_label: str,
    allow_motion: bool,
    confirm: Callable[[dict[str, Any]], str],
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Explicit mock transport only. No live factory parameter or network fallback.

    confirm is an application/test callback; the console always uses a terminal.
    Configuration and environment values are never accepted as confirmation.
    """
    if (
        config.connection is not None
        or config.safety != profile.safety
        or not isinstance(profile.safety, SafetyProfile)
    ):
        raise ConfigurationError(
            "rehearsal requires matching synthetic safety, no connection config"
        )
    clock = clock or SystemClock()
    connection = ConnectionConfig(
        host=host_label,
        controller_firmware=profile.firmware,
        protocol_compatibility_confirmed=True,
        tcp_control_owned=True,
        user_frame=profile.safety.user_frame,
        tool_frame=profile.safety.tool_frame,
    )
    stream = MockMotionSocket(initial)
    connections = 0

    def mock_factory(host: str, port: int, timeout: float) -> MockMotionSocket:
        nonlocal connections
        if connections or host != host_label or port != 29999 or timeout <= 0:
            raise ConfigurationError("unexpected mock connection")
        connections += 1
        return stream

    driver = LiveDobotMotionDriver(
        connection,
        profile,
        allow_motion=allow_motion,
        socket_factory=mock_factory,
        transport_is_mock=True,
        clock=clock,
    )
    report: dict[str, Any] = {
        "status": "BLOCKED",
        "PHYSICAL_SEND": False,
        "motion_commands_sent": 0,
        "robot_connections_opened": 0,
        "motion_ready": False,
        "request_control_attempted": False,
        "mock_transport": True,
        "evidence_provenance": "ALL POSE/JOINT/RIG/OWNERSHIP EVIDENCE IS SYNTHETIC",
        "phase4b_blocker": LIVE_BLOCKER,
    }
    try:
        driver.connect(allow_connection=True)
        driver.confirm_readiness(
            OperatorReadiness(
                "MOCK_OPERATOR",
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            )
        )
        start = driver.measure_start()
        target = tuple(v + d for v, d in zip(start.pose.xyz, delta, strict=True)) + (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        plan = build_motion_plan(
            uuid4().hex,
            (target,),
            start,
            start.pose,
            profile.safety,
            clock.monotonic(),
            profile.keepouts,
        )
        expiry = clock.monotonic() + profile.authority_lifetime
        review = {
            "host": host_label,
            "host_is_label_only": True,
            "robot_mode": int(start.mode),
            "current_native_pose_si": start.pose.values,
            "current_agent_state": (*start.pose.xyz, 0.0, 0.0, 0.0, 0.0),
            "user": plan.request.user,
            "tool": plan.request.tool,
            "exact_agent_target": plan.final_agent_pose,
            "exact_native_target": plan.request.pose.values,
            "translation_delta_m": translation_distance(start.pose, plan.final_pose_si),
            "angular_delta_rad": orientation_distance(start.pose, plan.final_pose_si),
            "MovL": plan.request.serialize(),
            "speed_percent": plan.request.speed_percent,
            "acceleration_percent": plan.request.acceleration_percent,
            "cp": 0,
            "plan_digest": plan_digest(plan, profile, connection),
            "authority_expires_at_monotonic": expiry,
            "authority_lifetime_seconds": profile.authority_lifetime,
            "authority_issued": False,
            "runtime_allow_motion": allow_motion,
            "PHYSICAL COMMAND COUNT": 1,
            "PHYSICAL_SEND": False,
            "provenance": "MOCK ONLY; no physical verification or authority",
        }
        report["review"] = review
        phrase = confirm(review)
        authority = driver.arm(plan, confirmation=phrase, expires_at=expiry)
        report["authority_details"] = asdict(authority.details)
        stream.expected_motion = plan.request.serialize().encode()
        stream.target = plan.final_pose_si
        result = driver.execute(plan, authority)
        report.update(
            status="DRY_RUN_OK"
            if result.state is LiveMotionState.COMPLETED and result.failure_cause is None
            else "BLOCKED",
            result=asdict(result),
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        driver.close()
        report.update(
            audit=driver.audit_records,
            connections_closed=stream.closed,
            mock_connections=connections,
            mock_motion_writes=sum(c.startswith(b"MovL(") for c in stream.sent),
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="in-memory transport only")
    modes.add_argument("--live", action="store_true", help="one supervised real 10mm experiment")
    parser.add_argument("--allow-motion", action="store_true", help="explicit runtime motion gate")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--host", help="mock label; live host must match the reviewed connection config"
    )
    parser.add_argument("--initial-native-si", nargs=6, type=float)
    parser.add_argument("--delta-xyz", nargs=3, type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evidence-dir", type=Path, help="required NEW directory for live evidence"
    )
    parser.add_argument("--json", action="store_true", help="print structured review and result")
    args = parser.parse_args(argv)
    # Also protects dependencies and accidental future regressions. Installed before
    # profile loading; absence of --dry-run never authorizes a network operation.
    if not args.live:
        sys.addaudithook(_deny_hardware)
    if not args.dry_run and not args.live:
        print(json.dumps({"status": "BLOCKED", "reason": LIVE_BLOCKER, "PHYSICAL_SEND": False}))
        return 2
    if args.live:
        from .live_trial import run_terminal_trial

        try:
            if args.initial_native_si is not None or args.delta_xyz is not None or args.output:
                raise ConfigurationError(
                    "live forbids synthetic starts, arbitrary deltas and --output; "
                    "use --evidence-dir"
                )
            if not all((args.config, args.profile, args.evidence_dir)):
                raise ConfigurationError(
                    "live requires --config, --profile and a NEW --evidence-dir"
                )
            config = load_config(args.config)
            settings = config.safety or config.local_micro_move
            if settings is None or config.connection is None:
                raise ConfigurationError("live requires complete connection and safety settings")
            if args.host is not None and args.host != config.connection.host:
                raise ConfigurationError(
                    "--host must match reviewed configuration, never override it"
                )
            profile = load_live_profile(args.profile, settings)
        except (Exception, KeyboardInterrupt) as exc:
            report = {"status": "BLOCKED", "reason": str(exc), "PHYSICAL_SEND": False}
        else:
            try:
                report = run_terminal_trial(
                    config, profile, args.evidence_dir, allow_motion=args.allow_motion
                )
            except (Exception, KeyboardInterrupt) as exc:
                # A workflow failure must never be misreported as proof of no send.
                # The durable journal is the source of truth for partial attempts.
                from .live_trial import failure_alert

                failure_alert(str(exc))
                report = {
                    "status": "BLOCKED"
                    if isinstance(exc, (ConfigurationError, MotionNotAuthorized, FileExistsError))
                    else "FAILED",
                    "reason": str(exc),
                    "PHYSICAL_SEND": False
                    if isinstance(exc, (ConfigurationError, MotionNotAuthorized, FileExistsError))
                    else None,
                    "instruction": "Inspect evidence; no retry. Hardware outcome UNKNOWN.",
                }
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["status"] in ("PASS", "MOCK_OK") else 2
    try:
        if not all((args.config, args.profile, args.initial_native_si, args.delta_xyz)):
            raise ConfigurationError(
                "dry-run requires --config, --profile, --initial-native-si, --delta-xyz"
            )
        config = load_config(args.config)
        if config.safety is None:
            raise ConfigurationError("explicit synthetic safety required")
        profile = load_live_profile(args.profile, config.safety)
        # The driver catches Ctrl-C while executing and attempts Stop on the worker's
        # stream. Before execution a KeyboardInterrupt cancels without sending motion.
        report = rehearse_micro_move(
            config,
            profile,
            PoseSI(*args.initial_native_si),
            tuple(args.delta_xyz),
            host_label=args.host or "192.0.2.1",
            allow_motion=args.allow_motion,
            confirm=terminal_confirmation,
        )
    except (Exception, KeyboardInterrupt) as exc:
        report = {"status": "BLOCKED", "reason": str(exc), "PHYSICAL_SEND": False}
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0 if report["status"] == "DRY_RUN_OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
