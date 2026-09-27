"""Metadata-only compatibility; never connects a driver, camera or model provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inspect_robots.compat import check_compatibility
from inspect_robots.conformance import check_embodiment

from .config import DobotConfig, load_config
from .embodiment import EEF_DIM_LABELS, DobotEmbodiment
from .errors import ConfigurationError


def run_readonly_preflight(config: DobotConfig) -> dict[str, Any]:
    """Validate connection metadata only; never construct a client or infer motion readiness."""
    connection = config.connection
    errors: list[str] = []
    warnings: list[str] = []
    if connection is None or connection.host is None:
        errors.append("read-only diagnostics require an explicit connection.host")
    if connection is not None:
        if connection.expected_protocol_version != "4.6.5":
            errors.append("only protocol document 4.6.5 is implemented")
        if not connection.protocol_compatibility_confirmed:
            warnings.append("actual firmware/document compatibility remains unverified")
        if connection.tcp_control_owned is not True:
            warnings.append("TCP ownership is unknown or declared unowned; no change is authorized")
    if config.camera is not None:
        warnings.append("physical camera backend is unavailable; preflight never opens a camera")
    return {
        "ok": not errors,
        "mode": "OFFLINE READ-ONLY CONFIGURATION",
        "scope": "connection metadata only; not action compatibility or hardware readiness",
        "host": None if connection is None else connection.host,
        "expected_protocol_version": (
            None if connection is None else connection.expected_protocol_version
        ),
        "hardware_connected": False,
        "motion_enabled": False,
        "motion_ready": False,
        "motion_commands_sent": 0,
        "request_control_attempted": False,
        "hardware_readiness_verified": False,
        "errors": errors,
        "warnings": warnings,
    }


def run_preflight(config: DobotConfig | None = None) -> dict[str, Any]:
    embodiment = DobotEmbodiment(config)
    report = check_embodiment(embodiment.info)
    errors = [i.message for i in report.issues if i.severity == "error"]
    warnings = [i.message for i in report.issues if i.severity == "warning"]
    if embodiment.config.control_hz is None:
        errors.append("explicit control_hz is required; no hardware rate is inferred")
    agent_compatible = False
    if not errors:
        try:
            import httpx
            from inspect_robots_agent import LLMAgentPolicy

            def prohibit_inference(request: httpx.Request) -> httpx.Response:
                raise RuntimeError("metadata preflight must never perform inference")

            policy = LLMAgentPolicy(
                model="offline-preflight",
                base_url="http://preflight.invalid/v1",
                env={},
                transport=httpx.MockTransport(prohibit_inference),
            )
            policy.bind(embodiment.info)
            compatibility = check_compatibility(policy, embodiment)
            errors.extend(i.message for i in compatibility.errors)
            warnings.extend(i.message for i in compatibility.warnings)
            agent_compatible = compatibility.ok
        except ImportError:
            errors.append(
                "agent package unavailable; provision pinned agent extra for compatibility checks"
            )
        except (ValueError, RuntimeError) as exc:
            errors.append(f"agent compatibility: {exc}")
    return {
        "ok": not errors,
        "mode": "OFFLINE METADATA ONLY",
        "phase": "2C",
        "action_dim": embodiment.info.action_space.dim,
        "action_labels": EEF_DIM_LABELS,
        "agent_compatible": agent_compatible,
        "hardware_connected": False,
        "motion_enabled": False,
        "motion_commands_sent": 0,
        "hardware_readiness_verified": False,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="explicit JSON metadata/safety configuration")
    parser.add_argument(
        "--read-only", action="store_true", help="validate connection metadata only, with no I/O"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config) if args.config else DobotConfig()
        report = run_readonly_preflight(config) if args.read_only else run_preflight(config)
    except ConfigurationError as exc:
        report = {
            "ok": False,
            "hardware_connected": False,
            "motion_commands_sent": 0,
            "errors": [str(exc)],
            "warnings": [],
        }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("OFFLINE METADATA ONLY — NO MOTION COMMANDS SENT")
        success = (
            "READ-ONLY CONFIG VALID (hardware unverified)"
            if args.read_only
            else "COMPATIBLE (not hardware readiness)"
        )
        print(success if report["ok"] else "NOT READY")
        for message in report["errors"]:
            print(f"ERROR: {message}")
        for message in report["warnings"]:
            print(f"WARNING: {message}")
    return 0 if report["ok"] else 1
