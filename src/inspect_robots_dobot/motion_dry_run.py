"""Offline CLI: scripted generic agent -> actual rollout -> one prospective MovL.

There is no live mode, host option or motion-enabling flag. The only authority
accepted is an explicit simulation flag. All model responses are in-memory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from inspect_robots.logging.json_log import JsonLogSink
from inspect_robots.scene import Scene
from inspect_robots.task import Task

from .camera import FakeCamera
from .chunks import DobotExecutionSession, StagedDobotEmbodiment
from .clock import FakeClock
from .config import DobotConfig, load_config
from .driver import FakeDobotDriver
from .errors import ConfigurationError, MotionNotAuthorized
from .safety import MotionAuthority
from .types import PoseSI


def run_dry_run(
    config: DobotConfig,
    initial_pose: PoseSI,
    targets: dict[str, float],
    *,
    allow_fake_motion: bool,
    log_dir: Path,
) -> dict[str, Any]:
    """No connection config is accepted, even if its host would otherwise be unused."""
    if allow_fake_motion is not True:
        raise MotionNotAuthorized("explicit --allow-fake-motion is required for simulation")
    if not all(math.isfinite(v) for v in initial_pose.values):
        raise ConfigurationError("initial synthetic pose must contain six finite SI values")
    if config.connection is not None:
        raise ConfigurationError("dry-run config must not contain a connection section")
    if config.safety is None or config.control_hz is None:
        raise ConfigurationError("dry-run requires explicit synthetic safety and control_hz")
    if not targets or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in targets.values()
    ):
        raise ConfigurationError("targets must be a nonempty object of finite numeric values")
    try:
        import httpx
        from inspect_robots_agent import LLMAgentPolicy
    except ImportError as exc:
        raise ConfigurationError(
            "install the optional agent dependencies before offline use"
        ) from exc

    clock = FakeClock(1.0)
    driver = FakeDobotDriver(
        initial_pose=initial_pose,
        initial_joints=(0.0,) * 6,
        profile=config.safety,
        clock=clock,
        authority=MotionAuthority(True),
        convergence_delay=0.05,
    )
    camera = FakeCamera(config.camera, clock) if config.camera is not None else None
    embodiment = StagedDobotEmbodiment(config, driver=driver, camera=camera)
    requests = 0

    def scripted(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        # Exactly one proposed move_to. Stop if rejected; no retry/corrective target.
        name = "move_to" if requests == 1 else "done"
        arguments = (
            {"targets": targets, "note": "Explicit offline synthetic target."}
            if requests == 1
            else {"summary": "End the single-proposal offline demonstration."}
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"offline-{requests}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    policy = LLMAgentPolicy(
        model="offline-scripted",
        base_url="http://offline.invalid/v1",
        env={},
        transport=httpx.MockTransport(scripted),
        pre_check=embodiment.pre_check,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    with DobotExecutionSession(embodiment) as session:
        logs = session.evaluate(
            Task(
                name="dobot-motion-dry-run",
                scenes=[
                    Scene(
                        id="synthetic", instruction="One scripted fake XYZ motion; never physical."
                    )
                ],
                scorer=[],
                max_steps=512,
            ),
            policy=policy,
            log_dir=str(log_dir),
            sinks=[JsonLogSink(str(log_dir))],
        )
    success = (
        len(embodiment.plans) == len(embodiment.execution_results) == 1
        and bool(logs)
        and all(log.status == "success" for log in logs)
    )
    return {
        "status": "DRY_RUN_OK" if success else "BLOCKED",
        "notice": "DRY RUN ONLY — NO MOTION COMMAND SENT",
        "PHYSICAL_SEND": False,
        "physical_authorized": False,
        "motion_ready": False,
        "simulation_authorized": True,
        "robot_connections_opened": 0,
        "request_control_attempted": False,
        "motion_commands_sent": 0,
        "connections_closed": True,
        "initial_agent_state": [*initial_pose.xyz, 0.0, 0.0, 0.0, 0.0],
        "initial_native_si": initial_pose.values,
        "requested_targets": targets,
        "prospective_movl_count": len(embodiment.plans),
        "plans": [asdict(plan) for plan in embodiment.plans],
        "commands_would_send": [plan.request.serialize() for plan in embodiment.plans],
        "results": [asdict(result) for result in embodiment.execution_results],
        "audit": embodiment.audit_records,
        "framework_status": [log.status for log in logs],
        "framework_log_dir": str(log_dir.resolve()),
        "model_api_calls": 0,
        "scripted_model_responses": requests,
    }


def _deny_hardware(event: str, args: tuple[Any, ...]) -> None:
    """Process-level backstop for the CLI, including accidental dependency I/O."""
    if event.startswith("socket."):
        raise RuntimeError("dry-run prohibits all socket operations")
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        name = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if name.startswith(("/dev/video", "/dev/media", "/dev/v4l", "/dev/bus/usb")):
            raise RuntimeError("dry-run prohibits physical camera/device access")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--initial-native-si",
        nargs=6,
        type=float,
        required=True,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="synthetic measured native pose: metres and native radians",
    )
    parser.add_argument("--targets", required=True, help='agent target JSON, e.g. {"x":0.315}')
    parser.add_argument("--allow-fake-motion", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("run-logs"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    sys.addaudithook(_deny_hardware)
    try:
        targets = json.loads(args.targets)
        if not isinstance(targets, dict):
            raise ConfigurationError("--targets must be a JSON object")
        report = run_dry_run(
            load_config(args.config),
            PoseSI(*args.initial_native_si),
            targets,
            allow_fake_motion=args.allow_fake_motion,
            log_dir=args.log_dir,
        )
    except Exception as exc:
        report = {
            "status": "BLOCKED",
            "reason": str(exc),
            "PHYSICAL_SEND": False,
            "motion_commands_sent": 0,
            "motion_ready": False,
        }
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print("DRY RUN ONLY — NO MOTION COMMAND SENT")
        print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "DRY_RUN_OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
