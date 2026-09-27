"""Command line entry point for Phase 6A Astra shadow evaluation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .astra_shadow import (
    CAMERA_ORDER,
    ShadowExecutor,
    run_fixture_shadow,
)
from .official_agent_shadow import (
    OFFICIAL_EFFORT,
    OFFICIAL_IMAGE_HORIZON,
    OFFICIAL_IMAGES,
    OFFICIAL_MODEL,
    MissingOpenAIKey,
    OfficialAgentConfig,
    run_official_shadow,
)


def _replay_response(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("replay response must be a JSON object")
    return data


def _replay_images(data: dict[str, Any], base: Path) -> dict[str, np.ndarray] | None:
    images = data.get("images")
    if images is None:
        return None
    if not isinstance(images, dict) or tuple(images) != CAMERA_ORDER:
        raise ValueError("replay images must name front_rgb, right_rgb, wrist_rgb in order")
    from .camera_v4l2 import decode_jpeg_ffmpeg

    sizes = {"front_rgb": (640, 480), "right_rgb": (640, 480), "wrist_rgb": (1280, 720)}
    result: dict[str, np.ndarray] = {}
    for name in CAMERA_ORDER:
        value = images[name]
        if not isinstance(value, str):
            raise ValueError(f"replay image path for {name} must be a string")
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        width, height = sizes[name]
        result[name] = decode_jpeg_ffmpeg(path.read_bytes(), width, height, timeout=5.0)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inspect-robots-dobot-astra-shadow")
    parser.add_argument(
        "--provider",
        choices=("fake", "official-agent"),
        default="fake",
        help="explicit deterministic fixture provider or pinned inspect-robots-agent",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--fixture", action="store_true", help="run the offline synthetic observation"
    )
    modes.add_argument(
        "--replay", type=Path, help="JSON file containing one structured FakeAstra response"
    )
    modes.add_argument(
        "--live-camera",
        action="store_true",
        help="reserved; physical camera mode is disabled in Phase 6A CLI",
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="bounded repeated static queries (1-10)"
    )
    parser.add_argument("--json", action="store_true", help="print JSON records")
    parser.add_argument("--evidence-dir", type=Path, default=Path("run-evidence/shadow"))
    parser.add_argument("--model", default=OFFICIAL_MODEL)
    parser.add_argument("--base-url", help="explicit upstream provider/gateway base URL")
    parser.add_argument("--api-key-env", help="gateway credential environment variable NAME")
    parser.add_argument("--wire", default="responses")
    parser.add_argument("--effort", default=OFFICIAL_EFFORT)
    parser.add_argument("--images", default=OFFICIAL_IMAGES)
    parser.add_argument("--image-horizon", type=int, default=OFFICIAL_IMAGE_HORIZON)
    parser.add_argument("--max-llm-calls", type=int, default=1)
    parser.add_argument("--max-speed-frac", type=float, default=0.1)
    parser.add_argument(
        "--mock-response",
        type=Path,
        help="official-agent only: JSON Responses output for deterministic offline testing",
    )
    args = parser.parse_args(argv)
    if args.steps < 1 or args.steps > 10:
        parser.error("--steps must be in [1,10]")
    if args.live_camera:
        parser.error("Phase 6A live-camera mode is disabled until an explicit reviewed gate")

    records = []
    if args.provider == "official-agent" and args.fixture:
        parser.error("official-agent requires --replay; --fixture is reserved for --provider fake")
    if args.provider == "fake" and args.mock_response:
        parser.error("--mock-response requires --provider official-agent")

    if args.fixture:
        for _ in range(args.steps):
            records.append(run_fixture_shadow())
    else:
        # Replay consumes a structured response while retaining the fixture's
        # robot state. It never opens a camera or robot connection.
        replay_data = _replay_response(args.replay)
        replay_images = _replay_images(replay_data, args.replay.parent)
        if args.provider == "fake":
            response = replay_data.get("response", replay_data)
            if not isinstance(response, dict):
                raise ValueError("replay response must be a JSON object")
            for _ in range(args.steps):
                records.append(run_fixture_shadow(response=response, images=replay_images))
        else:
            mocked: dict[str, Any] | None = None
            if args.mock_response is not None:
                mocked = json.loads(args.mock_response.read_text(encoding="utf-8"))
                if not isinstance(mocked, dict):
                    raise ValueError("--mock-response must contain a JSON object")
            config = OfficialAgentConfig(
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                wire=args.wire,
                effort=args.effort,
                images=args.images,
                image_horizon=args.image_horizon,
                max_llm_calls=args.max_llm_calls,
                max_speed_frac=args.max_speed_frac,
            )
            try:
                for _ in range(args.steps):
                    records.append(
                        run_official_shadow(
                            response_json=mocked,
                            images=replay_images,
                            config=config,
                            env=dict(os.environ),
                            require_live_key=mocked is None,
                        )
                    )
            except MissingOpenAIKey as exc:
                blocked = {
                    "phase": "6A.1",
                    "provider": "official-agent",
                    "status": "BLOCKED_MISSING_API_KEY",
                    "message": str(exc),
                    "execution": False,
                    "evidence": [],
                    "openai_live_requests": 0,
                    "nova_connections": 0,
                    "gripper_commands_sent": 0,
                }
                if args.json:
                    print(json.dumps(blocked, indent=2, sort_keys=True))
                else:
                    print(blocked["status"])
                    print(blocked["message"])
                return 1

    written = []
    for record in records:
        written.append(str(ShadowExecutor.write(record, args.evidence_dir)))
    output = {
        "phase": "6A.1" if args.provider == "official-agent" else "6A",
        "provider": args.provider,
        "execution": False,
        "records": [asdict(record) for record in records],
        "evidence": written,
        "astra_live": "BLOCKED_BY_CONFIGURATION"
        if args.provider == "fake"
        else "MOCKED"
        if args.mock_response
        else "QUERY_PERFORMED",
    }
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        for record in records:
            result = record.hypothetical_movl or record.rejection_reason
            print(f"{record.proposed_high_level_action} {record.selected_candidate}: {result}")
        print(f"evidence: {', '.join(written)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
