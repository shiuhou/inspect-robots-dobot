"""Offline fixture/schema and explicit API-only commands. No hardware entry point."""

from __future__ import annotations

import argparse
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .api import (
    DreamIntegrationError,
    InferenceContract,
    compatibility,
    connect_client,
    discover,
    infer_once,
    nova2_config,
    validate_local_schema,
)
from .observation import fixture
from .records import load_record, save_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, help="private raw key file; never printed")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("schema")
    fixture_parser = sub.add_parser("fixture")
    fixture_parser.add_argument("--output", type=Path, required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--embodiment-id", default="unregistered-nova2")
    inference = sub.add_parser("infer-recorded")
    inference.add_argument("--observation", type=Path, required=True)
    inference.add_argument("--contract", type=Path, required=True)
    inference.add_argument("--prompt", required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--timeout-ms", type=int, default=15000)
    args = parser.parse_args()
    client: Any = None
    try:
        if args.command == "schema":
            validate_local_schema()
            print(
                json.dumps(
                    {
                        "local_sdk_validation": "PASS",
                        "server_registered": False,
                        "checkpoint_compatible": False,
                        "config": nova2_config(),
                    }
                )
            )
            return 0
        if args.command == "fixture":
            obs, configs = fixture()
            save_record(args.output, obs, configs, max_age=0.5, max_skew=0.1)
            print(json.dumps({"fixture_validation": "PASS", "execution": "DISCONNECTED"}))
            return 0
        if args.key_file:
            info = args.key_file.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise DreamIntegrationError("key file must be user-owned regular file, mode 0600")
            os.environ["DREAM_API_KEY"] = args.key_file.read_text().strip()
        client = connect_client()
        if args.command == "probe":
            print(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": m.id,
                                "embodiment": m.embodiment,
                                "servable": m.servable,
                                "cameras": m.cameras,
                                "nova2_blocker": compatibility(m, args.embodiment_id),
                            }
                            for m in discover(client)
                        ]
                    },
                    indent=2,
                )
            )
            return 0
        obs, configs, max_age, max_skew = load_record(args.observation)
        contract = InferenceContract(**json.loads(args.contract.read_text()))
        if args.output.exists():
            raise DreamIntegrationError("output exists; choose a new evidence directory")
        result = infer_once(
            client,
            obs,
            configs,
            contract,
            args.prompt,
            validation_time=obs.collected_at,
            max_age=max_age,
            max_skew=max_skew,
            timeout_ms=args.timeout_ms,
        )
        result["contract"] = asdict(contract)
        result["freshness_evaluation"] = (
            "original capture time; recorded replay, not live freshness"
        )
        save_record(args.output, obs, configs, max_age=max_age, max_skew=max_skew, result=result)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        reason = (
            str(exc)
            if isinstance(exc, DreamIntegrationError)
            else "invalid local input or SDK failure"
        )
        print(json.dumps({"status": "BLOCKED", "reason": reason, "execution": "DISCONNECTED"}))
        return 2
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                print(json.dumps({"status": "BLOCKED", "reason": "Dream client cleanup failed"}))
                return 2


if __name__ == "__main__":
    raise SystemExit(main())
