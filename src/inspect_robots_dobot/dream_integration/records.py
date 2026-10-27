"""Offline records retain original timestamps; replay never implies live freshness."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..camera import Frame
from ..config import CameraConfig
from ..types import PoseSI, RobotMode, RobotSnapshot
from .observation import GripperMeasurement, Nova2Observation


def save_record(
    directory: Path,
    obs: Nova2Observation,
    configs: dict[str, CameraConfig],
    *,
    max_age: float,
    max_skew: float,
    result: dict[str, Any] | None = None,
) -> None:
    obs.validate(configs, now=obs.collected_at, max_age=max_age, max_skew=max_skew)
    directory.mkdir(parents=True, exist_ok=False)
    meta: dict[str, Any] = {
        "schema": 1,
        "clock": "original_host_monotonic",
        "kind": "fixture" if obs.fixture else "recorded",
        "collected_at": obs.collected_at,
        "state": asdict(obs.state),
        "gripper_state": None if obs.gripper_state is None else asdict(obs.gripper_state),
        "calibration_evidence": obs.calibration_evidence,
        "configs": {k: asdict(v) for k, v in configs.items()},
        "max_age": max_age,
        "max_skew": max_skew,
        "result": result,
        "execution": "DISCONNECTED",
        "units": {"joints": "rad", "tcp_xyz": "m", "tcp_angles": "rad_extrinsic_xyz"},
        "frames": {"user_index": obs.state.user_frame, "tool_index": obs.state.tool_frame},
        "validity": {
            "host_freshness_checked_at_capture": True,
            "sensor_exposure_synchronization_verified": False,
            "gripper_feedback_available": obs.gripper_state is not None,
            "calibration_evidence": obs.calibration_evidence,
        },
    }
    for name in ("front_rgb", "wrist_rgb"):
        frame = getattr(obs, name)
        np.save(directory / f"{name}.npy", frame.rgb, allow_pickle=False)
        meta[name] = {k: v for k, v in asdict(frame).items() if k != "rgb"}
    (directory / "observation.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")


def load_record(directory: Path) -> tuple[Nova2Observation, dict[str, CameraConfig], float, float]:
    meta = json.loads((directory / "observation.json").read_text())
    if (
        meta.get("units") != {"joints": "rad", "tcp_xyz": "m", "tcp_angles": "rad_extrinsic_xyz"}
        or meta["schema"] != 1
        or meta["clock"] != "original_host_monotonic"
        or meta["kind"] not in ("fixture", "recorded")
    ):
        raise ValueError("unsupported observation record")
    frames = [
        Frame(np.load(directory / f"{name}.npy", allow_pickle=False), **meta[name])
        for name in ("front_rgb", "wrist_rgb")
    ]
    raw = meta["state"]
    state = RobotSnapshot(
        PoseSI(**raw["pose"]),
        tuple(raw["joints"]),
        RobotMode(raw["mode"]),
        tuple(raw["errors"]),
        raw["command_id"],
        raw["observed_at"],
        raw["user_frame"],
        raw["tool_frame"],
        raw["joints_synthetic"],
    )
    g = meta["gripper_state"]
    obs = Nova2Observation(
        frames[0],
        frames[1],
        state,
        None if g is None else GripperMeasurement(**g),
        meta["collected_at"],
        meta["calibration_evidence"],
        meta["kind"] == "fixture",
    )
    configs = {k: CameraConfig(**v) for k, v in meta["configs"].items()}
    obs.validate(configs, now=obs.collected_at, max_age=meta["max_age"], max_skew=meta["max_skew"])
    return obs, configs, meta["max_age"], meta["max_skew"]
