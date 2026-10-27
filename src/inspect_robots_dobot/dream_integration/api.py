"""SDK 1.0.9 boundary. No registration, training, device access or action dispatch."""

from __future__ import annotations

import base64
import importlib
import io
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import CameraConfig
from .observation import Nova2Observation


class DreamIntegrationError(ValueError):
    """Sanitized errors: upstream exception messages may contain private data."""


def nova2_config() -> dict[str, Any]:
    return {
        "state": {"right_arm": [0, 6], "right_gripper": [6, 7]},
        "action": {"right_arm": [0, 6], "right_gripper": [6, 7]},
        "video": {"front": "observation.images.front_rgb", "wrist": "observation.images.wrist_rgb"},
        "head_camera": "front",
    }


def sdk() -> Any:
    # Disable SDK diagnostic files; policy() still uploads its own background
    # recordings (SDK behavior), so only call it for an explicit inference request.
    os.environ.setdefault("DREAM_LOG", "off")
    module = importlib.import_module("dream")
    if module.__version__ != "1.0.9":
        raise DreamIntegrationError("requires inspected dreamlabs-client 1.0.9")
    return module


def validate_local_schema() -> None:
    sdk()
    # No public standalone schema validator in 1.0.9. Use the SAME local parser
    # add_embodiment calls, without calling that network-writing method.
    parser = importlib.import_module("shared.datasets.embodiments")
    body = parser.check_registration(nova2_config())
    parser.check_state_width(body)
    parser.check_action_width(body)


def connect_client() -> Any:
    key = os.environ.get("DREAM_API_KEY", "").strip()
    if not key:
        raise DreamIntegrationError("DREAM_API_KEY missing; configure it securely outside the repo")
    try:
        return sdk().DreamClient(api_key=key)
    except Exception:
        raise DreamIntegrationError("Dream client configuration failed") from None


def discover(client: Any) -> list[Any]:
    try:
        models = client.get_models()
        if not isinstance(models, list):
            raise TypeError()
        for model in models:
            if not isinstance(model.id, str) or type(model.servable) is not bool:
                raise TypeError()
        return models
    except Exception:
        raise DreamIntegrationError(
            "Dream model discovery failed (API/configuration/timeout)"
        ) from None


def compatibility(model: Any, embodiment_id: str) -> str | None:
    if not embodiment_id or embodiment_id == "yam":
        return "an independently registered Nova2 embodiment id is required"
    if model.embodiment == "yam":
        return "YAM checkpoint: two arms and overhead/wrist_l/wrist_r; Nova2 is unsupported here"
    if model.servable is not True or model.embodiment != embodiment_id:
        return "checkpoint not servable or does not match the approved Nova2 embodiment id"
    if model.parts_withheld or model.embodiment_config != nova2_config():
        return "checkpoint does not expose the exact candidate Nova2 state/action schema"
    if model.cameras is None or sorted(model.cameras) != ["front", "wrist"]:
        return "checkpoint camera roles do not match front/wrist"
    return None


@dataclass(frozen=True)
class InferenceContract:
    """Locally reviewed training semantics, NOT proof of vendor/model approval.

    All joint targets are absolute radians in Nova2 controller joint order J1-J6.
    Gripper targets are normalized 0=closed,1=open, backed by measured calibration.
    These semantics must match the training data; a schema alone cannot prove them.
    """

    checkpoint: str
    embodiment_id: str
    action_hz: float
    horizon: int
    user_frame: int
    tool_frame: int
    evidence: str
    joint_units: str = "rad"
    gripper_units: str = "normalized_0_closed_1_open"
    action_semantics: str = "absolute_joint_targets"

    def validate(self) -> None:
        if (
            not self.checkpoint
            or not self.evidence
            or not self.embodiment_id
            or self.embodiment_id == "yam"
            or not math.isfinite(self.action_hz)
            or self.action_hz <= 0
            or type(self.horizon) is not int
            or not 1 <= self.horizon <= 1024
            or self.joint_units != "rad"
            or self.gripper_units != "normalized_0_closed_1_open"
            or self.action_semantics != "absolute_joint_targets"
            or any(type(v) is not int or v < 0 for v in (self.user_frame, self.tool_frame))
        ):
            raise DreamIntegrationError("invalid or unreviewed inference contract")


def decode_action(action: Any, contract: InferenceContract) -> dict[str, Any]:
    contract.validate()
    try:
        parts = action["per_key"]
        if set(parts) != {"action.right_arm", "action.right_gripper"}:
            raise ValueError()
        arm = np.asarray(parts["action.right_arm"])
        grip = np.asarray(parts["action.right_gripper"])
        if arm.dtype.kind not in "fiu" or grip.dtype.kind not in "fiu":
            raise ValueError()
        arm, grip = arm.astype(float), grip.astype(float)
        if grip.shape == (contract.horizon,):
            grip = grip[:, None]
        if arm.shape != (contract.horizon, 6) or grip.shape != (contract.horizon, 1):
            raise ValueError()
        if not np.isfinite(arm).all() or not np.isfinite(grip).all():
            raise ValueError()
        if (grip < 0).any() or (grip > 1).any():
            raise ValueError()
        return {
            "joint_targets_rad": arm.tolist(),
            "gripper_targets": grip.tolist(),
            "time_from_start_s": (np.arange(contract.horizon) / contract.action_hz).tolist(),
            "execution": "DISCONNECTED",
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        raise DreamIntegrationError("malformed Dream action chunk") from None


def infer_once(
    client: Any,
    obs: Nova2Observation,
    configs: dict[str, CameraConfig],
    contract: InferenceContract,
    prompt: str,
    *,
    validation_time: float,
    max_age: float,
    max_skew: float,
    timeout_ms: int = 15000,
) -> dict[str, Any]:
    contract.validate()
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 2**31 - 1:
        raise DreamIntegrationError("invalid API timeout")
    obs.validate(configs, now=validation_time, max_age=max_age, max_skew=max_skew)
    models = discover(client)
    model = next((m for m in models if m.id == contract.checkpoint), None)
    reason = (
        "checkpoint unavailable" if model is None else compatibility(model, contract.embodiment_id)
    )
    if reason:
        raise DreamIntegrationError(reason)
    if obs.fixture or obs.state.joints_synthetic:
        raise DreamIntegrationError("synthetic fixture cannot be uploaded as Nova2 inference")
    if not prompt.strip() or not obs.calibration_evidence or obs.state.errors:
        raise DreamIntegrationError(
            "prompt, calibration evidence and error-free measured state required"
        )
    if (obs.state.user_frame, obs.state.tool_frame) != (contract.user_frame, contract.tool_frame):
        raise DreamIntegrationError("observation coordinate frames differ from training contract")
    g = obs.gripper_state
    if (
        g is None
        or g.units != contract.gripper_units
        or not g.calibration_evidence
        or not 0 <= g.position <= 1
    ):
        raise DreamIntegrationError(
            "calibrated measured gripper feedback required; no dummy state allowed"
        )
    # Import only after compatibility; fixture/probe paths need no image encoder.
    image = importlib.import_module("PIL.Image")
    frames = {}
    for role, frame in [("front", obs.front_rgb), ("wrist", obs.wrist_rgb)]:
        buffer = io.BytesIO()
        image.fromarray(frame.rgb).resize((320, 176)).save(buffer, format="JPEG", quality=85)
        frames[f"video.{role}"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    state = {
        "state.right_arm": np.asarray(obs.joint_positions, dtype=np.float32),
        "state.right_gripper": np.array([g.position], dtype=np.float32),
        "prompt": prompt,
    }
    started = time.perf_counter()
    try:
        client.policy(
            {
                "checkpoint": contract.checkpoint,
                "embodiment": nova2_config(),
                "strict": True,
                "timeout_ms": timeout_ms,
                "action_hz": contract.action_hz,
            }
        )
        client.set_observation(
            frames, state, timestamp=min(obs.front_rgb.timestamp, obs.wrist_rgb.timestamp)
        )
        raw = client.get_action()
    except Exception:
        raise DreamIntegrationError(
            "Dream inference failed (API/configuration/timeout); no retry"
        ) from None
    decoded = decode_action(raw, contract)
    return {
        "checkpoint": contract.checkpoint,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "action": decoded,
        "observation_kind": "recorded_or_measured",
        "execution": "DISCONNECTED",
    }
