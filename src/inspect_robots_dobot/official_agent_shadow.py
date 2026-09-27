"""Official ``inspect-robots-agent`` integration for Phase 6A.1.

This module is intentionally a small boundary around the pinned upstream
``LLMAgentPolicy``.  Provider resolution, Responses serialization, image
encoding, tool validation and ActionChunk construction remain upstream.  The
Dobot repository only supplies the embodiment, its pre-check, the staged
approver path, and the terminal shadow record.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import ActionChunk, Observation
from inspect_robots_agent import LLMAgentPolicy

from .astra_shadow import (
    CAMERA_ORDER,
    PHASE6A_TASK,
    AstraDecision,
    AstraResponse,
    Candidate,
    ShadowExecutor,
    ShadowRecord,
    build_fixture_observation,
)
from .chunks import DobotExecutionSession
from .errors import ConfigurationError, SafetyRejected
from .gripper import semantic_target, validate_normalized

OFFICIAL_MODEL = "openai/gpt-6-astra"
OFFICIAL_WIRE = "responses"
OFFICIAL_EFFORT = "medium"
OFFICIAL_IMAGES = "always"
OFFICIAL_IMAGE_HORIZON = 2


class MissingOpenAIKey(ConfigurationError):
    """Missing provider credential; the legacy exception name is preserved."""


@dataclass(frozen=True)
class OfficialAgentConfig:
    model: str = OFFICIAL_MODEL
    wire: str = OFFICIAL_WIRE
    effort: str = OFFICIAL_EFFORT
    images: str = OFFICIAL_IMAGES
    image_horizon: int = OFFICIAL_IMAGE_HORIZON
    max_llm_calls: int = 1
    max_speed_frac: float = 0.1
    base_url: str | None = None
    api_key_env: str | None = None


def require_openai_key(
    env: dict[str, str] | None = None, *, key_env: str = "OPENAI_API_KEY"
) -> None:
    """Fail closed for the selected key name; preserve the OpenAI default."""
    source = os.environ if env is None else env
    if not source.get(key_env):
        raise MissingOpenAIKey(
            f"{key_env} is not configured. Official Astra shadow integration is ready "
            f"but live query is blocked. Set {key_env} manually, then rerun."
        )


def _tool_call_from_transcript(transcript: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not transcript:
        return {}
    for message in reversed(transcript):
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            call = calls[0]
            if isinstance(call, dict):
                function = call.get("function")
                if isinstance(function, dict):
                    return {
                        "name": function.get("name"),
                        "arguments": function.get("arguments"),
                        "id": call.get("id"),
                    }
    return {}


def _tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    arguments = tool_call.get("arguments")
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_gripper_proposal(tool_call: dict[str, Any]) -> str:
    values = _tool_arguments(tool_call).get("targets")
    if not isinstance(values, dict) or "gripper" not in values:
        return "hold"
    try:
        normalized = validate_normalized(values["gripper"], label="gripper target")
    except SafetyRejected:
        return "target"
    return {"CLOSED": "grasp", "OPEN": "release"}.get(semantic_target(normalized), "target")


def _request_summary(body: dict[str, Any]) -> dict[str, Any]:
    """Keep evidence reproducible without storing image data URLs."""
    tools = body.get("tools", [])
    names = [tool.get("name") for tool in tools if isinstance(tool, dict)]
    return {
        "model": body.get("model"),
        "wire": OFFICIAL_WIRE,
        "reasoning": body.get("reasoning"),
        "store": body.get("store"),
        "tool_names": names,
        "tool_definitions": tools,
        "image_policy": OFFICIAL_IMAGES,
        "image_horizon": OFFICIAL_IMAGE_HORIZON,
        "image_roles": list(CAMERA_ORDER),
    }


def _redact_credential(value: Any, credential: str) -> Any:
    """Scrub a credential echoed in upstream provider errors or model content."""
    if isinstance(value, str):
        return value.replace(credential, "[REDACTED]") if credential else value
    if isinstance(value, list):
        return [_redact_credential(item, credential) for item in value]
    if isinstance(value, dict):
        return {key: _redact_credential(item, credential) for key, item in value.items()}
    return value


def _official_candidates(
    chunk: ActionChunk, observation: Observation, tool_name: str
) -> tuple[Candidate, ...]:
    if tool_name in {"done", "give_up"}:
        return (
            Candidate(
                f"official-{tool_name}",
                tool_name,
                (0.0, 0.0, 0.0),
                f"official inspect-robots-agent {tool_name} tool call",
            ),
        )
    current = np.asarray(observation.state["eef_state"], dtype=np.float64)
    target = np.asarray(chunk.actions[-1].data, dtype=np.float64)
    delta = (
        float(target[0] - current[0]),
        float(target[1] - current[1]),
        float(target[2] - current[2]),
    )
    return (
        Candidate(
            "official-move_to",
            "move_to",
            delta,
            "official inspect-robots-agent move_to tool call",
        ),
    )


def run_official_shadow(
    *,
    response_json: dict[str, Any] | None = None,
    images: dict[str, np.ndarray] | None = None,
    config: OfficialAgentConfig | None = None,
    env: dict[str, str] | None = None,
    require_live_key: bool = False,
) -> ShadowRecord:
    """Run one official-agent decision through the Dobot shadow boundary.

    ``response_json`` is an explicit deterministic test seam.  With it, the
    upstream Responses client uses an in-memory transport and no provider
    request is made.  A future live invocation must set ``require_live_key``;
    missing credentials are rejected before an HTTP client is constructed.
    """
    if config is None:
        config = OfficialAgentConfig()
    if not config.base_url and config.model != OFFICIAL_MODEL:
        raise ConfigurationError("Phase 6A.1 requires model=openai/gpt-6-astra without --base-url")
    if config.wire != OFFICIAL_WIRE:
        raise ConfigurationError(
            "Phase 6A.1 requires wire=responses and POST <base_url>/responses. "
            "A Chat Completions-only gateway is incompatible; no wire fallback is performed."
        )
    if config.effort != OFFICIAL_EFFORT:
        raise ConfigurationError("Phase 6A.1 requires effort=medium")
    # Match the pinned upstream resolver's explicit-base-url key default.
    key_env = (config.api_key_env or "OPENROUTER_API_KEY") if config.base_url else "OPENAI_API_KEY"
    if require_live_key:
        require_openai_key(env, key_env=key_env)
    credential = (os.environ if env is None else env).get(key_env, "")

    embodiment, observation = build_fixture_observation()
    # The official-agent path is explicitly the Phase 6A.2 shadow boundary;
    # enable normalized gripper validation before upstream tool pre_check runs.
    embodiment._allow_shadow_gripper = True
    if images is not None:
        if tuple(images) != CAMERA_ORDER:
            embodiment.close()
            raise ConfigurationError("official replay images must contain the three ordered roles")
        from inspect_robots.types import Observation as ObservationType

        observation = ObservationType(
            images=images,
            state=observation.state,
            instruction=observation.instruction,
            image_times=observation.image_times,
            state_time=observation.state_time,
            extra=observation.extra,
        )

    captured: list[dict[str, Any]] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        if response_json is None:
            raise RuntimeError("official live transport is not enabled by the shadow runner")
        return httpx.Response(200, json=response_json)

    transport = None if response_json is None else httpx.MockTransport(transport_handler)
    policy = LLMAgentPolicy(
        model=config.model,
        base_url=config.base_url
        or ("http://official-agent.mock/v1" if response_json is not None else None),
        api_key_env=config.api_key_env,
        wire=config.wire,
        effort=config.effort,
        images=config.images,
        image_horizon=config.image_horizon,
        max_llm_calls=config.max_llm_calls,
        max_speed_frac=config.max_speed_frac,
        env=None if env is None else env,
        transport=transport,
        pre_check=embodiment.pre_check,
    )
    scene = Scene(id="phase6a1-official", instruction=PHASE6A_TASK)
    policy.bind(embodiment.info)
    policy.reset(scene)
    precheck: str | None = None
    approval: str | None = None
    rejection: str | None = None
    chunk: ActionChunk | None = None
    plan = None
    chunk_identity: str | None = None
    gripper_target: float | None = None
    started = time.monotonic()
    try:
        with DobotExecutionSession(embodiment, allow_shadow_gripper=True) as session:
            store: dict[str, Any] = {}
            action = session.controller.next_action(policy, observation, 0, store)
            pending = embodiment._chunk
            if pending is None:
                raise SafetyRejected("REJECT: official agent did not register a chunk")
            chunk_identity = pending.identity
            chunk = ActionChunk(
                actions=list(pending.actions),
                control_hz=policy.info.control_hz,
                meta=dict(session.controller.chunk_metadata),
            )
            rows = np.asarray([item.data for item in chunk.actions], dtype=np.float64)
            rows.flags.writeable = False
            precheck = embodiment.pre_check(rows)
            if precheck is not None:
                rejection = precheck
            else:
                for index in range(len(chunk)):
                    current_action = (
                        action
                        if index == 0
                        else session.controller.next_action(policy, observation, index, store)
                    )
                    reviewed = session.approver.review(current_action, store)
                    staged = embodiment.shadow_stage(reviewed)
                    if staged is not None:
                        plan = staged
                approval = "approved"
            gripper_target = embodiment.shadow_gripper_target
    except (SafetyRejected, ValueError, ConfigurationError, RuntimeError) as exc:
        rejection = _redact_credential(str(exc), credential)
    finally:
        if embodiment._session is not None:
            embodiment.close()

    transcript = _redact_credential(policy.transcript(), credential)
    tool_call = _tool_call_from_transcript(transcript)
    gripper_proposal = _tool_gripper_proposal(tool_call)
    raw = {"tool_call": tool_call, "transcript": transcript}
    response = AstraResponse(
        provider="official-inspect-robots-agent",
        model=config.model,
        raw=raw,
        latency_s=time.monotonic() - started,
    )
    candidates: tuple[Candidate, ...]
    tool_name = str(tool_call.get("name") or "move_to")
    if chunk is None:
        candidates = (
            Candidate(
                f"official-{tool_name}",
                tool_name,
                (0.0, 0.0, 0.0),
                "official tool call did not yield a motion chunk",
            ),
        )
        decision = AstraDecision(
            tool_name,
            f"official-{tool_name}",
            "official tool call rejected",
            None,
            False,
            True,
            gripper_proposal,
            (0.0, 0.0, 0.0),
            raw,
        )
    else:
        candidates = _official_candidates(chunk, observation, tool_name)
        delta = candidates[0].delta_m
        decision = AstraDecision(
            tool_name,
            candidates[0].candidate_id,
            str(tool_call.get("arguments", "")),
            None,
            False,
            True,
            gripper_proposal,
            delta,
            raw,
        )
    request = (
        _request_summary(captured[0])
        if captured
        else {
            "model": config.model,
            "wire": config.wire,
            "effort": config.effort,
            "image_policy": config.images,
            "image_horizon": config.image_horizon,
            "image_roles": list(CAMERA_ORDER),
        }
    )
    request["run_id"] = "phase6a1-official-shadow"
    request["upstream_policy"] = "inspect_robots_agent.LLMAgentPolicy"
    request["base_url"] = config.base_url
    request["api_key_env"] = config.api_key_env
    request["image_shapes"] = {
        name: list(np.asarray(observation.images[name]).shape) for name in CAMERA_ORDER
    }
    return ShadowExecutor().record(
        observation=observation,
        response=response,
        decision=decision,
        candidates=candidates,
        chunk=chunk,
        plan=plan,
        pre_check_result=precheck,
        approver_result=approval,
        rejection_reason=rejection,
        request=request,
        gripper_target_normalized=(
            gripper_target
            if chunk is not None and gripper_proposal not in {"none", "hold"}
            else None
        ),
        chunk_identity=chunk_identity,
    )
