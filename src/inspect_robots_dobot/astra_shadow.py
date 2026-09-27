"""Phase 6A Astra shadow path.

This module deliberately has no network, Dashboard, feedback, or gripper imports.  A
provider is injected, and the default fixture provider is deterministic.  The full
path uses the existing absolute ``move_to`` action semantics and the existing staged
chunk/approver checks, then stops at a pure prospective MovL record.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import numpy as np
from inspect_robots.policy import Policy
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from .chunks import DobotExecutionSession, StagedDobotEmbodiment
from .config import DobotConfig, SafetyProfile
from .driver import FakeDobotDriver
from .errors import ConfigurationError, SafetyRejected
from .gripper import ShadowGripperExecutor, validate_normalized
from .motion import CartesianMotionPlan
from .safety import MotionAuthority
from .types import PoseSI

CAMERA_ORDER = ("front_rgb", "right_rgb", "wrist_rgb")
TRANSLATION_ACTIONS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
SEMANTIC_ACTIONS = ("grasp", "release", "observe_again")
PHASE6A_TASK = "Pick up the blue block and place it inside the box."


class AstraUnavailable(RuntimeError):
    """Raised when an external Astra provider is not configured."""


class AstraProvider(Protocol):
    provider: str
    model: str

    def query(self, request: dict[str, Any]) -> AstraResponse: ...


@dataclass(frozen=True)
class AstraResponse:
    provider: str
    model: str
    raw: Any
    latency_s: float | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class AstraDecision:
    action: str
    candidate_id: str
    rationale_summary: str
    confidence: float | None
    terminate: bool
    observe_again: bool
    gripper_proposal: str
    delta_m: tuple[float, float, float]
    raw_response: Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    action: str
    delta_m: tuple[float, float, float]
    description: str
    physical_execution: str = "SHADOW_ONLY"


@dataclass(frozen=True)
class ShadowRecord:
    observation_id: str
    decision_id: str
    run_id: str
    provider: str
    model: str
    raw_response: Any
    model_response_sha256: str
    request_metadata: dict[str, Any]
    proposed_high_level_action: str
    candidate_set: tuple[dict[str, Any], ...]
    selected_candidate: str | None
    rejected_alternatives: tuple[str, ...]
    move_to_parameters: dict[str, Any] | None
    action_chunk: dict[str, Any] | None
    number_of_waypoints: int
    final_target: tuple[float, ...] | None
    orientation_delta: float | None
    gripper_proposal: str
    gripper_execution: str
    gripper_action_available: bool
    gripper_state_input: dict[str, Any]
    gripper_target_normalized: float | None
    gripper_semantic_target: str | None
    gripper_backend: str
    hypothetical_gripper: str | None
    physical_gripper_connected: bool
    physical_gripper_command_sent: bool
    gripper_commands_sent: int
    pre_check_result: str | None
    approver_result: str | None
    rejection_reason: str | None
    hypothetical_dobot_target: tuple[float, ...] | None
    hypothetical_movl: str | None
    execution: bool
    execution_performed: bool
    nova_connections: int
    nova_dashboard_connections: int
    nova_feedback_connections: int
    motion_commands_sent: int
    live_authorities_created: int
    gripper_serial_connections: int
    image_roles: tuple[str, ...]
    image_hashes: dict[str, str]
    robot_state_visible: dict[str, Any]
    robot_state_hidden: tuple[str, ...]
    task_instruction: str
    latency_s: float | None


class FakeAstra:
    """Structured offline provider used by fixtures and tests."""

    provider = "fake"

    def __init__(self, response: dict[str, Any] | str, *, model: str = "fake-astra-v0") -> None:
        self.model = model
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def query(self, request: dict[str, Any]) -> AstraResponse:
        self.requests.append(request)
        return AstraResponse(self.provider, self.model, self.response, latency_s=0.0)


class MissingAstra:
    """Explicit marker for the absent live Astra integration."""

    provider = "astra"
    model = "unconfigured"

    def query(self, request: dict[str, Any]) -> AstraResponse:
        del request
        raise AstraUnavailable("ASTRA_LIVE_CLIENT_BLOCKED: no Astra SDK/HTTP client is configured")


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def observation_image_hashes(observation: Observation) -> dict[str, str]:
    return {
        name: hashlib.sha256(np.asarray(observation.images[name]).tobytes()).hexdigest()
        for name in CAMERA_ORDER
        if name in observation.images
    }


def visible_robot_state(observation: Observation) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key in ("eef_state", "dobot_native_pose"):
        if key in observation.state:
            state[key] = np.asarray(observation.state[key], dtype=np.float64).tolist()
    dobot = observation.extra.get("dobot", {})
    if isinstance(dobot, dict):
        for key in ("robot_mode", "command_id", "user_frame", "tool_frame"):
            if key in dobot:
                state[key] = dobot[key]
        if isinstance(dobot.get("gripper"), dict):
            state["gripper"] = dict(dobot["gripper"])
    return state


def generate_candidates(observation: Observation, profile: SafetyProfile) -> tuple[Candidate, ...]:
    """Offer bounded translation candidates; orientation is pinned and gripper is semantic."""
    raw = observation.state.get("eef_state")
    if raw is None:
        raise SafetyRejected("REJECT: Astra candidate generation needs eef_state")
    current = np.asarray(raw, dtype=np.float64)
    if current.shape != (7,) or not np.all(np.isfinite(current)):
        raise SafetyRejected(
            "REJECT: Astra candidate generation needs finite seven-value eef_state"
        )
    step = min(0.005, profile.max_translation_step / 2.0)
    candidates: list[Candidate] = []
    for label, axis, sign in (
        ("+X", 0, 1),
        ("-X", 0, -1),
        ("+Y", 1, 1),
        ("-Y", 1, -1),
        ("+Z", 2, 1),
        ("-Z", 2, -1),
    ):
        delta = [0.0, 0.0, 0.0]
        delta[axis] = sign * step
        target = current[:3] + delta
        if (
            np.all(target >= np.asarray(profile.workspace_low))
            and np.all(target <= np.asarray(profile.workspace_high))
            and target[2] >= profile.minimum_tcp_z
        ):
            candidates.append(
                Candidate(
                    label,
                    "translate",
                    (float(delta[0]), float(delta[1]), float(delta[2])),
                    f"bounded {label} translation of {step:g} m",
                )
            )
    candidates.extend(
        Candidate(name, name, (0.0, 0.0, 0.0), f"semantic {name} proposal")
        for name in SEMANTIC_ACTIONS
    )
    return tuple(candidates)


class AstraPolicyAdapter:
    """Build deterministic model context and parse one structured next action."""

    def __init__(self, provider: AstraProvider, *, task_instruction: str = PHASE6A_TASK) -> None:
        self.provider = provider
        self.task_instruction = task_instruction

    def request(
        self, observation: Observation, candidates: tuple[Candidate, ...]
    ) -> dict[str, Any]:
        if tuple(observation.images) != CAMERA_ORDER:
            raise ConfigurationError(
                "Phase 6A Observation must contain exactly front_rgb, right_rgb, wrist_rgb in order"
            )
        return {
            "task": self.task_instruction,
            "image_roles": list(CAMERA_ORDER),
            "images": [
                {
                    "role": name,
                    "shape": list(np.asarray(observation.images[name]).shape),
                    "sha256": observation_image_hashes(observation)[name],
                }
                for name in CAMERA_ORDER
                if name in observation.images
            ],
            "robot_state": visible_robot_state(observation),
            "action_semantics": {
                "interface": "move_to",
                "position_units": "metres",
                "rotation_units": "radians",
                "orientation": "HOLD / pinned yaw,pitch,roll",
                "gripper_action_available": True,
                "gripper_execution": "SHADOW_ONLY",
                "gripper_semantics": "0=closed, 1=open",
                "one_next_action": True,
            },
            "candidates": [asdict(candidate) for candidate in candidates],
        }

    def parse(self, response: AstraResponse, candidates: tuple[Candidate, ...]) -> AstraDecision:
        raw = response.raw
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("Astra response is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError("Astra response must be a JSON object")
        action = raw.get("action")
        candidate_id = raw.get("candidate_id")
        rationale = raw.get("rationale_summary", "")
        if not isinstance(action, str) or not action:
            raise ValueError("Astra response missing action")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Astra response missing candidate_id")
        if not isinstance(rationale, str):
            raise ValueError("rationale_summary must be a string")
        selected = next(
            (candidate for candidate in candidates if candidate.candidate_id == candidate_id), None
        )
        if selected is None:
            raise ValueError(f"unsupported candidate_id {candidate_id!r}")
        if action != selected.action:
            raise ValueError("Astra action does not match selected candidate")
        confidence = raw.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError("confidence must be a finite number in [0,1]")
        terminate = raw.get("terminate", False)
        observe_again = raw.get("observe_again", True)
        if type(terminate) is not bool or type(observe_again) is not bool:
            raise ValueError("terminate and observe_again must be booleans")
        gripper = raw.get("gripper_proposal", "hold")
        if not isinstance(gripper, str):
            raise ValueError("gripper_proposal must be a string")
        if gripper not in {"none", "hold", "grasp", "release", "target"}:
            raise ValueError("gripper_proposal must be one of none, hold, grasp, release, target")
        return AstraDecision(
            action,
            candidate_id,
            rationale,
            None if confidence is None else float(confidence),
            terminate,
            observe_again,
            gripper,
            selected.delta_m,
            raw,
        )

    def decide(
        self, observation: Observation, candidates: tuple[Candidate, ...]
    ) -> tuple[AstraResponse, AstraDecision, dict[str, Any]]:
        request = self.request(observation, candidates)
        started = time.monotonic()
        response = self.provider.query(request)
        if response.latency_s is None:
            response = AstraResponse(
                response.provider,
                response.model,
                response.raw,
                time.monotonic() - started,
                response.temperature,
                response.top_p,
                response.seed,
            )
        return response, self.parse(response, candidates), request


def move_to_chunk(
    observation: Observation, decision: AstraDecision, profile: SafetyProfile
) -> ActionChunk:
    """Map a bounded candidate to generic absolute ``move_to`` waypoints."""
    if decision.action not in {"translate", "grasp", "release"}:
        raise ValueError("semantic Astra action has no robot action chunk")
    state = np.asarray(observation.state["eef_state"], dtype=np.float64)
    target = state.copy()
    if decision.action == "translate":
        target[:3] += np.asarray(decision.delta_m)
    target[3:6] = state[3:6]
    if decision.action == "grasp" or decision.gripper_proposal == "grasp":
        target[6] = 0.0
    elif decision.action == "release" or decision.gripper_proposal == "release":
        target[6] = 1.0
    else:
        target[6] = validate_normalized(state[6], label="current gripper")
    axis_limit = profile.max_translation_step / math.sqrt(3)
    steps = max(1, int(math.ceil(max(abs(value) for value in decision.delta_m) / axis_limit)))
    rows = [state + (target - state) * fraction for fraction in np.linspace(1 / steps, 1, steps)]
    rows[-1] = target
    actions = [
        Action(np.asarray(row, dtype=np.float64), {"chunk_final": index == len(rows) - 1})
        for index, row in enumerate(rows)
    ]
    return ActionChunk(
        actions=actions,
        control_hz=10.0,
        meta={"interface": "move_to", "candidate_id": decision.candidate_id},
    )


class ShadowExecutor:
    """Terminal Phase 6A boundary. It records a plan and has no execution method."""

    def record(
        self,
        *,
        observation: Observation,
        response: AstraResponse,
        decision: AstraDecision,
        candidates: tuple[Candidate, ...],
        chunk: ActionChunk | None,
        plan: CartesianMotionPlan | None,
        pre_check_result: str | None,
        approver_result: str | None,
        rejection_reason: str | None,
        request: dict[str, Any],
        gripper_target_normalized: float | None = None,
        chunk_identity: str | None = None,
        task_instruction: str = PHASE6A_TASK,
    ) -> ShadowRecord:
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.candidate_id == decision.candidate_id
            ),
            None,
        )
        explicit_gripper = decision.gripper_proposal not in {"none", "hold"}
        if gripper_target_normalized is None and chunk is not None and explicit_gripper:
            gripper_target_normalized = validate_normalized(
                float(chunk.actions[-1].data[6]), label="gripper target"
            )
        gripper_record = None
        if gripper_target_normalized is not None and approver_result == "approved":
            gripper_record = ShadowGripperExecutor().record_target(gripper_target_normalized)
        move_params = (
            None
            if chunk is None or decision.action not in {"translate", "move_to", "grasp", "release"}
            else {
                "tool": "move_to",
                "target": list(chunk.actions[-1].data),
                "candidate_id": decision.candidate_id,
                "gripper_target_normalized": gripper_target_normalized,
                "gripper_target_explicit": explicit_gripper,
            }
        )
        hypothetical = None if plan is None else plan.request.serialize()
        return ShadowRecord(
            observation_id=str(observation.extra.get("observation_id", uuid4().hex)),
            decision_id=uuid4().hex,
            run_id=str(request.get("run_id", uuid4().hex)),
            provider=response.provider,
            model=response.model,
            raw_response=response.raw,
            model_response_sha256=_json_hash(response.raw),
            proposed_high_level_action=decision.action,
            request_metadata=request,
            candidate_set=tuple(asdict(candidate) for candidate in candidates),
            selected_candidate=selected.candidate_id if selected else None,
            rejected_alternatives=tuple(
                candidate.candidate_id
                for candidate in candidates
                if candidate.candidate_id != decision.candidate_id
            ),
            move_to_parameters=move_params,
            action_chunk=None
            if chunk is None
            else {
                "length": len(chunk),
                "meta": dict(chunk.meta),
                "identity": chunk_identity or getattr(plan, "chunk_id", None),
            },
            number_of_waypoints=0 if chunk is None else len(chunk),
            final_target=None if plan is None else plan.final_pose_si.values,
            orientation_delta=None if plan is None else 0.0,
            gripper_proposal=decision.gripper_proposal,
            gripper_execution="SHADOW_ONLY",
            gripper_action_available=True,
            gripper_state_input=(
                dict(observation.extra.get("dobot", {}).get("gripper", {}))
                if isinstance(observation.extra.get("dobot", {}).get("gripper", {}), dict)
                else {"normalized": float(observation.state["eef_state"][6]), "source": "unknown"}
            ),
            gripper_target_normalized=(
                None if gripper_record is None else float(gripper_record["target_normalized"])
            ),
            gripper_semantic_target=(
                None if gripper_record is None else str(gripper_record["semantic_target"])
            ),
            gripper_backend="shadow",
            hypothetical_gripper=(
                None if gripper_record is None else str(gripper_record["would_execute"])
            ),
            physical_gripper_connected=False,
            physical_gripper_command_sent=False,
            gripper_commands_sent=0,
            pre_check_result=pre_check_result,
            approver_result=approver_result,
            rejection_reason=rejection_reason,
            hypothetical_dobot_target=None if plan is None else plan.request.pose.values,
            hypothetical_movl=None if hypothetical is None else "WOULD_SEND: " + hypothetical,
            execution=False,
            execution_performed=False,
            nova_connections=0,
            nova_dashboard_connections=0,
            nova_feedback_connections=0,
            motion_commands_sent=0,
            live_authorities_created=0,
            gripper_serial_connections=0,
            image_roles=tuple(CAMERA_ORDER),
            image_hashes=observation_image_hashes(observation),
            robot_state_visible=visible_robot_state(observation),
            robot_state_hidden=(
                "joint configuration beyond synthetic fixture",
                "safety authority internals",
                "live Nova state",
            ),
            task_instruction=task_instruction,
            latency_s=response.latency_s,
        )

    @staticmethod
    def write(record: ShadowRecord, evidence_dir: Path) -> Path:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        path = evidence_dir / f"{record.run_id}-{record.decision_id}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
        path.write_text(
            json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    def execute(self, **kwargs: Any) -> ShadowRecord:
        """Named terminal boundary; this method only records and never executes."""
        return self.record(**kwargs)


def fixture_profile() -> SafetyProfile:
    return SafetyProfile(
        (0.1, -0.2, 0.01),
        (0.5, 0.2, 0.5),
        0.08,
        0,
        0,
        0.02,
        0.1,
        5,
        5,
        0.0001,
        0.001,
        0.5,
        0.25,
        0.01,
    )


def build_fixture_observation() -> tuple[StagedDobotEmbodiment, Observation]:
    profile = fixture_profile()
    from .clock import FakeClock

    clock = FakeClock(1.0)
    driver = FakeDobotDriver(
        initial_pose=PoseSI(0.3, 0.0, 0.2, 0.1, -0.2, 0.3),
        initial_joints=(0.0,) * 6,
        profile=profile,
        clock=clock,
        authority=MotionAuthority(True),
    )
    embodiment = StagedDobotEmbodiment(
        DobotConfig(profile, control_hz=10.0), driver=driver, clock=clock
    )
    observation = embodiment.reset(Scene(id="phase6a-fixture", instruction=PHASE6A_TASK))
    images = {
        name: np.full((8, 8, 3), index * 50, dtype=np.uint8)
        for index, name in enumerate(CAMERA_ORDER, 1)
    }
    observation = Observation(
        images=images,
        state=observation.state,
        instruction=observation.instruction,
        image_times=observation.image_times,
        state_time=observation.state_time,
        extra={**observation.extra, "observation_id": "fixture-observation-001"},
    )
    return embodiment, observation


def run_fixture_shadow(
    *,
    response: dict[str, Any] | None = None,
    images: dict[str, np.ndarray] | None = None,
) -> ShadowRecord:
    embodiment, observation = build_fixture_observation()
    embodiment._allow_shadow_gripper = True
    if images is not None:
        if tuple(images) != CAMERA_ORDER:
            raise ConfigurationError("replay images must contain the three ordered camera roles")
        observation = Observation(
            images=images,
            state=observation.state,
            instruction=observation.instruction,
            image_times=observation.image_times,
            state_time=observation.state_time,
            extra=observation.extra,
        )
    profile = fixture_profile()
    provider = FakeAstra(
        response
        or {
            "action": "translate",
            "candidate_id": "+X",
            "rationale_summary": "The blue block is to the right of the gripper.",
            "confidence": 0.8,
            "observe_again": True,
            "terminate": False,
            "gripper_proposal": "none",
        }
    )
    adapter = AstraPolicyAdapter(provider)
    candidates = generate_candidates(observation, profile)
    response_obj, decision, request = adapter.decide(observation, candidates)
    request["run_id"] = "phase6a-fixture"
    chunk: ActionChunk | None = None
    plan: CartesianMotionPlan | None = None
    chunk_identity: str | None = None
    precheck: str | None = None
    approval: str | None = None
    rejection: str | None = None
    try:
        try:
            chunk = move_to_chunk(observation, decision, profile)
        except ValueError as exc:
            if decision.action in SEMANTIC_ACTIONS:
                approval = "semantic proposal recorded; no robot action"
            else:
                rejection = str(exc)
            return ShadowExecutor().execute(
                observation=observation,
                response=response_obj,
                decision=decision,
                candidates=candidates,
                chunk=None,
                plan=None,
                pre_check_result=None,
                approver_result=approval,
                rejection_reason=rejection,
                request=request,
            )
        waypoints = np.asarray([action.data for action in chunk.actions], dtype=np.float64)
        waypoints.flags.writeable = False
        precheck = embodiment.pre_check(waypoints)
        if precheck is not None:
            rejection = precheck
        else:
            with DobotExecutionSession(embodiment, allow_shadow_gripper=True) as session:
                policy = _StaticChunkPolicy(chunk)
                store: dict[str, Any] = {}
                if embodiment._chunk is not None:
                    chunk_identity = embodiment._chunk.identity
                for index in range(len(chunk)):
                    action = session.controller.next_action(
                        cast(Policy, policy), observation, index, store
                    )
                    reviewed = session.approver.review(action, store)
                    plan = embodiment.shadow_stage(reviewed)
                approval = "approved"
    except (SafetyRejected, ValueError, ConfigurationError) as exc:
        rejection = str(exc)
    finally:
        if embodiment._session is not None:
            embodiment.close()
    return ShadowExecutor().execute(
        observation=observation,
        response=response_obj,
        decision=decision,
        candidates=candidates,
        chunk=chunk,
        plan=plan,
        pre_check_result=precheck,
        approver_result=approval,
        rejection_reason=rejection,
        request=request,
        gripper_target_normalized=(
            embodiment.shadow_gripper_target
            if decision.gripper_proposal not in {"none", "hold"}
            else None
        ),
        chunk_identity=chunk_identity,
    )


class _StaticChunkPolicy:
    def __init__(self, chunk: ActionChunk) -> None:
        self.chunk = chunk

    def act(self, observation: Observation) -> ActionChunk:
        del observation
        return self.chunk
