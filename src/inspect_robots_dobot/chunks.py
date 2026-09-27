"""Local controller, approval ledger and fake-only staged execution lifecycle.

Execution labels are not capabilities. An active session and an in-memory,
one-use review record are required at step(), even for simulated execution.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict, dataclass, replace
from types import TracebackType
from typing import Any
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from inspect_robots.approver import Approver, ClampApprover, DeltaLimitApprover
from inspect_robots.log import EvalLog
from inspect_robots.logging.sink import LogSink
from inspect_robots.policy import Policy
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

from .audit import DobotAuditSink
from .camera import CameraReader
from .clock import Clock
from .config import DobotConfig
from .driver import FakeDobotDriver, MotionReceipt
from .embodiment import DobotEmbodiment
from .errors import ConfigurationError, SafetyRejected
from .motion import (
    CartesianMotionPlan,
    KeepoutBox,
    MotionExecutionResult,
    Waypoints,
    build_motion_plan,
    require_translation_profile,
    validate_cartesian_path,
)
from .types import PoseSI, RobotSnapshot

_BUFFER_KEY = "_controller_action_buffer"
_INFER_KEY = "_controller_inferences"
_LABELS = ("dobot_chunk_id", "chunk_index", "chunk_length", "chunk_final")
_STOP_LABELS = ("request_stop", "stop_reason", "stop_detail", "stop_hindsight")


@dataclass
class _Chunk:
    identity: str
    waypoints: Waypoints
    actions: tuple[Action, ...]
    start: RobotSnapshot
    stop: bool
    expected_labels: tuple[dict[str, Any], ...]
    next_index: int = 0
    issued: Action | None = None
    approved: Action | None = None
    issued_labels: dict[str, Any] | None = None


class StagedDobotEmbodiment(DobotEmbodiment):
    """Explicit Phase 3 mode. The Phase 2C per-waypoint simulator stays available.

    Use DobotExecutionSession as a context manager, including for manual playback.
    This class inherits the fake-only driver gate; it cannot accept a real driver.
    """

    def __init__(
        self,
        config: DobotConfig,
        *,
        driver: FakeDobotDriver | None = None,
        camera: CameraReader | None = None,
        clock: Clock | None = None,
        keepouts: tuple[KeepoutBox, ...] = (),
    ) -> None:
        super().__init__(config, driver=driver, camera=camera, clock=clock)
        self._allow_shadow_gripper = False
        require_translation_profile(self._profile())
        if not isinstance(keepouts, tuple) or not all(isinstance(b, KeepoutBox) for b in keepouts):
            raise ConfigurationError("keepouts must be an explicit tuple of KeepoutBox values")
        self.keepouts = keepouts
        self._session: DobotExecutionSession | None = None
        self._chunk: _Chunk | None = None
        self._plans: list[CartesianMotionPlan] = []
        self._results: list[MotionExecutionResult] = []
        self._shadow_gripper_target: float | None = None
        self.info = replace(
            self.info,
            docs=(self.info.docs or "")
            + " Phase 3 staged dry-run: intermediate approved steps only "
            "stage targets and return unchanged measured state. One complete straight XYZ "
            "chunk yields one prospective MovL and one fake move/settle. No live backend.",
        )

    @property
    def plans(self) -> tuple[CartesianMotionPlan, ...]:
        return tuple(self._plans)

    @property
    def execution_results(self) -> tuple[MotionExecutionResult, ...]:
        return tuple(self._results)

    @property
    def shadow_gripper_target(self) -> float | None:
        """Final normalized target from the last fully approved shadow chunk."""

        return self._shadow_gripper_target

    @property
    def pending_chunk_id(self) -> str | None:
        return None if self._chunk is None else self._chunk.identity

    def _require_session(self) -> DobotExecutionSession:
        if self._session is None or not self._session.active:
            raise SafetyRejected(
                "REJECT: staged execution requires an active DobotExecutionSession"
            )
        return self._session

    def abort_chunk(self, reason: str) -> None:
        pending, self._chunk = self._chunk, None
        if pending is not None:
            self._audit.append(
                {
                    "kind": "chunk_aborted",
                    "chunk_id": pending.identity,
                    "staged_count": pending.next_index,
                    "reason": reason,
                    "timestamp": self._clock.monotonic(),
                    "physical_sent": False,
                }
            )
        if self._session is not None:
            self._session.controller.discard()

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self.abort_chunk("reset")
        self._shadow_gripper_target = None
        return super().reset(scene, seed=seed)

    def close(self) -> None:
        self.abort_chunk("close")
        super().close()

    def pre_check(self, waypoints: npt.NDArray[np.float64]) -> str | None:
        message = super().pre_check(waypoints)
        if message is None:
            try:
                validate_cartesian_path(
                    tuple(tuple(float(v) for v in row) for row in waypoints),
                    self._reference(),
                    self._held_pose(),
                    self._profile(),
                    self._clock.monotonic(),
                    self.keepouts,
                )
            except SafetyRejected as exc:
                message = str(exc)
        self._audit.append(
            {
                "kind": "staged_pre_check",
                "accepted": message is None,
                "reason": message,
                "physical_sent": False,
            }
        )
        return message

    def _begin_chunk(self, chunk: ActionChunk) -> tuple[Action, ...]:
        self._require_session()
        if self._chunk is not None:
            raise SafetyRejected(
                "REJECT: stale incomplete chunk cannot be combined with a new chunk"
            )
        originals = tuple(chunk.actions)
        if not originals:
            raise SafetyRejected("REJECT: empty chunk")
        stop = any(bool(a.meta.get("request_stop")) for a in originals)
        if stop:
            if len(originals) != 1 or originals[0].meta.get("chunk_final"):
                raise SafetyRejected("REJECT: policy stop must be a standalone non-motion action")
        else:
            for i, action in enumerate(originals):
                marker = action.meta.get("chunk_final", False)
                if type(marker) is not bool or marker != (i == len(originals) - 1):
                    raise SafetyRejected(
                        "REJECT: exactly the last motion action must mark chunk_final"
                    )
        if any(any(k in a.meta for k in _LABELS[:3]) for a in originals):
            raise SafetyRejected("REJECT: policy must not supply execution identity metadata")
        values: list[tuple[float, ...]] = []
        for action in originals:
            data = np.asarray(action.data, dtype=np.float64)
            if data.shape != (7,) or not np.all(np.isfinite(data)):
                raise SafetyRejected("REJECT: each staged action must be finite shape (7,)")
            values.append(tuple(float(v) for v in data))
        waypoints = tuple(values)
        current = self._connected_driver().snapshot()
        if not stop:
            validate_cartesian_path(
                waypoints,
                current,
                self._held_pose(),
                self._profile(),
                self._clock.monotonic(),
                self.keepouts,
            )
        identity = uuid4().hex
        actions = tuple(
            Action(
                np.array(row),
                {
                    **copy.deepcopy(dict(original.meta)),
                    "dobot_chunk_id": identity,
                    "chunk_index": i,
                    "chunk_length": len(originals),
                    "chunk_final": not stop and i == len(originals) - 1,
                },
            )
            for i, (row, original) in enumerate(zip(waypoints, originals, strict=True))
        )
        labels = tuple(
            {k: copy.deepcopy(action.meta.get(k)) for k in (*_LABELS, *_STOP_LABELS)}
            for action in actions
        )
        self._chunk = _Chunk(identity, waypoints, actions, current, stop, labels)
        self._audit.append(
            {
                "kind": "chunk_registered",
                "chunk_id": identity,
                "chunk_length": len(actions),
                "waypoints": waypoints,
                "start_pose": current.pose.values,
                "timestamp": self._clock.monotonic(),
                "source_camera_timestamp": self._last_camera_time,
                "physical_sent": False,
            }
        )
        return actions

    def _check_issued(self, action: Action) -> _Chunk:
        self._require_session()
        pending = self._chunk
        if pending is None or pending.issued is None or pending.issued_labels is None:
            raise SafetyRejected("REJECT: no current issued chunk action (stale or duplicate)")
        if any(
            type(action.meta.get(k)) is not type(v) or action.meta.get(k) != v
            for k, v in pending.issued_labels.items()
        ):
            raise SafetyRejected("REJECT: execution identity/final/stop metadata was modified")
        if (
            not np.array_equal(action.data, pending.waypoints[pending.next_index])
            or action.meta.get("clamped")
            or action.meta.get("delta_clamped")
        ):
            raise SafetyRejected("REJECT: framework modified Cartesian waypoint; abort whole chunk")
        return pending

    def step(self, action: Action) -> StepResult:
        try:
            pending = self._check_issued(action)
            if pending.approved is not action:
                raise SafetyRejected("REJECT: missing one-use framework approval for this action")
            pending.approved = None
            if pending.stop:
                self.abort_chunk("policy stop")
                return StepResult(self._observe(), info={"dobot_staging": "policy_stop_no_motion"})
            self._validate_action(action)
            current = self._connected_driver().snapshot()
            if current.pose != pending.start.pose or current.command_id != pending.start.command_id:
                raise SafetyRejected(
                    "REJECT: measured start changed while staging; replan required"
                )
            pending.next_index += 1
            self._audit.append(
                {
                    "kind": "waypoint_staged",
                    "chunk_id": pending.identity,
                    "chunk_index": pending.next_index - 1,
                    "chunk_length": len(pending.actions),
                    "action": pending.waypoints[pending.next_index - 1],
                    "framework_approved": True,
                    "measured_pose": current.pose.values,
                    "timestamp": self._clock.monotonic(),
                    "physical_sent": False,
                }
            )
            if pending.next_index < len(pending.actions):
                pending.issued = None
                pending.issued_labels = None
                return StepResult(
                    self._observe(),
                    info={
                        "dobot_staging": {
                            "chunk_id": pending.identity,
                            "staged_count": pending.next_index,
                            "physical_sent": False,
                        }
                    },
                )
            plan = build_motion_plan(
                pending.identity,
                pending.waypoints,
                current,
                self._held_pose(),
                self._profile(),
                self._clock.monotonic(),
                self.keepouts,
            )
            self._plans.append(plan)
            self._audit.append(
                {
                    "kind": "prospective_movl",
                    **asdict(plan),
                    "protocol_command": plan.request.serialize(),
                }
            )
            # Sole target dispatch: inherited fake-only gate, settling and camera postselection.
            result = super().step(action)
            record = result.info["dobot_motion"]
            execution = MotionExecutionResult(
                plan.chunk_id,
                plan.request.serialize(),
                int(record["command_id"]),
                self._reference().pose,
                float(record["position_residual"]),
                float(record["orientation_residual"]),
                float(record["settle_duration"]),
            )
            self._results.append(execution)
            self._chunk = None
            return replace(result, info={**result.info, "dobot_execution": asdict(execution)})
        except BaseException as exc:
            self.abort_chunk(f"step failed: {type(exc).__name__}: {exc}")
            raise

    def _dispatch_linear(self, target: PoseSI, current: RobotSnapshot) -> MotionReceipt:
        pending = self._chunk
        if pending is None or pending.next_index != len(pending.actions):
            raise SafetyRejected("REJECT: complete staged plan required at fake dispatch")
        if current.pose != pending.start.pose or current.command_id != pending.start.command_id:
            raise SafetyRejected("REJECT: measured start changed at fake dispatch")
        verified = build_motion_plan(
            pending.identity,
            pending.waypoints,
            current,
            self._held_pose(),
            self._profile(),
            self._clock.monotonic(),
            self.keepouts,
        )
        if target != verified.final_pose_si or verified.request != self._plans[-1].request:
            raise SafetyRejected("REJECT: dispatch target differs from approved prospective plan")
        return super()._dispatch_linear(target, current)

    def shadow_stage(self, action: Action) -> CartesianMotionPlan | None:
        """Consume one approved waypoint without changing fake state or dispatching motion.

        Phase 6A needs the same identity, start stability, path and target checks as
        staged execution, but its terminal boundary is ``ShadowExecutor``.  This
        method intentionally never calls ``FakeDobotDriver.move_linear``.
        """
        pending = self._check_issued(action)
        if pending.approved is not action:
            raise SafetyRejected("REJECT: missing one-use framework approval for shadow staging")
        pending.approved = None
        if pending.stop:
            self.abort_chunk("policy stop")
            return None
        self._validate_action(action)
        current = self._connected_driver().snapshot()
        if current.pose != pending.start.pose or current.command_id != pending.start.command_id:
            raise SafetyRejected("REJECT: measured start changed while shadow staging")
        pending.next_index += 1
        self._audit.append(
            {
                "kind": "shadow_waypoint_staged",
                "chunk_id": pending.identity,
                "chunk_index": pending.next_index - 1,
                "chunk_length": len(pending.actions),
                "action": pending.waypoints[pending.next_index - 1],
                "framework_approved": True,
                "physical_sent": False,
                "shadow_only": True,
            }
        )
        pending.issued = None
        pending.issued_labels = None
        if pending.next_index < len(pending.actions):
            return None
        final = np.asarray(pending.waypoints[-1], dtype=np.float64)
        arm_reference = np.asarray((*current.pose.xyz, 0.0, 0.0, 0.0), dtype=np.float64)
        self._shadow_gripper_target = float(final[6])
        if all(
            np.allclose(np.asarray(row[:6], dtype=np.float64), arm_reference, atol=1e-12, rtol=0.0)
            for row in pending.waypoints
        ):
            self._audit.append(
                {
                    "kind": "shadow_gripper_target",
                    "chunk_id": pending.identity,
                    "chunk_length": len(pending.actions),
                    "target_normalized": self._shadow_gripper_target,
                    "execution": False,
                    "physical_sent": False,
                }
            )
            self._chunk = None
            return None
        plan = build_motion_plan(
            pending.identity,
            pending.waypoints,
            current,
            self._held_pose(),
            self._profile(),
            self._clock.monotonic(),
            self.keepouts,
        )
        self._plans.append(plan)
        self._audit.append(
            {
                "kind": "shadow_prospective_movl",
                **asdict(plan),
                "protocol_command": plan.request.serialize(),
                "execution": False,
                "physical_sent": False,
            }
        )
        self._chunk = None
        return plan


class DobotChunkController:
    """Pinned Controller protocol with complete playback and explicit action issuance."""

    def __init__(
        self, embodiment: StagedDobotEmbodiment, replan_interval: int | None = None
    ) -> None:
        if replan_interval is not None and (
            type(replan_interval) is not int or replan_interval < 1
        ):
            raise ConfigurationError("replan_interval must be a positive integer or None")
        self.embodiment, self.replan_interval = embodiment, replan_interval
        self._store: dict[str, Any] | None = None
        self._next_t: int | None = None
        self.chunk_metadata: dict[str, Any] = {}

    def discard(self) -> None:
        if self._store is not None:
            self._store.pop(_BUFFER_KEY, None)
            # A staged waypoint was approved, not executed. Rewind generic delta
            # history to the last measured observation when abandoning a proposal.
            sample = self.embodiment._last
            if sample is not None:
                DeltaLimitApprover.rewind_reference(
                    self._store, np.array((*sample.pose.xyz, 0.0, 0.0, 0.0, 0.0))
                )
        self._store, self._next_t = None, None
        self.chunk_metadata = {}

    def next_action(
        self, policy: Policy, observation: Observation, t: int, store: dict[str, Any]
    ) -> Action:
        try:
            self.embodiment._require_session()
            if self.embodiment._session is None or self.embodiment._session.controller is not self:
                raise SafetyRejected("REJECT: controller is not bound to the active session")
            if self._store is not None and self._store is not store:
                raise SafetyRejected("REJECT: trial store changed before lifecycle cleanup")
            if self._next_t is not None and t != self._next_t:
                raise SafetyRejected("REJECT: nonconsecutive controller step")
            pending = self.embodiment._chunk
            if pending is not None and pending.issued is not None:
                raise SafetyRejected(
                    "REJECT: previous action not consumed; cannot skip approval/step"
                )
            self._store = store
            buffer: deque[Action] = store.setdefault(_BUFFER_KEY, deque())
            if not buffer:
                if pending is not None:
                    raise SafetyRejected("REJECT: incomplete chunk lost its action buffer")
                chunk = policy.act(observation)
                if self.replan_interval is not None and self.replan_interval < len(chunk):
                    raise ConfigurationError(
                        "Dobot staged MovL execution requires full-chunk playback; "
                        f"replan_interval={self.replan_interval} would truncate chunk "
                        f"length={len(chunk)} before its physical commit"
                    )
                actions = self.embodiment._begin_chunk(chunk)
                self.chunk_metadata = copy.deepcopy(dict(chunk.meta))
                buffer.extend(actions)
                store.setdefault(_INFER_KEY, []).append((chunk.inference_latency_s, len(actions)))
            pending = self.embodiment._chunk
            if pending is None:
                raise SafetyRejected("REJECT: unregistered controller buffer")
            action = buffer.popleft()
            if action is not pending.actions[pending.next_index]:
                raise SafetyRejected("REJECT: controller buffer/action order changed")
            pending.issued = action
            pending.issued_labels = pending.expected_labels[pending.next_index]
            self.embodiment._check_issued(action)
            self._next_t = t + 1
            return action
        except BaseException as exc:
            self.embodiment.abort_chunk(f"controller failed: {type(exc).__name__}: {exc}")
            raise


class DobotChunkApprover:
    """Wrap the entire framework chain; record approval only after every guard succeeds."""

    def __init__(self, embodiment: StagedDobotEmbodiment, guards: tuple[Approver, ...]) -> None:
        self.embodiment, self.guards = embodiment, guards

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        try:
            pending = self.embodiment._check_issued(action)
            session = self.embodiment._require_session()
            if session.approver is not self or session.controller._store is not store:
                raise SafetyRejected("REJECT: approval belongs to a different session/store")
            if pending.issued is not action or pending.approved is not None:
                raise SafetyRejected("REJECT: duplicate or unissued approval request")
            reviewed = action
            for guard in self.guards:
                incoming = reviewed
                reviewed = guard.review(incoming, store)
                self.embodiment._check_issued(incoming)  # also detect in-place edits
                self.embodiment._check_issued(reviewed)
                self.embodiment._audit.append(
                    {
                        "kind": "waypoint_approval",
                        "chunk_id": pending.identity,
                        "chunk_index": pending.next_index,
                        "guard": type(guard).__name__,
                        "accepted": True,
                        "timestamp": self.embodiment._clock.monotonic(),
                        "physical_sent": False,
                    }
                )
            pending.approved = reviewed
            return reviewed
        except BaseException as exc:
            self.embodiment.abort_chunk(f"approval failed: {type(exc).__name__}: {exc}")
            raise


class _StagedAuditSink(DobotAuditSink):
    def __init__(self, embodiment: StagedDobotEmbodiment) -> None:
        super().__init__(embodiment)
        self.staged = embodiment

    def on_trial_end(self, record: TrialRecord) -> None:
        self.staged.abort_chunk(f"trial ended: {record.termination_reason or record.status}")
        super().on_trial_end(record)


class DobotExecutionSession:
    """Required lifecycle; eval() uses the real public framework, not a rollout fork.

    Manual callers must keep controller/review/step inside the with block. Exception
    unwinding, even outside framework callbacks, discards partial chunks and closes.
    Custom guards are appended to mandatory clamp/delta/local guards, never replace them.
    """

    def __init__(
        self,
        embodiment: StagedDobotEmbodiment,
        *,
        replan_interval: int | None = None,
        extra_approvers: tuple[Approver, ...] = (),
        allow_shadow_gripper: bool = False,
    ) -> None:
        self.embodiment = embodiment
        self.allow_shadow_gripper = allow_shadow_gripper
        self.controller = DobotChunkController(embodiment, replan_interval)
        space = embodiment.info.action_space
        self.approver = DobotChunkApprover(
            embodiment,
            (
                ClampApprover(space),
                DeltaLimitApprover(space),
                *extra_approvers,
                embodiment.contribute_guardrails(space).approvers[0][1],
            ),
        )
        self.sink = _StagedAuditSink(embodiment)
        self.active = False

    def __enter__(self) -> DobotExecutionSession:
        if self.active or self.embodiment._session is not None:
            raise ConfigurationError("staged embodiment already belongs to a session")
        self.active = True
        self.embodiment._session = self
        self.embodiment._allow_shadow_gripper = self.allow_shadow_gripper
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.embodiment.close()
        finally:
            self.embodiment._allow_shadow_gripper = False
            self.embodiment._session = None
            self.active = False

    def evaluate(
        self, task: Task, *, policy: Policy, log_dir: str, sinks: list[LogSink] | None = None
    ) -> list[EvalLog]:
        """No user-supplied controller/approver override; mandatory guards stay installed."""
        from inspect_robots import eval as robot_eval

        self.embodiment._require_session()
        try:
            return robot_eval(
                task,
                policy=policy,
                embodiment=self.embodiment,
                controller=self.controller,
                approver=self.approver,
                log_dir=log_dir,
                sinks=[self.sink, *(sinks or [])],
            )
        finally:
            self.embodiment.abort_chunk("evaluation returned or raised")
