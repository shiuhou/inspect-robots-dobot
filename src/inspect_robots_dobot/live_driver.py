"""Explicit single-use live backend, developed and validated with mocks only.

The generic agent/staged embodiment cannot select this driver. The terminal CLI
requires deliberate current-session readiness and review. Software Stop remains
best-effort; the physically present operator's tested E-stop is the hard channel.
"""

from __future__ import annotations

import copy
import math
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any
from uuid import uuid4

from .clock import Clock, SystemClock
from .config import ConnectionConfig, LocalMicroMoveSettings, SafetyProfile
from .errors import (
    CommandRejected,
    ConfigurationError,
    DriverFault,
    MotionNotAuthorized,
    SafetyRejected,
    TransportTimeout,
)
from .live_authority import AuthorityDetails, LiveMotionAuthority, plan_digest
from .live_channel import MotionCancelled, _LiveDashboardChannel
from .live_profile import LiveMotionProfile, OperatorReadiness
from .local_envelope import LocalEnvelopeSafety, LocalMicroMoveEnvelope, envelope_safety
from .motion import CartesianMotionPlan, build_motion_plan, decimal_micro_profile
from .protocol import (
    DashboardResponse,
    Query,
    decode_command_id,
    decode_errors,
    decode_joints,
    decode_mode,
    decode_pose,
)
from .safety import validate_position, validate_snapshot, validate_target
from .transport import Deadline, SocketFactory, open_socket
from .types import NativeDecimalPose, RobotMode, RobotSnapshot
from .units import orientation_distance, translation_distance


class LiveMotionState(str, Enum):
    IDLE = "IDLE"
    ARMED_FOR_ONE_PLAN = "ARMED_FOR_ONE_PLAN"
    SENDING = "SENDING"
    ACCEPTED = "ACCEPTED"
    EXECUTING = "EXECUTING"
    SETTLING = "SETTLING"
    COMPLETED = "COMPLETED"
    REJECTED_BEFORE_SEND = "REJECTED_BEFORE_SEND"
    AMBIGUOUS_ACCEPTANCE = "AMBIGUOUS_ACCEPTANCE"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"
    FAULTED = "FAULTED"
    TIMEOUT = "TIMEOUT"


_TRANSITIONS = {
    LiveMotionState.IDLE: {
        LiveMotionState.ARMED_FOR_ONE_PLAN,
        LiveMotionState.REJECTED_BEFORE_SEND,
    },
    LiveMotionState.ARMED_FOR_ONE_PLAN: {
        LiveMotionState.SENDING,
        LiveMotionState.REJECTED_BEFORE_SEND,
    },
    LiveMotionState.SENDING: {
        LiveMotionState.REJECTED_BEFORE_SEND,
        LiveMotionState.ACCEPTED,
        LiveMotionState.AMBIGUOUS_ACCEPTANCE,
        LiveMotionState.FAULTED,
    },
    LiveMotionState.ACCEPTED: {
        LiveMotionState.EXECUTING,
        LiveMotionState.SETTLING,
        LiveMotionState.ABORTING,
        LiveMotionState.FAULTED,
        LiveMotionState.TIMEOUT,
    },
    LiveMotionState.EXECUTING: {
        LiveMotionState.SETTLING,
        LiveMotionState.ABORTING,
        LiveMotionState.FAULTED,
        LiveMotionState.TIMEOUT,
    },
    LiveMotionState.SETTLING: {
        LiveMotionState.EXECUTING,
        LiveMotionState.COMPLETED,
        LiveMotionState.ABORTING,
        LiveMotionState.FAULTED,
        LiveMotionState.TIMEOUT,
    },
    LiveMotionState.AMBIGUOUS_ACCEPTANCE: {LiveMotionState.ABORTING},
    LiveMotionState.FAULTED: {LiveMotionState.ABORTING},
    LiveMotionState.TIMEOUT: {LiveMotionState.ABORTING},
    LiveMotionState.ABORTING: {LiveMotionState.ABORTED, LiveMotionState.FAULTED},
    LiveMotionState.COMPLETED: {LiveMotionState.FAULTED},
}


@dataclass(frozen=True)
class LiveMotionResult:
    state: LiveMotionState
    failure_cause: str | None
    command_id: int | None
    motion_write_attempts: int
    stop_attempts: int
    standstill_confirmed: bool
    final_sample: RobotSnapshot | None
    authority_consumed: bool
    acceptance_ambiguous: bool
    transport_is_mock: bool
    physical_send_attempted: bool
    connections_closed: bool


class LiveDobotMotionDriver:
    """One plan, one motion attempt, one abort episode per instance. No reset/retry.

    Constructing this class performs no I/O. connect() requires separate explicit
    connection permission and only opens one stream. Ownership/power/enable stay
    operator-controlled. Operator readiness is attached to this connected session.
    """

    is_simulated = False

    def __init__(
        self,
        connection: ConnectionConfig,
        profile: LiveMotionProfile,
        *,
        allow_motion: bool = False,
        socket_factory: SocketFactory = open_socket,
        clock: Clock | None = None,
        transport_is_mock: bool = False,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        failure_alert: Callable[[str], None] | None = None,
    ) -> None:
        if type(allow_motion) is not bool or type(transport_is_mock) is not bool:
            raise ConfigurationError("runtime flags must be explicit booleans")
        if transport_is_mock and socket_factory is open_socket:
            raise ConfigurationError("mock mode requires an injected transport")
        if not isinstance(profile, LiveMotionProfile):
            raise ConfigurationError("complete LiveMotionProfile required")
        if isinstance(profile.safety, LocalEnvelopeSafety):
            raise ConfigurationError(
                "a bound envelope cannot be imported into another live session"
            )
        self._connection, self._profile = connection, profile
        self._allow_motion, self._transport_is_mock = allow_motion, transport_is_mock
        self.clock = clock or SystemClock()
        self._event_sink, self._failure_alert = event_sink, failure_alert
        self._handling_failure = False
        self._cancel = threading.Event()
        self._worker = threading.Lock()
        self._channel = _LiveDashboardChannel(
            connection, profile, self.clock, socket_factory, self._cancel, self._audit_event
        )
        self._state = LiveMotionState.IDLE
        self._events: list[dict[str, Any]] = []
        self._session = uuid4().hex
        self._readiness: OperatorReadiness | None = None
        self._authority: LiveMotionAuthority | None = None
        self._issued_details: AuthorityDetails | None = None
        self._consumed = False
        self._ever_connected = False
        self._closed = False
        self._connections_closed = False
        self._result: LiveMotionResult | None = None
        self._bound_envelope: LocalMicroMoveEnvelope | None = None

    @property
    def validation_safety(self) -> SafetyProfile:
        if not isinstance(self.profile.safety, SafetyProfile):
            raise MotionNotAuthorized("local envelope has not been bound to a measured start")
        return self.profile.safety

    @property
    def connection(self) -> ConnectionConfig:
        return self._connection

    @property
    def profile(self) -> LiveMotionProfile:
        return self._profile

    @property
    def allow_motion(self) -> bool:
        return self._allow_motion

    @property
    def transport_is_mock(self) -> bool:
        return self._transport_is_mock

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if not self._worker.acquire(blocking=False):
            raise MotionNotAuthorized("another session operation is active")
        try:
            yield
        finally:
            try:
                # A concurrent close/reset while connecting/arming cannot leave
                # an open socket or a newly minted capability behind.
                if self._cancel.is_set():
                    self._invalidate()
                    self._closed = True
                    if self._state in (LiveMotionState.IDLE, LiveMotionState.ARMED_FOR_ONE_PLAN):
                        self._transition(LiveMotionState.REJECTED_BEFORE_SEND, "session cancelled")
                    self._channel.close()
            finally:
                self._worker.release()

    @property
    def state(self) -> LiveMotionState:
        return self._state

    @property
    def audit_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            copy.deepcopy(self._events + [{"kind": "wire", **r} for r in self._channel.records])
        )

    @property
    def result(self) -> LiveMotionResult | None:
        return self._result

    @property
    def connections_closed(self) -> bool:
        return self._connections_closed

    def _record(self, kind: str, **values: Any) -> None:
        self._events.append(
            {
                "kind": kind,
                "timestamp": self.clock.monotonic(),
                "session_nonce": self._session,
                **values,
            }
        )
        self._audit_event(copy.deepcopy(self._events[-1]))

    def _audit_event(self, event: dict[str, Any]) -> None:
        if self._event_sink is not None:
            try:
                self._event_sink(event)
            except Exception:
                # Persistence failure must not prevent an already-required Stop
                # or socket cleanup. Normal execution propagates and aborts.
                if not self._handling_failure:
                    raise

    def _transition(self, state: LiveMotionState, reason: str) -> None:
        if state is self._state:
            return
        if state not in _TRANSITIONS.get(self._state, set()):
            raise DriverFault(f"forbidden transition {self._state.value} -> {state.value}")
        previous, self._state = self._state, state
        self._record("transition", previous=previous.value, state=state.value, reason=reason)

    def connect(self, *, allow_connection: bool = False) -> None:
        with self._exclusive():
            self._connect(allow_connection=allow_connection)

    def _connect(self, *, allow_connection: bool = False) -> None:
        if allow_connection is not True:
            raise MotionNotAuthorized("explicit current-session connection permission required")
        if self._closed or self._ever_connected or self._cancel.is_set():
            raise MotionNotAuthorized("closed/used live session cannot reconnect")
        if self.connection.expected_protocol_version != "4.6.5":
            raise ConfigurationError("only V4.6.5 is source-verified")
        self._ever_connected = True
        self._channel.connect()
        self._record("connected", host=self.connection.host, automatic_commands=[])

    def confirm_readiness(self, readiness: OperatorReadiness) -> None:
        with self._exclusive():
            self._confirm_readiness(readiness)

    def _confirm_readiness(self, readiness: OperatorReadiness) -> None:
        if not self._channel.connected or self._state is not LiveMotionState.IDLE:
            raise MotionNotAuthorized("readiness must be confirmed on an idle connected session")
        readiness.require()
        self._readiness = readiness
        self._record("operator_readiness", **asdict(readiness), provenance="operator/controller UI")

    def request_stop(self) -> None:
        """Thread-safe cancellation request; worker performs bounded Stop, never a second writer."""
        self._cancel.set()

    def reset(self) -> None:
        self.close()
        raise MotionNotAuthorized(
            "live session reset invalidates authority; construct a new reviewed session"
        )

    def close(self) -> None:
        if self._closed:
            return  # Never rewrite a previous close failure as success on a second call.
        self.request_stop()
        # The worker checks cancellation between bounded I/O operations. Do not race
        # its Stop send by closing underneath it; the executing worker owns cleanup.
        if not self._worker.acquire(blocking=False):
            return
        try:
            self._handling_failure = True
            self._invalidate()
            self._closed = True
            if self._state in (LiveMotionState.IDLE, LiveMotionState.ARMED_FOR_ONE_PLAN):
                self._transition(LiveMotionState.REJECTED_BEFORE_SEND, "closed without sending")
            self._channel.close()
            self._connections_closed = True
        finally:
            self._worker.release()

    def _invalidate(self) -> None:
        self._authority = None
        self._issued_details = None
        self._readiness = None
        self._bound_envelope = None

    def _gates(self) -> None:
        if self.allow_motion is not True:
            raise MotionNotAuthorized("runtime --allow-motion is absent")
        if self._closed or not self._channel.connected:
            raise MotionNotAuthorized("live session is not connected")
        if self._cancel.is_set():
            raise MotionCancelled("operator cancelled session")
        if self._readiness is None:
            raise MotionNotAuthorized("current-session operator readiness is absent")
        self._readiness.require()
        c, s = self.connection, self.profile.safety
        # After the first measurement the local settings are replaced by the
        # session-bound envelope projection. Keep the narrowly-scoped exception
        # attached to that same session rather than treating the projection as a
        # general production profile.
        local_micro_move = isinstance(s, LocalMicroMoveSettings) or self._bound_envelope is not None
        if c.tcp_control_owned is not True and not (
            local_micro_move and self._readiness.tcp_control_owned
        ):
            raise MotionNotAuthorized(
                "verified TCP ownership declaration is required; no RequestControl"
            )
        if not local_micro_move and (
            not c.protocol_compatibility_confirmed or c.controller_firmware != self.profile.firmware
        ):
            raise MotionNotAuthorized("operator-verified firmware compatibility is required")
        if (c.user_frame, c.tool_frame) != (s.user_frame, s.tool_frame):
            raise MotionNotAuthorized("verified explicit user/tool frames must match the profile")

    def _validate_plan(self, plan: CartesianMotionPlan) -> None:
        if (
            not isinstance(plan, CartesianMotionPlan)
            or plan.starting_measured_state.joints_synthetic
        ):
            raise SafetyRejected("live plan must originate from measured, non-synthetic state")
        start = plan.starting_measured_state
        envelope = self._bound_envelope
        if plan.local_envelope != envelope:
            raise SafetyRejected("plan envelope does not belong to this measured session")
        if envelope is not None and (
            envelope.session_nonce != self._session
            or plan.chunk_id != envelope.chunk_id
            or self.clock.monotonic() >= envelope.expires_at
            or plan.request.native_decimal != envelope.target_native
        ):
            raise SafetyRejected("local envelope session/target/expiry mismatch")
        regenerated = build_motion_plan(
            plan.chunk_id,
            plan.staged_waypoints,
            start,
            start.pose,
            self.validation_safety,
            envelope.bound_at if envelope is not None else start.observed_at,
            self.profile.keepouts,
            micro_move_direction=plan.micro_move_direction,
        )
        if asdict(regenerated) != asdict(plan):
            raise SafetyRejected("reviewed plan fields/validation/native target are inconsistent")
        if (
            translation_distance(start.pose, plan.final_pose_si)
            <= self.profile.start_position_tolerance
        ):
            raise SafetyRejected("micro-move displacement must exceed start-position uncertainty")

    def _sample(self, deadline: Deadline, *, stopping: bool = False) -> RobotSnapshot:
        started = self.clock.monotonic()

        def query(q: Query) -> DashboardResponse:
            return self._channel.query(q, deadline=deadline, stop_monitor=stopping)

        before = decode_mode(query(Query.MODE))
        errors = decode_errors(query(Query.ERRORS))
        if before not in (RobotMode.ENABLED_IDLE, RobotMode.RUNNING) or errors:
            self._record("controller_fault", mode=int(before), active_errors=errors)
            raise DriverFault(f"mode={before.name}, alarms={errors}; pose may be unavailable")
        pose_response = query(Query.POSE)
        pose = decode_pose(pose_response)
        native_decimal = NativeDecimalPose(
            tuple(v.strip() for v in pose_response.payload.split(","))
        )
        joints = decode_joints(query(Query.JOINTS))
        command_id = decode_command_id(query(Query.COMMAND_ID))
        after = decode_mode(query(Query.MODE))
        errors = decode_errors(query(Query.ERRORS))
        if after not in (RobotMode.ENABLED_IDLE, RobotMode.RUNNING) or errors:
            raise DriverFault(
                f"controller state during acquisition: {before}/{after}, alarms={errors}"
            )
        c = self.connection
        if c.user_frame is None or c.tool_frame is None:
            raise ConfigurationError("live samples require explicit user/tool frames")
        # RUNNING->IDLE during acquisition is normal, but that sample is not
        # convergence evidence: pose could have been acquired before arrival.
        conservative_mode = after if before == after else RobotMode.RUNNING
        sample = RobotSnapshot(
            pose,
            joints,
            conservative_mode,
            errors,
            command_id,
            started,
            c.user_frame,
            c.tool_frame,
            native_decimal=native_decimal,
        )
        self._validate_measurement(sample)
        self._record("measurement", **asdict(sample), stopping=stopping)
        return sample

    def _validate_measurement(self, sample: RobotSnapshot) -> None:
        age = self.clock.monotonic() - sample.observed_at
        if not math.isfinite(age) or not 0 <= age <= self.profile.max_measurement_age:
            raise DriverFault(f"stale/invalid measurement age={age}")
        if sample.errors or sample.mode not in (RobotMode.ENABLED_IDLE, RobotMode.RUNNING):
            raise DriverFault(f"invalid controller mode/errors: {sample.mode}/{sample.errors}")
        if (sample.user_frame, sample.tool_frame) != (
            self.profile.safety.user_frame,
            self.profile.safety.tool_frame,
        ):
            raise DriverFault("measurement frames changed")
        if not all(math.isfinite(v) for v in (*sample.pose.values, *sample.joints)):
            raise DriverFault("measurement contains NaN/Inf")
        if isinstance(self.profile.safety, SafetyProfile):
            validate_position(sample.pose, self.profile.safety, label="measured TCP")
        elif self._state is not LiveMotionState.IDLE or self._bound_envelope is not None:
            raise MotionNotAuthorized("unbound local profile cannot monitor or execute motion")

    def measure_start(self) -> RobotSnapshot:
        with self._exclusive():
            return self._measure_start()

    def _measure_start(self) -> RobotSnapshot:
        self._gates()
        if self._state is not LiveMotionState.IDLE:
            raise MotionNotAuthorized("start measurement requires a new idle session")
        if self._bound_envelope is not None:
            raise MotionNotAuthorized("local start/envelope cannot be remeasured or regenerated")
        sample = self._sample(Deadline(self.clock, self.profile.max_measurement_age))
        if isinstance(self.profile.safety, LocalMicroMoveSettings):
            if sample.native_decimal is None:
                raise SafetyRejected("native decimal measurement required for local envelope")
            now = self.clock.monotonic()
            envelope = LocalMicroMoveEnvelope(
                self.profile.safety,
                sample,
                sample.native_decimal.translated_10mm("+Z"),
                self._session,
                uuid4().hex,
                now,
                now + self.profile.authority_lifetime,
            )
            self._profile = replace(self.profile, safety=envelope_safety(envelope))
            self._bound_envelope = envelope
            self._record(
                "local_envelope_bound",
                envelope=asdict(envelope),
                local_low=envelope.low,
                local_high=envelope.high,
                table_z_known=False,
                general_workspace=False,
            )
        validate_snapshot(sample, self.validation_safety, self.clock.monotonic())
        return sample

    def _validate_start(self, plan: CartesianMotionPlan, sample: RobotSnapshot) -> None:
        self._validate_measurement(sample)
        validate_snapshot(sample, self.validation_safety, self.clock.monotonic())
        start = plan.starting_measured_state
        if sample.command_id != start.command_id:
            raise SafetyRejected("command ID changed since plan review")
        if translation_distance(sample.pose, start.pose) > self.profile.start_position_tolerance:
            raise SafetyRejected("measured start position changed beyond reviewed tolerance")
        if orientation_distance(sample.pose, start.pose) > self.profile.start_orientation_tolerance:
            raise SafetyRejected("measured start orientation changed beyond reviewed tolerance")
        # Sensor noise tolerance does not change or regenerate the reviewed target.
        validation_profile = self.validation_safety
        if plan.micro_move_direction is not None:
            native = plan.request.native_decimal
            if native is None or sample.native_decimal is None:
                raise SafetyRejected("10mm experiment requires native decimal measurements")
            validation_profile = decimal_micro_profile(
                sample.native_decimal, native, validation_profile
            )
        validate_target(plan.final_pose_si, sample, validation_profile, self.clock.monotonic())
        if translation_distance(sample.pose, plan.final_pose_si) > 0.020:
            raise SafetyRejected("actual start-to-target displacement exceeds 20mm")
        if any(box.intersects(sample.pose, plan.final_pose_si) for box in self.profile.keepouts):
            raise SafetyRejected("actual start-to-target segment intersects keepout")

    def arm(
        self, plan: CartesianMotionPlan, *, confirmation: str, expires_at: float | None = None
    ) -> LiveMotionAuthority:
        with self._exclusive():
            return self._arm(plan, confirmation=confirmation, expires_at=expires_at)

    def _arm(
        self, plan: CartesianMotionPlan, *, confirmation: str, expires_at: float | None = None
    ) -> LiveMotionAuthority:
        try:
            self._gates()
            if self._state is not LiveMotionState.IDLE or self._authority is not None:
                raise MotionNotAuthorized("only one authority may be issued per session")
            if confirmation != "MOVE ONCE":
                raise MotionNotAuthorized("exact operator confirmation MOVE ONCE required")
            self._validate_plan(plan)
            self._validate_start(
                plan, self._sample(Deadline(self.clock, self.profile.max_measurement_age))
            )
            self._gates()
            now = self.clock.monotonic()
            expiry = now + self.profile.authority_lifetime if expires_at is None else expires_at
            if self._bound_envelope is not None:
                if expires_at is not None and expires_at != self._bound_envelope.expires_at:
                    raise MotionNotAuthorized(
                        "local envelope expiry cannot be extended or replaced"
                    )
                expiry = self._bound_envelope.expires_at
            if (
                not math.isfinite(expiry)
                or not now < expiry <= now + self.profile.authority_lifetime
            ):
                raise MotionNotAuthorized(
                    "reviewed authority expiry is elapsed or exceeds lifetime"
                )
            details = AuthorityDetails(
                plan_digest(plan, self.profile, self.connection),
                self._session,
                now,
                expiry,
            )
            authority = LiveMotionAuthority(details)
            self._authority, self._issued_details = authority, details
            self._transition(LiveMotionState.ARMED_FOR_ONE_PLAN, "operator confirmed exact plan")
            self._record("authority_issued", **asdict(details))
            return authority
        except BaseException:
            if self._state is LiveMotionState.IDLE:
                self._transition(LiveMotionState.REJECTED_BEFORE_SEND, "arming gates rejected")
            self._invalidate()
            raise

    def _verify_authority(self, plan: CartesianMotionPlan, authority: LiveMotionAuthority) -> None:
        self._gates()
        details = self._issued_details
        if authority is not self._authority or details is None or self._consumed:
            raise MotionNotAuthorized(
                "authority is absent, foreign, expired session or already consumed"
            )
        if authority.details != details or details.session_nonce != self._session:
            raise MotionNotAuthorized("authority details/session were modified")
        if not details.issued_at <= self.clock.monotonic() < details.expires_at:
            raise MotionNotAuthorized("one-use authority expired")
        if details.plan_digest != plan_digest(plan, self.profile, self.connection):
            raise MotionNotAuthorized("authority plan/profile/host digest mismatch")

    def _pause(self, deadline: Deadline, *, stopping: bool = False) -> None:
        if not stopping and self._cancel.is_set():
            raise MotionCancelled("operator cancelled")
        # <=100ms configured poll interval; I/O has its own bounded cancellation points.
        self.clock.sleep(min(self.profile.safety.poll_interval, deadline.remaining()))

    def _monitor(self, plan: CartesianMotionPlan, command_id: int) -> RobotSnapshot:
        deadline = Deadline(self.clock, self.profile.safety.settle_timeout)
        count = 0
        last_time = -float("inf")
        while True:
            sample = self._sample(deadline)
            if self._cancel.is_set():
                raise MotionCancelled("operator cancelled during measurement")
            pos = translation_distance(sample.pose, plan.final_pose_si)
            angle = orientation_distance(sample.pose, plan.final_pose_si)
            if (
                orientation_distance(sample.pose, plan.starting_measured_state.pose)
                > self.profile.start_orientation_tolerance
            ):
                raise DriverFault("unexpected orientation drift in translation-only execution")
            if any(box.intersects(sample.pose, sample.pose) for box in self.profile.keepouts):
                raise DriverFault("measured TCP entered keepout")
            converged = (
                sample.command_id == command_id
                and sample.mode is RobotMode.ENABLED_IDLE
                and pos <= self.profile.safety.position_tolerance
                and angle <= self.profile.safety.orientation_tolerance
                and sample.observed_at > last_time
            )
            count = count + 1 if converged else 0
            last_time = sample.observed_at
            self._transition(
                LiveMotionState.SETTLING if converged else LiveMotionState.EXECUTING,
                "measured convergence" if converged else "awaiting ID/mode/pose convergence",
            )
            self._record(
                "residual",
                command_id=sample.command_id,
                accepted_id=command_id,
                position=pos,
                orientation=angle,
                consecutive=count,
                measurement_time=sample.observed_at,
            )
            deadline.remaining()
            if count >= self.profile.consecutive_samples:
                self._transition(LiveMotionState.COMPLETED, "consecutive measured arrival samples")
                return sample
            self._pause(deadline)

    def _abort(self, reason: str) -> tuple[bool, RobotSnapshot | None]:
        self._transition(LiveMotionState.ABORTING, reason)
        previous: RobotSnapshot | None = None
        anchor: RobotSnapshot | None = None
        count = 0
        try:
            self._channel.stop_once()
            self._record("stop_acknowledged", standstill_confirmed=False)
            deadline = Deadline(self.clock, self.profile.stop_timeout)
            while True:
                sample = self._sample(deadline, stopping=True)
                stable = (
                    sample.mode is RobotMode.ENABLED_IDLE
                    and previous is not None
                    and anchor is not None
                    and sample.observed_at > previous.observed_at
                    and sample.command_id == anchor.command_id
                    and translation_distance(sample.pose, anchor.pose)
                    <= self.profile.standstill_position_tolerance
                    and orientation_distance(sample.pose, anchor.pose)
                    <= self.profile.standstill_orientation_tolerance
                )
                count = count + 1 if stable else (1 if sample.mode is RobotMode.ENABLED_IDLE else 0)
                if not stable:
                    anchor = sample
                previous = sample
                self._record("standstill_sample", **asdict(sample), consecutive=count)
                deadline.remaining()
                if count >= self.profile.consecutive_samples:
                    self._transition(LiveMotionState.ABORTED, "measured idle standstill after Stop")
                    return True, sample
                self._pause(deadline, stopping=True)
        except BaseException as exc:
            self._record("stop_unverified", reason=str(exc), standstill_confirmed=False)
            self._transition(
                LiveMotionState.FAULTED, "Stop/standstill unverified; human inspection required"
            )
            return False, previous

    def execute(
        self, plan: CartesianMotionPlan, authority: LiveMotionAuthority
    ) -> LiveMotionResult:
        if not self._worker.acquire(blocking=False):
            raise MotionNotAuthorized("motion worker already active")
        if self._closed or self._consumed:
            self._worker.release()
            raise MotionNotAuthorized("terminal/used live session; no resend or second abort")
        command_id = None
        final = None
        standstill = False
        ambiguous = False
        cause = None
        try:
            if self._state is not LiveMotionState.ARMED_FOR_ONE_PLAN:
                raise MotionNotAuthorized("session is not armed for an unused plan")
            self._verify_authority(plan, authority)
            self._validate_plan(plan)
            sample = self._sample(Deadline(self.clock, self.profile.max_measurement_age))
            self._validate_start(plan, sample)

            def consume_before_send() -> None:
                self._validate_start(plan, sample)
                self._verify_authority(plan, authority)
                self._consumed = True
                self._transition(
                    LiveMotionState.SENDING, "authority consumed before one write attempt"
                )
                self._record(
                    "authority_consumed",
                    plan_digest=authority.details.plan_digest,
                    raw_request=plan.request.serialize(),
                )
                # Durable review/audit I/O can take time. Recheck after it, with no
                # more file I/O before sendall. Consumption never authorizes a retry.
                self._validate_start(plan, sample)
                details = authority.details
                if self.clock.monotonic() >= details.expires_at:
                    raise MotionNotAuthorized("authority expired while persisting send evidence")
                self._gates()

            command_id = self._channel._move_once(plan.request, consume_before_send)
            self._transition(
                LiveMotionState.ACCEPTED, f"queue ResultID={command_id}; not completion"
            )
            final = self._monitor(plan, command_id)
        except BaseException as exc:
            cause = f"{type(exc).__name__}: {exc}"
            self._handling_failure = True
            if self._failure_alert is not None:
                with suppress(Exception):
                    self._failure_alert(cause)
            if self._channel.motion_attempts:
                if self._state is LiveMotionState.SENDING:
                    ambiguous = not isinstance(exc, CommandRejected)
                    self._transition(
                        LiveMotionState.AMBIGUOUS_ACCEPTANCE
                        if ambiguous
                        else LiveMotionState.FAULTED,
                        cause,
                    )
                elif isinstance(exc, TransportTimeout):
                    self._transition(LiveMotionState.TIMEOUT, cause)
                elif not isinstance(exc, (MotionCancelled, KeyboardInterrupt)):
                    self._transition(LiveMotionState.FAULTED, cause)
                standstill, final = self._abort(cause)
            elif self._state in (
                LiveMotionState.IDLE,
                LiveMotionState.ARMED_FOR_ONE_PLAN,
                LiveMotionState.SENDING,
            ):
                self._transition(LiveMotionState.REJECTED_BEFORE_SEND, cause)
        finally:
            self._handling_failure = True
            self._invalidate()
            try:
                self._channel.close()
                self._connections_closed = True
            except Exception as exc:
                self._record("close_failed", reason=str(exc))
                cause = cause or f"close failed: {exc}"
            self._closed = True
            self._worker.release()
        self._result = LiveMotionResult(
            self._state,
            cause,
            command_id,
            self._channel.motion_attempts,
            self._channel.stop_attempts,
            standstill,
            final,
            self._consumed,
            ambiguous,
            self.transport_is_mock,
            bool(self._channel.motion_attempts) and not self.transport_is_mock,
            self._connections_closed,
        )
        self._record("result", **asdict(self._result))
        return self._result
