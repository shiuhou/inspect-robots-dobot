"""Single-arm relative 6-DoF EEF embodiment; fake-only independent enforcement."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.approver import GuardrailContribution
from inspect_robots.embodiment import SELF_PACED, EmbodimentBase, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult

from .camera import CameraReader, LatestFrameReader, validate_frame
from .clock import Clock, SystemClock
from .config import CameraConfig, DobotConfig, SafetyProfile, validate_camera_hardware_mapping
from .driver import FakeDobotDriver, MotionReceipt
from .errors import CameraFault, ConfigurationError, DriverFault, PhaseUnavailable, SafetyRejected
from .gripper import ShadowGripper
from .safety import (
    DobotApprover,
    WaypointPreCheck,
    action_target,
    validate_orientation,
    validate_position,
    validate_snapshot,
    validate_target,
)
from .settle import wait_until_settled
from .types import PoseSI, RobotSnapshot
from .units import orientation_distance, to_native, translation_distance

EEF_DIM_LABELS = ("x", "y", "z", "yaw", "pitch", "roll", "gripper")


def _audit_action(action: Action) -> Any:
    """Keep rejected NaN/Inf actions JSON-safe too; never log arbitrary metadata."""

    def clean(value: Any) -> Any:
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, (float, int)) and not math.isfinite(value):
            return str(value)
        if value is None or isinstance(value, (str, bool, float, int)):
            return value
        return f"<{type(value).__name__}>"

    return clean(np.asarray(action.data).tolist())


def build_info(config: DobotConfig) -> EmbodimentInfo:
    """Unconfigured bounds remain absent; conformance/preflight must report them."""
    if config.local_micro_move is not None:
        raise ConfigurationError("local micro-move envelope is not a Robocurve/Astra profile")
    profile = config.safety
    low = high = None
    max_step: tuple[float | None, ...] | None = None
    if profile is not None:
        low = np.array(
            (*profile.workspace_low[:2], profile.minimum_tcp_z, *profile.orientation_low, 0.0)
        )
        high = np.array((*profile.workspace_high, *profile.orientation_high, 1.0))
        # Per-axis declared budgets conservatively imply the Euclidean limit.
        component = profile.max_translation_step / math.sqrt(3)
        # Rotation triangle inequality: sum of the three axis turns bounds SO(3).
        rotation_steps = tuple(
            profile.max_orientation_step / 3 if lo < hi else None
            for lo, hi in zip(profile.orientation_low, profile.orientation_high, strict=True)
        )
        max_step = (component, component, component, *rotation_steps, 0.1)
    semantics = ActionSemantics(
        control_mode="eef_abs_pose",
        rotation_repr="none",
        gripper="continuous",
        frame="base",
        dim_labels=EEF_DIM_LABELS,
        max_step=max_step,
    )
    cameras = tuple(CameraSpec(c.name, c.height, c.width) for c in config.cameras)
    if config.camera is not None:
        cameras = (CameraSpec("table", config.camera.height, config.camera.width),)
    return EmbodimentInfo(
        name="dobot_nova",
        action_space=Box((7,), low=low, high=high, semantics=semantics),
        observation_space=ObservationSpace(
            cameras=cameras,
            state=StateSpec(
                fields=(
                    StateField("eef_state", (7,), "m+rad+normalized"),
                    StateField("gripper_state", (1,), "normalized"),
                    StateField("joint_pos", (6,), "rad"),
                    StateField("dobot_native_pose", (6,), "m+rad (native extrinsic XYZ)"),
                )
            ),
        ),
        control_hz=config.control_hz,
        is_simulated=True,
        capabilities=frozenset({SELF_PACED}),
        docs="SIMULATED ONLY. Absolute x/y/z in metres in base user frame 0. "
        "Agent yaw,pitch,roll radians mean Rz(yaw) Ry(-pitch) Rx(roll) R_reset. "
        "Reference is measured at reset; zero preserves it. Tool frame is explicit. "
        "Orientation axes are pinned at zero unless explicitly configured. "
        "The normalized gripper dimension is active: 0 means closed and 1 means open. "
        "The current Phase 6A.2 executor records gripper proposals as shadow-only; "
        "no physical gripper command is sent. "
        "No collision-free claim. Agent steps are bounded, settled point-to-point moves.",
    )


class DobotEmbodiment(EmbodimentBase):
    def __init__(
        self,
        config: DobotConfig | None = None,
        *,
        driver: FakeDobotDriver | None = None,
        camera: CameraReader | Mapping[str, CameraReader] | None = None,
        clock: Clock | None = None,
        gripper: ShadowGripper | None = None,
    ) -> None:
        self.config = config or DobotConfig()
        self.info = build_info(self.config)
        if driver is not None and (
            not isinstance(driver, FakeDobotDriver) or not driver.is_simulated
        ):
            raise PhaseUnavailable("Phase 2C embodiment only accepts FakeDobotDriver")
        self._driver = driver
        self._cameras: dict[str, tuple[CameraReader, CameraConfig]] = {}
        if config is not None and config.cameras:
            if camera is None:
                from .camera_v4l2 import V4L2MjpegFrameSource

                validate_camera_hardware_mapping(config.cameras)
                self._cameras = {
                    spec.name: (
                        LatestFrameReader(
                            V4L2MjpegFrameSource(spec),
                            spec.frame_config,
                            startup_timeout=spec.startup_timeout,
                        ),
                        spec.frame_config,
                    )
                    for spec in config.cameras
                }
            elif isinstance(camera, Mapping):
                if set(camera) != {spec.name for spec in config.cameras}:
                    raise ConfigurationError(
                        "injected cameras must match all three configured names"
                    )
                self._cameras = {
                    spec.name: (camera[spec.name], spec.frame_config) for spec in config.cameras
                }
            else:
                raise ConfigurationError("three named cameras require a named reader mapping")
        elif camera is not None:
            if isinstance(camera, Mapping) or self.config.camera is None:
                raise ConfigurationError(
                    "legacy camera requires one reader and explicit camera config"
                )
            self._cameras = {"table": (camera, self.config.camera)}
        self._clock = clock or (driver.clock if driver is not None else SystemClock())
        if driver is not None and driver.clock is not self._clock:
            raise ConfigurationError("driver and embodiment must share the same monotonic clock")
        if driver is not None and driver.profile != self.config.safety:
            raise ConfigurationError("driver and embodiment safety profiles must match")
        self._last_camera_identity: dict[str, tuple[int, int]] = {}
        self._held: PoseSI | None = None
        self._last: RobotSnapshot | None = None
        self._last_camera_time: float | None = None
        self._instruction: str | None = None
        self._faulted = False
        self._gripper = gripper or ShadowGripper()
        self._allow_shadow_gripper = False
        self._audit: list[dict[str, Any]] = []

    @property
    def audit_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._audit))

    def _profile(self) -> SafetyProfile:
        if self.config.safety is None or self.config.control_hz is None:
            raise ConfigurationError(
                "explicit safety profile and control_hz are required; no rig defaults"
            )
        return self.config.safety

    def _connected_driver(self) -> FakeDobotDriver:
        if self._driver is None:
            raise PhaseUnavailable(
                "no fake driver supplied; real read-only driver cannot execute embodiment actions"
            )
        return self._driver

    def _reference(self) -> RobotSnapshot:
        if self._last is None:
            raise DriverFault("reset and obtain an observation before requesting actions")
        return self._last

    def _held_pose(self) -> PoseSI:
        if self._held is None:
            raise DriverFault("reset has not established the measured reference orientation")
        return self._held

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        profile = self._profile()
        driver = self._connected_driver()
        if self._faulted:
            raise DriverFault(
                "embodiment is fault-latched; close and inspect before constructing a new session"
            )
        if self.config.camera is not None and not self._cameras:
            raise ConfigurationError(
                "legacy table camera requires an injected source; "
                "use named cameras for physical capture"
            )
        driver.connect()
        try:
            current = driver.capture_orientation_reference()
            validate_snapshot(current, profile, self._clock.monotonic())
            validate_position(current.pose, profile, label="reset pose")
            self._held, self._instruction = current.pose, scene.instruction
            for reader, _ in self._cameras.values():
                reader.start()
            return self._observe()
        except BaseException:
            self.close()
            raise

    def _observe(
        self,
        *,
        after: float | None = None,
        target: PoseSI | None = None,
        receipt: MotionReceipt | None = None,
    ) -> Observation:
        images: dict[str, npt.NDArray[np.uint8]] = {}
        times: dict[str, float] = {}
        camera_info: dict[str, Any] = {"available": False, "reason": "explicit no-camera mode"}
        next_identity: dict[str, tuple[int, int]] = {}
        if self._cameras:
            frame_info: dict[str, Any] = {}
            for name, (reader, camera_config) in self._cameras.items():
                try:
                    frame = reader.latest(after=after)
                    now = self._clock.monotonic()
                    validate_frame(frame, camera_config, now=now, after=after)
                except CameraFault as exc:
                    raise CameraFault(f"{name}: {exc}") from exc
                identity = (frame.generation, frame.sequence)
                previous = self._last_camera_identity.get(name)
                if after is not None and previous is not None and identity <= previous:
                    raise DriverFault(f"{name} did not advance to a new frame generation/sequence")
                images[name] = frame.rgb.copy()
                times[name] = frame.timestamp
                next_identity[name] = identity
                frame_info[name] = {
                    "host_receive_time_monotonic": frame.timestamp,
                    "acquisition_started_at": frame.acquisition_started_at,
                    "published_at": frame.published_at,
                    "age_s": now - frame.timestamp,
                    "timestamp_source": frame.timestamp_source,
                    "exposure_time_verified": False,
                    "sequence": frame.sequence,
                    "generation": frame.generation,
                    "width": camera_config.width,
                    "height": camera_config.height,
                }
            camera_info = {
                "available": True,
                "frames": frame_info,
                "host_receive_skew_s": max(times.values()) - min(times.values()),
                "exposure_time_verified": False,
            }
            if len(frame_info) == 1:
                camera_info.update(next(iter(frame_info.values())))
        sample = self._connected_driver().snapshot()
        validate_snapshot(sample, self._profile(), self._clock.monotonic())
        validate_position(sample.pose, self._profile(), label="observed pose")
        try:
            orientation = validate_orientation(
                sample.pose, self._held_pose(), self._profile(), label="observed orientation"
            )
        except SafetyRejected as exc:
            raise DriverFault(f"orientation changed while acquiring observation: {exc}") from exc
        if (
            target is not None
            and receipt is not None
            and (
                sample.command_id != receipt.command_id
                or translation_distance(sample.pose, target) > self._profile().position_tolerance
                or orientation_distance(sample.pose, target) > self._profile().orientation_tolerance
            )
        ):
            raise DriverFault("robot left the settled target while acquiring observation")
        self._last = sample
        self._last_camera_time = max(times.values()) if times else None
        self._last_camera_identity = next_identity
        return Observation(
            images=images,
            state={
                "eef_state": np.array((*sample.pose.xyz, *orientation, self._gripper.read())),
                "gripper_state": np.array([self._gripper.read()]),
                "joint_pos": np.array(sample.joints),
                "dobot_native_pose": np.array(sample.pose.values),
            },
            instruction=self._instruction,
            image_times=times,
            state_time=sample.observed_at,
            extra={
                "dobot": {
                    "simulated": True,
                    "robot_mode": int(sample.mode),
                    "active_errors": sample.errors,
                    "command_id": sample.command_id,
                    "feedback_timestamp": None,
                    "state_timestamp_source": "host_monotonic_fake_sample",
                    "joints_synthetic": sample.joints_synthetic,
                    "camera": camera_info,
                    "user_frame": sample.user_frame,
                    "tool_frame": sample.tool_frame,
                    "orientation_reference_native_si": self._held_pose().values[3:],
                    "gripper": {
                        "normalized": self._gripper.state.normalized,
                        "semantic": self._gripper.state.semantic,
                        "source": self._gripper.state.source,
                        "execution_mode": "shadow",
                    },
                }
            },
        )

    def pre_check(self, waypoints: npt.NDArray[np.float64]) -> str | None:
        if self._faulted:
            return "embodiment is fault-latched; operator inspection required"
        check = WaypointPreCheck(
            self._reference, self._held_pose, self._profile(), self._clock.monotonic
        )
        result = check(waypoints)
        if result is None and not self._allow_shadow_gripper:
            values = np.asarray(waypoints, dtype=np.float64)
            if values.shape == (len(values), 7) and any(
                float(row[6]) != self._gripper.read() for row in values
            ):
                result = (
                    "REJECT: physical gripper execution is disabled; use the Phase 6A.2 shadow path"
                )
        self._audit.append(
            {
                "kind": "pre_check",
                "timestamp": self._clock.monotonic(),
                "waypoints": _audit_action(Action(waypoints)),
                "reference_pose": self._reference().pose.values,
                "accepted": result is None,
                "reason": result,
                "physical_sent": False,
            }
        )
        return result

    def _validate_action(self, action: Action) -> None:
        if self._faulted:
            raise DriverFault("embodiment is fault-latched; operator inspection required")
        if action.meta.get("clamped") or action.meta.get("delta_clamped"):
            raise SafetyRejected("REJECT: target modified by framework approver; resubmit")
        current = self._connected_driver().snapshot()
        vector = np.asarray(action.data, dtype=np.float64)
        target = action_target(action.data, current, self._held_pose(), self._profile())
        if (
            vector.shape == (7,)
            and vector[6] != self._gripper.read()
            and not self._allow_shadow_gripper
        ):
            raise SafetyRejected(
                "physical gripper execution is disabled; use the Phase 6A.2 shadow path"
            )
        validate_target(
            target, current, self._profile(), self._clock.monotonic(), reference=self._held_pose()
        )
        self._connected_driver().authority.require(is_simulated=True)

    def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
        return GuardrailContribution(
            approvers=(("dobot-reject", DobotApprover(self._validate_action)),)
        )

    def _dispatch_linear(self, target: PoseSI, current: RobotSnapshot) -> MotionReceipt:
        """Fake dispatch hook; staged execution adds its final plan/start check here."""
        return self._connected_driver().move_linear(target)

    def step(self, action: Action) -> StepResult:
        record: dict[str, Any] = {
            "kind": "motion_attempt",
            "timestamp": self._clock.monotonic(),
            "requested_action": _audit_action(action),
            "source_camera_timestamp": self._last_camera_time,
            "physical_authorized": False,
            "physical_sent": False,
            "framework_approver_result": "not universally available; see framework approval events",
            "protocol_command": None,
            "controller_response": None,
        }
        self._audit.append(record)
        receipt = None
        send_attempted = False
        try:
            reference = self._reference()
            record["observation_pose"] = reference.pose.values
            record["observation_timestamp"] = reference.observed_at
            self._validate_action(action)
            driver, profile = self._connected_driver(), self._profile()
            current = driver.snapshot()
            target = action_target(action.data, current, self._held_pose(), profile)
            translation, orientation = validate_target(
                target, current, profile, self._clock.monotonic(), reference=self._held_pose()
            )
            record.update(
                initial_pose=current.pose.values,
                target_si=target.values,
                target_native=to_native(target).values,
                translation_delta=translation,
                orientation_delta=orientation,
                user_frame=profile.user_frame,
                tool_frame=profile.tool_frame,
                speed_percent=profile.speed_percent,
                safety_result="accepted",
                active_errors=current.errors,
                simulation_authorized=True,
                orientation_reference_native_si=self._held_pose().values[3:],
            )
            send_attempted = True
            receipt = self._dispatch_linear(target, current)
            record["command_id"] = receipt.command_id
            settled = wait_until_settled(
                driver, target, receipt, profile, self._clock, reference=self._held_pose()
            )
            record.update(
                robot_settled=True,
                settled_at=settled.settled_at,
                settle_duration=settled.duration,
                position_residual=settled.position_residual,
                orientation_residual=settled.orientation_residual,
            )
            assert self.config.control_hz is not None
            remaining = 1 / self.config.control_hz - (self._clock.monotonic() - receipt.accepted_at)
            if remaining > 0:
                self._clock.sleep(remaining)
            observation = self._observe(after=settled.settled_at, target=target, receipt=receipt)
            final = self._reference()
            record.update(
                status="settled",
                command_id=receipt.command_id,
                final_pose=final.pose.values,
                position_residual=translation_distance(final.pose, target),
                orientation_residual=orientation_distance(final.pose, target),
                settle_duration=settled.duration,
                settled_at=settled.settled_at,
                observed_mode_transition=[int(current.mode), int(settled.final.mode)],
                fake_accepted_mode="RUNNING",
            )
            return StepResult(observation, info={"dobot_motion": copy.deepcopy(record)})
        except BaseException as exc:
            record.update(
                status="failed",
                reason=str(exc),
                failure_timestamp=self._clock.monotonic(),
                acknowledgement_received=receipt is not None,
            )
            if send_attempted:
                self._faulted = True
                try:
                    self._connected_driver().stop()
                    record["stop_requested"] = True
                except Exception as stop_exc:
                    record["stop_error"] = str(stop_exc)
                try:
                    failed_state = self._connected_driver().snapshot()
                    record.update(
                        final_pose=failed_state.pose.values,
                        final_mode=int(failed_state.mode),
                        active_errors=failed_state.errors,
                    )
                except Exception as state_exc:
                    record["final_state_error"] = str(state_exc)
            raise

    def close(self) -> None:
        try:
            errors = []
            for reader, _ in self._cameras.values():
                try:
                    reader.close()
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                raise DriverFault(f"camera close failed: {errors}")
        finally:
            if self._driver is not None:
                self._driver.close()
            self._last = self._held = None
            self._last_camera_identity.clear()
