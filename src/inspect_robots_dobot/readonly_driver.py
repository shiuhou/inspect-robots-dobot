"""Real telemetry behind the driver contract; all actuator methods fail before IO."""

from __future__ import annotations

from .dashboard import DobotDashboardClient
from .driver import MotionReceipt
from .errors import MotionNotAuthorized, PhaseUnavailable, QueryUnavailable
from .feedback import RawFeedback
from .feedback_client import DobotFeedbackClient
from .types import JointPositions, PoseSI, RobotMode, RobotSnapshot


class ReadOnlyDobotDriver:
    """Dashboard query adapter; the action-capable embodiment remains fake-only.

    snapshot is sequential telemetry, not an atomic controller sample. Its timestamp
    is the beginning of acquisition to avoid making old measurements look newer.
    Explicit frames are required for snapshot; individual pose queries can instead
    report the controller-global frame selection as unknown.
    """

    is_simulated = False

    def __init__(
        self, dashboard: DobotDashboardClient, feedback: DobotFeedbackClient | None = None
    ) -> None:
        self.dashboard, self.feedback = dashboard, feedback

    def connect(self) -> None:
        # Independent feedback connection lets health diagnose either port alone.
        self.dashboard.connect()

    def close(self) -> None:
        try:
            self.dashboard.close()
        finally:
            if self.feedback is not None:
                self.feedback.close()

    def robot_mode(self) -> RobotMode:
        return self.dashboard.robot_mode()

    def _require_pose_query(self) -> None:
        mode = self.robot_mode()
        if mode in (RobotMode.ERROR, RobotMode.POWER_OFF):
            raise QueryUnavailable(
                f"GetPose/GetAngle unavailable in {mode.name} (V4.6.5 p160); no recovery attempted"
            )

    def get_pose(self) -> PoseSI:
        self._require_pose_query()
        return self.dashboard.get_pose()

    def get_joints(self) -> JointPositions:
        self._require_pose_query()
        return self.dashboard.get_joints()

    def get_errors(self) -> tuple[int, ...]:
        return self.dashboard.get_errors()

    def read_feedback(self) -> RawFeedback:
        if self.feedback is None:
            raise QueryUnavailable("feedback client was not configured")
        return self.feedback.read_sample()

    def snapshot(self) -> RobotSnapshot:
        config = self.dashboard.config
        if config.user_frame is None or config.tool_frame is None:
            raise QueryUnavailable(
                "snapshot requires explicit user/tool indices; no frame inferred"
            )
        started = self.dashboard.clock.monotonic()
        before = self.robot_mode()
        pose, joints, errors = self.get_pose(), self.get_joints(), self.get_errors()
        command_id = self.dashboard.current_command_id()
        after = self.robot_mode()
        if before != after:
            raise QueryUnavailable("RobotMode changed during sequential diagnostic acquisition")
        return RobotSnapshot(
            pose,
            joints,
            after,
            errors,
            command_id,
            started,
            config.user_frame,
            config.tool_frame,
            joints_synthetic=False,
        )

    def move_linear(self, target: PoseSI) -> MotionReceipt:
        raise MotionNotAuthorized("read-only driver has no physical motion implementation")

    def set_gripper(self, value: float) -> None:
        raise MotionNotAuthorized("read-only driver has no gripper actuation implementation")

    def stop(self) -> None:
        raise PhaseUnavailable("Stop changes controller state; absent from Phase 2A query surface")
