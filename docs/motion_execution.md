# Motion execution contract

The generic agent emits bounded end-effector actions. The adapter validates every
target against the latest measured observation and records approval decisions. The
offline path then simulates settling with `FakeDobotDriver` and returns measured
state; it does not send TCP commands.

The live path is intentionally a separate one-shot boundary. It binds a reviewed
plan to the measured start, session and expiry, serializes one native `MovL`, and
monitors command ID, mode and measured pose. The controller's accepted command is
not treated as arrival. A timeout, alarm, stale telemetry or lost acknowledgement
blocks completion and cannot trigger a retry.

No hidden chunk compression, fabricated intermediate observations, custom IK,
servo streaming or automatic recovery is provided. If a future integration needs
multi-segment execution, it must define approval scope, progress reporting and
interruption semantics before code is added.
