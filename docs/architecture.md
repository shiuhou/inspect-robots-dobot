# Architecture

The adapter has four deliberately separate boundaries:

1. **Offline embodiment.** `DobotEmbodiment` uses `FakeDobotDriver` for staged
   seven-dimensional end-effector actions. It validates measured starts, frames,
   path limits, settling and camera freshness without opening a real connection.
2. **Read-only diagnostics.** `DobotDashboardClient`, `DobotFeedbackClient` and
   `ReadOnlyDobotDriver` expose bounded status queries. Their send surface is a
   fixed query allowlist; they never change controller state.
3. **Optional physical observation.** The V4L2 backend drains named MJPEG streams
   and publishes immutable HWC RGB frames with host timestamps. Camera freshness
   does not establish sensor exposure time, calibration or robot safety.
4. **Separately gated motion.** The live driver accepts one reviewed plan through
   a one-use authority and monitors acknowledgement, mode and measured pose. It is
   not used by the generic offline embodiment and does not retry ambiguous writes.

The action order is `x/y/z/yaw/pitch/roll/gripper`, using metres, radians and a
normalized gripper value (`0=closed`, `1=open`). Native Dobot pose values remain
millimetres and degrees at the Dashboard boundary. The adapter keeps native Euler
conversion separate from the relative agent orientation representation and does
not implement custom IK.

The Astra shadow path ends at a recorded decision. The gripper is shadow-only.
No default path powers, enables, homes or moves a robot.

## Data flow

```text
Astra policy or fixture
        -> action chunk and adapter pre-checks
        -> staged validation / approval
        -> fake settle and measured observation

Dashboard queries -> bounded replies -> read-only diagnostics
V4L2 MJPEG       -> latest-frame reader -> named RGB observations
reviewed plan    -> one-use live authority -> one command and monitor
```

All transport and camera validation in the test suite uses fakes or in-memory
streams. A successful offline run proves software behavior only; it is not a
claim about firmware, reachability, collision clearance or physical safety.
