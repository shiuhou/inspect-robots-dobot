# Known limits

The adapter intentionally leaves these rig-specific facts unset:

- Nova model and firmware compatibility.
- TCP ownership and controller operating mode prerequisites.
- Tool/TCP calibration, user-frame transforms and payload/center of mass.
- Table height, collision geometry, keep-outs and reachable workspace.
- Gripper wiring, polarity, output index and safe state.
- Camera exposure timing, intrinsics, distortion, extrinsics and hand-eye
  calibration.
- Physical emergency-stop testing and operator readiness.

The feedback decoder validates packet structure and a documented field subset. Raw
joint/cartesian feedback is not silently converted into calibrated SI state. Host
receive time is not controller generation time. Camera publication time is not
sensor exposure time.

The staged executor can validate synthetic paths and settle a fake driver. The
controller's actual trajectory, queue behavior, orientation branch, interruption
timing and collision response still require a separately reviewed physical test.
The live boundary therefore allows one reviewed command and never retries an
ambiguous write or silently widens its limits.

The physical camera configuration is opt-in and requires stable `/dev/v4l/by-id`
or `/dev/v4l/by-path` paths. No camera is opened during import, preflight or
offline tests.
