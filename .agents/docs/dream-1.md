# Dream-1 / Nova2: API connectivity and offline boundary

## Verified result and compatibility

On 2026-10-27 an authenticated **model discovery** call succeeded with installed
`dreamlabs-client==1.0.9`. Only these checkpoints were listed:

| Checkpoint | Embodiment | Camera roles | Nova2 inference |
| --- | --- | --- | --- |
| `dreamlabs/dream-1-beta` | `yam` | overhead, wrist_l, wrist_r | BLOCKED |
| `dreamlabs/dream-1-beta-flash` | `yam` | overhead, wrist_l, wrist_r | BLOCKED |

No real Nova2 observation was uploaded, no action chunk was obtained from the
cloud, and no hardware was connected. Authentication is not Nova2 integration
acceptance. The compatible-checkpoint success path is tested with a fake cloud
client, real image encoding and production validation, not with a Nova2 model.

The candidate single-arm schema passes the SDK's own **local** registration,
state-width and action-width checks. It is **not registered**, does not grant
model access and is **not a fine-tuned checkpoint**. It names one `right_arm`
(six joints; “right” is the SDK part name, not a second physical arm), one
`right_gripper`, and the actual front/wrist views:

```json
{
  "state": {"right_arm": [0, 6], "right_gripper": [6, 7]},
  "action": {"right_arm": [0, 6], "right_gripper": [6, 7]},
  "video": {"front": "observation.images.front_rgb", "wrist": "observation.images.wrist_rgb"},
  "head_camera": "front"
}
```

The installed SDK documents `robot="other_embodiment"` and client-side EEF IK,
including custom arm files. This means public YAM metadata alone does **not**
prove all cross-embodiment use impossible. In that mode the base model ignores
robot state, joint-head output is refused, and camera/pose adaptation is required.
Nova2 is absent from `dream.client_ik.ARMS`. No verified Nova2 arm file, calibrated
training TCP, two-camera/no-missing-arm procedure, or vendor compatibility evidence
was found. We did not invoke that path, pad a second arm, duplicate a view into a
missing camera, or interpret YAM joint targets as Nova2 targets. This implementation
conservatively supports only a matching custom checkpoint and the exact schema
above; other vendor-approved configurations require a focused adapter review.

## Commands

From this component's repository root:

```bash
uv sync --frozen --extra agent --extra dream
uv run --offline python -m inspect_robots_dobot.dream_integration schema
uv run --offline python -m inspect_robots_dobot.dream_integration fixture \
  --output run-evidence/dream-fixture-01
uv run --offline pytest -q
uv run --offline mypy src/inspect_robots_dobot
uv run --offline ruff check .
uv run --offline ruff format --check .
```

The optional extra pins the inspected SDK, leaving ordinary adapter installations
provider-independent. The already installed `~/dream-api-env` was used to inspect
SDK 1.0.9 and initially probe the API; no replacement environment is required.
The project environment adds the same SDK plus Pillow so the commands above use
one reproducible environment with the adapter's existing dependencies.

Explicit cloud **metadata only** (no images, policy, hardware or paid training):

```bash
# DREAM_API_KEY must be configured in this shell; never print its value.
uv run --offline python -m inspect_robots_dobot.dream_integration probe
# Alternatively use a private, user-owned 0600 file containing ONLY the key:
uv run --offline python -m inspect_robots_dobot.dream_integration \
  --key-file /path/outside/repository/private-key-file probe
```

`uv --offline` concerns dependency resolution, not API networking. Probe exit 0
means discovery succeeded; inspect each model's `nova2_blocker`. Missing credentials,
SDK/configuration/API errors and inference incompatibility exit 2 with sanitized
messages. Key files are never sourced as shell code; nonregular, symlinked, foreign
owned and group/world-accessible files are refused. No keys or raw API errors are
written to reports. SDK diagnostic file logging defaults off in this boundary.

Future **explicit cloud inference on a recorded observation**, only after model
access, data semantics and upload authorization are established:

```bash
uv run --offline python -m inspect_robots_dobot.dream_integration infer-recorded \
  --observation /path/to/private/record-directory \
  --contract /path/to/private/reviewed-contract.json \
  --prompt 'the recorded task' --timeout-ms 15000 \
  --output run-evidence/dream-inference-01
```

Contract format (replace every example with reviewed training facts):

```json
{
  "checkpoint": "your-account/nova2-checkpoint",
  "embodiment_id": "emb_your_registered_nova2_id",
  "action_hz": 20.0,
  "horizon": 48,
  "user_frame": 0,
  "tool_frame": 0,
  "evidence": "reference to reviewed training metadata and calibration evidence",
  "joint_units": "rad",
  "gripper_units": "normalized_0_closed_1_open",
  "action_semantics": "absolute_joint_targets"
}
```

20 Hz is an example, **not a verified Nova2 rate**. SDK documentation describes
48 rows at the training dataset's FPS. The contract must record the actual model's
horizon/rate; action shape must match it exactly. No scheduling or execution occurs.
The checkpoint id must be returned by discovery; its servability, registered body,
complete config and camera roles must match. A local evidence string is a reference
for review, not cryptographic verification of calibration or vendor approval.

`DreamClient.policy()` itself enables the SDK's background observation recording
to Dream Labs. Consequently inference is an explicit upload of the recorded views
and state, including SDK recording; it is never part of schema/fixture/probe paths.
The runner makes one `get_action()` call, no automatic retry, including pool warming
and rate limits. Client timeout bounds that SDK inference request; SDK discovery
has its own HTTP timeout/retry behavior. Latency includes policy/encoding-independent
request preparation and inference, and is reported only after a valid result.

## Observation reuse and records

`ObservationAdapter` consumes an existing `StateReader` and two already running
`CameraReader`s. It does not construct, start, connect or close devices.
`DashboardMeasurements` adapts an **already connected** `DobotDashboardClient` using
only GetAngle, GetPose, RobotMode and GetErrorID. The existing protocol decoder
converts measured GetAngle degrees to radians and GetPose millimetres/degrees to
metres/native extrinsic XYZ radians. The configured user/tool indices are preserved
and required; they are not inferred from robot mode or meshes. No protocol changes
were made. Raw 30004 QActual/ToolVectorActual units remain unverified in this adapter.

Joint and pose reads are sequential. State timestamp is the conservative start of
that query interval; camera Frame timestamps retain their original acquisition,
publication, receive, sequence, generation and timestamp-source fields. Validation
checks dimensions, uint8 RGB, age, finite numbers, timestamp order, cross-source skew,
six joints and known robot modes. Age and skew bounds are caller-provided rig
requirements, not magic safety settings. Host freshness/skew do not establish sensor
exposure synchronization, controller sample freshness or calibration.

Unknown gripper feedback is `None`. Existing x-trainer Feetech feedback may be
provided as a `GripperMeasurement` by an authorized caller; commanded position is
not feedback, and servo ticks must not be called normalized opening without measured
endpoints/polarity evidence. Missing feedback is valid for read-only inspection but
blocks inference. Calibration evidence defaults unknown and also blocks inference.
The simulator's inactive gripper placeholder is never used as a physical reading.

Use `save_record(path, observation, configs, max_age=..., max_skew=...)` with a new
private directory. It writes `observation.json` plus front/wrist RGB `.npy` files
(no pickle). `load_record` validates them using their **original capture clock**;
it never rebases timestamps or claims an old record is a current live observation.
Generated actions and time offsets are saved alongside a successful inference in a
new directory. The record carries explicit units, frame indices, validity limits,
original timestamps, calibration status and `execution=DISCONNECTED`. Records are
ignored evidence, not public fixtures; the deterministic fixture generator is code.

## Adaptation and remaining work

The SDK supports single-arm joint-based custom bodies without inventing a left arm.
The documented route is: register the reviewed embodiment, validate a LeRobot v2.1
dataset locally, upload, wait for processing, request fine-tuning, and discover the
resulting `fine_tuned_model`. **None of registration, upload or training was done.**
Registration alone does not make a YAM checkpoint compatible.

Demonstrations must include synchronized measured J1–J6 state, calibrated measured
gripper opening, actual arm/gripper action targets with their correct temporal
alignment, front/wrist videos, consistent FPS, task labels and episode boundaries.
For this candidate contract both state and action are seven values, joints in
radians and gripper 0 closed/1 open. Convert the existing x-trainer recordings with
an audited mapping; Teach/Repeat command-only records are not automatically a
complete Dream dataset, and the generic dual-arm LeRobot converter is not Nova2
validation. SDK `validate_dataset` expects one LeRobot v2.1 dataset or a directory
of datasets: `meta/info.json`, `meta/tasks.jsonl`, matching parquet vectors and
`videos/chunk-NNN/<video-key>/<episode>.mp4`. All sessions need consistent FPS and
features. At least one episode is a structural minimum, **not evidence of sufficient
training data**; ask Dream Labs for Nova2/task-specific demonstration quantity and
coverage. `use_eef=True` currently requires both arms' six named EEF parts in the
legacy no-parts schema, so do not enable it on this single-arm candidate.

Next ask Dream Labs whether an approved Nova2 single-arm/two-camera zero-shot arm
file/path exists without fabricated inputs; otherwise confirm custom joint fine-tune
support, account entitlement, checkpoint access, costs, data/URDF/TCP requirements
and expected FPS/units. Additional approval/access requirements are account/server
policy and were not verified. Fine-tuning is a documented route, not a proven
mandatory requirement for every possible Nova2 setup. After obtaining a compatible
checkpoint, validate a real recorded observation offline and request one cloud chunk.
Hardware capture still requires a separately reviewed current-session procedure.
Execution and broker integration require a separate task and safety review.

## Validation and sources

Tests run: full socket/device-blocked suite **989 PASS** (930 existing + 59 Dream),
strict mypy **PASS**, Ruff lint/format **PASS**. SDK local schema validation **PASS**;
authenticated discovery **PASS**. Physical capture, Nova2 cloud inference and physical
execution **NOT_RUN**. No firmware, visual grounding or hardware readiness claim.

Inspected existing Dashboard/types/camera/FakeDobot/motion boundary, x-trainer
feedback/gripper/recorder interfaces, nova-mujoco action contract/FK and the separate
Argon checkout's Nova contracts, runner and supervision. Argon files were not changed.

Primary sources inspected on 2026-10-27:

- [Dream Labs](https://dreamlabs.ai/): product context.
- [Platform](https://platform.dreamlabs.ai/): application shell accessible; authenticated
  API documentation UI was not independently read.
- [Official SDK 1.0.9 package documentation](https://pypi.org/project/dreamlabs-client/1.0.9/),
  retrieved through its PyPI JSON release metadata: inference/FaaS contracts.
- Installed `dream.facade`, `dream.models`, `dream.datasets`, `dream.client_ik`,
  `dream.credentials`, `dream.client` and `shared.datasets.embodiments`: signatures,
  parser, supported arms, metadata, shapes, units and recording behavior. The local
  schema validator uses the version-pinned internal registration parser because SDK
  1.0.9 exposes no public standalone embodiment validation method.
