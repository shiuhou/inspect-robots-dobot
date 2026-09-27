# inspect-robots-dobot

Hardware-independent Astra/Robocurve adapter for Dobot Nova. The package keeps
planning, validation and diagnostics separate from physical control so the default
installation can be tested without a robot, camera, network or model credentials.

## What is included

- A seven-dimensional end-effector contract: `x`, `y`, `z`, `yaw`, `pitch`,
  `roll`, and normalized `gripper`.
- `FakeDobotDriver` and staged execution for deterministic offline tests.
- Query-only Dashboard and feedback clients, health reporting, and metadata
  preflight. These clients do not request control, power on, enable, clear faults,
  or move the robot.
- A V4L2 MJPEG camera backend for the named `front_rgb`, `right_rgb`, and
  `wrist_rgb` streams. Device access is opt-in and never happens during import or
  ordinary tests.
- A separately gated one-shot motion backend and operator CLI. It is a research
  boundary, not a general motion service or a hardware safety certification.
- Astra shadow and gripper shadow paths. Physical gripper control is not included.

## Offline setup

Requires Python 3.10+ and `uv`.

```bash
uv sync --frozen --extra agent
uv run --offline pytest
uv run --offline mypy src/inspect_robots_dobot
uv run --offline ruff check .
uv run --offline ruff format --check .
```

The tests use injected transports and fake devices. They must remain independent
of hardware, network services and provider credentials.

## Useful commands

```bash
# Metadata and query-only checks; neither command opens a socket by itself.
uv run --offline inspect-robots-dobot-preflight --json
uv run --offline inspect-robots-dobot-health --json

# Synthetic motion proposal and fake settling.
uv run --offline inspect-robots-dobot-motion-dry-run \
  --config examples/fake_motion.json \
  --initial-native-si 0.3 0 0.2 0 0 0 \
  --targets '{"x":0.315}' --allow-fake-motion --json

# Deterministic Astra shadow run.
uv run --offline inspect-robots-dobot-astra-shadow \
  --fixture --json --evidence-dir run-evidence/shadow
```

The `examples/` files are sanitized fixtures. They are not production rig
profiles. A live invocation requires a separately reviewed configuration, explicit
operator confirmation, a current-session safety check and the runtime flag; no
setting in this repository grants that authority automatically.

## Repository layout

```text
src/inspect_robots_dobot/   implementation and command-line entry points
tests/                      offline tests with fake transports and devices
examples/                   sanitized configuration and replay fixtures
docs/                       current architecture, safety, source and bring-up notes
```

Start with:

- [Architecture](docs/architecture.md)
- [Safety boundaries](docs/safety.md)
- [Read-only bring-up](docs/hardware_bringup.md)
- [Known limits](docs/known_unknowns.md)
- [Motion execution](docs/motion_execution.md)
- [Source contract](docs/source_contract.md)

For Dobot TCP/IP facts, use the workspace reference
`/home/dsa/project/dobot/dobot_tcp_ip_reference.md`. The original PDF in the
workspace remains authoritative for visual verification and ambiguous tables.

## Scope boundary

This repository owns the adapter and its offline contracts. It does not contain
Leader serial settings, camera credentials, raw hardware traces, provider keys,
or upstream `inspect-robots` source. Keep local runtime settings outside the
repository and publish component changes here before updating the workspace meta
repository's submodule pin.
