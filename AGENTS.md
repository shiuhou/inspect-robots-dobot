# inspect-robots-dobot contributor rules

- Keep tests hardware, network and credential independent. Use injected sockets,
  fake clocks and fake cameras.
- Never add implicit `RequestControl`, `PowerOn`, `EnableRobot`, `ClearError`,
  homing, retry or motion behavior to query-only or offline paths.
- Physical motion requires a separately reviewed configuration, an explicit runtime
  flag and current-session operator authorization. Do not treat fake execution,
  protocol parsing or a saved report as hardware safety evidence.
- Do not infer firmware compatibility, TCP ownership, tool/TCP calibration, user
  frames, gripper polarity, payload or collision clearance.
- Read `/home/dsa/project/dobot/dobot_tcp_ip_reference.md` before changing protocol
  behavior. Use the original PDF in the workspace to resolve visual or extraction
  ambiguity; do not invent missing protocol facts.
- Keep source facts, assumptions and runtime observations distinct in docs. Do not
  add raw traces, local endpoints, serial identifiers, credentials or generated
  evidence to the public component.
- Validate focused changes with `uv run --offline pytest`, strict `mypy`, Ruff
  lint and Ruff format checks before committing.

## Ownership and publication

This checkout publishes to `https://github.com/shiuhou/inspect-robots-dobot.git`.
Commit adapter code, tests and current documentation here first. After the commit
is reachable on `main`, update the parent `/home/dsa/project/dobot` meta repository
submodule SHA and `components.lock.yaml`, then publish the meta update. Never copy
component source into the meta repository and never include other checkout's dirty
work in a focused commit.
