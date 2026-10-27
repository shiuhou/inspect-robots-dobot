# Read-only task viewer

The viewer reads local evidence JSON and renders task state, proposed/executed
positions and outcomes. It cannot approve, stop, dispatch a robot action, start
cameras or invoke a model. Closing the viewer does not stop a robot.

From the component root, export a standalone page without networking:

```bash
uv run --offline python -m inspect_robots_dobot.task_viewer \
  --source examples/task_viewer_demo.json --output /tmp/nova-task-demo.html
node tests/task_viewer_dom_smoke.cjs /tmp/nova-task-demo.html
```

Output files are never overwritten. Use a fresh filename for another export.

An explicit local listener is also available:

```bash
uv run --offline python -m inspect_robots_dobot.task_viewer \
  --source /path/to/local/evidence --serve --no-camera-preview
```

It binds only 127.0.0.1:8766. Without `--no-camera-preview`, it may read an already
running loopback camera-preview health endpoint (default port 8765) and link its
fixed front/right/wrist streams; it never starts that service or opens V4L2.
`uv --offline` only controls dependency resolution, not runtime networking.

A producer may explicitly supply task-published JPEGs with `--runtime-frames`.
`--ready-file` records the actual listener port/PID and permits an ephemeral port
when the requested port is occupied. This release does not integrate automatic
viewer startup into the live robot runner. That mixed local candidate remains
under separate review.

Offline tests cover bounded file reads, redaction, unavailable/stale preview,
read-only routes, no-overwrite export and CLI option forwarding. The DOM smoke
executes the shipped JavaScript without a browser, device or network.
