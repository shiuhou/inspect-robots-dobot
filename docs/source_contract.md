# Source contract

The adapter is built against the pinned `inspect-robots` and
`inspect-robots-agent` revisions in `pyproject.toml`. Their source is a dependency
and is not copied or edited here.

For Dobot TCP/IP behavior, the primary machine-readable reference is
`/home/dsa/project/dobot/dobot_tcp_ip_reference.md`. The original
`Dobot TCP_IP二次开发接口文档_V4.6.5_20251015_cn.pdf` in the workspace remains
authoritative when a table, figure or formula needs visual verification.

The implementation relies on these documented facts:

- Dashboard is `29999`; feedback is `30004`.
- Replies contain `ErrorID`, values and the echoed command. The client validates
  framing and echo instead of accepting arbitrary text.
- `RobotMode` values include enabled idle (`5`), running (`7`), error (`9`) and
  collision (`11`).
- `GetPose` accepts a matching user/tool pair or neither. Native Cartesian values
  are millimetres and degrees; joints are degrees.
- `MovL` returns a queue/result identifier. Accepted is not completed; completion
  requires the documented ID/mode checks plus measured pose tolerances.
- `Stop` is a controller command and is not a certified physical emergency stop.
- `GetPose` and `GetAngle` are unavailable in error or power-off states; the
  adapter reports that condition rather than trying to recover automatically.

The feedback decoder validates the documented packet size, sentinel and a small
field subset. Unspecified units, frame meaning, firmware compatibility, ownership,
speed scaling and controller timestamps remain unknown. No protocol behavior is
inferred from a successful socket connection or packet shape alone.
