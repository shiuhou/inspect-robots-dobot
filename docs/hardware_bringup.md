# Read-only bring-up

The default commands below are metadata-only or explicitly query-only. They do
not grant ownership, power, enable, error recovery or motion authority.

```bash
uv run --offline inspect-robots-dobot-preflight --json
uv run --offline inspect-robots-dobot-preflight \
  --config examples/readonly_connection.json --read-only --json
uv run --offline inspect-robots-dobot-health --json
```

`examples/readonly_connection.json` intentionally has no host. Before any live
diagnostic, copy it to a private local file and fill in a literal controller IP
after checking the physical installation, emergency stop and controller UI. Do
not commit that local file.

The health client uses Dashboard `29999` and feedback `30004`, with the protocol
version explicitly configured. It does not call `RequestControl`, `PowerOn`,
`EnableRobot`, `ClearError`, `Stop` or any motion command. Unknown firmware,
ownership, frames and TCP calibration remain unknown until the operator verifies
them independently.

Live motion is outside this bring-up procedure. It requires a separate reviewed
profile, a fresh measured start, one-use authority, an interactive operator and a
physical emergency stop. Never reuse a fake workspace or a sanitized example as a
production safety profile. Never retry an ambiguous motion write.
