# Safety boundaries

Motion authority is disabled by default. Configuration, connectivity, a passing
test, or a valid protocol reply cannot grant it.

## Offline paths

The fake driver and staged embodiment are the normal development path. They use
synthetic geometry and in-memory transports. Their command records are proposals,
not executable robot commands. The read-only Dashboard client has a fixed query
allowlist and the feedback client has no send path.

## Live boundary

The live driver is deliberately separate. A valid run needs all of the following:

- a reviewed private configuration with the actual firmware and frames recorded;
- a fresh measured start and an exact one-use plan;
- an interactive operator who verifies ownership, workspace, gripper state and a
  tested physical emergency stop;
- an explicit runtime motion flag and bounded command/monitor timeouts.

The driver sends at most one motion command. A lost acknowledgement or partial
write is uncertain and permanently consumes the authority; the code does not
resend, auto-recover, clear faults or widen the envelope. Software `Stop` is a
best-effort controller command and is not a substitute for the physical E-stop.

## Gripper and cameras

The gripper dimension is normalized for planning but physical gripper control is
not implemented. Never infer polarity or wiring from a shadow record. Camera
frames are observations only; freshness does not prove exposure after movement or
provide calibration.

For API facts, use the workspace TCP/IP reference and verify ambiguous tables
against the original PDF. Do not replace an unknown hardware fact with a guessed
default.
