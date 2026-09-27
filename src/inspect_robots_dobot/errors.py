"""Typed failures that halt unsafe framework rollouts."""

from inspect_robots.errors import ConfigError, EmbodimentFault, SafetyAbort


class ConfigurationError(ConfigError):
    """A required explicit configuration value is absent or invalid."""


class MotionNotAuthorized(SafetyAbort):
    """Motion/actuation capability was not explicitly granted."""


class SafetyRejected(SafetyAbort):
    """An action violates a local safety invariant."""


class DriverFault(EmbodimentFault):
    """Controller state or an unavailable driver prevents execution."""


class SettleTimeout(DriverFault):
    """Accepted target did not converge before the deadline."""


class CameraFault(EmbodimentFault):
    """A configured camera cannot provide an eligible fresh frame."""


class PhaseUnavailable(ConfigurationError):
    """This phase does not implement the requested hardware capability."""


class ProtocolError(DriverFault):
    """Malformed or unexpected protocol data."""


class CommandRejected(ProtocolError):
    """The controller returned a nonzero ErrorID."""

    def __init__(self, error_id: int, command: str) -> None:
        self.error_id = error_id
        self.command = command
        super().__init__(f"controller rejected {command}: ErrorID={error_id}")


class TransportError(DriverFault):
    """Connection, EOF or I/O failure; no automatic reconnect/retry."""


class TransportTimeout(TransportError):
    """A bounded connection or complete query/packet deadline expired."""


class QueryUnavailable(DriverFault):
    """A documented state or declared control ownership prevents a query."""


class ControlNotOwned(QueryUnavailable):
    """Operator declared TCP control unowned; no ownership change was attempted."""


class UnsupportedProtocol(ProtocolError):
    """Requested document/layout version has not been implemented or verified."""


class UnverifiedField(ProtocolError):
    """A field's units, frame or representation have not been source-verified."""


class StaleFeedback(DriverFault):
    """No currently eligible host-received feedback sample exists."""
