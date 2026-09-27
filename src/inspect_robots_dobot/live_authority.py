"""Ephemeral identity capability; serializable digests are labels, not authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .config import ConnectionConfig
from .live_profile import LiveMotionProfile
from .motion import CartesianMotionPlan


def plan_digest(
    plan: CartesianMotionPlan, profile: LiveMotionProfile, connection: ConnectionConfig
) -> str:
    payload = {
        "schema": "dobot-one-shot-v1",
        "plan": asdict(plan),
        "profile": asdict(profile),
        "connection": asdict(connection),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AuthorityDetails:
    plan_digest: str
    session_nonce: str
    issued_at: float
    expires_at: float
    max_commands: int = 1


class LiveMotionAuthority:
    """Possession alone is insufficient: the issuing driver's identity ledger must match.

    Directly constructing this type cannot authorize a write. It has no config or
    pickle representation; copying creates no second spendable permission.
    """

    __slots__ = ("_details",)

    def __init__(self, details: AuthorityDetails) -> None:
        self._details = details

    @property
    def details(self) -> AuthorityDetails:
        return self._details

    def __reduce__(self) -> Any:
        raise TypeError("LiveMotionAuthority is session-local and cannot be serialized/copied")

    def __repr__(self) -> str:
        return "<LiveMotionAuthority: ephemeral one-use capability>"
