"""Pure orientation math. All angles here are radians, never protocol degrees.

Dobot: Rz(rz) Ry(ry) Rx(rx), TCP V4.6.5 pp80–81.
Agent: Rz(yaw) Ry(-pitch) Rx(roll) R_reference, pinned YAM implementation.
No IK, coordinate calibration, or physical trajectory guarantee is implied.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .types import PoseSI

Rotation = npt.NDArray[np.float64]
DEFAULT_SINGULARITY_MARGIN = 1e-6
_ROUNDOFF = 1e-14


def _rotation(matrix: npt.ArrayLike) -> Rotation:
    result = np.asarray(matrix, dtype=np.float64)
    if (
        result.shape != (3, 3)
        or not np.all(np.isfinite(result))
        or not np.allclose(result.T @ result, np.eye(3), atol=1e-10, rtol=0)
        or not math.isclose(float(np.linalg.det(result)), 1.0, abs_tol=1e-10)
    ):
        raise ValueError("expected a finite proper orthonormal 3x3 rotation")
    return result


def _margin(value: float) -> None:
    if not math.isfinite(value) or not 0 < value < math.pi / 2:
        raise ValueError("singularity margin must be finite in (0,pi/2)")


def _zero(value: float) -> float:
    # Remove matrix multiplication roundoff only, not sensor error or rig limits.
    return 0.0 if abs(value) < _ROUNDOFF else value


def wrap_angle(value: float) -> float:
    """YAM's canonical (-pi,pi] interval; no target/bound clamping."""
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    wrapped = (value + math.pi) % (2 * math.pi) - math.pi
    return _zero(math.pi if wrapped <= -math.pi else wrapped)


def dobot_native_to_rotation(rx: float, ry: float, rz: float) -> Rotation:
    """Native fixed-axis X,Y,Z in radians (not agent-relative coordinates)."""
    if not all(math.isfinite(v) for v in (rx, ry, rz)):
        raise ValueError("native angles must be finite")
    cx, cy, cz = (math.cos(v) for v in (rx, ry, rz))
    sx, sy, sz = (math.sin(v) for v in (rx, ry, rz))
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def native_rotation(pose: PoseSI) -> Rotation:
    return dobot_native_to_rotation(pose.rx, pose.ry, pose.rz)


def rotation_to_dobot_native(rotation: npt.ArrayLike) -> tuple[float, float, float]:
    """Canonical rx,ry,rz radians; ry in [-pi/2,pi/2].

    At native gimbal lock choose rx=0 and solve rz. This reproduces the rotation,
    not necessarily the controller's chosen Euler branch. Future motion encoding
    must review branch continuity independently; this is not a hardware sender.
    """
    r = _rotation(rotation)
    cos_y = math.hypot(float(r[0, 0]), float(r[1, 0]))
    ry = math.atan2(-float(r[2, 0]), cos_y)
    if cos_y < _ROUNDOFF:
        rx, rz = 0.0, math.atan2(-float(r[0, 1]), float(r[1, 1]))
    else:
        rx = math.atan2(float(r[2, 1]), float(r[2, 2]))
        rz = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return wrap_angle(rx), _zero(ry), wrap_angle(rz)


def agent_relative_to_rotation(
    yaw: float,
    pitch: float,
    roll: float,
    reference: npt.ArrayLike,
    *,
    singularity_margin: float = DEFAULT_SINGULARITY_MARGIN,
) -> Rotation:
    """World/base-axis relative rotation, left-multiplied onto reset reference."""
    _margin(singularity_margin)
    if not all(math.isfinite(v) for v in (yaw, pitch, roll)):
        raise ValueError("agent orientation must be finite")
    if abs(pitch) >= math.pi / 2 - singularity_margin:
        raise ValueError("agent pitch is inside the configured singularity exclusion")
    return dobot_native_to_rotation(roll, -pitch, yaw) @ _rotation(reference)


def rotation_to_agent_relative(
    rotation: npt.ArrayLike,
    reference: npt.ArrayLike,
    *,
    singularity_margin: float = DEFAULT_SINGULARITY_MARGIN,
) -> tuple[float, float, float]:
    _margin(singularity_margin)
    r = _rotation(rotation) @ _rotation(reference).T
    pitch = math.asin(float(np.clip(r[2, 0], -1.0, 1.0)))
    if abs(pitch) >= math.pi / 2 - singularity_margin:
        raise ValueError("measured relative pitch is inside the configured singularity exclusion")
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    return wrap_angle(yaw), _zero(pitch), wrap_angle(roll)


def rotation_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """SO(3) geodesic in [0,pi], stable at identity and angular wrap boundaries."""
    r = _rotation(a).T @ _rotation(b)
    skew = np.array((r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]))
    sine = float(np.linalg.norm(skew)) / 2
    cosine = float(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))
    return _zero(math.atan2(sine, cosine))


def orientation_distance(a: PoseSI, b: PoseSI) -> float:
    return rotation_distance(native_rotation(a), native_rotation(b))


def interpolate_rotation(start: npt.ArrayLike, target: npt.ArrayLike, fraction: float) -> Rotation:
    """Shortest SO(3) interpolation for deterministic fake tracking only.

    Agent ActionChunk interpolation still follows upstream's non-wrapping Euler
    coordinates. This models convergence between two accepted fake waypoints.
    At exactly pi the ambiguous axis sign is chosen deterministically.
    """
    a, b = _rotation(start), _rotation(target)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("interpolation fraction must be finite in [0,1]")
    if fraction == 0:
        return np.array(a, dtype=np.float64, copy=True)
    if fraction == 1:
        return np.array(b, dtype=np.float64, copy=True)
    angle = rotation_distance(a, b)
    if angle == 0:
        return np.array(a, dtype=np.float64, copy=True)
    relative = a.T @ b
    skew = np.array(
        (
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        )
    )
    if math.pi - angle < 1e-6:
        _, vectors = np.linalg.eigh(relative + relative.T)
        axis = vectors[:, -1]
        if np.linalg.norm(skew) > _ROUNDOFF:
            if np.dot(axis, skew) < 0:
                axis = -axis
        elif axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
    else:
        axis = skew / np.linalg.norm(skew)
    x, y, z = axis
    cross = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    theta = fraction * angle
    return np.asarray(
        a @ (np.eye(3) + math.sin(theta) * cross + (1 - math.cos(theta)) * (cross @ cross)),
        dtype=np.float64,
    )
