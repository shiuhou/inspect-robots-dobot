"""Auditable Dashboard-boundary conversion and native orientation geometry."""

from __future__ import annotations

import math

# Backward-compatible imports; all orientation geometry lives in transforms.py.
from .transforms import native_rotation as native_rotation
from .transforms import orientation_distance as orientation_distance
from .types import NativePose, PoseSI


def mm_to_m(value: float) -> float:
    return value / 1000.0


def m_to_mm(value: float) -> float:
    return value * 1000.0


def deg_to_rad(value: float) -> float:
    return math.radians(value)


def rad_to_deg(value: float) -> float:
    return math.degrees(value)


def to_native(pose: PoseSI) -> NativePose:
    return NativePose(*(m_to_mm(v) for v in pose.xyz), *(rad_to_deg(v) for v in pose.values[3:]))


def from_native(pose: NativePose) -> PoseSI:
    return PoseSI(*(mm_to_m(v) for v in pose.values[:3]), *(deg_to_rad(v) for v in pose.values[3:]))


def translation_distance(a: PoseSI, b: PoseSI) -> float:
    return math.dist(a.xyz, b.xyz)
