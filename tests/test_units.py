import math

import numpy as np
import pytest

from inspect_robots_dobot.types import NativePose, PoseSI
from inspect_robots_dobot.units import (
    deg_to_rad,
    from_native,
    m_to_mm,
    mm_to_m,
    native_rotation,
    orientation_distance,
    rad_to_deg,
    to_native,
    translation_distance,
)


@pytest.mark.parametrize("metres", [0, -0.1, 0.123456, 2.0])
def test_translation_roundtrip(metres):
    assert m_to_mm(metres) == pytest.approx(metres * 1000)
    assert mm_to_m(m_to_mm(metres)) == pytest.approx(metres)


@pytest.mark.parametrize("degrees", [0, -180, 90, 360, -720, 12.345])
def test_angle_roundtrip(degrees):
    assert rad_to_deg(deg_to_rad(degrees)) == pytest.approx(degrees)
    assert deg_to_rad(180) == math.pi


def test_full_pose_boundary():
    native = NativePose(300, -100, 200, 180, -90, 45)
    si = from_native(native)
    assert si.values == pytest.approx((0.3, -0.1, 0.2, math.pi, -math.pi / 2, math.pi / 4))
    assert to_native(si).values == pytest.approx(native.values)


def test_native_fixed_axis_rotation_order():
    pose = PoseSI(0, 0, 0, math.pi / 2, 0, math.pi / 2)
    # Rx then Rz: unit Y becomes unit Z; unit X becomes unit Y.
    rotation = native_rotation(pose)
    np.testing.assert_allclose(rotation @ [0, 1, 0], [0, 0, 1], atol=1e-14)
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-14)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)


def test_orientation_distance_handles_wrap_and_compound_rotation():
    origin = PoseSI(0, 0, 0, 0, 0, 0)
    assert orientation_distance(origin, PoseSI(0, 0, 0, 0, 0, 2 * math.pi)) == 0
    assert orientation_distance(origin, PoseSI(0, 0, 0, 0, 0, math.pi / 2)) == pytest.approx(
        math.pi / 2
    )
    a = PoseSI(0, 0, 0, 0.4, -0.7, 1.2)
    b = PoseSI(0, 0, 0, -0.2, 0.8, 0.1)
    assert orientation_distance(a, b) == pytest.approx(orientation_distance(b, a))


def test_diagonal_translation_distance():
    assert translation_distance(
        PoseSI(0, 0, 0, 0, 0, 0), PoseSI(0.003, 0.004, 0, 0, 0, 0)
    ) == pytest.approx(0.005)
