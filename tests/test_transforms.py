"""Independent axis matrices lock signs/order; round trips alone cannot do that."""

import math
from itertools import product

import numpy as np
import pytest

from inspect_robots_dobot.transforms import (
    agent_relative_to_rotation,
    dobot_native_to_rotation,
    interpolate_rotation,
    rotation_distance,
    rotation_to_agent_relative,
    rotation_to_dobot_native,
)


def rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize("index,expected", [(0, rz(0.4)), (1, ry(-0.4)), (2, rx(0.4))])
def test_positive_agent_axes_against_independent_matrices(index, expected):
    values = [0.0] * 3
    values[index] = 0.4
    actual = agent_relative_to_rotation(*values, np.eye(3))
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(rotation_to_agent_relative(expected, np.eye(3)), values, atol=1e-15)


@pytest.mark.parametrize("index,expected", [(0, rx(0.4)), (1, ry(0.4)), (2, rz(0.4))])
def test_positive_native_axes_against_independent_matrices(index, expected):
    values = [0.0] * 3
    values[index] = 0.4
    np.testing.assert_allclose(dobot_native_to_rotation(*values), expected, atol=1e-15)
    np.testing.assert_allclose(rotation_to_dobot_native(expected), values, atol=1e-15)


def test_tool_direction_signs_and_world_axis_left_composition():
    down = np.array([0, 0, -1])
    assert (agent_relative_to_rotation(0, 0.2, 0, np.eye(3)) @ down)[0] > 0
    assert (agent_relative_to_rotation(0, 0, 0.2, np.eye(3)) @ down)[1] > 0
    assert (agent_relative_to_rotation(0.2, 0, 0, np.eye(3)) @ np.array([1, 0, 0]))[1] > 0
    reference = rz(0.7) @ ry(0.2) @ rx(-0.4)
    expected = rz(0.3) @ ry(-0.5) @ rx(0.6) @ reference
    actual = agent_relative_to_rotation(0.3, 0.5, 0.6, reference)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert not np.allclose(actual, reference @ rz(0.3) @ ry(-0.5) @ rx(0.6))


@pytest.mark.parametrize("reference", [np.eye(3), rz(0.7) @ ry(-0.4) @ rx(2.4), ry(math.pi / 2)])
@pytest.mark.parametrize(
    "angles",
    [
        (0, 0, 0),
        (0.4, 0, 0),
        (0, 0.6, 0),
        (0, 0, -0.8),
        (0.9, -0.5, 0.3),
        (math.pi - 1e-8, 1.2, -math.pi + 1e-8),
        (-math.pi + 1e-8, -1.2, math.pi - 1e-8),
    ],
)
def test_relative_roundtrip_with_independently_constructed_reference(reference, angles):
    yaw, pitch, roll = angles
    expected = rz(yaw) @ ry(-pitch) @ rx(roll) @ reference
    np.testing.assert_allclose(agent_relative_to_rotation(*angles, reference), expected, atol=1e-14)
    np.testing.assert_allclose(rotation_to_agent_relative(expected, reference), angles, atol=1e-14)


def test_dense_native_and_agent_roundtrips():
    reference = rz(0.3) @ ry(1.1) @ rx(-2.5)
    for x, y, z in product((-3, -1, 0, 1, 3), (-1.4, -0.4, 0, 0.8, 1.4), (-3, 0, 3)):
        expected = rz(z) @ ry(y) @ rx(x)
        decoded = rotation_to_dobot_native(expected)
        np.testing.assert_allclose(decoded, (x, y, z), atol=1e-13)
        np.testing.assert_allclose(dobot_native_to_rotation(*decoded), expected, atol=1e-13)
        target = agent_relative_to_rotation(z, y, x, reference)
        np.testing.assert_allclose(
            rotation_to_agent_relative(target, reference), (z, y, x), atol=1e-13
        )


@pytest.mark.parametrize("pitch", [math.pi / 2, -math.pi / 2, 2.2, -2.2])
def test_native_gimbal_and_noncanonical_angles_preserve_rotation(pitch):
    expected = rz(0.7) @ ry(pitch) @ rx(-0.3)
    decoded = rotation_to_dobot_native(expected)
    np.testing.assert_allclose(dobot_native_to_rotation(*decoded), expected, atol=1e-14)
    if abs(pitch) == math.pi / 2:
        assert decoded[0] == 0


@pytest.mark.parametrize("pitch", [math.pi / 2, -math.pi / 2, math.pi / 2 - 0.02])
def test_configured_relative_singularity_exclusion_is_enforced(pitch):
    with pytest.raises(ValueError, match="singularity"):
        agent_relative_to_rotation(0, pitch, 0, np.eye(3), singularity_margin=0.05)
    with pytest.raises(ValueError, match="singularity"):
        rotation_to_agent_relative(ry(-pitch), np.eye(3), singularity_margin=0.05)


def test_wrap_and_geodesic_interpolation_do_not_spin_the_long_way():
    epsilon = 1e-5
    a, b = rz(math.pi - epsilon), rz(-math.pi + epsilon)
    assert rotation_distance(a, b) == pytest.approx(2 * epsilon)
    np.testing.assert_allclose(interpolate_rotation(a, b, 0.5), rz(math.pi), atol=1e-14)
    assert rotation_to_agent_relative(rz(-math.pi), np.eye(3))[0] == math.pi
    assert rotation_distance(rx(math.pi), rx(-math.pi)) == 0


@pytest.mark.parametrize("angle", [0, 1e-8, 0.6, math.pi - 1e-8, math.pi])
def test_interpolation_endpoints_and_angular_fraction(angle):
    a, b = rz(0.3) @ ry(0.1), rz(0.3) @ ry(0.1) @ rx(angle)
    for fraction in (0, 0.25, 0.5, 1):
        actual = interpolate_rotation(a, b, fraction)
        np.testing.assert_allclose(actual, a @ rx(angle * fraction), atol=1e-12)
        assert rotation_distance(a, actual) == pytest.approx(angle * fraction, abs=1e-12)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((3, 3)), np.eye(4), np.diag([1, 1, -1]), np.full((3, 3), np.nan), np.eye(3) * 1.001],
)
def test_malformed_rotation_is_never_silently_projected(bad):
    with pytest.raises(ValueError, match="rotation"):
        rotation_to_dobot_native(bad)
    with pytest.raises(ValueError, match="rotation"):
        rotation_to_agent_relative(np.eye(3), bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_angles_reject(value):
    with pytest.raises(ValueError, match="finite"):
        dobot_native_to_rotation(0, value, 0)
    with pytest.raises(ValueError, match="finite"):
        agent_relative_to_rotation(value, 0, 0, np.eye(3))
