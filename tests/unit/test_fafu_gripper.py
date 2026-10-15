import math

import numpy as np
import pytest

from wrs_agent.env.fafu_gripper import (
    MAX_ANGLE,
    MAX_OPEN,
    MIN_OPEN,
    angle_for_width,
    contact_tf,
    width_for_angle,
)


def fingertip_contacts(angle):
    """Rotate the two original mesh tips around their original hinge axes."""
    contacts = []
    for hinge, tip, signed_angle in (
        ([0.10705, -0.0675], [0.17010080790519714, -0.041], angle),
        ([0.10705, 0.0715], [0.17010080790519714, 0.045], -angle),
    ):
        c, s = math.cos(signed_angle), math.sin(signed_angle)
        rotation = np.array([[c, -s], [s, c]])
        hinge = np.asarray(hinge)
        contacts.append(hinge + rotation @ (np.asarray(tip) - hinge))
    return np.asarray(contacts)


def test_original_opening_and_servo_limit_have_the_calibrated_endpoints():
    assert MAX_OPEN == pytest.approx(0.086)
    assert MAX_ANGLE == pytest.approx(math.pi / 3)
    assert MIN_OPEN == pytest.approx(0.00329279725, abs=1e-10)
    assert angle_for_width(MAX_OPEN) == pytest.approx(0.0, abs=1e-12)
    assert angle_for_width(MIN_OPEN) == pytest.approx(MAX_ANGLE, abs=1e-12)
    assert width_for_angle(0.0) == pytest.approx(MAX_OPEN)
    assert width_for_angle(MAX_ANGLE) == pytest.approx(MIN_OPEN)


@pytest.mark.parametrize("width", [0.086, 0.060, 0.024476])
def test_reported_opening_matches_the_rotated_original_finger_contacts(width):
    angle = angle_for_width(width)
    contacts = fingertip_contacts(angle)
    assert contacts[1, 1] - contacts[0, 1] == pytest.approx(width, abs=1e-10)
    assert width_for_angle(angle) == pytest.approx(width, abs=1e-10)
    # The grasp frame must track the actual moving finger tips, including their
    # original two-millimetre offset from the URDF's tool_link centre line.
    midpoint = contacts.mean(axis=0)
    tf = contact_tf(width)
    assert tf[:2, 3] + [0.165, 0.0] == pytest.approx(midpoint, abs=1e-10)
    assert tf[1, 3] == pytest.approx(0.002, abs=1e-12)


def test_closing_retracts_the_contact_tcp_without_rotating_its_axes():
    widths = np.linspace(MAX_OPEN, MIN_OPEN, 17)
    angles = [angle_for_width(float(width)) for width in widths]
    frames = [contact_tf(float(width)) for width in widths]
    assert np.all(np.diff(angles) > 0)
    assert np.all(np.diff([tf[0, 3] for tf in frames]) < 0)
    assert frames[0][0, 3] == pytest.approx(0.00510080790519714, abs=1e-12)
    assert frames[-1][0, 3] < -0.049
    for tf in frames:
        assert tf.shape == (4, 4)
        assert np.all(np.isfinite(tf))
        assert tf[:3, :3] == pytest.approx(np.eye(3))
        assert tf[2, 3] == pytest.approx(0.0)
        assert tf[3] == pytest.approx([0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize("angle", [0.0, MAX_ANGLE / 2, MAX_ANGLE])
def test_angle_feedback_recovers_the_commanded_position(angle):
    width = width_for_angle(angle)
    assert MIN_OPEN <= width <= MAX_OPEN
    assert angle_for_width(width) == pytest.approx(angle, abs=1e-12)


@pytest.mark.parametrize("function", [angle_for_width, contact_tf])
@pytest.mark.parametrize("width", [
    MIN_OPEN - 0.0001, MAX_OPEN + 0.0001, -0.01, 0.0,
    math.nan, math.inf, -math.inf,
])
def test_opening_commands_reject_impossible_or_nonfinite_widths(function, width):
    with pytest.raises(ValueError):
        function(width)


@pytest.mark.parametrize("angle", [
    -0.0001, MAX_ANGLE + 0.0001, math.nan, math.inf, -math.inf,
])
def test_angle_feedback_rejects_out_of_range_or_nonfinite_values(angle):
    with pytest.raises(ValueError):
        width_for_angle(angle)
