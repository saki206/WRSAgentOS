import math

import numpy as np
import pytest

from wrs_agent.env.fafu_gripper import MAX_OPEN, MIN_OPEN, closing_travel, contact_tf


@pytest.mark.parametrize("width", [MIN_OPEN, 0.023, 0.02447608876694598, 0.040, 0.060, MAX_OPEN])
def test_parallel_closure_keeps_the_grasp_frame_on_the_deeper_inner_pad(width):
    # Original inner faces, sampled 8 mm behind the tips instead of at their edge.
    left_pad = np.array([0.16210080790519714, -0.041, 0.])
    right_pad = np.array([0.16210080790519714, 0.045, 0.])
    left_pad[1] += closing_travel(width)
    right_pad[1] -= closing_travel(width)
    assert right_pad[1]-left_pad[1] == pytest.approx(width, abs=1e-12)
    tf = contact_tf(width)
    np.testing.assert_allclose(tf[:3, 3]+[0.165, 0., 0.], (left_pad+right_pad)/2, atol=1e-12)
    np.testing.assert_array_equal(tf[:3, :3], np.eye(3))
    np.testing.assert_array_equal(tf[3], [0., 0., 0., 1.])


def test_closing_does_not_retract_or_rotate_the_tool_frame():
    reference = contact_tf(MAX_OPEN)
    for width in np.linspace(MAX_OPEN, MIN_OPEN, 17):
        np.testing.assert_array_equal(contact_tf(float(width)), reference)
    assert closing_travel(MAX_OPEN) == 0
    assert closing_travel(MIN_OPEN) > 0.04


@pytest.mark.parametrize("function", [closing_travel, contact_tf])
@pytest.mark.parametrize("width", [
    MIN_OPEN - 0.0001, MAX_OPEN + 0.0001, -0.01, 0.0,
    math.nan, math.inf, -math.inf,
])
def test_opening_commands_reject_impossible_or_nonfinite_widths(function, width):
    with pytest.raises(ValueError, match="width_out_of_range"):
        function(width)
