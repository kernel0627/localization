"""Independent numerical checks of reference-error propagation and cycle order."""

import numpy as np
import pytest

from scripts.q1_1.localization import pairwise_angles
from scripts.q1_3.analyze_iterative_reference import (
    active_cycle,
    cycle_matrix,
    linear_model,
    observation_derivative,
    positions_from_error,
)
from scripts.q1_3.local_adjustment import LocalSettings


@pytest.mark.parametrize(
    "receiver,tx", [(3, (0, 1, 2)), (8, (0, 5, 7)), (4, (0, 3, 6))]
)
def test_observation_derivative_includes_moving_reference_errors(receiver, tx):
    h = 1e-3
    numerical = np.empty((3, 16))
    for column in range(16):
        points = [
            positions_from_error(sign * h * np.eye(16)[column]) for sign in (1, -1)
        ]
        angles = [pairwise_angles(p[receiver], p[list(tx)]) for p in points]
        numerical[:, column] = (angles[0] - angles[1]) / (2 * h)
    np.testing.assert_allclose(
        observation_derivative(receiver, tx), numerical, atol=1e-10, rtol=1e-7
    )


@pytest.mark.parametrize("gain", [0.25, 0.5, 1.0])
def test_cycle_linearization_matches_original_nonlinear_controller(gain):
    matrices, choices, _ = linear_model(gain)
    a = cycle_matrix(matrices)
    assert min(row["relative_selection_gap"] for row in choices) > 0.004
    assert max(row["own_block_identity_error"] for row in choices) < 1e-12
    # Independent original solver checks product order, signs and synchronous updates.
    direction = np.random.default_rng(43).normal(size=16)
    direction /= np.linalg.norm(direction)
    h = 1e-3
    values = []
    expected = [
        (r["phase"], r["receiver_id"], tuple(map(int, r["selected_pair"].split("-"))))
        for r in choices
    ]
    for sign in (1, -1):
        value, selection, clips, failures = active_cycle(
            sign * h * direction, LocalSettings(gain=gain)
        )
        assert selection == expected
        assert clips == failures == 0
        values.append(value)
    np.testing.assert_allclose(
        (values[0] - values[1]) / (2 * h), a @ direction, atol=3e-6, rtol=1e-4
    )
