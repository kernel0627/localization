"""Regression tests for the frozen two-configuration analysis."""

import numpy as np

from scripts.q1_3.analyze_two_configuration import SCHEDULE, blocks, selected_matrices
from scripts.q1_3.analyze_iterative_reference import cycle_matrix


def test_two_configuration_matrix_matches_original_solver_on_random_direction():
    from scripts.q1_3.analyze_two_configuration import active_cycle
    from scripts.q1_3.local_adjustment import LocalSettings

    matrices, _, _ = selected_matrices(0.5)
    direction = np.random.default_rng(60).normal(size=16)
    direction /= np.linalg.norm(direction)
    h = 1e-3
    plus = active_cycle(h * direction, LocalSettings())[0]
    minus = active_cycle(-h * direction, LocalSettings())[0]
    np.testing.assert_allclose(
        (plus - minus) / (2 * h),
        cycle_matrix(matrices) @ direction,
        rtol=2e-4,
        atol=4e-6,
    )


def test_reference_core_is_closed_and_other_self_blocks_are_quarter_identity():
    assert SCHEDULE == ((1, 4, 5), (1, 7, 8))
    matrix = cycle_matrix(selected_matrices(0.5)[0])
    _, rows = blocks(matrix, 0.5)
    assert rows[0]["upper_right_max_abs"] < 1e-12
    assert rows[0]["other_self_block_max_error_from_expected_I"] < 1e-12
