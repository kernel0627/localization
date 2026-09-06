"""Small deterministic checks for Gaussian all-drone threshold integration."""

import numpy as np

from scripts.q1_3.analyze_two_configuration_precision_probability import (
    integrate_all_drone_probability,
    union_failure_bound,
)


def test_zero_covariance_always_meets_positive_precision_thresholds():
    counts = integrate_all_drone_probability(
        np.zeros((16, 16)), (0.001, 0.01), 1000, [17, 2, 4], batch_size=200
    )
    np.testing.assert_array_equal(counts, [1000, 1000])


def test_union_bound_is_conservative_without_block_independence_assumption():
    covariance = np.zeros((16, 16))
    for drone in range(8):
        covariance[2 * drone : 2 * drone + 2, 2 * drone : 2 * drone + 2] = 0.25 * np.eye(2)
    epsilon = 2.0
    radial_tail = np.exp(-epsilon**2 / (2 * 0.25))
    exact_independent_failure = 1 - (1 - radial_tail) ** 8
    assert union_failure_bound(covariance, epsilon) >= exact_independent_failure
