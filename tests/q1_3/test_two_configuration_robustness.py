"""Semantic checks for the two-configuration robustness runner."""
# ruff: noqa: E402

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from appendix1.optimize_schedule import simulate
from scripts.q1_2.run_validation import table_positions
from scripts.q1_3.run_robustness import initial_condition
from scripts.q1_3.simulation_noise import SimulationNoise
from scripts.q1_3.run_two_configuration_robustness import DiagnosticFactory
from scripts.q1_3.two_configuration_noise import TwoConfigurationNoise


SCHEDULE = ((1, 4, 5), (1, 7, 8))


def table1():
    points = table_positions()
    return np.array([points[i] for i in range(10)])


def test_compatible_streams_reproduce_legacy_white_noise_interface():
    position = initial_condition(4)[2]
    transmitters = initial_condition(4)[[0, 1, 4, 5]]
    old = SimulationNoise(.01, .01, 2026090505)
    new = TwoConfigurationNoise(.01, .01, 0, 2026090505)
    np.testing.assert_array_equal(
        old.observe(position, transmitters, 7, 2),
        new.observe(position, transmitters, 7, 2),
    )
    assert old.execute(2, .03, 7, 2) == new.execute(2, .03, 7, 2)


def test_link_bias_is_fixed_per_public_link_and_does_not_change_white_stream():
    noisy = TwoConfigurationNoise(.01, 0, .001, 99)
    assert noisy.link_bias_rad(3, 4) == noisy.link_bias_rad(3, 4)
    assert noisy.link_bias_rad(3, 4) != noisy.link_bias_rad(3, 5)
    with_bias = noisy.observe(table1()[2], table1()[[0, 1, 4, 5]], 1, 2)
    without_bias = TwoConfigurationNoise(.01, 0, 0, 99).observe(
        table1()[2], table1()[[0, 1, 4, 5]], 1, 2
    )
    assert not np.array_equal(with_bias, without_bias)
    bias_only = TwoConfigurationNoise(0, 0, .001, 99)
    # Slot 1 and 3 have the same public transmitter IDs, so their fixed-link
    # perturbations repeat exactly even though no controller sees those offsets.
    np.testing.assert_array_equal(
        bias_only.observe(table1()[2], table1()[[0, 1, 4, 5]], 1, 2),
        bias_only.observe(table1()[2], table1()[[0, 1, 4, 5]], 3, 2),
    )


def test_zero_custom_noise_preserves_frozen_two_configuration_trajectory():
    exact = simulate(table1(), SCHEDULE, max_slots=100)
    zero = simulate(table1(), SCHEDULE, max_slots=100, noise=TwoConfigurationNoise())
    assert exact[0]["final_positions"] == zero[0]["final_positions"]
    assert exact[1] == zero[1]


def test_diagnostic_wrapper_does_not_change_an_exact_controller_trajectory():
    exact = simulate(table1(), SCHEDULE, max_slots=100)
    records = []
    diagnosed = simulate(
        table1(), SCHEDULE, max_slots=100,
        controller_factory=DiagnosticFactory(records),
    )
    assert exact == diagnosed
    assert len(records) == 6 * diagnosed[0]["measurement_slots"]
