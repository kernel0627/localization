"""Noise geometry, seeded streams and original-controller regression checks."""

import numpy as np
import pytest

from scripts.q1_1.localization import fy_position, pairwise_angles
from scripts.q1_3.run_iterative_reference_baseline import simulate_adjustment
from scripts.q1_3.simulation_noise import SimulationNoise
from scripts.q1_3.run_robustness import initial_condition, aggregate


def test_zero_noise_preserves_original_trajectory_and_costs():
    initial = initial_condition(-1)
    a = simulate_adjustment(initial, max_epochs=1, retain_details=False)
    b = simulate_adjustment(
        initial, max_epochs=1, retain_details=False, noise=SimulationNoise()
    )
    np.testing.assert_array_equal(a["final_positions"], b["final_positions"])
    assert a["history"] == b["history"]
    assert a["summary"]["status"] == b["summary"]["status"]


def test_noisy_angles_share_rays_and_have_expected_covariance():
    position = np.zeros(2)
    directions = np.deg2rad([10, 40, 80, 130])
    transmitters = 100 * np.column_stack([np.cos(directions), np.sin(directions)])
    noise = SimulationNoise(bearing_std_deg=0.01)
    errors = np.array(
        [
            noise.observe(position, transmitters, t, 3)
            - pairwise_angles(position, transmitters)
            for t in range(5000)
        ]
    )
    # Angles 01 + 12 = 02 for these ordered rays. Independent pair errors would fail.
    np.testing.assert_allclose(errors[:, 0] + errors[:, 3], errors[:, 1], atol=1e-14)
    variance = np.deg2rad(0.01) ** 2
    covariance = np.cov(errors.T)
    assert covariance[0, 0] == pytest.approx(2 * variance, rel=0.07)
    assert covariance[0, 1] == pytest.approx(variance, rel=0.1)
    assert covariance[0, 3] == pytest.approx(-variance, rel=0.1)


def test_streams_are_reproducible_separate_and_preserve_fixed_references():
    q = np.array([fy_position(i) for i in range(10)])
    noise = SimulationNoise(0.01, 0.01, 123)
    a = noise.observe(q[3], q[[0, 1, 4, 7]], 10, 3)
    np.testing.assert_array_equal(a, noise.observe(q[3], q[[0, 1, 4, 7]], 10, 3))
    assert not np.array_equal(a, noise.observe(q[3], q[[0, 1, 4, 7]], 11, 3))
    assert noise.execute(0, 0, 10, 3) == (0, 0)
    result = simulate_adjustment(q, max_epochs=1, noise=noise, retain_details=False)
    np.testing.assert_array_equal(result["final_positions"][:2], q[:2])
    assert (
        result["summary"]["transmitter_uses"]
        == 4 * result["summary"]["measurement_slots"]
    )
    with pytest.raises(ValueError):
        SimulationNoise(-1)


def test_failed_trials_remain_in_rate_and_quantile_denominators():
    rows = []
    for status, stopped, error in [
        ("quiet_full_cycle", True, 0.001),
        ("max_epochs_reached", False, 0.002),
        ("local_fit_failed", False, 3.0),
    ]:
        rows.append(
            dict(
                condition="exact",
                trial=len(rows),
                status=status,
                stopped=stopped,
                final_below_1cm=error < 0.01,
                joint_success_1cm=stopped and error < 0.01,
                final_below_10cm=error < 0.1,
                first_1cm_slot=None,
                max_position_error_m=error,
                rms_position_error_m=error,
                last28_mean_max_error_m=error,
                measurement_slots=560,
                transmitter_uses=2240,
                movement_m=10,
            )
        )
    result = aggregate(rows)[0]
    assert result["runs"] == 3 and result["joint_success_rate"] == 1 / 3
    assert result["budget_exhausted_runs"] == 1 and result["fit_failure_runs"] == 1
    assert result["max_position_error_m_p95"] > 2


def test_linear_ray_derivative_and_periodic_noise_covariance():
    from scripts.q1_3.analyze_noise_floor import noise_covariance, ray_angle_derivative

    directions = np.deg2rad([145, -30, 70])
    with pytest.raises(ValueError):
        ray_angle_derivative(np.array([0, np.pi]))
    j = ray_angle_derivative(directions)

    def angles(values):
        from itertools import combinations

        return np.array(
            [
                abs(
                    np.arctan2(
                        np.sin(values[a] - values[b]), np.cos(values[a] - values[b])
                    )
                )
                for a, b in combinations(range(3), 2)
            ]
        )

    numeric = np.column_stack(
        [
            (angles(directions + 1e-7 * d) - angles(directions - 1e-7 * d)) / (2e-7)
            for d in np.eye(3)
        ]
    )
    np.testing.assert_allclose(j, numeric, atol=3e-9)
    a, q, p, phases, residual = noise_covariance()
    assert residual < 1e-12
    assert np.linalg.eigvalsh(p).min() > 0
    assert len(phases) == 28
    # Independent iterative covariance propagation reaches the Lyapunov solution.
    iterative = np.zeros((16, 16))
    for _ in range(20):
        iterative = a @ iterative @ a.T + q
    np.testing.assert_allclose(iterative, p, rtol=1e-10, atol=1e-8)
