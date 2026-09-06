"""Checks for the two-configuration periodic stochastic model."""

import numpy as np

from scripts.q1_3.analyze_two_configuration_robustness import (
    covariance_operator,
    fixed_link_bias_response,
    phase_matrices_and_injections,
    periodic_second_moment,
    second_moment_step,
)


def test_second_moment_operator_matches_direct_map():
    rng = np.random.default_rng(620)
    matrix = rng.normal(size=(16, 16)) / 20
    command = rng.normal(size=(16, 16)) / 30
    covariance = rng.normal(size=(16, 16))
    covariance = covariance @ covariance.T
    sigma = 0.07
    direct = second_moment_step(
        covariance, matrix, command, np.zeros((16, 16)), sigma
    )
    operator = covariance_operator(matrix, command, sigma)
    vectorized = (operator @ covariance.reshape(-1, order="F")).reshape(
        16, 16, order="F"
    )
    np.testing.assert_allclose(vectorized, direct, rtol=1e-12, atol=1e-12)


def test_iid_periodic_covariance_is_psd_and_closes_at_both_phases():
    matrices, commands, injections, _ = phase_matrices_and_injections(0.5)
    result = periodic_second_moment(
        matrices, commands, [injection @ injection.T for injection in injections]
    )
    assert result["cycle_spectral_radius"] < 1
    assert result["relative_residual"] < 1e-11
    for covariance in result["phase_covariances"].values():
        assert np.linalg.eigvalsh(covariance).min() > -1e-10


def test_fixed_link_bias_has_one_shared_cross_phase_loading_and_closes():
    matrices, _, injections, links = phase_matrices_and_injections(1.0)
    result = fixed_link_bias_response(matrices, injections)
    assert len(links) < sum(np.count_nonzero(injection.any(axis=0)) for injection in injections)
    assert result["closure_relative_residual"] < 1e-12
    phase_559 = result["phase_loadings"][559]
    phase_560 = result["phase_loadings"][560]
    np.testing.assert_allclose(
        phase_560, matrices[1] @ phase_559 + injections[1], rtol=0, atol=1e-11
    )


def test_independent_execution_error_is_mean_square_stable_for_stress_levels():
    matrices, commands, _, _ = phase_matrices_and_injections(0.5)
    zeros = [np.zeros((16, 16)), np.zeros((16, 16))]
    for sigma in (0.01, 0.05, 0.10):
        result = periodic_second_moment(matrices, commands, zeros, sigma)
        assert result["cycle_spectral_radius"] < 1
        assert result["relative_residual"] < 1e-12
