"""Regression checks for the Q2 noise/gain analysis contract."""

import json

import numpy as np
import pytest

from scripts.q2.analyze_noise_gain import (
    GAINS,
    N_BOOTSTRAP,
    N_MAIN,
    RECEIVER_IDS,
    _noise_inputs_for_case,
    azimuth_difference_matrix,
    bootstrap_covariance,
    fold_unsigned_difference,
    linear_monte_carlo,
    linear_theory,
    nonlinear_fixed_budget,
    receiver_noise_contract,
    row_selection_comparison,
)


def test_azimuth_noise_contract_has_shared_covariance_and_fold_visibility() -> None:
    sigma_deg = 0.05
    sigma = np.deg2rad(sigma_deg)
    sigma_phi = sigma / np.sqrt(2.0)
    rows = receiver_noise_contract(sigma_deg)
    assert len(rows) == len(RECEIVER_IDS)
    for row in rows:
        d = np.asarray(json.loads(row["D_selected"]), dtype=float)
        covariance = np.asarray(json.loads(row["Sigma_theta_selected"]), dtype=float)
        np.testing.assert_allclose(np.diag(covariance), sigma**2, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(covariance, sigma_phi**2 * d @ d.T, rtol=1e-12, atol=1e-18)
        assert np.isfinite(d).all()
    # The pairwise construction is continuous away from folds and explicitly
    # returns the unsigned 0/pi representatives.
    assert fold_unsigned_difference(0.0, 2.0 * np.pi - 0.1) == pytest.approx(0.1)
    assert fold_unsigned_difference(0.0, np.pi) == pytest.approx(np.pi)
    _, d_full = azimuth_difference_matrix(13, selected_only=False)
    assert np.isnan(d_full).any()


def test_bootstrap_measurement_covariance_is_not_a_locked_reference_covariance() -> None:
    result = bootstrap_covariance(0.05)
    sigma_measurement = np.asarray(result["Sigma_C_measurement"])
    theory = linear_theory(0.05, 0.5, 0.5)
    sigma_locked = np.asarray(theory["covariance_reference_error"])
    assert np.trace(sigma_locked) < np.trace(sigma_measurement)
    assert np.linalg.norm(np.asarray(theory["cross_covariance_receiver_error"])[0, 1]) > 0.0


def test_gain_stability_condition_and_steady_white_noise_formula() -> None:
    with pytest.raises(ValueError, match="0 < eta"):
        linear_theory(0.05, 0.0, 0.5)
    with pytest.raises(ValueError, match="0 < eta"):
        linear_theory(0.05, 0.5, 2.0)
    for eta in GAINS:
        result = linear_theory(0.05, eta, eta)
        np.testing.assert_allclose(
            result["white_noise_steady_multiplier"], eta / (2.0 - eta)
        )


def test_vectorized_linear_monte_carlo_matches_finite_horizon_prediction() -> None:
    trials = 2500
    rng = np.random.default_rng(12345)
    sigma_deg = 0.05
    base = rng.normal(0.0, np.deg2rad(sigma_deg), size=(trials, N_BOOTSTRAP, 2))
    azimuth = rng.normal(
        0.0,
        np.deg2rad(sigma_deg) / np.sqrt(2.0),
        size=(trials, N_MAIN, len(RECEIVER_IDS), 3),
    )
    initial_reference = np.array([0.08, -0.06])
    initial_receivers = np.zeros((len(RECEIVER_IDS), 2))
    result = linear_monte_carlo(
        sigma_deg,
        0.5,
        0.5,
        base_noise=base,
        azimuth_noise=azimuth,
        e_c0=initial_reference,
        e_receiver0=initial_receivers,
    )
    theory = linear_theory(
        sigma_deg,
        0.5,
        0.5,
        e_c0=initial_reference,
        e_receiver0=initial_receivers,
    )
    assert abs(result["position_rms_d"] / theory["position_rms_d"] - 1.0) < 0.04


def test_covariance_selection_comparison_keeps_production_choice_visible() -> None:
    rows = row_selection_comparison(0.05)
    selected = [row for row in rows if row["production_selected"]]
    assert len(selected) == len(RECEIVER_IDS)
    # With equal independent azimuth noise, any two regular differences carry
    # the same covariance information.  The existing geometric selector is a
    # conditioning/tie-break choice, so the comparison should report a tie,
    # without silently replacing the production pair.
    assert all(row["geometric_selector_matches_trace"] for row in selected)
    assert all(row["geometric_selector_matches_lambda_max"] for row in selected)


def test_fixed_budget_nonlinear_run_reports_stop_boundary_and_cost() -> None:
    seed, sigma_deg = 11, 0.01
    base, azimuth = _noise_inputs_for_case(seed, sigma_deg)
    rng = np.random.default_rng(seed + 900_000)
    receiver_errors = rng.normal(size=(len(RECEIVER_IDS), 2))
    receiver_errors /= np.maximum(np.linalg.norm(receiver_errors, axis=1, keepdims=True), 1e-12)
    receiver_errors *= 0.04
    result, failures, trace = nonlinear_fixed_budget(
        seed=seed,
        sigma_deg=sigma_deg,
        eta_c=0.5,
        eta_r=0.5,
        base_noise=base,
        azimuth_noise=azimuth,
        initial_reference_error=np.array([0.08, -0.06]),
        initial_receiver_errors=receiver_errors,
    )
    assert result["online_status"] == "not_evaluated_fixed_budget"
    assert result["measurement_slots"] == 2 * N_BOOTSTRAP + N_MAIN
    assert result["transmitter_uses"] == 4 * N_BOOTSTRAP + 3 * N_MAIN
    assert result["original_angle_stop_hits"] == 0
    assert result["failure_count"] == len(failures)
    assert len(trace) == N_BOOTSTRAP + N_MAIN * len(RECEIVER_IDS)
    assert np.isfinite(result["final_positions"]).all()
    assert all(np.isfinite(np.asarray(item["action"])).all() for item in trace)
