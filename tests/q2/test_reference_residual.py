"""Checks on residual propagation and the finite calibration interface."""

import numpy as np
import pytest

from scripts.q2.analyze_reference_residual import linear_audit, propagation_blocks
from scripts.q2.run_triangle_reference import _bootstrap_record, make_initial_state, run_case
from scripts.q2.triangle_reference import TARGET_TEMPLATE


def test_propagation_derivatives_match_observations_and_solver() -> None:
    _, audit = linear_audit()
    for key in (
        "max_bootstrap_derivative_error", "max_receiver_derivative_error",
        "max_apex_derivative_error", "max_solver_candidate_derivative_error",
    ):
        assert audit[key] < 2e-8, (key, audit[key])
    # A horizontal apex shift at FY13 has an independently visible symmetry:
    # the apparent receiver shift is equal and opposite on the base line.
    np.testing.assert_allclose(propagation_blocks(13)[2], [[-1., 0.], [0., 0.]], atol=1e-12)


def test_finite_bootstrap_keeps_the_actual_calibration_residual() -> None:
    case = make_initial_state(0.1, 11)
    state, summary, rows, actions = _bootstrap_record(
        case=case, gain=0.5, max_rounds=30, position_tolerance=0.005,
    )
    assert summary["status"] == "converged"
    assert rows[-1]["event"] == "stop_estimated_position_threshold"
    assert rows[-1]["estimated_position_error_d"] <= 0.005
    assert rows[-2]["estimated_position_error_d"] > 0.005
    residual = state[0]-TARGET_TEMPLATE[0]
    assert 1e-4 < np.linalg.norm(residual) <= 0.005
    np.testing.assert_allclose(
        residual, 0.5**len(actions)*(case.positions[0]-TARGET_TEMPLATE[0]), atol=1e-12,
    )
    np.testing.assert_array_equal(state[1:], case.positions[1:])


def test_online_stop_does_not_imply_historical_shape_accuracy() -> None:
    result = run_case(0.1, 11, 0.5, 30, bootstrap_position_tolerance=0.005)
    assert result["online_status"] == "success"
    assert result["status"] == "failure"
    assert "shape_error_above_1e-6d" in result["failure_types"]
    assert 1e-6 < result["final_max_position_error_d"] < 0.01


def test_position_stop_uses_estimate_not_simulator_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deliberately substitute a biased angle estimator. An online rule must
    # follow its estimate even when the offline simulator knows it is wrong.
    monkeypatch.setattr("scripts.q2.run_triangle_reference.estimate_apex",
                        lambda observed: TARGET_TEMPLATE[0].copy())
    case = make_initial_state(0.1, 11)
    state, summary, rows, actions = _bootstrap_record(
        case=case, gain=0.5, max_rounds=30, position_tolerance=0.005,
    )
    assert summary["status"] == "converged"
    assert rows[-1]["estimated_position_error_d"] == 0
    assert not actions
    assert np.linalg.norm(state[0]-TARGET_TEMPLATE[0]) > 0.005
    np.testing.assert_array_equal(state, case.positions)


@pytest.mark.parametrize("tolerance", [0., -1., np.nan, np.inf])
def test_invalid_position_budget_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _bootstrap_record(case=make_initial_state(0., None), gain=0.5,
                          max_rounds=30, position_tolerance=tolerance)
