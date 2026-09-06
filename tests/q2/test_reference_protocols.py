import numpy as np

from scripts.q2.compare_reference_protocols import (
    FAIR_RECEIVER_ESTIMATE_BUDGET,
    estimate_receiver_with_anchors,
    run_dynamic_case,
    run_staged_budget_case,
)
from scripts.q2.triangle_reference import (
    RECEIVER_IDS,
    TARGET_ANCHORS,
    TARGET_TEMPLATE,
    receiver_angles,
)


def test_known_anchor_local_solver_recovers_current_geometry() -> None:
    anchors = TARGET_ANCHORS.copy()
    anchors[0] += np.array([0.02, -0.01])
    receiver_id = 12
    actual = TARGET_TEMPLATE[receiver_id - 1] + np.array([0.03, -0.02])
    result = estimate_receiver_with_anchors(
        receiver_id,
        receiver_angles(actual, anchors),
        anchors,
    )
    assert result["success"] is True
    np.testing.assert_allclose(result["position"], actual, atol=1e-8)
    assert len(result["selected_indices"]) == 2


def test_known_anchor_estimator_does_not_silently_use_target_apex() -> None:
    anchors = TARGET_ANCHORS.copy()
    anchors[0] += np.array([0.08, -0.03])
    receiver_id = 12
    actual = TARGET_TEMPLATE[receiver_id - 1] + np.array([0.01, -0.02])
    observed = receiver_angles(actual, anchors)
    result = estimate_receiver_with_anchors(receiver_id, observed, anchors)
    assert result["success"] is True
    np.testing.assert_allclose(result["position"], actual, atol=1e-8)
    assert np.linalg.norm(result["position"] - TARGET_TEMPLATE[receiver_id - 1]) > 0.01


def test_dynamic_round_accounting_and_current_anchor_payload() -> None:
    result = run_dynamic_case(0.1, 11, 1.0, 1.0, 0.005, max_rounds=30)
    assert result["status"] == "success"
    rounds = result["main"]["rounds"]
    assert result["measurement_slots"] == 3 * rounds
    assert result["tx_uses"] == 7 * rounds
    assert result["relay_scalar_count"] == 2 * rounds
    assert len(result["main"]["rounds_detail"]) == rounds
    for round_i in range(1, rounds + 1):
        calibration = next(
            row
            for row in result["rows"]
            if row.get("stage") == "dynamic_calibration" and row["round"] == round_i
        )
        receiver_rows = [
            row
            for row in result["rows"]
            if row.get("stage") == "dynamic_main" and row["round"] == round_i
        ]
        assert receiver_rows
        for row in receiver_rows:
            assert row["c_hat_x_used"] == calibration["c_hat_x"]
            assert row["c_hat_y_used"] == calibration["c_hat_y"]


def test_staged_gains_are_independent_and_cost_is_reconstructible() -> None:
    result = run_staged_budget_case(0.1, 11, 0.5, 1.0, 0.005, max_rounds=30)
    assert result["status"] == "success"
    assert result["bootstrap"]["gain"] == 0.5
    assert result["main"]["status"] == "converged"
    assert result["eta_r"] == 1.0
    assert result["measurement_slots"] == 2 * result["bootstrap"]["rounds"] + result["main"]["broadcast_slots"]
    assert result["tx_uses"] == 4 * result["bootstrap"]["rounds"] + result["main"]["tx_uses"]
    assert result["relay_scalar_count"] == 2 * result["bootstrap"]["rounds"]
    main_rows = [row for row in result["rows"] if row["stage"] == "main_budget"]
    assert set(row["receiver_id"] for row in main_rows) == set(RECEIVER_IDS)


def test_static_budget_reports_reference_residual_excess_separately() -> None:
    result = run_staged_budget_case(0.1, 11, 0.5, 0.5, 0.005, max_rounds=30)
    assert result["online_status"] == "success"
    # The observable receiver stop and the complete final-position budget are
    # distinct contracts when the static reference is still offset.
    assert result["final_position_budget_excess_d"] > 0.0
    assert result["final_position_budget_pass"] is False


def test_shared_budget_allocates_reference_residual_before_receiver_stop() -> None:
    result = run_staged_budget_case(
        0.1,
        11,
        0.5,
        0.5,
        0.005,
        max_rounds=30,
        position_budget=FAIR_RECEIVER_ESTIMATE_BUDGET,
        budget_group="fair_shared_position_budget",
    )
    assert result["budget_group"] == "fair_shared_position_budget"
    assert result["receiver_estimate_budget_d"] == 0.0045
    assert result["main"]["position_budget_d"] == 0.0045
    assert result["final_position_budget_pass"] is True
