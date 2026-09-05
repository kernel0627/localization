"""Checks for local information, dual-bias correction and public rotation."""

from dataclasses import replace

import numpy as np
import pytest

from scripts.q1_1.localization import fy_position, pairwise_angles
from scripts.q1_2.run_validation import table_positions
from scripts.q1_3.local_adjustment import (
    LocalSettings,
    ReceiverController,
    decide_local_adjustment,
    estimate_bias,
    execute_relative_polar_step,
    public_schedule,
)
from scripts.q1_3.run_adjustment import simulate_adjustment


def nominal():
    return np.array([fy_position(i) for i in range(10)])


def test_polar_bias_recovers_both_components_with_calibrated_references():
    q = nominal()
    i, pair = 3, (1, 6)
    theta = np.deg2rad(80 + 0.7)
    point = 107 * np.array([np.cos(theta), np.sin(theta)])
    angles = pairwise_angles(point, q[[0, *pair]])
    estimate, state, residual, success, _, _ = estimate_bias(i, pair, angles)
    assert success and residual < 1e-10
    np.testing.assert_allclose(estimate, point, atol=1e-7)
    assert state[0] == pytest.approx(7, abs=1e-7)
    assert state[1] / 100 == pytest.approx(np.deg2rad(0.7), abs=1e-8)


def test_public_schedule_has_legal_transmitters_and_covers_every_receiver():
    schedule = public_schedule()
    assert len(schedule) == 28 and len(set(schedule)) == 28
    assert all(len(tx) == 3 and 1 in tx and len(set(tx)) == 3 for tx in schedule)
    for i in range(2, 10):
        assert sum(i not in tx for tx in schedule) == 21
        assert sum(i in tx for tx in schedule) == 7


def test_decision_uses_own_angles_and_third_transmitter_consistency():
    table = table_positions()
    points = np.array([table[i] for i in range(10)])
    i, tx = 3, (1, 2, 4)
    angles = pairwise_angles(points[i], points[[0, *tx]])
    decision = decide_local_adjustment(i, tx, angles)
    assert len(decision.candidates) == 3
    assert decision.selected.score == max(
        c.score for c in decision.candidates if c.success
    )
    # Changing a drone that this receiver does not observe leaves its decision identical.
    points[7] += [8, -4]
    same_angles = pairwise_angles(points[i], points[[0, *tx]])
    assert decide_local_adjustment(i, tx, same_angles) == decision
    assert decision.radial_step_m != 0 and decision.angular_step_rad != 0


def test_relative_actuator_applies_two_commands_and_gain_limits():
    point = np.array([0.0, 110.0])
    after = execute_relative_polar_step(point, -3, -0.02)
    assert np.linalg.norm(after) == pytest.approx(107)
    assert np.arctan2(after[1], after[0]) == pytest.approx(np.pi / 2 - 0.02)
    q = nominal()
    tx = (1, 4, 7)
    point = 112 * q[3] / 100
    settings = LocalSettings(max_radial_step_m=1, max_angular_step_rad=0.001)
    decision = decide_local_adjustment(
        3, tx, pairwise_angles(point, q[[0, *tx]]), settings
    )
    assert abs(decision.radial_step_m) <= 1
    assert abs(decision.angular_step_rad) <= 0.001


def test_local_hold_requires_persistent_small_errors_and_reactivates():
    q = nominal()
    controller = ReceiverController(3)
    tx = (1, 4, 7)
    angles = pairwise_angles(q[3], q[[0, *tx]])
    for _ in range(20):
        assert controller.decide(tx, angles).status == "adjust"
    held = controller.decide(tx, angles)
    assert held.status == "within_tolerance"
    assert held.radial_step_m == held.angular_step_rad == 0
    disturbed = pairwise_angles(q[3] * 1.01, q[[0, *tx]])
    assert controller.decide(tx, disturbed).status == "adjust"


def test_table1_converges_using_local_stopping_and_keeps_transmitters_fixed():
    table = table_positions()
    initial = np.array([table[i] for i in range(10)])
    run = simulate_adjustment(initial)
    summary = run["summary"]
    assert summary["status"] == "quiet_full_cycle"
    assert summary["final_metrics"]["max_position_error_m"] < 0.01
    assert summary["failed_local_fits"] == 0
    np.testing.assert_array_equal(run["final_positions"][:2], initial[:2])
    states = np.array([[r["x_m"], r["y_m"]] for r in run["positions"]]).reshape(
        -1, 10, 2
    )
    for slot, tx in enumerate(public_schedule() * summary["epochs"], start=1):
        np.testing.assert_array_equal(
            states[slot, [0, *tx]], states[slot - 1, [0, *tx]]
        )
    last_cycle = run["steps"][-28 * 6 :]
    assert all(r["status"] == "within_tolerance" for r in last_cycle)
    np.testing.assert_array_equal(initial, np.array([table[i] for i in range(10)]))


def test_max_epoch_limit_and_invalid_inputs_are_explicit():
    table = table_positions()
    run = simulate_adjustment(
        np.array([table[i] for i in range(10)]), max_epochs=1, retain_details=False
    )
    assert run["summary"]["status"] == "max_epochs_reached"
    with pytest.raises(ValueError):
        replace(LocalSettings(), gain=2)
    with pytest.raises(ValueError):
        decide_local_adjustment(3, (1, 3, 5), np.ones(6))
    with pytest.raises(ValueError):
        decide_local_adjustment(3, (1, 4, 5), np.ones(3))
