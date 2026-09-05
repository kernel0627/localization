"""Independent geometric and information-flow checks for Q1(3)."""

import numpy as np
import pytest

from scripts.q1_1.localization import GeometryError, pairwise_angles
from scripts.q1_2.run_validation import table_positions
from scripts.q1_3.audit_information import alternative_for_receiver
from scripts.q1_3.joint_localization import (
    ideal_formation,
    jacobian_diagnostics,
    joint_jacobian,
    localize_formation,
    movement_commands,
    observation_layout,
    predict_angles,
)
from scripts.q1_3.run_joint_baseline import simulate_adjustment
from scripts.q1_3.transmitter_selection import enumerate_designs, select_design

CONFIGS = ((2, 8), (3, 9))


def test_receiver_only_angles_admit_distinct_nearby_formations():
    for points in (ideal_formation(), table_array()):
        for receiver_id in range(2, 10):
            alternative = alternative_for_receiver(points, receiver_id)
            np.testing.assert_array_equal(alternative[:2], points[:2])
            assert np.linalg.norm(alternative[receiver_id] - points[receiver_id]) > 0.1
            for pair in CONFIGS:
                if receiver_id in pair:
                    continue
                transmitters = [0, 1, *pair]
                np.testing.assert_allclose(
                    pairwise_angles(points[receiver_id], points[transmitters]),
                    pairwise_angles(
                        alternative[receiver_id], alternative[transmitters]
                    ),
                    atol=1e-12,
                    rtol=0,
                )


def table_array():
    table = table_positions()
    return np.array([table[i] for i in range(10)])


def independent_observations(points, configurations=CONFIGS):
    angles = []
    for pair in configurations:
        transmitters = (0, 1, *pair)
        for receiver in range(2, 10):
            if receiver not in transmitters:
                angles.extend(
                    pairwise_angles(points[receiver], points[list(transmitters)])
                )
    return np.array(angles)


def test_joint_angles_match_existing_q1_1_and_jacobian_matches_finite_difference():
    points = table_array()
    layout = observation_layout(CONFIGS)
    assert predict_angles(points, layout) == pytest.approx(
        independent_observations(points), abs=1e-13
    )
    step = 1e-4
    numerical = np.zeros((72, 16))
    for column in range(16):
        plus, minus = points.copy(), points.copy()
        i, axis = 2 + column // 2, column % 2
        plus[i, axis] += step
        minus[i, axis] -= step
        numerical[:, column] = (
            independent_observations(plus) - independent_observations(minus)
        ) / (2 * step)
    np.testing.assert_allclose(joint_jacobian(points, layout), numerical, atol=2e-10)


def test_enumeration_rank_counts_and_design_selection():
    singles, designs = enumerate_designs()
    assert len(singles) == 28 and {r["rank"] for r in singles} == {14}
    assert len(designs) == 210 and {r["rank"] for r in designs} == {16}
    selected = select_design(designs)
    assert (selected["a"], selected["b"]) == ([2, 8], [3, 9])
    assert selected["sigma_min_rad_per_m"] == max(
        r["sigma_min_rad_per_m"] for r in designs
    )
    # Design identities are invariant to a change of length units.
    _, scaled = enumerate_designs(radius=250)
    selected_scaled = select_design(scaled)
    assert selected_scaled["a"] == selected["a"]
    assert selected_scaled["b"] == selected["b"]
    assert selected_scaled["sigma_min_rad_per_m"] == pytest.approx(
        selected["sigma_min_rad_per_m"] / 2.5
    )


def test_joint_recovery_from_independent_angles_with_unknown_transmitter_offsets():
    points = table_array()
    result = localize_formation(CONFIGS, independent_observations(points))
    assert result.success
    np.testing.assert_allclose(result.positions, points, atol=1e-7)
    # Diverse nearby formations, including transverse rather than radial offsets.
    rng = np.random.default_rng(20220905)
    for _ in range(12):
        perturbed = ideal_formation()
        perturbed[2:] += rng.uniform(-5, 5, (8, 2))
        result = localize_formation(CONFIGS, independent_observations(perturbed))
        assert result.success
        np.testing.assert_allclose(result.positions, perturbed, atol=1e-7)


def test_static_slots_hold_roles_and_two_round_recovery():
    initial = table_array()
    run = simulate_adjustment(initial, CONFIGS)
    np.testing.assert_array_equal(initial, table_array())
    assert run["metrics"][0]["max_position_error_m"] == pytest.approx(
        12.011139, abs=1e-6
    )
    assert run["metrics"][-1]["max_position_error_m"] < 1e-7
    for round_number, held in enumerate(CONFIGS, start=1):
        before, after = run["states"][round_number - 1 : round_number + 1]
        np.testing.assert_array_equal(before[[0, 1, *held]], after[[0, 1, *held]])
        angles = [
            r["angle_rad"] for r in run["observations"] if r["round"] == round_number
        ]
        np.testing.assert_allclose(angles, independent_observations(before), atol=1e-13)
    # Controller uses the estimated error: injected bias yields opposite final error.
    biased = initial.copy()
    biased[4] += [0.5, -0.25]
    commands = movement_commands(biased, CONFIGS[0])
    np.testing.assert_allclose(
        initial[4] + commands[4] - ideal_formation()[4], [-0.5, 0.25]
    )


def test_underidentified_stopped_and_inconsistent_fits_are_rejected():
    nominal = ideal_formation()
    single = (CONFIGS[0],)
    result = localize_formation(single, independent_observations(nominal, single))
    assert not result.success and result.diagnostics["rank"] == 14
    table_diagnostics = jacobian_diagnostics(
        joint_jacobian(table_array(), observation_layout(single))
    )
    assert table_diagnostics["rank"] == 16
    angles = independent_observations(table_array())
    assert not localize_formation(CONFIGS, angles, max_nfev=1).success
    angles[0] += 0.02
    assert not localize_formation(CONFIGS, angles).success


def test_invalid_geometry_and_reference_assumptions_fail_explicitly():
    with pytest.raises(ValueError):
        observation_layout(((2, 2),))
    with pytest.raises(ValueError):
        simulate_adjustment(table_array(), ((2, 8), (2, 9)))
    shifted = table_array()
    shifted[1, 0] += 1
    with pytest.raises(ValueError):
        simulate_adjustment(shifted, CONFIGS)
    collision = table_array()
    collision[4] = collision[0]
    with pytest.raises(GeometryError):
        predict_angles(collision, observation_layout(CONFIGS))
    collinear = ideal_formation()
    collinear[4] = [50, 0]
    with pytest.raises(GeometryError):
        joint_jacobian(collinear, observation_layout(CONFIGS))
