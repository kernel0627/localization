"""Targeted tests for the finite Q2 geometry-boundary audit."""

import numpy as np

from scripts.q2.analyze_geometry_boundaries import (
    _receiver_row,
    _right_worst_direction,
    bootstrap_boundary_rows,
    counterexample_rows,
    main_stage_rows,
)
from scripts.q2.analyze_reference_residual import propagation_blocks
from scripts.q2.triangle_reference import (
    RECEIVER_IDS,
    TARGET_ANCHORS,
    angle_jacobian,
    receiver_angles,
)


def test_bootstrap_interior_and_exterior_base_limits_are_different() -> None:
    rows = bootstrap_boundary_rows()
    interior = [row for row in rows if row["path"] == "base_interior_x2"][-1]
    exterior = [row for row in rows if row["path"] == "base_exterior_x_minus1"][-1]
    assert interior["angle_sum_rad"] < 1e-7
    assert np.pi - exterior["angle_sum_rad"] < 1e-7
    assert interior["sigma_min_rad_per_d"] < 1e-8
    assert exterior["sigma_min_rad_per_d"] < 1e-8


def test_one_collinear_angle_does_not_make_all_receiver_rows_invalid() -> None:
    point = TARGET_ANCHORS[0] + 1e-4 * (TARGET_ANCHORS[1] - TARGET_ANCHORS[0])
    jacobian = angle_jacobian(point, TARGET_ANCHORS, allow_degenerate=True)
    assert np.isnan(jacobian[0]).all()
    assert np.isfinite(jacobian[1:]).all()

    row = _receiver_row(
        "reference_ray", 0, 2, point, t=1e-4, anchor_label="FY01_toward_FY11"
    )
    assert row["nonsmooth_jacobian_rows"] == "0"
    assert row["selected_rank"] == 2
    assert row["estimator_success"] is True


def test_circumcircle_is_rank_deficient_and_has_same_arc_observations() -> None:
    center = np.array([2.0, 2.0 * np.sqrt(3.0) / 3.0])
    radius = 4.0 / np.sqrt(3.0)
    points = [
        center + radius * np.array([np.cos(theta), np.sin(theta)])
        for theta in (0.0, 0.3)
    ]
    observations = [receiver_angles(point, TARGET_ANCHORS) for point in points]
    assert np.linalg.norm(observations[0] - observations[1]) < 1e-12
    jacobian = angle_jacobian(points[0], TARGET_ANCHORS, allow_degenerate=True)
    assert np.linalg.matrix_rank(jacobian, tol=1e-10) == 1

    default = _receiver_row(
        "reference_circumcircle", 0, 2, points[0], t=0.0,
        anchor_label="reference_circumcircle", circle_theta=0.0,
    )
    assert default["selected_rank"] == 1
    assert default["estimator_success"] is False


def test_worst_direction_is_the_right_singular_direction() -> None:
    matrices = {receiver_id: propagation_blocks(receiver_id)[2] for receiver_id in RECEIVER_IDS}
    receiver_id = max(RECEIVER_IDS, key=lambda identifier: np.linalg.norm(matrices[identifier], 2))
    direction = _right_worst_direction(matrices[receiver_id])
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert np.isclose(
        np.linalg.norm(matrices[receiver_id] @ direction),
        np.linalg.norm(matrices[receiver_id], 2),
        atol=1e-12,
    )


def test_counterexample_and_main_stage_outputs_retain_failures_and_residual() -> None:
    counterexamples = counterexample_rows()
    assert any(row["family"] == "bootstrap_mirror" for row in counterexamples)
    assert sum(not row["accepted"] for row in counterexamples) >= 1

    direction = _right_worst_direction(propagation_blocks(2)[2])
    rows = main_stage_rows(
        {
            "global_worst_receiver_id": 2,
            "global_worst_right_singular_vector": direction,
        }
    )
    assert len(rows) == 30
    assert {row["receiver_offset_label"] for row in rows} == {
        "Q", "plus_0.05_x", "plus_0.05_y", "plus_0.10_x", "plus_0.10_y"
    }
    assert all(row["status"] == "converged" for row in rows)
    row = next(row for row in rows if row["amplitude_d"] == 0.005 and row["sign"] == 1)
    assert np.isclose(row["final_apex_error_d"], 0.005, atol=1e-12)
    assert row["final_team_max_error_d"] > 0.005
