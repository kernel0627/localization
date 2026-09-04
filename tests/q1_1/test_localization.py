from __future__ import annotations

import numpy as np
import pytest

from scripts.q1_1.localization import (
    GeometryError,
    cosine_jacobian,
    cosine_residuals,
    find_local_candidates,
    fy_position,
    localize_receiver,
    pairwise_angles,
    polar_to_cartesian,
)


def test_chat_example_angles_and_exact_recovery() -> None:
    anchors = np.vstack([fy_position(0), fy_position(1), fy_position(5)])
    true_position = polar_to_cartesian(112.0, 80.21)
    observed = pairwise_angles(true_position, anchors)

    assert np.rad2deg(observed) == pytest.approx(
        [46.050, 46.231, 92.282], abs=0.001
    )

    result = localize_receiver(anchors, observed, fy_position(3))

    assert result.success
    assert np.linalg.norm(result.position - true_position) < 1e-7
    assert result.residual_norm < 1e-10
    assert result.jacobian_singular_values[-1] > 0.0


def test_analytic_jacobian_matches_central_difference() -> None:
    anchors = np.vstack([fy_position(0), fy_position(1), fy_position(4)])
    receiver = polar_to_cartesian(105.0, 119.75)
    observed = pairwise_angles(receiver, anchors)
    analytic = cosine_jacobian(receiver, anchors, observed)

    step = 1e-5
    numeric = np.empty_like(analytic)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        numeric[:, axis] = (
            cosine_residuals(receiver + offset, anchors, observed)
            - cosine_residuals(receiver - offset, anchors, observed)
        ) / (2.0 * step)

    assert analytic == pytest.approx(numeric, abs=2e-10)


def test_small_angle_noise_keeps_local_solution_near_truth() -> None:
    anchors = np.vstack([fy_position(0), fy_position(1), fy_position(5)])
    true_position = polar_to_cartesian(112.0, 80.21)
    observed = pairwise_angles(true_position, anchors)
    noisy = np.clip(
        observed + np.deg2rad(np.array([0.10, -0.08, 0.05])),
        0.0,
        np.pi,
    )

    result = localize_receiver(anchors, noisy, fy_position(3))

    assert result.success
    assert np.linalg.norm(result.position - true_position) < 1.0
    assert result.residual_norm < 0.01


def test_unoriented_angles_can_have_two_global_candidates() -> None:
    anchors = np.vstack([fy_position(0), fy_position(1), fy_position(5)])
    true_position = polar_to_cartesian(112.0, 80.21)
    observed = pairwise_angles(true_position, anchors)
    starts = np.vstack(
        [fy_position(drone_id) for drone_id in (2, 3, 4, 6, 7, 8, 9)]
    )

    candidates = find_local_candidates(anchors, observed, starts)

    assert len(candidates) == 2
    assert min(
        np.linalg.norm(candidate.position - true_position) for candidate in candidates
    ) < 1e-7
    assert any(
        np.linalg.norm(candidate.position - np.array([-13.6159, -76.0614])) < 1e-3
        for candidate in candidates
    )
    assert all(candidate.residual_norm < 1e-10 for candidate in candidates)


def test_receiver_cannot_coincide_with_transmitter() -> None:
    anchors = np.vstack([fy_position(0), fy_position(1), fy_position(5)])

    with pytest.raises(GeometryError):
        pairwise_angles(fy_position(0), anchors)
