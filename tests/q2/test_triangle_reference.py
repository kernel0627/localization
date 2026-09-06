import numpy as np
import pytest

from scripts.q2.triangle_reference import (
    RECEIVER_IDS,
    TARGET_ANCHORS,
    angle_jacobian,
    bootstrap_angles,
    bootstrap_from_angles,
    estimate_receiver,
    receiver_angles,
    selected_angle_pairs,
    selected_angle_sigma_min,
    template,
)


def test_canonical_template_and_bootstrap_roundtrip() -> None:
    points = template()
    assert points.shape == (15, 2)
    np.testing.assert_allclose(points[[0, 10, 14]], TARGET_ANCHORS)
    observed = bootstrap_angles(points)
    np.testing.assert_allclose(bootstrap_from_angles(observed), points[0], atol=1e-12)


def test_bootstrap_angles_reject_invalid_triangle_angles() -> None:
    with pytest.raises(ValueError, match="nondegenerate triangle"):
        bootstrap_from_angles((np.deg2rad(100.0), np.deg2rad(100.0)))


def test_all_receivers_have_selected_rank_two_smooth_pair() -> None:
    points = template()
    for receiver_id in RECEIVER_IDS:
        result = estimate_receiver(
            receiver_id,
            receiver_angles(points[receiver_id - 1], TARGET_ANCHORS),
        )
        assert result["success"] is True
        np.testing.assert_allclose(result["position"], points[receiver_id - 1], atol=1e-10)
        assert len(selected_angle_pairs(receiver_id)) == 2
        assert selected_angle_sigma_min(receiver_id) > 0.3


def test_perturbed_receiver_uses_full_triangle_residual_check() -> None:
    points = template()
    receiver_id = 12
    actual = points[receiver_id - 1] + np.array([0.03, -0.02])
    result = estimate_receiver(receiver_id, receiver_angles(actual, TARGET_ANCHORS))
    assert result["success"] is True
    np.testing.assert_allclose(result["position"], actual, atol=1e-8)
    assert result["max_angle_residual"] <= 1e-8


def test_full_triangle_residual_rejects_inconsistent_unselected_angle() -> None:
    points = template()
    receiver_id = 12
    observed = receiver_angles(points[receiver_id - 1], TARGET_ANCHORS)
    inconsistent = observed.copy()
    inconsistent[2] = np.pi - 1e-3
    result = estimate_receiver(receiver_id, inconsistent)
    assert result["success"] is False
    assert result["max_angle_residual"] > 1e-4


def test_regular_angle_jacobian_matches_central_difference() -> None:
    points = template()
    receiver = points[7] + np.array([0.07, -0.04])
    analytic = angle_jacobian(receiver, TARGET_ANCHORS)
    step = 1e-5
    numeric = np.empty_like(analytic)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        numeric[:, axis] = (
            receiver_angles(receiver + offset, TARGET_ANCHORS)
            - receiver_angles(receiver - offset, TARGET_ANCHORS)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, atol=2e-9)


def test_geometry_failure_is_reported_without_truth_fallback() -> None:
    initial = np.array([0.0, 0.0])
    result = estimate_receiver(12, np.array([0.1, 0.2, np.pi]), initial=initial)
    assert result["success"] is False
    np.testing.assert_allclose(result["position"], initial)
    assert result["nfev"] == 0
    assert "geometry/solver failure" in result["message"]
