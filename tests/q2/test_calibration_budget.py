import numpy as np
import pytest

from scripts.q2.calibration_budget import apex_angle_box, equal_angle_budget
from scripts.q2.triangle_reference import TARGET_ANCHORS, bootstrap_from_angles


@pytest.mark.parametrize("center,widths", [
    ([np.pi/3, np.pi/3], [0.001, 0.001]),
    ([0.7, 1.3], [0.1, 0.02]),
    ([0.9, 1.0], [0., 0.05]),
])
def test_angle_box_encloses_interior_and_attains_vertex_bound(center, widths) -> None:
    result = apex_angle_box(center, widths)
    bound = result["maximum_position_error_d"]
    for a in np.linspace(center[0]-widths[0], center[0]+widths[0], 11):
        for b in np.linspace(center[1]-widths[1], center[1]+widths[1], 11):
            assert np.linalg.norm(bootstrap_from_angles([a, b])-TARGET_ANCHORS[0]) <= bound+1e-12
    vertices = np.array(result["vertices"])
    assert np.max(np.linalg.norm(vertices-TARGET_ANCHORS[0], axis=1)) == bound


def test_finite_angle_bound_includes_second_order_growth() -> None:
    epsilon = np.deg2rad(0.1)
    bound = apex_angle_box(np.full(2, np.pi/3), epsilon)["maximum_position_error_d"]
    assert bound > 8*epsilon
    np.testing.assert_allclose(bound, 2*np.tan(np.pi/3+epsilon)-2*np.sqrt(3), atol=1e-13)


def test_inverse_budget_and_monotonicity() -> None:
    epsilons = [equal_angle_budget(tau) for tau in (0.002, 0.0048, 0.005, 0.01)]
    assert np.all(np.diff(epsilons) > 0)
    for tau, epsilon in zip((0.002, 0.0048, 0.005, 0.01), epsilons):
        np.testing.assert_allclose(apex_angle_box(np.full(2, np.pi/3), epsilon)["maximum_position_error_d"], tau, atol=1e-13)


@pytest.mark.parametrize("center,width", [([0.1, 0.1], 0.1), ([1.5, 1.5], 0.2), ([1., 1.], -0.1)])
def test_unsupported_angle_domains_are_rejected(center, width) -> None:
    with pytest.raises(ValueError):
        apex_angle_box(center, width)
