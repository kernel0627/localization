from __future__ import annotations

import numpy as np
import pytest

from scripts.q1_1.localization import fy_position, pairwise_angles, polar_to_cartesian
from scripts.q1_2.identification import identify_anonymous_emitter
from scripts.q1_2.run_validation import table_positions


def observations(receiver, emitter_id):
    return pairwise_angles(receiver, [fy_position(i) for i in (0, 1, emitter_id)])


def test_all_56_table_pairs_recover_identity_and_position():
    for k, truth in table_positions().items():
        if k < 2:
            continue
        for q in range(2, 10):
            if q == k:
                continue
            result = identify_anonymous_emitter(k, observations(truth, q))
            assert result.status == "selected", (k, q)
            assert result.selected.emitter_id == q, (k, q)
            assert result.selected.position == pytest.approx(truth, abs=1e-6)
            assert {h.emitter_id for h in result.hypotheses} == set(range(2, 10)) - {k}
            for h in result.hypotheses:
                for candidate in h.candidates:
                    assert observations(
                        candidate.position, h.emitter_id
                    ) == pytest.approx(
                        observations(truth, q),
                        abs=1e-8,
                    )


def test_circle_completion_recovers_when_local_optimizer_is_stopped():
    truth = polar_to_cartesian(112, 80.21)
    result = identify_anonymous_emitter(3, observations(truth, 5), max_nfev=1)
    assert result.selected.emitter_id == 5
    assert result.selected.source == "circle_intersection"
    assert result.selected.position == pytest.approx(truth, abs=1e-6)
    assert any(h.local_status == "not_converged" for h in result.hypotheses)


def test_nearly_tied_candidates_are_reported_without_arbitrary_id_choice():
    truth = polar_to_cartesian(112, 80.21)
    result = identify_anonymous_emitter(
        3,
        observations(truth, 5),
        distance_tie_tolerance=1000,
    )
    assert result.status == "ambiguous"
    assert result.selected is None


def test_inconsistent_three_angle_data_are_rejected():
    # Three planar rays cannot have all three pairwise angles equal to 45 deg.
    result = identify_anonymous_emitter(3, np.deg2rad([45, 45, 45]))
    assert result.status == "no_consistent_candidate"
    assert result.selected is None


def test_danger_circle_with_continuous_solutions_is_not_claimed_unique():
    anchors = np.array([fy_position(i) for i in (0, 1, 2)])
    center = np.linalg.solve(2 * anchors[1:], np.sum(anchors[1:] ** 2, axis=1))
    ideal = fy_position(3)
    truth = center + np.linalg.norm(center) * (ideal - center) / np.linalg.norm(
        ideal - center
    )
    result = identify_anonymous_emitter(3, observations(truth, 2))
    assert result.status == "degenerate"
    assert result.selected is None
    assert result.continuum_distance_bound == pytest.approx(33.45308, abs=1e-4)
    assert any(h.continuum_distance_bounds for h in result.hypotheses)


def test_radius_scaling_changes_positions_but_not_identity():
    truth = polar_to_cartesian(112, 80.21)
    observed = observations(truth, 5)
    result = identify_anonymous_emitter(3, observed, radius=250)
    assert result.selected.emitter_id == 5
    assert result.selected.position == pytest.approx(2.5 * truth, abs=1e-6)


@pytest.mark.parametrize(
    "receiver_id,angles,kwargs",
    [
        (1, [1, 1, 2], {}),
        (3.5, [1, 1, 2], {}),
        (3, [1, 2], {}),
        (3, [1, float("nan"), 2], {}),
        (3, [0, 1, 1], {}),
        (3, [1, 1, np.pi], {}),
        (3, [1, 1, 2], {"radius": 0}),
        (3, [1, 1, 2], {"residual_tolerance": -1}),
        (3, [1, 1, 2], {"distance_tie_tolerance": -1}),
        (3, [1, 1, 2], {"max_nfev": 0}),
    ],
)
def test_invalid_inputs_fail_explicitly(receiver_id, angles, kwargs):
    with pytest.raises(ValueError):
        identify_anonymous_emitter(receiver_id, angles, **kwargs)
