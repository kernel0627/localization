"""Enumerate disjoint two-slot designs using only the nominal formation."""

from itertools import combinations

import numpy as np

from scripts.q1_3.joint_localization import (
    ideal_formation,
    jacobian_diagnostics,
    joint_jacobian,
    observation_layout,
)


def enumerate_designs(radius: float = 100.0) -> tuple[list[dict], list[dict]]:
    nominal = ideal_formation(radius)
    pairs = tuple(combinations(range(2, 10), 2))
    jacobians = {
        pair: joint_jacobian(nominal, observation_layout((pair,))) for pair in pairs
    }
    singles = [
        {"pair": list(pair), **jacobian_diagnostics(jacobians[pair])} for pair in pairs
    ]
    designs = []
    for a, b in combinations(pairs, 2):
        if set(a).isdisjoint(b):
            designs.append(
                {
                    "a": list(a),
                    "b": list(b),
                    **jacobian_diagnostics(np.vstack((jacobians[a], jacobians[b]))),
                }
            )
    return singles, designs


def select_design(designs: list[dict]) -> dict:
    feasible = [row for row in designs if row["rank"] == 16]
    if not feasible:
        raise ValueError("no locally identifiable design")
    best_score = max(row["sigma_min_rad_per_m"] for row in feasible)
    # Resolve symmetry-related floating-point ties deterministically by ID.
    tied = [
        row
        for row in feasible
        if np.isclose(row["sigma_min_rad_per_m"], best_score, rtol=1e-10, atol=0)
    ]
    best = min(tied, key=lambda row: (*row["a"], *row["b"]))
    return {**best, "tie_count": len(tied), "tie_relative_tolerance": 1e-10}
