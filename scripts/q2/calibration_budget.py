"""Exact upper-branch apex bounds from two bounded base-angle intervals.

Four intersecting boundary rays form a bounded convex quadrilateral when
both lower angles are positive and the two upper angles sum to less than pi.
The maximum distance to a fixed target occurs at one of its vertices.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import norm

from scripts.q2.triangle_reference import TARGET_ANCHORS, bootstrap_from_angles
from scripts.q2.run_triangle_reference import _write_csv


def apex_angle_box(
    observed: ArrayLike, half_width: ArrayLike | float,
    target: ArrayLike = TARGET_ANCHORS[0],
) -> dict:
    """Enclose every feasible apex under deterministic angle bounds.

    ``half_width`` bounds errors in radians. A standard deviation alone is
    not a deterministic bound. The known upper-side branch is required.
    """
    center = np.asarray(observed, dtype=float)
    widths = np.broadcast_to(np.asarray(half_width, dtype=float), (2,)).copy()
    goal = np.asarray(target, dtype=float)
    if center.shape != (2,) or goal.shape != (2,):
        raise ValueError("observed and target must each have shape (2,)")
    if not np.isfinite(center).all() or not np.isfinite(widths).all() or not np.isfinite(goal).all():
        raise ValueError("angle box and target must be finite")
    if np.any(widths < 0):
        raise ValueError("angle half widths must be nonnegative")
    lower, upper = center-widths, center+widths
    if np.any(lower <= 0) or float(upper.sum()) >= np.pi:
        raise ValueError("angle box must lie strictly in the bounded upper-triangle domain")
    vertices = np.array([
        bootstrap_from_angles([alpha, beta])
        for alpha, beta in product((lower[0], upper[0]), (lower[1], upper[1]))
    ])
    distances = np.linalg.norm(vertices-goal, axis=1)
    return {
        "lower_angles_rad": lower.tolist(), "upper_angles_rad": upper.tolist(),
        "vertices": vertices.tolist(),
        "maximum_position_error_d": float(distances.max()),
        "maximizing_vertex": int(distances.argmax()),
    }


def equal_angle_budget(position_budget: float) -> float:
    """Largest equal base-angle target residual fitting an apex budget."""
    if not np.isfinite(position_budget) or position_budget <= 0:
        raise ValueError("position budget must be finite and positive")
    target_angles = np.full(2, np.pi/3)
    return float(brentq(
        lambda epsilon: apex_angle_box(target_angles, epsilon)["maximum_position_error_d"]-position_budget,
        0., np.pi/6-1e-6, xtol=1e-15,
    ))


def gaussian_angle_half_width(
    sigma_rad: float, samples_per_angle: int, *, family_error: float = 0.01,
    planned_checks: int = 1,
) -> float:
    """Simultaneous two-angle mean interval with a declared error budget.

    Bonferroni splits the family error across two angles and a predeclared
    finite number of checks. Known Gaussian marginal variances and independent
    repeated samples within each angle are assumed. Across-angle/check
    independence is not needed for the union bound.
    """
    if not np.isfinite(sigma_rad) or sigma_rad <= 0:
        raise ValueError("sigma_rad must be finite and positive")
    if not isinstance(samples_per_angle, (int, np.integer)) or samples_per_angle <= 0:
        raise ValueError("samples_per_angle must be a positive integer")
    if not isinstance(planned_checks, (int, np.integer)) or planned_checks <= 0:
        raise ValueError("planned_checks must be a positive integer")
    if not 0 < family_error < 1:
        raise ValueError("family_error must lie in (0,1)")
    z = float(norm.ppf(1-family_error/(4*planned_checks)))
    return z*sigma_rad/np.sqrt(samples_per_angle)


def gaussian_minimum_samples(
    position_budget: float, sigma_rad: float, *, planned_checks: int = 1,
    family_error: float = 0.01,
) -> int:
    """Best-centered K: interval is centered exactly at target base angles.

    Actual measured means away from target need the full apex_angle_box test
    and may require more samples or another adjustment.
    """
    epsilon = equal_angle_budget(position_budget)
    one_sample_width = gaussian_angle_half_width(
        sigma_rad, 1, family_error=family_error, planned_checks=planned_checks,
    )
    return max(1, int(np.ceil((one_sample_width/epsilon)**2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/q2/calibration_budget"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for degree in (0.01, 0.05, 0.1):
        epsilon = float(np.deg2rad(degree))
        result = apex_angle_box(np.full(2, np.pi/3), epsilon)
        rows.append({"each_angle_bound_deg": degree,
                     "linear_apex_bound_d": 8*epsilon,
                     "exact_apex_bound_d": result["maximum_position_error_d"]})
    budget_rows = []
    for budget in (0.002, 0.0048, 0.005, 0.01):
        epsilon = equal_angle_budget(budget)
        budget_rows.append({"apex_position_budget_d": budget,
                            "equal_angle_bound_rad": epsilon,
                            "equal_angle_bound_deg": float(np.rad2deg(epsilon)),
                            "verified_exact_bound_d": apex_angle_box(np.full(2, np.pi/3), epsilon)["maximum_position_error_d"]})
    _write_csv(args.output_dir/"angle_to_position.csv", rows)
    _write_csv(args.output_dir/"position_to_angle.csv", budget_rows)
    statistical_rows = []
    for sigma_deg in (0.01, 0.05, 0.1):
        for budget in (0.0048, 0.005):
            for checks in (1, 30):
                count = gaussian_minimum_samples(budget, np.deg2rad(sigma_deg), planned_checks=checks)
                half_width = gaussian_angle_half_width(np.deg2rad(sigma_deg), count, planned_checks=checks)
                bound = apex_angle_box(np.full(2, np.pi/3), half_width)["maximum_position_error_d"]
                statistical_rows.append({
                    "sigma_each_angle_deg": sigma_deg, "apex_budget_d": budget,
                    "planned_checks": checks, "family_error_probability": 0.01,
                    "samples_per_angle_best_centered": count,
                    "one_check_measurement_slots": 2*count,
                    "one_check_tx_uses": 4*count,
                    "angle_mean_interval_half_width_deg": np.rad2deg(half_width),
                    "best_centered_exact_apex_radius_d": bound,
                })
    _write_csv(args.output_dir/"gaussian_confidence_budget.csv", statistical_rows)
    summary = {
        "deterministic_contract": "Bounded intervals for two base angles; canonical upper branch; alpha>0, beta>0, alpha+beta<pi. Geometry alone assumes no noise distribution.",
        "gaussian_contract": "Known Gaussian single-sample variance; fixed independent samples per angle within each stationary batch; predeclared maximum checks; Bonferroni family error control. Sample requirements assume means centered exactly at target, not guaranteed acceptance.",
        "angle_to_position": rows, "position_to_angle": budget_rows,
        "gaussian_confidence_budget": statistical_rows,
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
