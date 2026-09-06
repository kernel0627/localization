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
    summary = {
        "contract": "Deterministic bounds for two base angles; canonical upper branch; interval lies in alpha>0, beta>0, alpha+beta<pi. No noise distribution assumed.",
        "angle_to_position": rows, "position_to_angle": budget_rows,
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
