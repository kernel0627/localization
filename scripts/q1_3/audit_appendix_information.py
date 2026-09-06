"""Independent information audit for local alternating bootstrap certificates.

Joint readings below are offline mathematical diagnostics, never controller
inputs. The counterexample uses the existing geometric ambiguity construction.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from scripts.q1_1.localization import angle_jacobian, pairwise_angles
from scripts.q1_2.run_validation import write_csv
from scripts.q1_3.analyze_iterative_reference import (
    nominal_positions,
    linear_model,
    observation_derivative,
    polar_basis,
    positions_from_error,
    state_slice,
)
from scripts.q1_3.audit_information import alternative_for_receiver
from scripts.q1_3.local_adjustment import estimate_bias, public_schedule

ROOT = Path(__file__).resolve().parents[2]


def columns(pair):
    return [j for i in pair for j in range(state_slice(i).start, state_slice(i).stop)]


def stacked_angles(points, pair):
    return np.concatenate(
        [pairwise_angles(points[x], points[[0, 1, y]]) for x, y in (pair, pair[::-1])]
    )


def joint_derivative(pair):
    return np.vstack(
        [
            observation_derivative(x, (0, 1, y))[:, columns(pair)]
            for x, y in (pair, pair[::-1])
        ]
    )


def audit(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    q = nominal_positions()
    geometry, checks = [], []
    for pair in combinations(range(2, 10), 2):
        h = joint_derivative(pair)
        singular = np.linalg.svd(h, compute_uv=False)
        local_ranks = [
            int(np.linalg.matrix_rank(h[k : k + 3], tol=1e-10)) for k in (0, 3)
        ]
        nuisance_ranks = [
            int(np.linalg.matrix_rank(h[:3, 2:], tol=1e-10)),
            int(np.linalg.matrix_rank(h[3:, :2], tol=1e-10)),
        ]
        estimates = []
        matrices = []
        for offset, (x, y) in enumerate((pair, pair[::-1])):
            j = angle_jacobian(q[x], q[[0, 1, y]]) @ polar_basis(x)
            k = np.linalg.pinv(j) @ h[3 * offset : 3 * offset + 3]
            estimates.append(k)
            matrix = np.eye(4)
            matrix[2 * offset : 2 * offset + 2] -= 0.5 * k
            matrices.append(matrix)
        fit_map = np.vstack(estimates)
        cycle = matrices[1] @ matrices[0]
        geometry.append(
            {
                "x": pair[0],
                "y": pair[1],
                "joint_rank": int(np.linalg.matrix_rank(h, tol=1e-10)),
                "joint_sigma_min_rad_per_m": float(singular[-1]),
                "joint_condition_number": float(singular[0] / singular[-1]),
                "x_own_rank": local_ranks[0],
                "y_own_rank": local_ranks[1],
                "x_nuisance_rank": nuisance_ranks[0],
                "y_nuisance_rank": nuisance_ranks[1],
                "joint_fitted_bias_map_sigma_min": float(
                    np.linalg.svd(fit_map, compute_uv=False)[-1]
                ),
                "alternating_cycle_rho_gain_half": float(
                    max(abs(np.linalg.eigvals(cycle)))
                ),
            }
        )
        # Direct observed-angle differences independently check reference terms.
        step = 1e-3
        numerical = np.empty((6, 4))
        for k, col in enumerate(columns(pair)):
            e = np.zeros(16)
            e[col] = step
            numerical[:, k] = (
                stacked_angles(positions_from_error(e), pair)
                - stacked_angles(positions_from_error(-e), pair)
            ) / (2 * step)
        checks.append(
            {
                "x": pair[0],
                "y": pair[1],
                "step_m": step,
                "max_entry_error": float(np.max(np.abs(numerical - h))),
            }
        )

    examples = []
    witnesses = []
    pair = (2, 5)
    for receiver, reference in (pair, pair[::-1]):
        for probe_rotation in (1e-4, 1e-3):
            moved = alternative_for_receiver(q, receiver, probe_rotation)
            witness = q.copy()
            witness[list(pair)] = moved[list(pair)]
            own = pairwise_angles(witness[receiver], witness[[0, 1, reference]])
            nominal = pairwise_angles(q[receiver], q[[0, 1, reference]])
            _, state, residual, success, _, _ = estimate_bias(
                receiver, (1, reference), own
            )
            other_diff = pairwise_angles(
                witness[reference], witness[[0, 1, receiver]]
            ) - pairwise_angles(q[reference], q[[0, 1, receiver]])
            case_id = f"receiver_{receiver:02d}_rotation_{probe_rotation:g}"
            examples.append(
                {
                    "case_id": case_id,
                    "receiver_id": receiver,
                    "reference_id": reference,
                    "construction_parameter_rad": probe_rotation,
                    "own_true_error_m": float(
                        np.linalg.norm(witness[receiver] - q[receiver])
                    ),
                    "reference_true_error_m": float(
                        np.linalg.norm(witness[reference] - q[reference])
                    ),
                    "max_own_angle_difference_rad": float(max(abs(own - nominal))),
                    "other_max_angle_difference_rad": float(max(abs(other_diff))),
                    "estimated_radial_m": float(state[0]),
                    "estimated_tangential_m": float(state[1]),
                    "fit_residual_rad": residual,
                    "fit_success": success,
                    "fixed_reference_max_difference_m": float(
                        np.max(np.abs(witness[:2] - q[:2]))
                    ),
                }
            )
            witnesses.append({"case_id": case_id, "positions": witness.tolist()})

    write_csv(output_dir / "joint_and_local_ranks.csv", geometry)
    write_csv(output_dir / "observation_derivative_checks.csv", checks)
    write_csv(output_dir / "local_zero_estimate_counterexamples.csv", examples)
    (output_dir / "counterexample_positions.json").write_text(
        json.dumps(witnesses, indent=2) + "\n"
    )
    assert len(geometry) == 28 and all(r["joint_rank"] == 4 for r in geometry)
    assert all(r["x_own_rank"] == r["y_own_rank"] == 2 for r in geometry)
    assert all(r["x_nuisance_rank"] == r["y_nuisance_rank"] == 1 for r in geometry)
    assert max(r["max_entry_error"] for r in checks) < 1e-9
    assert all(
        r["max_own_angle_difference_rad"] < 1e-12
        and r["own_true_error_m"] > 0.001
        and r["fixed_reference_max_difference_m"] == 0
        and r["fit_success"]
        and abs(r["estimated_radial_m"]) + abs(r["estimated_tangential_m"]) < 1e-9
        for r in examples
    )
    source_paths = [
        Path(__file__),
        ROOT / "scripts/q1_3/audit_information.py",
        ROOT / "scripts/q1_3/analyze_iterative_reference.py",
        ROOT / "scripts/q1_3/local_adjustment.py",
        ROOT / "scripts/q1_1/localization.py",
    ]
    summary = {
        "purpose": "Offline information audit, not a new localization/controller input",
        "pair_count": len(geometry),
        "joint_rank_4_pairs": sum(r["joint_rank"] == 4 for r in geometry),
        "min_joint_sigma_min": min(r["joint_sigma_min_rad_per_m"] for r in geometry),
        "max_joint_sigma_min": max(r["joint_sigma_min_rad_per_m"] for r in geometry),
        "max_derivative_entry_error": max(r["max_entry_error"] for r in checks),
        "counterexamples": len(examples),
        "max_own_angle_difference_rad": max(
            r["max_own_angle_difference_rad"] for r in examples
        ),
        "scope": "A single receiver's instantaneous local fit/residual cannot certify both anchors. This does not exclude time feedback, preplanned protocols, or future angle-only certificates under additional explicit conditions.",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    # The optimized schedule has a four-aircraft feedback core; the remaining
    # four receive every slot and never enter anyone else's reference set.
    matrices, _, _ = linear_model()
    schedule = public_schedule()
    cycle = matrices[schedule.index((1, 7, 8))] @ matrices[schedule.index((1, 4, 5))]
    core_ids, follower_ids = (4, 5, 7, 8), (2, 3, 6, 9)
    core, followers = columns(core_ids), columns(follower_ids)
    acc = cycle[np.ix_(core, core)]
    acf = cycle[np.ix_(core, followers)]
    aff = cycle[np.ix_(followers, followers)]
    structure = {
        "core_ids": core_ids,
        "follower_ids": follower_ids,
        "follower_to_core_block_max_abs": float(np.max(np.abs(acf))),
        "follower_self_block_error_vs_quarter_identity": float(
            np.max(np.abs(aff - 0.25 * np.eye(8)))
        ),
        "core_cycle_rho": float(max(abs(np.linalg.eigvals(acc)))),
        "core_two_cycle_norm_2": float(np.linalg.norm(acc @ acc, 2)),
        "scope": "Active smooth local map at nominal geometry. Core physical observations and local controller state do not depend on followers; global logging termination remains separate. Near-repeated eigenvalue splitting must not be interpreted as precision of rho.",
    }
    assert structure["follower_to_core_block_max_abs"] < 1e-12
    assert structure["follower_self_block_error_vs_quarter_identity"] < 1e-12
    (output_dir / "schedule_structure.json").write_text(
        json.dumps(structure, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    audit(ROOT / "outputs/q1_3/appendix_information_audit")
