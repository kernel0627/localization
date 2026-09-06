"""Local linearization of main's active feedback and retrospective cost audit.

This analysis does not change the controller. The smooth active map excludes
hold memory; the implemented finite-tolerance protocol is audited separately.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from scripts.q1_1.localization import angle_jacobian, fy_position, pairwise_angles
from scripts.q1_2.run_validation import table_positions, write_csv
from scripts.q1_3.local_adjustment import (
    LocalSettings,
    decide_local_adjustment,
    execute_relative_polar_step,
    public_schedule,
)
from scripts.q1_3.run_iterative_reference_baseline import simulate_adjustment

ROOT = Path(__file__).resolve().parents[2]
DIMENSION = 16


def nominal_positions():
    return np.array([fy_position(i) for i in range(10)])


def polar_basis(drone_id):
    theta = np.deg2rad(40 * (drone_id - 1))
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def state_slice(drone_id):
    return slice(2 * (drone_id - 2), 2 * (drone_id - 1))


def positions_from_error(error):
    """e_i=(delta r_i, R*delta theta_i), both in metres; FY00/01 fixed."""
    points = nominal_positions()
    for i in range(2, 10):
        radial, tangential = np.asarray(error)[state_slice(i)]
        theta = np.deg2rad(40 * (i - 1)) + tangential / 100
        points[i] = (100 + radial) * np.array([np.cos(theta), np.sin(theta)])
    return points


def error_from_positions(points):
    result = np.empty(DIMENSION)
    for i in range(2, 10):
        theta = np.arctan2(points[i, 1], points[i, 0])
        delta = (theta - np.deg2rad(40 * (i - 1)) + np.pi) % (2 * np.pi) - np.pi
        result[state_slice(i)] = [np.linalg.norm(points[i]) - 100, 100 * delta]
    return result


def observation_derivative(receiver_id, transmitter_ids):
    """Analytic derivative of actual observed angles with respect to all 16 e's."""
    q = nominal_positions()
    rows = []
    for a, b in combinations(transmitter_ids, 2):
        u, v = q[a] - q[receiver_id], q[b] - q[receiver_id]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        cosine = np.dot(u, v) / (nu * nv)
        sine = np.sqrt(1 - cosine**2)
        du = -(v / (nu * nv) - cosine * u / nu**2) / sine
        dv = -(u / (nu * nv) - cosine * v / nv**2) / sine
        row = np.zeros(DIMENSION)
        row[state_slice(receiver_id)] = -(du + dv) @ polar_basis(receiver_id)
        for drone_id, derivative in ((a, du), (b, dv)):
            if drone_id >= 2:
                row[state_slice(drone_id)] = derivative @ polar_basis(drone_id)
        rows.append(row)
    return np.array(rows)


def linear_model(gain=0.5):
    """Return 28 synchronous step matrices and selection/propagation evidence."""
    q = nominal_positions()
    matrices, choices, blocks = [], [], []
    for phase, tx in enumerate(public_schedule(), 1):
        matrix = np.eye(DIMENSION)
        for i in range(2, 10):
            if i in tx:
                continue
            candidates = []
            for pair in combinations(tx, 2):
                anchors = q[[0, *pair]]
                j = angle_jacobian(q[i], anchors) @ polar_basis(i)
                sigma = float(np.linalg.svd(j, compute_uv=False)[-1])
                candidates.append((sigma, pair, j))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            sigma, pair, j = candidates[0]
            propagation = np.linalg.pinv(j) @ observation_derivative(i, (0, *pair))
            matrix[state_slice(i)] -= gain * propagation
            choices.append(
                {
                    "phase": phase,
                    "receiver_id": i,
                    "transmitters": "-".join(map(str, (0, *tx))),
                    "selected_pair": "-".join(map(str, pair)),
                    "best_sigma": sigma,
                    "second_sigma": candidates[1][0],
                    "relative_selection_gap": (sigma - candidates[1][0]) / sigma,
                    "own_block_identity_error": float(
                        np.linalg.norm(propagation[:, state_slice(i)] - np.eye(2))
                    ),
                }
            )
            for a in pair:
                if a < 2:
                    continue
                block = propagation[:, state_slice(a)]
                blocks.append(
                    {
                        "phase": phase,
                        "receiver_id": i,
                        "reference_id": a,
                        "b_rr": block[0, 0],
                        "b_rt": block[0, 1],
                        "b_tr": block[1, 0],
                        "b_tt": block[1, 1],
                        "operator_norm": float(np.linalg.norm(block, 2)),
                    }
                )
        matrices.append(matrix)
    return matrices, choices, blocks


def cycle_matrix(matrices):
    product = np.eye(DIMENSION)
    for matrix in matrices:
        product = matrix @ product
    return product


def active_cycle(error, settings=LocalSettings()):
    """Run actual local fits, selection and limiting, without hold memory.

    Only this diagnostic removes the hold state, to test the smooth feedback
    core. It is never used as a replacement simulator or performance result.
    """
    points = positions_from_error(error)
    selected, clipped, failures = [], 0, 0
    for phase, tx in enumerate(public_schedule(), 1):
        before, after = points.copy(), points.copy()
        for i in range(2, 10):
            if i in tx:
                continue
            angles = pairwise_angles(before[i], before[[0, *tx]])
            decision = decide_local_adjustment(i, tx, angles, settings)
            if decision.selected is None:
                failures += 1
                continue
            selected.append((phase, i, decision.selected.pair))
            clipped += int(
                abs(settings.gain * decision.selected.radial_bias_m)
                > settings.max_radial_step_m
                or abs(settings.gain * decision.selected.angular_bias_rad)
                > settings.max_angular_step_rad
            )
            after[i] = execute_relative_polar_step(
                before[i], decision.radial_step_m, decision.angular_step_rad
            )
        points = after
    return error_from_positions(points), selected, clipped, failures


def validate_cycle_derivatives(output_dir):
    """Central differences through the original nonlinear solver, at three scales."""
    rows = []
    for gain in (0.25, 0.5, 1.0):
        matrices, choices, _ = linear_model(gain)
        analytic = cycle_matrix(matrices)
        expected = [
            (
                r["phase"],
                r["receiver_id"],
                tuple(map(int, r["selected_pair"].split("-"))),
            )
            for r in choices
        ]
        for h in (1e-3, 1e-4, 1e-5):
            numerical = np.empty_like(analytic)
            switches, clips, failures = 0, 0, 0
            for column in range(DIMENSION):
                direction = np.eye(DIMENSION)[column] * h
                values = []
                for sign in (1, -1):
                    value, selected, nclip, nfail = active_cycle(
                        sign * direction, LocalSettings(gain=gain)
                    )
                    values.append(value)
                    switches += sum(s != e for s, e in zip(selected, expected))
                    clips += nclip
                    failures += nfail
                numerical[:, column] = (values[0] - values[1]) / (2 * h)
            row = {
                "gain": gain,
                "difference_step_m": h,
                "absolute_frobenius_error": float(np.linalg.norm(numerical - analytic)),
                "relative_frobenius_error": float(
                    np.linalg.norm(numerical - analytic) / np.linalg.norm(analytic)
                ),
                "max_entry_error": float(np.max(np.abs(numerical - analytic))),
                "selection_changes": switches,
                "clipped_decisions": clips,
                "failed_fits": failures,
            }
            rows.append(row)
            print(json.dumps({"derivative_check": row}), flush=True)
    write_csv(output_dir / "cycle_difference_checks.csv", rows)
    return rows


def evaluate_table1(output_dir):
    """One-factor parameter checks with main's original control and stopping."""
    table = table_positions()
    initial = np.array([table[i] for i in range(10)])
    default = LocalSettings()
    cases = [
        (f"gain_{gain:g}", replace(default, gain=gain)) for gain in (0.25, 0.5, 1.0)
    ]
    cases += [
        (f"radial_limit_{limit:g}m", replace(default, max_radial_step_m=limit))
        for limit in (2.5, 10.0)
    ]
    cases += [
        (
            f"angular_limit_{limit:g}deg",
            replace(default, max_angular_step_rad=np.deg2rad(limit)),
        )
        for limit in (1.0, 4.0)
    ]
    rows, threshold_rows = [], []
    for name, settings in cases:
        run = simulate_adjustment(initial, settings=settings, retain_details=False)
        summary = run["summary"]
        case_dir = output_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        write_csv(case_dir / "slot_metrics.csv", run["history"])
        (case_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        row = {
            "case": name,
            "gain": settings.gain,
            "max_radial_step_m": settings.max_radial_step_m,
            "max_angular_step_deg": float(np.rad2deg(settings.max_angular_step_rad)),
            **{
                key: summary[key]
                for key in (
                    "status",
                    "measurement_slots",
                    "transmitter_uses",
                    "total_endpoint_displacement_m",
                    "last_motion_slot",
                    "post_motion_confirmation_slots",
                    "post_motion_transmitter_uses",
                    "failed_local_fits",
                )
            },
            "max_position_error_m": summary["final_metrics"]["max_position_error_m"],
            "rms_position_error_m": summary["final_metrics"]["rms_position_error_m"],
        }
        rows.append(row)
        threshold_rows.extend(
            {"case": name, **r} for r in summary["precision_thresholds"]
        )
        print(json.dumps({"table1_evaluation": row}), flush=True)
    write_csv(output_dir / "table1_evaluation.csv", rows)
    write_csv(output_dir / "precision_costs.csv", threshold_rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/q1_3/main_analysis"
    )
    parser.add_argument(
        "--linear-only",
        action="store_true",
        help="Only export analytic matrices and selection evidence",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gain_rows = []
    power_rows = []
    for gain in (0.25, 0.5, 1.0):
        matrices, choices, blocks = linear_model(gain)
        a = cycle_matrix(matrices)
        eigenvalues = np.linalg.eigvals(a)
        gain_rows.append(
            {
                "gain": gain,
                "spectral_radius": float(np.max(np.abs(eigenvalues))),
                "cycle_operator_norm": float(np.linalg.norm(a, 2)),
                "two_cycle_operator_norm": float(np.linalg.norm(a @ a, 2)),
                "max_slot_operator_norm": max(
                    float(np.linalg.norm(m, 2)) for m in matrices
                ),
                "min_relative_selection_gap": min(
                    r["relative_selection_gap"] for r in choices
                ),
            }
        )
        for cycles in range(1, 9):
            power_rows.append(
                {
                    "gain": gain,
                    "cycles": cycles,
                    "operator_norm": float(
                        np.linalg.norm(np.linalg.matrix_power(a, cycles), 2)
                    ),
                }
            )
        np.savetxt(
            args.output_dir / f"cycle_jacobian_gain_{gain:g}.csv", a, delimiter=","
        )
        write_csv(
            args.output_dir / f"eigenvalues_gain_{gain:g}.csv",
            [
                {
                    "real": float(z.real),
                    "imaginary": float(z.imag),
                    "modulus": float(abs(z)),
                }
                for z in eigenvalues
            ],
        )
    write_csv(args.output_dir / "gain_stability.csv", gain_rows)
    write_csv(args.output_dir / "nominal_selection.csv", choices)
    write_csv(args.output_dir / "reference_error_blocks.csv", blocks)
    write_csv(args.output_dir / "cycle_power_norms.csv", power_rows)
    print(json.dumps(gain_rows, indent=2), flush=True)
    metadata = {
        "state_order": [
            f"FY{i:02d}_{component}"
            for i in range(2, 10)
            for component in ("dr_m", "R_dtheta_m")
        ],
        "controller_sha256": hashlib.sha256(
            (ROOT / "scripts/q1_3/local_adjustment.py").read_bytes()
        ).hexdigest(),
        "matrix_scope": "Ideal smooth local solution, unique nominal pair selection, "
        "no active clipping, continuous correction without hold memory. "
        "Matrices are floating-point evaluations of analytic derivatives.",
        "hold_scope": "The real protocol has receiver hold flags/counters. With all "
        "receivers held strictly inside thresholds, the position map is "
        "the identity. The active Jacobian does not prove convergence "
        "to zero for this finite-tolerance protocol.",
        "gain_stability": gain_rows,
    }
    if not args.linear_only:
        metadata["difference_checks"] = validate_cycle_derivatives(args.output_dir)
        metadata["table1_evaluation"] = evaluate_table1(args.output_dir)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
