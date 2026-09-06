"""Bounded, noiseless audit of a frozen FY01 calibration residual.

The production estimator and its rejection/stopping rules remain unchanged.
Truth is used only to generate observations and score the offline probes.
Run with ``conda run -n agent python -m scripts.q2.analyze_reference_residual``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.q2.triangle_reference import (
    ANCHOR_PAIRS,
    RECEIVER_IDS,
    SELECTED_ANGLE_INDICES,
    TARGET_ANCHORS,
    TARGET_TEMPLATE,
    angle_jacobian,
    bootstrap_angles,
    bootstrap_from_angles,
    estimate_receiver,
    receiver_angles,
)
from scripts.q2.run_triangle_reference import (
    _json_value,
    _main_record,
    _write_csv,
    make_initial_state,
    run_case,
)


def apex_angle_jacobian(position: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Derivative of the three unsigned angles with respect to anchor 0.

    Nonsmooth rows return NaNs, even when the apex is absent from that row.
    Only the production estimator's smooth selected rows enter the analysis.
    """
    result = np.zeros((3, 2))
    for row, (first, second) in enumerate(ANCHOR_PAIRS):
        u, v = anchors[first] - position, anchors[second] - position
        u2, v2 = u @ u, v @ v
        if min(u2, v2) <= 1e-24:
            raise ValueError("receiver coincides with anchor")
        cross = u[0] * v[1] - u[1] * v[0]
        if abs(cross) / np.sqrt(u2 * v2) <= 1e-10:
            result[row] = np.nan
        elif first == 0:
            result[row] = -np.sign(cross) * np.array([-u[1], u[0]]) / u2
        elif second == 0:
            result[row] = np.sign(cross) * np.array([-v[1], v[0]]) / v2
    return result


def bootstrap_jacobian(apex: np.ndarray) -> np.ndarray:
    """Two base-angle derivatives on the canonical upper half-plane."""
    x, y = apex
    if y <= 0:
        raise ValueError("bootstrap derivative requires upper branch")
    return np.array([
        [-y, x] / np.array(x*x + y*y),
        [y, 4-x] / np.array((4-x)**2 + y*y),
    ])


def propagation_blocks(receiver_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = TARGET_TEMPLATE[receiver_id - 1]
    selected = list(SELECTED_ANGLE_INDICES[receiver_id])
    j = angle_jacobian(point, TARGET_ANCHORS, allow_degenerate=True)[selected]
    b = apex_angle_jacobian(point, TARGET_ANCHORS)[selected]
    return j, b, np.linalg.solve(j, b)


def central_difference(function, point: np.ndarray, step: float) -> np.ndarray:
    return np.column_stack([
        (function(point + offset) - function(point - offset)) / (2*step)
        for offset in np.eye(2)*step
    ])


def linear_audit() -> tuple[list[dict], dict]:
    jc = bootstrap_jacobian(TARGET_ANCHORS[0])
    inv_jc = np.linalg.inv(jc)
    corners = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    rows = []
    max_j_error = max_b_error = max_g_error = 0.0
    for identifier in RECEIVER_IDS:
        point = TARGET_TEMPLATE[identifier-1]
        selected = list(SELECTED_ANGLE_INDICES[identifier])
        j, b, g = propagation_blocks(identifier)

        def angles_at_apex(apex):
            anchors = TARGET_ANCHORS.copy()
            anchors[0] = apex
            return receiver_angles(point, anchors)

        def estimated_at_apex(apex):
            # Retain the returned two-row candidate even when the complete
            # triangle check rejects it; acceptance is audited separately.
            return estimate_receiver(identifier, angles_at_apex(apex))["position"]

        for step in (1e-4, 1e-5):
            nj = central_difference(
                lambda p: receiver_angles(p, TARGET_ANCHORS)[selected], point, step
            )
            nb = central_difference(
                lambda c: angles_at_apex(c)[selected], TARGET_ANCHORS[0], step
            )
            ng = central_difference(estimated_at_apex, TARGET_ANCHORS[0], step)
            max_j_error = max(max_j_error, float(np.max(np.abs(j-nj))))
            max_b_error = max(max_b_error, float(np.max(np.abs(b-nb))))
            max_g_error = max(max_g_error, float(np.max(np.abs(g-ng))))
        rows.append({
            "receiver_id": identifier,
            "selected_indices": ";".join(map(str, selected)),
            "sigma_min_rad_per_d": np.linalg.svd(j, compute_uv=False)[-1],
            "g_00": g[0, 0], "g_01": g[0, 1],
            "g_10": g[1, 0], "g_11": g[1, 1],
            "g_spectral_norm": np.linalg.norm(g, 2),
            "angle_box_to_position_d_per_rad": max(
                np.linalg.norm(g @ inv_jc @ s) for s in corners
            ),
        })
    def base_angles(apex):
        points = TARGET_TEMPLATE.copy()
        points[0] = apex
        return bootstrap_angles(points)
    jc_error = max(float(np.max(np.abs(jc-central_difference(
        base_angles, TARGET_ANCHORS[0], h
    )))) for h in (1e-4, 1e-5))
    return rows, {
        "bootstrap_jacobian": jc,
        "bootstrap_sigma_min": np.linalg.svd(jc, compute_uv=False)[-1],
        "bootstrap_angle_box_factor": max(np.linalg.norm(inv_jc @ s) for s in corners),
        "g_max": max(row["g_spectral_norm"] for row in rows),
        "receiver_angle_box_factor": max(row["angle_box_to_position_d_per_rad"] for row in rows),
        "max_bootstrap_derivative_error": jc_error,
        "max_receiver_derivative_error": max_j_error,
        "max_apex_derivative_error": max_b_error,
        "max_solver_candidate_derivative_error": max_g_error,
    }


def residual_probes() -> tuple[list[dict], list[dict], list[dict]]:
    """Eight directions, five magnitudes, all receivers; 480 estimates."""
    rows, loops, observations = [], [], []
    for amplitude in (1e-5, 1e-3, 0.005, 0.01, 0.02):
        for direction in range(8):
            theta = direction*np.pi/4
            delta = amplitude*np.array([np.cos(theta), np.sin(theta)])
            anchors = TARGET_ANCHORS.copy()
            anchors[0] += delta
            for identifier in RECEIVER_IDS:
                point = TARGET_TEMPLATE[identifier-1]
                observed = receiver_angles(point, anchors)
                result = estimate_receiver(identifier, observed)
                bias = result["position"]-point
                prediction = propagation_blocks(identifier)[2] @ delta
                rows.append({
                    "amplitude_d": amplitude, "direction_deg": direction*45,
                    "delta_c_x": delta[0], "delta_c_y": delta[1],
                    "receiver_id": identifier,
                    "candidate_bias_x": bias[0], "candidate_bias_y": bias[1],
                    "candidate_bias_norm_d": np.linalg.norm(bias),
                    "linear_prediction_x": prediction[0], "linear_prediction_y": prediction[1],
                    "linear_error_d": np.linalg.norm(bias-prediction),
                    "accepted": result["success"],
                    "full_residual_rad": result["max_angle_residual"],
                    "message": result["message"],
                })
            # A controlled injection at the main-stage entry, not a complete
            # bootstrap protocol. Preserve every production failure/stop rule.
            if amplitude in (0.005, 0.02):
                for gain in (1., 0.5):
                    state = TARGET_TEMPLATE.copy()
                    state[0] += delta
                    final, summary, records, actions = _main_record(
                        case=make_initial_state(0., None), gain=gain,
                        state=state, max_rounds=30,
                    )
                    label = {"amplitude_d": amplitude, "direction_deg": direction*45, "gain": gain}
                    receiver_error = np.linalg.norm(
                        (final-TARGET_TEMPLATE)[np.array(RECEIVER_IDS)-1], axis=1
                    )
                    loops.append({
                        **label, "status": summary["status"],
                        "failure_count": summary["failure_count"],
                        "failures": json.dumps(summary["failures"], ensure_ascii=False),
                        "main_slots": summary["broadcast_slots"],
                        "main_tx_uses": summary["tx_uses"],
                        "main_actions": len(actions),
                        "main_displacement_d": sum(np.hypot(a["delta_x"], a["delta_y"]) for a in actions),
                        "receiver_max_error_d": max(receiver_error),
                        "full_team_max_error_d": max(amplitude, max(receiver_error)),
                        "final_max_angle_error_rad": summary["final_max_angle_error_rad"],
                    })
                    observations.extend({**label, **record} for record in records)
    return rows, loops, observations


def geometry_probes() -> tuple[list[dict], list[dict]]:
    base = []
    for height in (2*np.sqrt(3), 1., 0.1, 0.01, 0.001):
        j = bootstrap_jacobian(np.array([2., height]))
        singular = np.linalg.svd(j, compute_uv=False)
        base.append({"height_d": height, "sigma_min": singular[-1], "condition": singular[0]/singular[-1]})
    branches = []
    for identifier in RECEIVER_IDS:
        actual = TARGET_TEMPLATE[identifier-1]+np.array([0.03, -0.02])
        observed = receiver_angles(actual, TARGET_ANCHORS)
        target = TARGET_TEMPLATE[identifier-1]
        starts = [target, target+[0.2, 0.2], target+[-0.2, -0.2],
                  target*np.array([1., -1.]), np.array([2., 5.])]
        for index, initial in enumerate(starts):
            result = estimate_receiver(identifier, observed, initial=initial)
            branches.append({
                "receiver_id": identifier, "start_id": index,
                "initial_x": initial[0], "initial_y": initial[1],
                "actual_x": actual[0], "actual_y": actual[1],
                "candidate_x": result["position"][0], "candidate_y": result["position"][1],
                "accepted": result["success"],
                "position_error_d": np.linalg.norm(result["position"]-actual),
                "full_residual_rad": result["max_angle_residual"], "message": result["message"],
            })
    return base, branches


def threshold_comparison() -> tuple[list[dict], list[dict]]:
    """27 complete protocols: three seeds, three gains, three stop rules."""
    rows, cases = [], []
    for seed in (11, 23, 47):
        for gain in (1., 0.8, 0.5):
            for tolerance in (None, 0.005, 0.01):
                result = run_case(0.1, seed, gain, 30,
                                  bootstrap_position_tolerance=tolerance)
                label = "angle_1e-8" if tolerance is None else f"position_{tolerance:g}d"
                cases.append({"calibration_rule": label, **result})
                # Reconstruct the state and costs independently from action
                # logs, including the finite residual of the frozen apex.
                reconstructed = make_initial_state(0.1, seed).positions.copy()
                actions = result["bootstrap_actions"] + result["main_actions"]
                for action in actions:
                    reconstructed[action["receiver_id"]-1] += [action["delta_x"], action["delta_y"]]
                np.testing.assert_allclose(reconstructed, result["final_positions"], atol=1e-13, rtol=0)
                displacement = sum(np.hypot(a["delta_x"], a["delta_y"]) for a in actions)
                np.testing.assert_allclose(displacement, result["total_cumulative_move_d"], atol=1e-12)
                b, m = result["bootstrap"], result["main"]
                assert result["measurement_slots"] == 2*b["rounds"] + m["broadcast_slots"]
                assert result["tx_uses"] == 4*b["rounds"] + 3*m["broadcast_slots"]
                rows.append({
                    "seed": seed, "gain": gain, "calibration_rule": label,
                    "online_status": result["online_status"],
                    "historical_1e_minus6_status": result["status"],
                    "failure_types": json.dumps(result["failure_types"]),
                    "illustrative_0_01d_pass": result["online_status"] == "success" and result["final_max_position_error_d"] <= 0.01,
                    "final_apex_error_d": np.linalg.norm(result["final_positions"][0]-TARGET_TEMPLATE[0]),
                    "final_max_position_error_d": result["final_max_position_error_d"],
                    "bootstrap_slots": b["angle_slots"], "main_slots": m["broadcast_slots"],
                    "complete_slots": result["measurement_slots"], "complete_tx_uses": result["tx_uses"],
                    "bootstrap_relay_scalars": result["bootstrap_relay_scalars"],
                    "complete_displacement_d": result["total_cumulative_move_d"],
                })
    return rows, cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/q2/reference_residual"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blocks, linear = linear_audit()
    probes, loops, observations = residual_probes()
    base, branches = geometry_probes()
    thresholds, threshold_cases = threshold_comparison()
    (args.output_dir/"threshold_cases.json").write_text(
        json.dumps(_json_value(threshold_cases), indent=2, allow_nan=False)+"\n"
    )
    angle_box = []
    target_angles = bootstrap_angles(TARGET_TEMPLATE)
    for epsilon_deg in (0.01, 0.05, 0.1):
        epsilon = np.deg2rad(epsilon_deg)
        for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            c = bootstrap_from_angles(target_angles+epsilon*np.array([sx, sy]))
            angle_box.append({"epsilon_deg": epsilon_deg, "alpha_sign": sx, "beta_sign": sy,
                              "apex_error_d": np.linalg.norm(c-TARGET_ANCHORS[0]),
                              "linear_box_bound_d": linear["bootstrap_angle_box_factor"]*epsilon})
    tables = {
        "propagation_blocks": blocks, "residual_probes": probes,
        "main_stage_runs": loops, "main_stage_observations": observations,
        "bootstrap_conditioning": base, "multistart_probes": branches,
        "bootstrap_angle_box": angle_box,
        "threshold_comparison": thresholds,
    }
    for name, rows in tables.items():
        _write_csv(args.output_dir/f"{name}.csv", rows)
    summary = {
        "contract": "Noiseless frozen FY01 offset; canonical upper local branch; production estimator unchanged. Injected main-stage costs exclude bootstrap; threshold comparison includes the complete two-stage protocol.",
        "linear": linear,
        "residual_probes": {
            "count": len(probes), "rejected": sum(not row["accepted"] for row in probes),
            "by_amplitude": [{
                "amplitude_d": amplitude,
                "max_candidate_gain": max(row["candidate_bias_norm_d"]/amplitude for row in probes if row["amplitude_d"] == amplitude),
                "max_linear_error_d": max(row["linear_error_d"] for row in probes if row["amplitude_d"] == amplitude),
                "rejected": sum(not row["accepted"] for row in probes if row["amplitude_d"] == amplitude),
            } for amplitude in (1e-5, 1e-3, 0.005, 0.01, 0.02)],
        },
        "main_stage_runs": {"count": len(loops), "converged": sum(row["status"] == "converged" for row in loops)},
        "multistart": {"count": len(branches), "accepted": sum(row["accepted"] for row in branches),
                       "accepted_wrong_position": sum(row["accepted"] and row["position_error_d"] > 1e-6 for row in branches)},
        "threshold_comparison": {
            "count": len(thresholds),
            "online_success": sum(row["online_status"] == "success" for row in thresholds),
            "illustrative_0_01d_pass": sum(row["illustrative_0_01d_pass"] for row in thresholds),
            "historical_1e_minus6_success": sum(row["historical_1e_minus6_status"] == "success" for row in thresholds),
            "action_reconstruction_and_cost_checks": "passed for every complete case",
        },
        "limits": ["Linear coefficients are at the target only.", "Rejected two-row candidates are not successful localizations.",
                   "Finite thresholds tested on three fixed initial states, not an entire neighborhood.",
                   "No noisy stopping rule or global uniqueness is validated."],
    }
    (args.output_dir/"summary.json").write_text(json.dumps(_json_value(summary), indent=2, allow_nan=False)+"\n")
    print(json.dumps(_json_value(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
