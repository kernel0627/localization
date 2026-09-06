"""Finite geometry and branch-boundary audit for the FY01 triangle route.

The script keeps the production angle estimator unchanged and probes only
explicitly named paths.  It is deliberately finite: the output is evidence
about the listed paths, amplitudes, starts, and complete main-stage runs.

Run with::

    conda run -n agent python -m scripts.q2.analyze_geometry_boundaries
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.q2.analyze_reference_residual import propagation_blocks
from scripts.q2.run_triangle_reference import _main_record, _write_csv, make_initial_state
from scripts.q2.triangle_reference import (
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


DEFAULT_OUTPUT = Path("outputs/q2/geometry_boundaries")
ANCHOR_NAMES = ("FY01", "FY11", "FY15")
BOOTSTRAP_BASE_H = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8)
G_AMPLITUDES = (0.002, 0.0048, 0.005, 0.01, 0.02)
RECEIVER_OFFSET_SCENARIOS = {
    "Q": np.array([0.0, 0.0]),
    "plus_0.05_x": np.array([0.05, 0.0]),
    "plus_0.05_y": np.array([0.0, 0.05]),
    "plus_0.10_x": np.array([0.10, 0.0]),
    "plus_0.10_y": np.array([0.0, 0.10]),
}
RECEIVER_PATH_IDS = tuple(RECEIVER_IDS)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _finite_svd_metrics(matrix: np.ndarray) -> dict[str, Any]:
    finite_rows = np.isfinite(matrix).all(axis=1)
    regular = matrix[finite_rows]
    if len(regular) == 0:
        return {
            "regular_row_count": 0,
            "rank": 0,
            "sigma_min": float("nan"),
            "sigma_max": float("nan"),
            "inverse_amplification": float("nan"),
        }
    singular = np.linalg.svd(regular, compute_uv=False)
    rank = int(np.linalg.matrix_rank(regular, tol=1e-10))
    sigma_min = float(singular[-1]) if len(singular) >= 2 else float("nan")
    return {
        "regular_row_count": int(len(regular)),
        "rank": rank,
        "sigma_min": sigma_min,
        "sigma_max": float(singular[0]),
        "inverse_amplification": (
            float(1.0 / sigma_min)
            if np.isfinite(sigma_min) and sigma_min > 0.0
            else float("inf") if sigma_min == 0.0 else float("nan")
        ),
    }


def _triangle_state(apex: np.ndarray) -> np.ndarray:
    state = TARGET_TEMPLATE.copy()
    state[0] = apex
    return state


def _bootstrap_row(path: str, path_index: int, apex: np.ndarray) -> dict[str, Any]:
    observed = bootstrap_angles(_triangle_state(apex))
    interior = np.array([observed[0], observed[1], np.pi - observed.sum()])
    jacobian = _bootstrap_jacobian(apex)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    corners = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    estimate = bootstrap_from_angles(observed)
    return {
        "path": path,
        "path_index": path_index,
        "x_d": float(apex[0]),
        "h_d": float(apex[1]),
        "alpha_rad": float(observed[0]),
        "beta_rad": float(observed[1]),
        "angle_sum_rad": float(observed.sum()),
        "min_triangle_interior_angle_rad": float(np.min(interior)),
        "apex_interior_angle_rad": float(interior[2]),
        "sigma_min_rad_per_d": float(singular[-1]),
        "sigma_max_rad_per_d": float(singular[0]),
        "condition_number": float(singular[0] / singular[-1]),
        "angle_box_inverse_amplification_d_per_rad": float(
            max(np.linalg.norm(np.linalg.solve(jacobian, corner)) for corner in corners)
        ),
        "inverse_reconstruction_error_d": float(np.linalg.norm(estimate - apex)),
    }


def _bootstrap_jacobian(apex: np.ndarray) -> np.ndarray:
    x, h = apex
    if h <= 0.0:
        raise ValueError("bootstrap boundary audit uses the upper branch h > 0")
    return np.array(
        [
            [-h, x] / (x * x + h * h),
            [h, 4.0 - x] / ((4.0 - x) ** 2 + h * h),
        ]
    )


def bootstrap_boundary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path_index = 0
    for x, name in ((2.0, "base_interior_x2"), (-1.0, "base_exterior_x_minus1")):
        for h in BOOTSTRAP_BASE_H:
            rows.append(_bootstrap_row(name, path_index, np.array([x, h])))
            path_index += 1
    for h in (4.0, 10.0, 100.0, 1000.0):
        rows.append(_bootstrap_row("high_vertical_x2", path_index, np.array([2.0, h])))
        path_index += 1
    for t in (10.0, 100.0, 1000.0):
        rows.append(
            _bootstrap_row("far_diagonal_x2t_h_t", path_index, np.array([2.0 * t, t]))
        )
        path_index += 1
    for x in (10.0, 100.0, 1000.0):
        rows.append(_bootstrap_row("far_horizontal_h1", path_index, np.array([x, 1.0])))
        path_index += 1
    return rows


def _receiver_row(
    path: str,
    path_index: int,
    receiver_id: int,
    point: np.ndarray,
    *,
    t: float,
    anchor_label: str,
    circle_theta: float | None = None,
) -> dict[str, Any]:
    observed = receiver_angles(point, TARGET_ANCHORS)
    jacobian = angle_jacobian(point, TARGET_ANCHORS, allow_degenerate=True)
    selected = jacobian[list(SELECTED_ANGLE_INDICES[receiver_id])]
    full_metrics = _finite_svd_metrics(jacobian)
    selected_metrics = _finite_svd_metrics(selected)
    result = estimate_receiver(
        receiver_id,
        observed,
        initial=TARGET_TEMPLATE[receiver_id - 1],
    )
    candidate = np.asarray(result["position"], dtype=float)
    endpoint_singular = [
        int(index)
        for index, value in enumerate(observed)
        if min(float(value), float(np.pi - value)) <= 1e-10
    ]
    singular_rows = [int(index) for index, row in enumerate(jacobian) if not np.isfinite(row).all()]
    return {
        "path": path,
        "path_index": path_index,
        "receiver_id": receiver_id,
        "anchor": anchor_label,
        "t": float(t),
        "circle_theta_rad": "" if circle_theta is None else float(circle_theta),
        "point_x": float(point[0]),
        "point_y": float(point[1]),
        "angle_01_rad": float(observed[0]),
        "angle_02_rad": float(observed[1]),
        "angle_12_rad": float(observed[2]),
        "min_observed_angle_rad": float(np.min(observed)),
        "endpoint_zero_or_pi_rows": ";".join(map(str, endpoint_singular)),
        "nonsmooth_jacobian_rows": ";".join(map(str, singular_rows)),
        "full_regular_row_count": full_metrics["regular_row_count"],
        "full_rank": full_metrics["rank"],
        "full_sigma_min": full_metrics["sigma_min"],
        "selected_indices": ";".join(map(str, SELECTED_ANGLE_INDICES[receiver_id])),
        "selected_regular_row_count": selected_metrics["regular_row_count"],
        "selected_rank": selected_metrics["rank"],
        "selected_sigma_min": selected_metrics["sigma_min"],
        "selected_inverse_amplification": selected_metrics["inverse_amplification"],
        "estimator_success": bool(result["success"]),
        "candidate_x": float(candidate[0]),
        "candidate_y": float(candidate[1]),
        "candidate_error_to_path_d": float(np.linalg.norm(candidate - point)),
        "full_residual_rad": float(result["max_angle_residual"]),
        "estimator_message": str(result["message"]),
    }


def receiver_boundary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path_index = 0
    ray_t = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        start, end = TARGET_ANCHORS[first], TARGET_ANCHORS[second]
        label = f"{ANCHOR_NAMES[first]}_toward_{ANCHOR_NAMES[second]}"
        for t in ray_t:
            point = start + t * (end - start)
            for receiver_id in RECEIVER_PATH_IDS:
                rows.append(
                    _receiver_row(
                        "reference_ray", path_index, receiver_id, point,
                        t=t, anchor_label=label,
                    )
                )
            path_index += 1
    point_t = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
    for anchor_index in range(3):
        direction_angle = 0.37 + 0.51 * anchor_index
        direction = np.array([np.cos(direction_angle), np.sin(direction_angle)])
        anchor = TARGET_ANCHORS[anchor_index]
        label = f"{ANCHOR_NAMES[anchor_index]}_generic_ray"
        for t in point_t:
            point = anchor + t * direction
            for receiver_id in RECEIVER_PATH_IDS:
                rows.append(
                    _receiver_row(
                        "reference_point", path_index, receiver_id, point,
                        t=t, anchor_label=label,
                    )
                )
            path_index += 1

    center = np.array([2.0, 2.0 * np.sqrt(3.0) / 3.0])
    radius = 4.0 / np.sqrt(3.0)
    circle_offsets = (0.1, 0.01, 0.001, 0.0001, 0.0)
    for theta in (0.0, 0.3):
        unit = np.array([np.cos(theta), np.sin(theta)])
        for offset in circle_offsets:
            point = center + (radius + offset) * unit
            for receiver_id in RECEIVER_PATH_IDS:
                rows.append(
                    _receiver_row(
                        "reference_circumcircle", path_index, receiver_id, point,
                        t=offset, anchor_label="reference_circumcircle",
                        circle_theta=theta,
                    )
                )
            path_index += 1
    return rows


def _right_worst_direction(matrix: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(matrix)
    vector = vh[0]
    first = np.flatnonzero(np.abs(vector) > 1e-12)
    if len(first) and vector[first[0]] < 0:
        vector = -vector
    return vector


def g_worst_direction_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        receiver_id: propagation_blocks(receiver_id) for receiver_id in RECEIVER_IDS
    }
    global_receiver = max(
        RECEIVER_IDS,
        key=lambda identifier: (np.linalg.norm(blocks[identifier][2], 2), -identifier),
    )
    global_g = blocks[global_receiver][2]
    for receiver_id in RECEIVER_IDS:
        g = blocks[receiver_id][2]
        direction = _right_worst_direction(g)
        for sign in (1, -1):
            for amplitude in G_AMPLITUDES:
                delta = float(sign * amplitude) * direction
                anchors = TARGET_ANCHORS.copy()
                anchors[0] += delta
                for offset_label, offset in RECEIVER_OFFSET_SCENARIOS.items():
                    actual = TARGET_TEMPLATE[receiver_id - 1] + offset
                    try:
                        observed = receiver_angles(actual, anchors)
                        result = estimate_receiver(
                            receiver_id,
                            observed,
                            initial=TARGET_TEMPLATE[receiver_id - 1],
                        )
                        candidate = np.asarray(result["position"], dtype=float)
                        error = candidate - actual
                        candidate_x = float(candidate[0])
                        candidate_y = float(candidate[1])
                        residual = float(result["max_angle_residual"])
                        accepted = bool(result["success"])
                        message = str(result["message"])
                    except (ValueError, FloatingPointError) as exc:
                        error = np.array([np.nan, np.nan])
                        candidate_x = float("nan")
                        candidate_y = float("nan")
                        residual = float("inf")
                        accepted = False
                        message = str(exc)
                    prediction = g @ delta
                    rows.append(
                        {
                            "receiver_id": receiver_id,
                            "offset_label": offset_label,
                            "offset_x_d": float(offset[0]),
                            "offset_y_d": float(offset[1]),
                            "sign": sign,
                            "amplitude_d": amplitude,
                            "delta_c_x_d": float(delta[0]),
                            "delta_c_y_d": float(delta[1]),
                            "right_singular_vector_x": float(direction[0]),
                            "right_singular_vector_y": float(direction[1]),
                            "g_spectral_norm": float(np.linalg.norm(g, 2)),
                            "predicted_bias_x_d": float(prediction[0]),
                            "predicted_bias_y_d": float(prediction[1]),
                            "candidate_bias_x_d": float(error[0]),
                            "candidate_bias_y_d": float(error[1]),
                            "candidate_linear_error_d": float(np.linalg.norm(error - prediction)),
                            "accepted": accepted,
                            "candidate_x": candidate_x,
                            "candidate_y": candidate_y,
                            "full_residual_rad": residual,
                            "message": message,
                        }
                    )
    return rows, {
        "global_worst_receiver_id": int(global_receiver),
        "global_worst_g_spectral_norm": float(np.linalg.norm(global_g, 2)),
        "global_worst_right_singular_vector": _right_worst_direction(global_g),
        "amplitudes_d": list(G_AMPLITUDES),
        "offset_scenarios": {key: value for key, value in RECEIVER_OFFSET_SCENARIOS.items()},
        "row_count": len(rows),
        "accepted_count": int(sum(row["accepted"] for row in rows)),
        "rejected_count": int(sum(not row["accepted"] for row in rows)),
    }


def main_stage_rows(global_info: dict[str, Any]) -> list[dict[str, Any]]:
    receiver_id = int(global_info["global_worst_receiver_id"])
    direction = np.asarray(global_info["global_worst_right_singular_vector"], dtype=float)
    blocks = {identifier: propagation_blocks(identifier)[2] for identifier in RECEIVER_IDS}
    rows: list[dict[str, Any]] = []
    for offset_label, receiver_offset in RECEIVER_OFFSET_SCENARIOS.items():
        for sign in (1, -1):
            for amplitude in (0.0048, 0.005, 0.01):
                delta = float(sign * amplitude) * direction
                state = TARGET_TEMPLATE.copy()
                state[0] += delta
                state[np.asarray(RECEIVER_IDS) - 1] += receiver_offset
                final, summary, records, actions = _main_record(
                    case=make_initial_state(0.0, None),
                    gain=1.0,
                    state=state,
                    max_rounds=30,
                )
                receiver_errors = {
                    str(identifier): float(
                        np.linalg.norm(final[identifier - 1] - TARGET_TEMPLATE[identifier - 1])
                    )
                    for identifier in RECEIVER_IDS
                }
                predicted_receiver_errors = {
                    str(identifier): float(np.linalg.norm(blocks[identifier] @ delta))
                    for identifier in RECEIVER_IDS
                }
                failures = summary["failures"]
                rows.append(
                    {
                        "global_worst_receiver_id": receiver_id,
                        "receiver_offset_label": offset_label,
                        "receiver_offset_x_d": float(receiver_offset[0]),
                        "receiver_offset_y_d": float(receiver_offset[1]),
                        "sign": sign,
                        "amplitude_d": amplitude,
                        "delta_c_x_d": float(delta[0]),
                        "delta_c_y_d": float(delta[1]),
                        "status": summary["status"],
                        "failure_count": int(summary["failure_count"]),
                        "failures": json.dumps(failures, ensure_ascii=False, sort_keys=True),
                        "broadcast_slots": int(summary["broadcast_slots"]),
                        "tx_uses": int(summary["tx_uses"]),
                        "action_count": int(len(actions)),
                        "records": int(len(records)),
                        "final_apex_error_d": float(np.linalg.norm(final[0] - TARGET_TEMPLATE[0])),
                        "final_max_receiver_error_d": max(receiver_errors.values()),
                        "final_team_max_error_d": float(
                            max(np.linalg.norm(final - TARGET_TEMPLATE, axis=1))
                        ),
                        # This is the target-point G prediction for delta C only;
                        # the receiver offset is deliberately kept in the actual run.
                        "predicted_max_receiver_error_d": max(predicted_receiver_errors.values()),
                        "receiver_errors_json": json.dumps(receiver_errors, sort_keys=True),
                        "predicted_receiver_errors_json": json.dumps(
                            predicted_receiver_errors, sort_keys=True
                        ),
                        "final_max_angle_error_rad": float(summary["final_max_angle_error_rad"]),
                    }
                )
    return rows


def counterexample_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    upper = TARGET_TEMPLATE[0].copy()
    lower = np.array([upper[0], -upper[1]])
    upper_observed = bootstrap_angles(_triangle_state(upper))
    lower_observed = bootstrap_angles(_triangle_state(lower))
    mirror_estimate = bootstrap_from_angles(lower_observed)
    rows.append(
        {
            "family": "bootstrap_mirror",
            "case": "lower_apex_same_unsigned_base_angles",
            "receiver_id": "",
            "initial_id": "",
            "point_x": float(lower[0]),
            "point_y": float(lower[1]),
            "candidate_x": float(mirror_estimate[0]),
            "candidate_y": float(mirror_estimate[1]),
            "observation_difference": float(np.linalg.norm(lower_observed - upper_observed)),
            "position_error_d": float(np.linalg.norm(mirror_estimate - lower)),
            "accepted": True,
            "message": "unsigned bootstrap angles choose the canonical upper branch",
        }
    )

    center = np.array([2.0, 2.0 * np.sqrt(3.0) / 3.0])
    radius = 4.0 / np.sqrt(3.0)
    circle_points = [
        center + radius * np.array([np.cos(theta), np.sin(theta)])
        for theta in (0.0, 0.3)
    ]
    circle_observations = [receiver_angles(point, TARGET_ANCHORS) for point in circle_points]
    for index, (point, observed) in enumerate(zip(circle_points, circle_observations)):
        default_result = estimate_receiver(2, observed, initial=TARGET_TEMPLATE[1])
        on_circle_result = estimate_receiver(2, observed, initial=point)
        rows.extend(
            [
                {
                    "family": "circumcircle_default_start",
                    "case": f"theta_{index}",
                    "receiver_id": 2,
                    "initial_id": "template",
                    "point_x": float(point[0]),
                    "point_y": float(point[1]),
                    "candidate_x": float(default_result["position"][0]),
                    "candidate_y": float(default_result["position"][1]),
                    "observation_difference": float(
                        np.linalg.norm(observed - circle_observations[0])
                    ),
                    "position_error_d": float(
                        np.linalg.norm(default_result["position"] - point)
                    ),
                    "accepted": bool(default_result["success"]),
                    "message": str(default_result["message"]),
                },
                {
                    "family": "circumcircle_on_branch_start",
                    "case": f"theta_{index}",
                    "receiver_id": 2,
                    "initial_id": "point_itself",
                    "point_x": float(point[0]),
                    "point_y": float(point[1]),
                    "candidate_x": float(on_circle_result["position"][0]),
                    "candidate_y": float(on_circle_result["position"][1]),
                    "observation_difference": float(
                        np.linalg.norm(observed - circle_observations[0])
                    ),
                    "position_error_d": float(
                        np.linalg.norm(on_circle_result["position"] - point)
                    ),
                    "accepted": bool(on_circle_result["success"]),
                    "message": str(on_circle_result["message"]),
                },
            ]
        )

    receiver_id = 2
    actual = TARGET_TEMPLATE[receiver_id - 1] + np.array([0.03, -0.02])
    starts = {
        "template": TARGET_TEMPLATE[receiver_id - 1],
        "nearby": TARGET_TEMPLATE[receiver_id - 1] + np.array([0.2, 0.2]),
        "reflected": TARGET_TEMPLATE[receiver_id - 1] * np.array([1.0, -1.0]),
        "far": np.array([2.0, 5.0]),
    }
    observed = receiver_angles(actual, TARGET_ANCHORS)
    for initial_id, initial in starts.items():
        result = estimate_receiver(receiver_id, observed, initial=initial)
        rows.append(
            {
                "family": "receiver_multistart",
                "case": "actual_Q_plus_0.03_minus_0.02",
                "receiver_id": receiver_id,
                "initial_id": initial_id,
                "point_x": float(actual[0]),
                "point_y": float(actual[1]),
                "candidate_x": float(result["position"][0]),
                "candidate_y": float(result["position"][1]),
                "observation_difference": 0.0,
                "position_error_d": float(np.linalg.norm(result["position"] - actual)),
                "accepted": bool(result["success"]),
                "message": str(result["message"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_rows = bootstrap_boundary_rows()
    receiver_rows = receiver_boundary_rows()
    g_rows, g_info = g_worst_direction_rows()
    main_rows = main_stage_rows(g_info)
    counter_rows = counterexample_rows()

    _write_csv(args.output_dir / "bootstrap_paths.csv", bootstrap_rows)
    _write_csv(args.output_dir / "receiver_paths.csv", receiver_rows)
    _write_csv(args.output_dir / "g_worst_direction.csv", g_rows)
    _write_csv(args.output_dir / "main_stage_worst_direction.csv", main_rows)
    _write_csv(args.output_dir / "counterexamples.csv", counter_rows)

    main_stage_by_amplitude: dict[str, dict[str, Any]] = {}
    for amplitude in sorted({float(row["amplitude_d"]) for row in main_rows}):
        selected = [row for row in main_rows if row["amplitude_d"] == amplitude]
        final_team_errors = [float(row["final_team_max_error_d"]) for row in selected]
        main_stage_by_amplitude[str(amplitude)] = {
            "row_count": len(selected),
            "converged_count": sum(row["status"] == "converged" for row in selected),
            "failure_count": sum(row["status"] != "converged" for row in selected),
            "final_team_max_error_range_d": [min(final_team_errors), max(final_team_errors)],
            "broadcast_slots_range": [
                min(int(row["broadcast_slots"]) for row in selected),
                max(int(row["broadcast_slots"]) for row in selected),
            ],
            "tx_uses_range": [
                min(int(row["tx_uses"]) for row in selected),
                max(int(row["tx_uses"]) for row in selected),
            ],
        }

    circle_rows = [
        row
        for row in receiver_rows
        if row["path"] == "reference_circumcircle" and row["t"] == 0.0
    ]
    summary = {
        "contract": (
            "Finite noiseless geometry and branch-boundary audit; production estimator "
            "unchanged; rows are evidence only for listed paths, amplitudes, offsets, "
            "and starts."
        ),
        "bootstrap": {
            "row_count": len(bootstrap_rows),
            "paths": sorted({row["path"] for row in bootstrap_rows}),
            "interior_small_h_last_angle_sum": next(
                row["angle_sum_rad"]
                for row in reversed(bootstrap_rows)
                if row["path"] == "base_interior_x2"
            ),
            "exterior_small_h_last_angle_sum": next(
                row["angle_sum_rad"]
                for row in reversed(bootstrap_rows)
                if row["path"] == "base_exterior_x_minus1"
            ),
            "high_far_included": True,
        },
        "receiver_boundary": {
            "row_count": len(receiver_rows),
            "circumcircle_exact_row_count": len(circle_rows),
            "reference_ray_rows_with_one_nonsmooth_angle": sum(
                bool(row["nonsmooth_jacobian_rows"])
                and row["full_regular_row_count"] >= 2
                for row in receiver_rows
                if row["path"] == "reference_ray"
            ),
            "circumcircle_exact_selected_rank_deficient": sum(
                row["selected_rank"] < 2
                for row in circle_rows
            ),
        },
        "g_worst_direction": g_info,
        "main_stage": {
            "row_count": len(main_rows),
            "converged_count": sum(row["status"] == "converged" for row in main_rows),
            "failure_count": sum(row["status"] != "converged" for row in main_rows),
            "receiver_offset_scenarios": list(RECEIVER_OFFSET_SCENARIOS),
            "by_amplitude": main_stage_by_amplitude,
        },
        "counterexamples": {
            "row_count": len(counter_rows),
            "rejected_count": sum(not row["accepted"] for row in counter_rows),
            "circumcircle_observation_pair_difference": float(
                np.linalg.norm(
                    receiver_angles(
                        np.array([circle_rows[0]["point_x"], circle_rows[0]["point_y"]]),
                        TARGET_ANCHORS,
                    )
                    - receiver_angles(
                        np.array([circle_rows[12]["point_x"], circle_rows[12]["point_y"]]),
                        TARGET_ANCHORS,
                    )
                )
            ),
        },
        "limits": [
            "No noisy observations, execution errors, or global uniqueness claim.",
            "A finite path approaching a boundary does not establish behavior elsewhere.",
            "Circumcircle rank loss is local and exact for the listed reference triangle.",
            "Main-stage runs retain the true FY01 residual and report online stopping separately from position error.",
        ],
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(_json_value(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
