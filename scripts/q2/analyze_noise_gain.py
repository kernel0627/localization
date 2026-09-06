"""Q2 angle-noise, reference-residual, and gain analysis.

This module is deliberately a separate analysis runner.  It does not change
the production estimator or its strict three-angle acceptance rule.  The
runner records both the local candidate returned by the estimator and the
production ``success`` flag, preserving rejected branches as solver guards.

Run with::

    conda run -n agent python -m scripts.q2.analyze_noise_gain

The generated files are written below ``outputs/q2/noise_gain``.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from scripts.q2.analyze_reference_residual import bootstrap_jacobian, propagation_blocks
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


OUTPUT_DIR = Path("outputs/q2/noise_gain")
SCENARIO_SIGMA_DEG = (0.01, 0.05, 0.10)
GAINS = (0.25, 0.50, 0.75, 1.00)
NONLINEAR_GAIN_PAIRS = tuple((eta, eta) for eta in GAINS) + ((0.50, 0.25),)
N_BOOTSTRAP = 8
N_MAIN = 18
N_LINEAR_TRIALS = 8_000
NONLINEAR_CASES = (
    (11, 0.01),
    (23, 0.01),
    (47, 0.05),
    (71, 0.05),
    (101, 0.10),
    (131, 0.10),
)
PI = float(np.pi)
FOLD_TOL = 1e-10


def _json_value(value: Any) -> Any:
    """Convert numpy values to strict JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wrap_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle to ``[-pi, pi)`` without changing array shape."""

    return (np.asarray(angle) + PI) % (2.0 * PI) - PI


def fold_unsigned_difference(first: float, second: float) -> float:
    """Return the unsigned angle in ``[0, pi]`` with the 0/pi fold."""

    return float(abs(float(wrap_pi(first - second))))


def azimuth_angles(receiver: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Return the three labeled ray azimuths from one receiver."""

    vectors = np.asarray(anchors, dtype=float) - np.asarray(receiver, dtype=float)
    return np.arctan2(vectors[:, 1], vectors[:, 0])


def pairwise_from_azimuth(azimuth: np.ndarray) -> np.ndarray:
    """Generate the three unsigned angles from three noisy azimuths.

    ``azimuth`` may have shape ``(3,)`` or any leading batch shape ending in
    ``(3,)``.  The pair ordering follows ``ANCHOR_PAIRS`` in the production
    estimator.  The explicit fold is important near 0 and pi.
    """

    values = np.asarray(azimuth, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("azimuth must end in a length-three axis")
    result = np.empty(values.shape[:-1] + (3,), dtype=float)
    for row, (first, second) in enumerate(ANCHOR_PAIRS):
        result[..., row] = np.abs(wrap_pi(values[..., first] - values[..., second]))
    return result


def _azimuth_difference_sign(first: float, second: float) -> float | None:
    """Derivative sign for one smooth folded pair, or ``None`` at a fold."""

    difference = float(wrap_pi(first - second))
    if abs(difference) <= FOLD_TOL or abs(abs(difference) - PI) <= FOLD_TOL:
        return None
    return float(np.sign(difference))


def azimuth_difference_matrix(
    receiver_id: int,
    *,
    selected_only: bool = True,
) -> tuple[tuple[int, int], np.ndarray]:
    """Return the local ``D`` matrix mapping azimuth errors to angle errors.

    A selected row is smooth by construction.  Asking for all rows retains
    ``nan`` at a 0/pi fold, making the undefined linearization visible rather
    than silently assigning a sign.
    """

    receiver = TARGET_TEMPLATE[receiver_id - 1]
    azimuth = azimuth_angles(receiver, TARGET_ANCHORS)
    if selected_only:
        indices = tuple(int(value) for value in SELECTED_ANGLE_INDICES[receiver_id])
    else:
        indices = tuple(range(3))
    # Selected rows are regular and need a genuine linear map with zeros in
    # the unused azimuth columns.  The full three-row diagnostic retains NaN
    # only where the 0/pi fold makes a derivative undefined.
    matrix = np.zeros((len(indices), 3), dtype=float)
    for output_row, angle_index in enumerate(indices):
        first, second = ANCHOR_PAIRS[angle_index]
        sign = _azimuth_difference_sign(azimuth[first], azimuth[second])
        if sign is not None:
            matrix[output_row, first] = sign
            matrix[output_row, second] = -sign
        elif not selected_only:
            matrix[output_row, :] = np.nan
    return indices, matrix


def bootstrap_covariance(sigma_deg: float) -> dict[str, np.ndarray | float]:
    """Return base-angle and FY01 position noise covariance for one scenario."""

    sigma = np.deg2rad(float(sigma_deg))
    sigma_theta = sigma
    jacobian = bootstrap_jacobian(TARGET_ANCHORS[0])
    inverse = np.linalg.inv(jacobian)
    sigma_angles = sigma_theta**2 * np.eye(2)
    sigma_position = inverse @ sigma_angles @ inverse.T
    return {
        "sigma_theta_rad": sigma_theta,
        "sigma_theta_deg": float(sigma_deg),
        "sigma_angles": sigma_angles,
        "J_C": jacobian,
        "J_C_inverse": inverse,
        "Sigma_C_measurement": sigma_position,
    }


def receiver_noise_contract(sigma_deg: float) -> list[dict[str, Any]]:
    """Build selected-row D and angle covariance tables for all receivers."""

    sigma_theta = np.deg2rad(float(sigma_deg))
    sigma_azimuth = sigma_theta / np.sqrt(2.0)
    rows: list[dict[str, Any]] = []
    for receiver_id in RECEIVER_IDS:
        selected, d_selected = azimuth_difference_matrix(receiver_id, selected_only=True)
        _, d_full = azimuth_difference_matrix(receiver_id, selected_only=False)
        sigma_selected = sigma_azimuth**2 * (d_selected @ d_selected.T)
        j, b, g = propagation_blocks(receiver_id)
        j_inverse = np.linalg.inv(j)
        sigma_position = j_inverse @ sigma_selected @ j_inverse.T
        rows.append(
            {
                "sigma_theta_deg": float(sigma_deg),
                "sigma_azimuth_deg": float(np.rad2deg(sigma_azimuth)),
                "receiver_id": receiver_id,
                "selected_indices": ";".join(str(index) for index in selected),
                "selected_pairs": ";".join(
                    f"{first + 1}-{second + 1}"
                    for first, second in (ANCHOR_PAIRS[index] for index in selected)
                ),
                "D_selected": json.dumps(_json_value(d_selected)),
                "D_full": json.dumps(_json_value(d_full)),
                "Sigma_theta_selected": json.dumps(_json_value(sigma_selected)),
                "Sigma_receiver_measurement": json.dumps(_json_value(sigma_position)),
                "sigma_receiver_x": float(np.sqrt(sigma_position[0, 0])),
                "sigma_receiver_y": float(np.sqrt(sigma_position[1, 1])),
                "g_spectral_norm": float(np.linalg.norm(g, 2)),
                "selected_rows_are_smooth": bool(np.isfinite(d_selected).all()),
            }
        )
    return rows


def row_selection_comparison(sigma_deg: float = 0.05) -> list[dict[str, Any]]:
    """Compare geometric row selection with covariance-aware alternatives.

    The production selector maximizes the unweighted ``sigma_min(J)``.  With
    shared azimuth noise the angle covariance is ``sigma_phi^2 D D.T``; this
    table computes the resulting position covariance for every regular pair.
    It is diagnostic only and does not replace the production choice.
    """

    sigma_phi = np.deg2rad(float(sigma_deg)) / np.sqrt(2.0)
    rows: list[dict[str, Any]] = []
    for receiver_id in RECEIVER_IDS:
        point = TARGET_TEMPLATE[receiver_id - 1]
        jacobian_full = angle_jacobian(point, TARGET_ANCHORS, allow_degenerate=True)
        _, d_full = azimuth_difference_matrix(receiver_id, selected_only=False)
        regular = [
            index for index in range(3)
            if np.isfinite(jacobian_full[index]).all() and np.isfinite(d_full[index]).all()
        ]
        production = tuple(int(value) for value in SELECTED_ANGLE_INDICES[receiver_id])
        candidates: list[dict[str, Any]] = []
        for pair in combinations(regular, 2):
            j_pair = jacobian_full[list(pair)]
            d_pair = d_full[list(pair)]
            sigma_theta = sigma_phi**2 * (d_pair @ d_pair.T)
            inverse = np.linalg.inv(j_pair)
            sigma_position = inverse @ sigma_theta @ inverse.T
            singular = np.linalg.svd(j_pair, compute_uv=False)
            candidate = {
                "receiver_id": receiver_id,
                "pair_indices": ";".join(map(str, pair)),
                "pair_labels": ";".join(
                    f"{first + 1}-{second + 1}"
                    for first, second in (ANCHOR_PAIRS[index] for index in pair)
                ),
                "production_selected": bool(tuple(pair) == production),
                "sigma_min_J": float(singular[-1]),
                "trace_position_cov_d2": float(np.trace(sigma_position)),
                "lambda_max_position_cov_d2": float(np.linalg.eigvalsh(sigma_position)[-1]),
                "position_covariance": json.dumps(_json_value(sigma_position)),
            }
            candidates.append(candidate)
        best_trace_value = min(row["trace_position_cov_d2"] for row in candidates)
        best_lambda_value = min(row["lambda_max_position_cov_d2"] for row in candidates)
        trace_tolerance = 1e-10 * max(1.0, abs(best_trace_value))
        lambda_tolerance = 1e-10 * max(1.0, abs(best_lambda_value))
        for candidate in candidates:
            candidate["covariance_best_trace"] = abs(candidate["trace_position_cov_d2"] - best_trace_value) <= trace_tolerance
            candidate["covariance_best_lambda_max"] = abs(candidate["lambda_max_position_cov_d2"] - best_lambda_value) <= lambda_tolerance
            candidate["geometric_selector_matches_trace"] = bool(candidate["production_selected"] and candidate["covariance_best_trace"])
            candidate["geometric_selector_matches_lambda_max"] = bool(candidate["production_selected"] and candidate["covariance_best_lambda_max"])
            rows.append(candidate)
    return rows


def _geometric_sum(eta: float, count: int) -> float:
    """Return ``eta^2 sum_{j=0}^{count-1}(1-eta)^(2j)``."""

    if count <= 0:
        return 0.0
    a = 1.0 - float(eta)
    if abs(1.0 - a * a) < 1e-14:
        return float(count * eta * eta)
    return float(eta * eta * (1.0 - a ** (2 * count)) / (1.0 - a * a))


def _locked_reference_theory(
    eta_c: float,
    *,
    n_bootstrap: int,
    e_c0: np.ndarray,
    sigma_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the frozen FY01 error after bootstrap."""

    covariance = bootstrap_covariance(sigma_deg)
    a = 1.0 - float(eta_c)
    mean = a**n_bootstrap * np.asarray(e_c0, dtype=float)
    covariance_n = _geometric_sum(eta_c, n_bootstrap) * covariance["Sigma_C_measurement"]
    return mean, np.asarray(covariance_n, dtype=float)


def linear_theory(
    sigma_deg: float,
    eta_c: float,
    eta_r: float,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    n_main: int = N_MAIN,
    e_c0: np.ndarray | None = None,
    e_receiver0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Finite-horizon bias/covariance formula for all 12 receivers.

    ``e_receiver0`` is a deterministic ``(12,2)`` array by default.  The
    reference covariance appears in every cross-machine block because the
    reference is measured, locked, and then held fixed.
    """

    if not (0.0 < eta_c < 2.0 and 0.0 < eta_r < 2.0):
        raise ValueError("both gains must satisfy 0 < eta < 2")
    if e_c0 is None:
        e_c0 = np.array([0.08, -0.06], dtype=float)
    if e_receiver0 is None:
        e_receiver0 = np.zeros((len(RECEIVER_IDS), 2), dtype=float)
    e_c0 = np.asarray(e_c0, dtype=float)
    e_receiver0 = np.asarray(e_receiver0, dtype=float)
    if e_c0.shape != (2,) or e_receiver0.shape != (len(RECEIVER_IDS), 2):
        raise ValueError("initial error shapes are invalid")

    mean_c, covariance_c = _locked_reference_theory(
        eta_c, n_bootstrap=n_bootstrap, e_c0=e_c0, sigma_deg=sigma_deg
    )
    a_r = 1.0 - float(eta_r)
    attenuation = a_r**n_main
    main_sum = _geometric_sum(eta_r, n_main)
    sigma_theta = np.deg2rad(float(sigma_deg))
    sigma_azimuth = sigma_theta / np.sqrt(2.0)
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    g_matrices: list[np.ndarray] = []
    receiver_covariances: list[np.ndarray] = []
    for index, receiver_id in enumerate(RECEIVER_IDS):
        selected, d_selected = azimuth_difference_matrix(receiver_id, selected_only=True)
        sigma_theta_selected = sigma_azimuth**2 * (d_selected @ d_selected.T)
        j, _, g = propagation_blocks(receiver_id)
        v = np.linalg.inv(j)
        sigma_v = v @ sigma_theta_selected @ v.T
        mean_i = attenuation * e_receiver0[index] - (1.0 - attenuation) * g @ mean_c
        covariance_i = (
            (1.0 - attenuation) ** 2 * g @ covariance_c @ g.T
            + main_sum * sigma_v
        )
        means.append(mean_i)
        covariances.append(covariance_i)
        g_matrices.append(g)
        receiver_covariances.append(sigma_v)
    means_array = np.asarray(means)
    covariance_array = np.asarray(covariances)
    cross_covariance = np.empty((len(RECEIVER_IDS), len(RECEIVER_IDS), 2, 2))
    for i, g_i in enumerate(g_matrices):
        for j, g_j in enumerate(g_matrices):
            cross_covariance[i, j] = (
                (1.0 - attenuation) ** 2 * g_i @ covariance_c @ g_j.T
            )
            if i == j:
                cross_covariance[i, j] += main_sum * receiver_covariances[i]
    per_receiver_mse = np.sum(means_array**2, axis=1) + np.trace(covariance_array, axis1=1, axis2=2)
    return {
        "sigma_theta_deg": float(sigma_deg),
        "eta_c": float(eta_c),
        "eta_r": float(eta_r),
        "n_bootstrap": int(n_bootstrap),
        "n_main": int(n_main),
        "mean_reference_error": mean_c,
        "covariance_reference_error": covariance_c,
        "mean_receiver_error": means_array,
        "covariance_receiver_error": covariance_array,
        "cross_covariance_receiver_error": cross_covariance,
        "g_matrices": np.asarray(g_matrices),
        "receiver_measurement_covariances": np.asarray(receiver_covariances),
        "position_rms_d": float(np.sqrt(np.mean(per_receiver_mse))),
        "reference_rms_d": float(np.sqrt(np.sum(mean_c**2) + np.trace(covariance_c))),
        "white_noise_steady_multiplier": float(eta_r / (2.0 - eta_r)),
        "reference_white_noise_steady_multiplier": float(eta_c / (2.0 - eta_c)),
    }


def _apply_linear_bootstrap_noise(
    base_noise: np.ndarray,
    eta_c: float,
    e_c0: np.ndarray,
    sigma_deg: float,
) -> np.ndarray:
    covariance = bootstrap_covariance(sigma_deg)
    inverse = np.asarray(covariance["J_C_inverse"])
    measurement = np.einsum("ab,nkb->nka", inverse, base_noise)
    current = np.broadcast_to(e_c0, (base_noise.shape[0], 2)).copy()
    for round_index in range(base_noise.shape[1]):
        current = (1.0 - eta_c) * current - eta_c * measurement[:, round_index]
    return current


def linear_monte_carlo(
    sigma_deg: float,
    eta_c: float,
    eta_r: float,
    *,
    base_noise: np.ndarray,
    azimuth_noise: np.ndarray,
    e_c0: np.ndarray,
    e_receiver0: np.ndarray,
) -> dict[str, Any]:
    """Vectorized linear recurrence using shared noise arrays."""

    reference_error = _apply_linear_bootstrap_noise(base_noise, eta_c, e_c0, sigma_deg)
    current = np.broadcast_to(e_receiver0, (base_noise.shape[0], len(RECEIVER_IDS), 2)).copy()
    for round_index in range(azimuth_noise.shape[1]):
        update = np.empty_like(current)
        for receiver_index, receiver_id in enumerate(RECEIVER_IDS):
            selected, d_selected = azimuth_difference_matrix(receiver_id, selected_only=True)
            epsilon = np.einsum(
                "ab,nb->na", d_selected, azimuth_noise[:, round_index, receiver_index, :]
            )
            epsilon *= 1.0  # D already maps azimuth errors; scale is in the input.
            j, _, _ = propagation_blocks(receiver_id)
            candidate_noise = np.einsum("ab,nb->na", np.linalg.inv(j), epsilon)
            _, _, g = propagation_blocks(receiver_id)
            update[:, receiver_index, :] = (
                (1.0 - eta_r) * current[:, receiver_index, :]
                - eta_r * (np.einsum("ab,nb->na", g, reference_error) + candidate_noise)
            )
        current = update
    final = np.concatenate((reference_error[:, None, :], current), axis=1)
    position_rms = np.sqrt(np.mean(np.sum(current**2, axis=2)))
    max_error = np.max(np.linalg.norm(final, axis=2), axis=1)
    return {
        "final_reference_error": reference_error,
        "final_receiver_error": current,
        "position_rms_d": float(position_rms),
        "position_max_error_sample_d": float(np.max(max_error)),
        "position_p99_error_sample_d": float(np.quantile(max_error, 0.99)),
    }


def _initial_team(seed: int, *, reference_error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build one deterministic local initial state and its receiver errors."""

    rng = np.random.default_rng(seed + 900_000)
    perturbations = rng.normal(size=(len(RECEIVER_IDS), 2))
    norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
    perturbations = perturbations / np.maximum(norms, 1e-12) * 0.04
    team = TARGET_TEMPLATE.copy()
    team[0] += reference_error
    for index, receiver_id in enumerate(RECEIVER_IDS):
        team[receiver_id - 1] += perturbations[index]
    return team, perturbations


def nonlinear_fixed_budget(
    *,
    seed: int,
    sigma_deg: float,
    eta_c: float,
    eta_r: float,
    base_noise: np.ndarray,
    azimuth_noise: np.ndarray,
    initial_reference_error: np.ndarray,
    initial_receiver_errors: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one fixed-budget nonlinear trajectory and preserve failures."""

    team = TARGET_TEMPLATE.copy()
    team[0] += initial_reference_error
    for index, receiver_id in enumerate(RECEIVER_IDS):
        team[receiver_id - 1] += initial_receiver_errors[index]
    target_apex = TARGET_TEMPLATE[0]
    failures: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    bootstrap_displacement = 0.0
    main_displacement = 0.0
    bootstrap_failures = 0
    receiver_failures = 0
    production_acceptance_hits = 0
    original_angle_stop_hits = 0
    target_bootstrap_angles = bootstrap_angles(TARGET_TEMPLATE)

    # The two base-angle noises are fixed per case and replayed across gains.
    for round_index in range(N_BOOTSTRAP):
        true_angles = bootstrap_angles(team)
        observed = true_angles + base_noise[round_index]
        try:
            estimate = np.asarray(bootstrap_from_angles(observed), dtype=float)
            if not np.isfinite(estimate).all():
                raise ValueError("non-finite bootstrap estimate")
            error = estimate - target_apex
            action = -eta_c * error
            team[0] += action
            bootstrap_displacement += float(np.linalg.norm(action))
            trace_rows.append({
                "stage": "bootstrap", "round": round_index + 1,
                "receiver_id": "FY01", "observed": observed.copy(),
                "estimate": estimate.copy(), "action": action.copy(),
                "success": True, "max_angle_residual_rad": None,
            })
            if float(np.max(np.abs(observed - target_bootstrap_angles))) <= 1e-8:
                original_angle_stop_hits += 1
        except (ValueError, FloatingPointError) as error:
            bootstrap_failures += 1
            failures.append(
                {
                    "seed": seed,
                    "sigma_theta_deg": sigma_deg,
                    "eta_c": eta_c,
                    "eta_r": eta_r,
                    "stage": "bootstrap",
                    "round": round_index + 1,
                    "receiver_id": "FY01",
                    "failure": str(error),
                }
            )
            trace_rows.append({
                "stage": "bootstrap", "round": round_index + 1,
                "receiver_id": "FY01", "observed": observed.copy(),
                "estimate": None, "action": np.zeros(2),
                "success": False, "max_angle_residual_rad": None,
            })

    anchors = np.vstack((team[0], team[10], team[14]))
    for round_index in range(N_MAIN):
        for receiver_index, receiver_id in enumerate(RECEIVER_IDS):
            point = team[receiver_id - 1]
            true_azimuth = azimuth_angles(point, anchors)
            observed = pairwise_from_azimuth(
                true_azimuth + azimuth_noise[round_index, receiver_index]
            )
            result = estimate_receiver(receiver_id, observed)
            position = np.asarray(result["position"], dtype=float)
            if bool(result["success"]):
                production_acceptance_hits += 1
            if not bool(result["success"]):
                receiver_failures += 1
                failures.append(
                    {
                        "seed": seed,
                        "sigma_theta_deg": sigma_deg,
                        "eta_c": eta_c,
                        "eta_r": eta_r,
                        "stage": "main",
                        "round": round_index + 1,
                        "receiver_id": f"FY{receiver_id:02d}",
                        "failure": result["message"],
                        "max_angle_residual_rad": result["max_angle_residual"],
                        "nfev": result["nfev"],
                    }
                )
            # A rejected production estimate is preserved as a failure and
            # does not move the simulator.  This keeps fixed-budget scoring
            # faithful if a future noise/initial-state case fails.
            action = np.zeros(2, dtype=float)
            if bool(result["success"]) and np.isfinite(position).all():
                action = -eta_r * (position - TARGET_TEMPLATE[receiver_id - 1])
                team[receiver_id - 1] += action
                main_displacement += float(np.linalg.norm(action))
            trace_rows.append({
                "stage": "main", "round": round_index + 1,
                "receiver_id": f"FY{receiver_id:02d}",
                "observed": observed.copy(), "estimate": position.copy(),
                "action": action.copy(), "success": bool(result["success"]),
                "max_angle_residual_rad": float(result["max_angle_residual"]),
            })
            target_angles = receiver_angles(TARGET_TEMPLATE[receiver_id - 1], TARGET_ANCHORS)
            if float(np.max(np.abs(observed - target_angles))) <= 1e-8:
                original_angle_stop_hits += 1
        anchors = np.vstack((team[0], team[10], team[14]))

    errors = team - TARGET_TEMPLATE
    receiver_errors = errors[np.asarray(RECEIVER_IDS) - 1]
    all_norms = np.linalg.norm(errors, axis=1)
    summary = {
        "seed": int(seed),
        "sigma_theta_deg": float(sigma_deg),
        "eta_c": float(eta_c),
        "eta_r": float(eta_r),
        "n_bootstrap": N_BOOTSTRAP,
        "n_main": N_MAIN,
        "online_status": "not_evaluated_fixed_budget",
        "original_1e-8_stop_applicable": False,
        "production_acceptance_hits": int(production_acceptance_hits),
        "original_angle_stop_hits": int(original_angle_stop_hits),
        "bootstrap_failure_count": int(bootstrap_failures),
        "receiver_failure_count": int(receiver_failures),
        "failure_count": int(len(failures)),
        "final_reference_error_norm_d": float(np.linalg.norm(errors[0])),
        "final_receiver_rms_d": float(np.sqrt(np.mean(np.sum(receiver_errors**2, axis=1)))),
        "final_receiver_max_d": float(np.max(np.linalg.norm(receiver_errors, axis=1))),
        "final_team_rms_d": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "final_team_max_d": float(np.max(all_norms)),
        "bootstrap_action_displacement_d": float(bootstrap_displacement),
        "main_action_displacement_d": float(main_displacement),
        "total_action_displacement_d": float(bootstrap_displacement + main_displacement),
        "measurement_slots": int(2 * N_BOOTSTRAP + N_MAIN),
        "transmitter_uses": int(4 * N_BOOTSTRAP + 3 * N_MAIN),
        "final_positions": team,
    }
    return summary, failures, trace_rows


def _noise_inputs_for_case(seed: int, sigma_deg: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sigma_theta = np.deg2rad(float(sigma_deg))
    sigma_azimuth = sigma_theta / np.sqrt(2.0)
    base = rng.normal(0.0, sigma_theta, size=(N_BOOTSTRAP, 2))
    azimuth = rng.normal(
        0.0,
        sigma_azimuth,
        size=(N_MAIN, len(RECEIVER_IDS), 3),
    )
    return base, azimuth


def _linear_noise_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    e_c0 = np.array([0.08, -0.06], dtype=float)
    linear_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    noise_archive: dict[str, Any] = {}
    rng = np.random.default_rng(20260906)
    for sigma_deg in SCENARIO_SIGMA_DEG:
        contract_rows.extend(receiver_noise_contract(sigma_deg))
        base = rng.normal(0.0, np.deg2rad(sigma_deg), size=(N_LINEAR_TRIALS, N_BOOTSTRAP, 2))
        azimuth = rng.normal(
            0.0,
            np.deg2rad(sigma_deg) / np.sqrt(2.0),
            size=(N_LINEAR_TRIALS, N_MAIN, len(RECEIVER_IDS), 3),
        )
        for eta in GAINS:
            theory = linear_theory(sigma_deg, eta, eta, e_c0=e_c0)
            simulation = linear_monte_carlo(
                sigma_deg,
                eta,
                eta,
                base_noise=base,
                azimuth_noise=azimuth,
                e_c0=e_c0,
                e_receiver0=np.zeros((len(RECEIVER_IDS), 2)),
            )
            linear_rows.append(
                {
                    "sigma_theta_deg": sigma_deg,
                    "sigma_azimuth_deg": sigma_deg / np.sqrt(2.0),
                    "eta_c": eta,
                    "eta_r": eta,
                    "n_bootstrap": N_BOOTSTRAP,
                    "n_main": N_MAIN,
                    "trials": N_LINEAR_TRIALS,
                    "theory_reference_rms_d": theory["reference_rms_d"],
                    "theory_receiver_rms_d": theory["position_rms_d"],
                    "mc_reference_rms_d": float(
                        np.sqrt(np.mean(np.sum(simulation["final_reference_error"] ** 2, axis=1)))
                    ),
                    "mc_receiver_rms_d": simulation["position_rms_d"],
                    "mc_team_max_error_p99_d": simulation["position_p99_error_sample_d"],
                    "mc_team_max_error_sample_d": simulation["position_max_error_sample_d"],
                    "white_noise_steady_multiplier_eta_r": theory["white_noise_steady_multiplier"],
                    "white_noise_steady_multiplier_eta_c": theory["reference_white_noise_steady_multiplier"],
                    "shared_reference_covariance_trace_d2": float(
                        np.trace(theory["covariance_reference_error"])
                    ),
                }
            )
        # Keep one compact seed-independent archive for reproducibility.  Full
        # linear arrays are regenerated from this fixed master seed in tests.
        noise_archive[f"sigma_{sigma_deg:g}_base_seed"] = 20260906
    return linear_rows, contract_rows, noise_archive


def independent_gain_grid() -> list[dict[str, Any]]:
    """Enumerate all 4x4 ``(eta_C, eta_R)`` theory combinations."""

    rows: list[dict[str, Any]] = []
    initial_reference = np.array([0.08, -0.06], dtype=float)
    # Use the same nonzero receiver perturbation construction as the
    # nonlinear fixed-budget cases.  A zero receiver initial error would make
    # the grid answer only the steady-noise question and would understate the
    # transient cost of eta_R.
    _, initial_receivers = _initial_team(11, reference_error=initial_reference)
    for sigma_deg in SCENARIO_SIGMA_DEG:
        for eta_c in GAINS:
            for eta_r in GAINS:
                result = linear_theory(
                    sigma_deg, eta_c, eta_r, e_c0=initial_reference,
                    e_receiver0=initial_receivers,
                )
                rows.append({
                    "sigma_theta_deg": sigma_deg,
                    "eta_c": eta_c, "eta_r": eta_r,
                    "n_bootstrap": N_BOOTSTRAP, "n_main": N_MAIN,
                    "initial_receiver_error_rms_d": float(
                        np.sqrt(np.mean(np.sum(initial_receivers**2, axis=1)))
                    ),
                    "theory_reference_rms_d": result["reference_rms_d"],
                    "theory_receiver_rms_d": result["position_rms_d"],
                    "shared_reference_covariance_trace_d2": float(
                        np.trace(result["covariance_reference_error"])
                    ),
                    "white_noise_steady_multiplier_eta_c": result[
                        "reference_white_noise_steady_multiplier"
                    ],
                    "white_noise_steady_multiplier_eta_r": result[
                        "white_noise_steady_multiplier"
                    ],
                })
    return rows


def run_analysis(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Run all cheap and bounded nonlinear analyses and write artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    linear_rows, contract_rows, noise_archive = _linear_noise_tables()
    gain_grid_rows = independent_gain_grid()
    selection_rows = row_selection_comparison(0.05)
    nonlinear_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    nonlinear_traces: list[dict[str, Any]] = []
    input_archive: dict[str, Any] = {}
    initial_reference_error = np.array([0.08, -0.06], dtype=float)
    for seed, sigma_deg in NONLINEAR_CASES:
        base_noise, azimuth_noise = _noise_inputs_for_case(seed, sigma_deg)
        _, receiver_errors = _initial_team(seed, reference_error=initial_reference_error)
        input_archive[f"case_{seed}_{sigma_deg:g}_base"] = base_noise
        input_archive[f"case_{seed}_{sigma_deg:g}_azimuth"] = azimuth_noise
        input_archive[f"case_{seed}_{sigma_deg:g}_receiver_initial"] = receiver_errors
        for eta_c, eta_r in NONLINEAR_GAIN_PAIRS:
            result, failures, trace = nonlinear_fixed_budget(
                seed=seed,
                sigma_deg=sigma_deg,
                eta_c=eta_c,
                eta_r=eta_r,
                base_noise=base_noise,
                azimuth_noise=azimuth_noise,
                initial_reference_error=initial_reference_error,
                initial_receiver_errors=receiver_errors,
            )
            row = dict(result)
            row.pop("final_positions", None)
            nonlinear_rows.append(row)
            failure_rows.extend(failures)
            nonlinear_traces.append({
                "seed": seed, "sigma_theta_deg": sigma_deg,
                "eta_c": eta_c, "eta_r": eta_r,
                "summary": row,
                "final_positions": result["final_positions"],
                "trace": trace,
            })
    np.savez_compressed(output_dir / "nonlinear_noise_inputs.npz", **input_archive)
    _write_csv(output_dir / "noise_contract.csv", contract_rows)
    _write_csv(output_dir / "row_selection_comparison.csv", selection_rows)
    _write_csv(output_dir / "linear_monte_carlo.csv", linear_rows)
    _write_csv(output_dir / "independent_gain_grid.csv", gain_grid_rows)
    _write_csv(output_dir / "nonlinear_trajectories.csv", nonlinear_rows)
    _write_csv(output_dir / "nonlinear_failures.csv", failure_rows)
    (output_dir / "nonlinear_traces.json").write_text(
        json.dumps(_json_value(nonlinear_traces), indent=2,
                   ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    theory_example = linear_theory(0.05, 0.5, 0.5, e_c0=initial_reference_error)
    bootstrap = bootstrap_covariance(0.05)
    summary = {
        "contract": {
            "bootstrap": "每轮两底角独立 Gaussian；sigma_theta 为每条角行边际标准差",
            "main": "三条方位角先独立加 Gaussian，再按 0/pi fold 生成三条无向夹角；sigma_azimuth=sigma_theta/sqrt(2)",
            "selected_rows": "各接收机仅在线性协方差和局部控制更新中使用预选的两条光滑行；退化行保留 fold 观测和生产一致性失败",
            "execution": "坐标动作精确执行；无额外执行误差",
            "locking": "FY01 校准结束后固定；其随机误差是所有接收机的共同偏置源",
            "fixed_budget": f"n_bootstrap={N_BOOTSTRAP}, n_main={N_MAIN}; 不运行在线停止保证",
        },
        "gain_condition": {
            "linear_stability": "0 < eta_C < 2 且 0 < eta_R < 2",
            "local_scope": "目标附近、选定光滑行、连续调整、同一分支",
            "steady_white_noise_multiplier": "eta/(2-eta)",
            "interpretation": "固定参考残差不按白噪声稳态公式压缩；它进入共同固定偏置项",
        },
        "bootstrap_example_sigma_0.05deg": bootstrap,
        "example_theory_eta_0.5_sigma_0.05deg": theory_example,
        "linear_rows": len(linear_rows),
        "independent_gain_grid_rows": len(gain_grid_rows),
        "row_selection_rows": len(selection_rows),
        "row_selection_production_matches": {
            "trace": sum(bool(row["geometric_selector_matches_trace"]) for row in selection_rows if row["production_selected"]),
            "lambda_max": sum(bool(row["geometric_selector_matches_lambda_max"]) for row in selection_rows if row["production_selected"]),
        },
        "nonlinear_trajectories": len(nonlinear_rows),
        "nonlinear_failure_records": len(failure_rows),
        "nonlinear_failures_by_gain_pair": {
            f"{eta_c:g},{eta_r:g}": sum(
                row["eta_c"] == eta_c and row["eta_r"] == eta_r
                for row in failure_rows
            )
            for eta_c, eta_r in NONLINEAR_GAIN_PAIRS
        },
        "nonlinear_success_wording": "这些行都是固定预算轨迹；online_status 明确为 not_evaluated_fixed_budget，failure_count 保留生产 strict success 失败",
        "reproducibility": {
            "linear_master_seed": 20260906,
            "nonlinear_noise_file": "nonlinear_noise_inputs.npz",
            "nonlinear_cases": NONLINEAR_CASES,
            "nonlinear_gain_pairs": NONLINEAR_GAIN_PAIRS,
            "gains": GAINS,
        },
        "limits": [
            "线性协方差只在 canonical 局部支、固定参考、连续调整下成立",
            "样本最大误差是所运行 Monte Carlo 的样本统计，不是硬上界",
            "非线性轨迹仅为 30 个固定预算案例，不能替代全初态域或在线停止成功率",
            "原有 1e-8 全三角残差阈值不适合作为这些含噪观测的停止保证",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_value(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # The archive metadata is small and human-readable; the actual arrays are
    # in the compressed NPZ next to the CSV tables.
    (output_dir / "noise_archive.json").write_text(
        json.dumps(_json_value(noise_archive), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    summary = run_analysis(args.output_dir)
    print(json.dumps(_json_value(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
