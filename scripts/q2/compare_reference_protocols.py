"""Bounded comparison of staged and synchronised Q2 reference protocols.

The production runner deliberately keeps one gain for both stages and its
receiver solver is tied to the canonical anchors.  This audit composes its
private bootstrap and main-stage records to make the two gains explicit, then
implements the synchronised alternative locally.  The latter measures the
current FY01 position in every round and passes those three measured anchors
to a local receiver estimator; it never reads a moved position as an old
reference and never uses simulator truth at the estimator boundary.

Run with::

    conda run -n agent python -m scripts.q2.compare_reference_protocols

All coordinates are in the normalised template unit ``d``.  The default
comparison is noiseless and bounded to rho=.1, seeds 11/23/47, with the
position-budget online stop at .01d.  It also writes complete strict-angle
records so a relaxed position stop is never presented as a structural win.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.q2.run_triangle_reference import (  # noqa: E402
    C_INDEX,
    MAX_ROUNDS,
    _bootstrap_record,
    _json_value,
    _main_record,
    _write_csv,
    make_initial_state,
)
from scripts.q2.triangle_reference import (  # noqa: E402
    ANGLE_RESIDUAL_TOLERANCE,
    ANCHOR_PAIRS,
    RECEIVER_IDS,
    REFERENCE_IDS,
    TARGET_ANCHORS,
    TARGET_TEMPLATE,
    angle_jacobian,
    bootstrap_angles,
    bootstrap_from_angles,
    estimate_receiver,
    receiver_angles,
)


DEFAULT_OUTPUT = Path("outputs/q2/protocol_comparison")
RHO = 0.1
SEEDS = (11, 23, 47)
TAU_C_VALUES = (0.0048, 0.005)
GAIN_PAIRS = ((1.0, 1.0), (1.0, 0.75), (0.75, 0.75), (0.5, 0.5), (0.5, 1.0))
FINAL_POSITION_BUDGET = 0.01
FAIR_TAU_C = 0.005
FAIR_RECEIVER_ESTIMATE_BUDGET = 0.0045
STRICT_ANGLE_TOLERANCE = float(ANGLE_RESIDUAL_TOLERANCE)
MAX_NFEV = 500
# The local first-order FY01-to-receiver multiplier audited in
# docs/q2/参考残差传播与校准阈值.md.  It is reported as a budget diagnostic,
# not promoted to a nonlinear guarantee.
G_MAX_FIRST_ORDER = 1.040833


def _norm(value: Any) -> Any:
    """Convert numpy values recursively for JSON and CSV-friendly records."""

    return _json_value(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_norm(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _as_anchors(anchors: np.ndarray) -> np.ndarray:
    value = np.asarray(anchors, dtype=float)
    if value.shape != (3, 2) or not np.isfinite(value).all():
        raise ValueError("known anchors must have shape (3, 2) and be finite")
    return value


def _choose_local_rows(receiver_id: int, anchors: np.ndarray) -> tuple[int, int]:
    """Select two smooth rows at the template initial point for known anchors.

    This is intentionally a local branch: the initial point is the canonical
    template location for the labelled receiver, while the anchor geometry is
    the explicit current estimate supplied by the protocol.  Every candidate
    pair is scored by its minimum singular value, and the complete triangle is
    checked after solving.
    """

    point = TARGET_TEMPLATE[int(receiver_id) - 1]
    gradients = angle_jacobian(point, _as_anchors(anchors), allow_degenerate=True)
    candidates: list[tuple[float, tuple[int, int]]] = []
    for first in range(len(ANCHOR_PAIRS)):
        for second in range(first + 1, len(ANCHOR_PAIRS)):
            matrix = gradients[[first, second]]
            if not np.isfinite(matrix).all():
                continue
            sigma_min = float(np.linalg.svd(matrix, compute_uv=False)[-1])
            if sigma_min > 0.0:
                candidates.append((sigma_min, (first, second)))
    if not candidates:
        raise ValueError(f"FY{receiver_id:02d} has no smooth local two-row branch")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def estimate_receiver_with_anchors(
    receiver_id: int,
    observed3: np.ndarray,
    known_anchors: np.ndarray,
    *,
    initial: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate one receiver using explicit current anchors.

    The estimator boundary contains only labelled angles, three known anchor
    coordinates, and an optional local template initial point.  It does not
    access the true state.  Two rows chosen at that initial point are solved;
    all three rows are then evaluated and the same $10^{-8}$ rad acceptance
    contract as the production solver is applied.
    """

    identifier = int(receiver_id)
    observed = np.asarray(observed3, dtype=float)
    if identifier not in RECEIVER_IDS:
        raise ValueError("receiver_id must be a non-reference identifier")
    if observed.shape != (3,) or not np.isfinite(observed).all():
        raise ValueError("observed angles must have shape (3,) and be finite")
    if np.any(observed < 0.0) or np.any(observed > np.pi):
        raise ValueError("observed angles must lie in [0, pi]")
    anchors = _as_anchors(known_anchors)
    start = (
        TARGET_TEMPLATE[identifier - 1].copy()
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )
    if start.shape != (2,) or not np.isfinite(start).all():
        raise ValueError("initial must be one finite 2-D point")
    try:
        selected = _choose_local_rows(identifier, anchors)

        def residual(point: np.ndarray) -> np.ndarray:
            return receiver_angles(point, anchors)[list(selected)] - observed[list(selected)]

        def jacobian(point: np.ndarray) -> np.ndarray:
            rows = angle_jacobian(point, anchors, allow_degenerate=True)[list(selected)]
            if not np.isfinite(rows).all():
                raise ValueError("selected local row reached an angle singularity")
            return rows

        result = least_squares(
            residual,
            start,
            jac=jacobian,
            method="trf",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=MAX_NFEV,
        )
        position = np.asarray(result.x, dtype=float)
        predicted = receiver_angles(position, anchors)
        max_residual = float(np.max(np.abs(predicted - observed)))
        success = bool(result.success and max_residual <= STRICT_ANGLE_TOLERANCE)
        message = str(result.message)
        if result.success and not success:
            message = (
                f"solver converged but complete-triangle residual {max_residual:.3e} "
                f"exceeds {STRICT_ANGLE_TOLERANCE:.3e}"
            )
        return {
            "position": position,
            "success": success,
            "max_angle_residual": max_residual,
            "nfev": int(result.nfev),
            "selected_indices": list(selected),
            "message": message,
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return {
            "position": start,
            "success": False,
            "max_angle_residual": float("inf"),
            "nfev": 0,
            "selected_indices": [],
            "message": f"geometry/solver failure: {error}",
        }


def _initial_result(case: Any) -> dict[str, Any]:
    return {
        "rho": float(case.rho),
        "seed": case.seed if case.seed is not None else "ideal",
        "initial_max_error_d": float(case.normalized_max_deviation),
        "initial_rms_error_d": float(case.normalized_rms_deviation),
        "actual_spacing_d": float(case.actual_spacing),
    }


def _action_distance(actions: Iterable[dict[str, Any]]) -> float:
    return float(sum(np.hypot(float(row["delta_x"]), float(row["delta_y"])) for row in actions))


def _score_result(
    case: Any,
    state: np.ndarray,
    *,
    protocol: str,
    eta_c: float,
    eta_r: float,
    tau_c: float,
    stop_rule: str,
    measurement_slots: int,
    tx_uses: int,
    relay_scalars: int,
    rows: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    status: str,
    failures: list[str],
    bootstrap_summary: dict[str, Any] | None = None,
    main_summary: dict[str, Any] | None = None,
    receiver_estimate_budget: float | None = None,
    budget_group: str = "strict_angle",
    public_reference_scalars: int = 0,
) -> dict[str, Any]:
    errors = np.linalg.norm(np.asarray(state) - TARGET_TEMPLATE, axis=1)
    final_max_error = float(np.max(errors))
    budget_excess = max(0.0, final_max_error - FINAL_POSITION_BUDGET)
    budget_plus_reference = (
        None
        if receiver_estimate_budget is None
        else float(receiver_estimate_budget + G_MAX_FIRST_ORDER * tau_c)
    )
    return {
        **_initial_result(case),
        "protocol": protocol,
        "eta_c": float(eta_c),
        "eta_r": float(eta_r),
        "tau_c_d": float(tau_c),
        "final_position_budget_d": float(FINAL_POSITION_BUDGET),
        "receiver_estimate_budget_d": (
            None if receiver_estimate_budget is None else float(receiver_estimate_budget)
        ),
        "budget_group": budget_group,
        "stop_rule": stop_rule,
        "status": status,
        "online_status": "success" if status == "success" else "failure",
        "failures": sorted(set(failures)),
        "measurement_slots": int(measurement_slots),
        "tx_uses": int(tx_uses),
        "relay_scalar_count": int(relay_scalars),
        "public_reference_scalars": int(public_reference_scalars),
        "final_max_position_error_d": final_max_error,
        "final_position_budget_pass": bool(final_max_error <= FINAL_POSITION_BUDGET),
        "final_position_budget_excess_d": budget_excess,
        "first_order_reference_propagation_bound_d": float(G_MAX_FIRST_ORDER * tau_c),
        "online_budget_plus_reference_bound_d": budget_plus_reference,
        "final_rms_position_error_d": float(np.sqrt(np.mean(errors**2))),
        "cumulative_displacement_d": _action_distance(actions),
        "action_count": int(len(actions)),
        "bootstrap": bootstrap_summary or {},
        "main": main_summary or {},
        "rows": rows,
        "actions": actions,
        "final_positions": np.asarray(state),
    }


def run_staged_case(
    rho: float,
    seed: int,
    eta_c: float,
    eta_r: float,
    tau_c: float | None,
    *,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, Any]:
    """Run finite calibration followed by the production strict-angle stage."""

    case = make_initial_state(rho, seed)
    state, bootstrap, bootstrap_rows, bootstrap_actions = _bootstrap_record(
        case=case, gain=eta_c, max_rounds=max_rounds, position_tolerance=tau_c
    )
    bootstrap = {**bootstrap, "gain": float(eta_c)}
    failures: list[str] = []
    rows = [{"stage": "bootstrap", **row} for row in bootstrap_rows]
    actions = [{"stage": "bootstrap", **row} for row in bootstrap_actions]
    if bootstrap["status"] != "converged":
        failures.append(str(bootstrap.get("failure", "bootstrap_failure")))
        return _score_result(
            case,
            state,
            protocol="staged_fixed_reference",
            eta_c=eta_c,
            eta_r=eta_r,
            tau_c=0.0 if tau_c is None else tau_c,
            stop_rule="strict_main_angle_after_finite_calibration",
            measurement_slots=2 * int(bootstrap["rounds"]),
            tx_uses=int(bootstrap["tx_uses"]),
            relay_scalars=int(bootstrap["relay_scalar_count"]),
            rows=rows,
            actions=actions,
            status="failure",
            failures=failures,
            bootstrap_summary=bootstrap,
            receiver_estimate_budget=None,
            budget_group="strict_angle",
            public_reference_scalars=int(bootstrap["relay_scalar_count"]),
        )
    state, main, main_rows, main_actions = _main_record(
        case=case, gain=eta_r, state=state, max_rounds=max_rounds
    )
    main = {**main, "gain": float(eta_r)}
    rows.extend({"stage": "main", **row} for row in main_rows)
    actions.extend({"stage": "main", **row} for row in main_actions)
    if main["status"] != "converged":
        failures.extend(str(value) for value in main.get("failures", {}).values())
    return _score_result(
        case,
        state,
        protocol="staged_fixed_reference",
        eta_c=eta_c,
        eta_r=eta_r,
        tau_c=0.0 if tau_c is None else tau_c,
        stop_rule="strict_main_angle_after_finite_calibration",
        measurement_slots=2 * int(bootstrap["rounds"]) + int(main["broadcast_slots"]),
        tx_uses=int(bootstrap["tx_uses"] + main["tx_uses"]),
        relay_scalars=int(bootstrap["relay_scalar_count"]),
        rows=rows,
        actions=actions,
        status="success" if not failures else "failure",
        failures=failures,
        bootstrap_summary=bootstrap,
        main_summary=main,
        receiver_estimate_budget=None,
        budget_group="strict_angle",
        public_reference_scalars=int(bootstrap["relay_scalar_count"]),
    )


def _run_fixed_position_main(
    case: Any,
    state: np.ndarray,
    eta_r: float,
    max_rounds: int,
    position_budget: float,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Static-reference main stage with an observable estimated-position stop."""

    current = np.asarray(state, dtype=float).copy()
    active = set(RECEIVER_IDS)
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    failures: dict[int, str] = {}
    rounds = 0
    for round_i in range(1, max_rounds + 1):
        if not active:
            break
        rounds += 1
        observed: dict[int, np.ndarray] = {}
        pending: list[int] = []
        for receiver_id in sorted(active):
            angles = np.asarray(
                receiver_angles(
                    current[receiver_id - 1], current[np.asarray(REFERENCE_IDS) - 1]
                ),
                dtype=float,
            )
            observed[receiver_id] = angles
            rows.append(
                {
                    "stage": "main_budget",
                    "round": round_i,
                    "receiver_id": receiver_id,
                    "measurement_slot": round_i,
                    "measurement_tx_count": 3,
                    "angle_error_to_target_rad": float(
                        np.max(np.abs(angles - receiver_angles(TARGET_TEMPLATE[receiver_id - 1], TARGET_ANCHORS)))
                    ),
                    "estimated_x": "",
                    "estimated_y": "",
                    "estimated_position_error_d": "",
                    "estimator_success": "",
                    "estimator_residual_rad": "",
                    "delta_x": 0.0,
                    "delta_y": 0.0,
                    "action_applied": False,
                    "event": "observe",
                    "failure": "",
                }
            )
            pending.append(receiver_id)
        for receiver_id in pending:
            estimate = estimate_receiver(
                receiver_id, observed[receiver_id], initial=TARGET_TEMPLATE[receiver_id - 1]
            )
            row = next(
                item
                for item in reversed(rows)
                if item["stage"] == "main_budget"
                and item["round"] == round_i
                and item["receiver_id"] == receiver_id
            )
            position = np.asarray(estimate["position"], dtype=float)
            row["estimated_x"] = float(position[0])
            row["estimated_y"] = float(position[1])
            row["estimated_position_error_d"] = float(
                np.linalg.norm(position - TARGET_TEMPLATE[receiver_id - 1])
            )
            row["estimator_success"] = bool(estimate["success"])
            row["estimator_residual_rad"] = float(estimate["max_angle_residual"])
            if not bool(estimate["success"]):
                failure = f"receiver_estimation_failure:{estimate['message']}"
                row["event"] = "failure"
                row["failure"] = failure
                failures[receiver_id] = failure
                active.remove(receiver_id)
                continue
            if row["estimated_position_error_d"] <= position_budget:
                row["event"] = "stop_estimated_position_budget"
                active.remove(receiver_id)
                continue
            delta = -float(eta_r) * (position - TARGET_TEMPLATE[receiver_id - 1])
            row["delta_x"] = float(delta[0])
            row["delta_y"] = float(delta[1])
            row["action_applied"] = True
            row["event"] = "observe_and_move"
            current[receiver_id - 1] += delta
            actions.append(
                {
                    "stage": "main_budget",
                    "round": round_i,
                    "receiver_id": receiver_id,
                    "delta_x": float(delta[0]),
                    "delta_y": float(delta[1]),
                    "true_position_after_x": float(current[receiver_id - 1, 0]),
                    "true_position_after_y": float(current[receiver_id - 1, 1]),
                }
            )
    for receiver_id in sorted(active):
        failures[receiver_id] = "receiver_budget_exhausted"
    final_errors = {
        receiver_id: float(
            np.max(
                np.abs(
                    receiver_angles(
                        current[receiver_id - 1], current[np.asarray(REFERENCE_IDS) - 1]
                    )
                    - receiver_angles(TARGET_TEMPLATE[receiver_id - 1], TARGET_ANCHORS)
                )
            )
        )
        for receiver_id in RECEIVER_IDS
    }
    summary = {
        "status": "converged" if not failures else "failure",
        "failure_count": len(failures),
        "failures": {str(key): value for key, value in failures.items()},
        "broadcast_slots": rounds,
        "tx_uses": rounds * 3,
        "stop_rule": "estimated_receiver_position",
        "position_budget_d": float(position_budget),
        "final_max_angle_error_rad": max(final_errors.values()),
        "receiver_status": {
            str(identifier): ("failure" if identifier in failures else "converged")
            for identifier in RECEIVER_IDS
        },
    }
    return current, summary, rows, actions


def run_staged_budget_case(
    rho: float,
    seed: int,
    eta_c: float,
    eta_r: float,
    tau_c: float,
    *,
    max_rounds: int = MAX_ROUNDS,
    position_budget: float = FINAL_POSITION_BUDGET,
    budget_group: str = "unallocated_reference_budget",
) -> dict[str, Any]:
    """Run staged finite calibration with the common observable position stop."""

    case = make_initial_state(rho, seed)
    state, bootstrap, bootstrap_rows, bootstrap_actions = _bootstrap_record(
        case=case, gain=eta_c, max_rounds=max_rounds, position_tolerance=tau_c
    )
    bootstrap = {**bootstrap, "gain": float(eta_c)}
    rows = [{"stage": "bootstrap", **row} for row in bootstrap_rows]
    actions = [{"stage": "bootstrap", **row} for row in bootstrap_actions]
    failures: list[str] = []
    main: dict[str, Any] = {}
    if bootstrap["status"] == "converged":
        state, main, main_rows, main_actions = _run_fixed_position_main(
            case, state, eta_r, max_rounds, position_budget
        )
        main = {**main, "gain": float(eta_r)}
        rows.extend(main_rows)
        actions.extend(main_actions)
        failures.extend(str(value) for value in main.get("failures", {}).values())
    else:
        failures.append(str(bootstrap.get("failure", "bootstrap_failure")))
    return _score_result(
        case,
        state,
        protocol="staged_fixed_reference",
        eta_c=eta_c,
        eta_r=eta_r,
        tau_c=0.0 if tau_c is None else tau_c,
        stop_rule="estimated_position_budget_after_finite_calibration",
        measurement_slots=2 * int(bootstrap["rounds"]) + int(main.get("broadcast_slots", 0)),
        tx_uses=int(bootstrap["tx_uses"] + main.get("tx_uses", 0)),
        relay_scalars=int(bootstrap["relay_scalar_count"]),
        rows=rows,
        actions=actions,
        status="success" if not failures else "failure",
        failures=failures,
        bootstrap_summary=bootstrap,
        main_summary=main,
        receiver_estimate_budget=position_budget,
        budget_group=budget_group,
        public_reference_scalars=int(bootstrap["relay_scalar_count"]),
    )


def run_dynamic_case(
    rho: float,
    seed: int,
    eta_c: float,
    eta_r: float,
    tau_c: float,
    *,
    max_rounds: int = MAX_ROUNDS,
    position_budget: float = FINAL_POSITION_BUDGET,
    budget_group: str = "unallocated_reference_budget",
) -> dict[str, Any]:
    """Run the synchronised current-reference protocol.

    At each round every node is stationary during two bottom-angle slots and
    one (1,11,15) broadcast slot.  The two measured angles yield ``C_hat``;
    the three-anchor estimator receives ``[C_hat, FY11, FY15]``.  Only after
    all estimates have been computed are FY01 and all active receivers moved
    together.  The stop uses observable ``C_hat`` and receiver estimates.
    """

    case = make_initial_state(rho, seed)
    current = case.positions.copy()
    active_receivers = set(RECEIVER_IDS)
    calibration_active = True
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    failures: list[str] = []
    c_hat_error = float("inf")
    receiver_estimate_errors: dict[int, float] = {}
    round_summaries: list[dict[str, Any]] = []

    for round_i in range(1, max_rounds + 1):
        anchors_true = current[np.asarray(REFERENCE_IDS) - 1].copy()
        observed_bootstrap = np.asarray(bootstrap_angles(current), dtype=float)
        try:
            c_hat = np.asarray(bootstrap_from_angles(observed_bootstrap), dtype=float)
        except (ValueError, FloatingPointError) as error:
            failures.append(f"bootstrap_estimation_failure:{error}")
            break
        c_hat_error = float(np.linalg.norm(c_hat - TARGET_ANCHORS[0]))
        rows.append(
            {
                "stage": "dynamic_calibration",
                "round": round_i,
                "measurement_slot_count": 2,
                "measurement_tx_count": 4,
                "relay_scalar_count": 2,
                "alpha_rad": float(observed_bootstrap[0]),
                "beta_rad": float(observed_bootstrap[1]),
                "c_hat_x": float(c_hat[0]),
                "c_hat_y": float(c_hat[1]),
                "c_hat_error_to_target_d": c_hat_error,
                "event": "measure_current_reference",
                "failure": "",
            }
        )
        # FY11/FY15 are fixed by the coordinate contract and therefore use
        # their known canonical coordinates; only FY01 is inferred this round.
        known_anchors = np.vstack((c_hat, TARGET_ANCHORS[1], TARGET_ANCHORS[2]))
        pending: list[tuple[int, dict[str, Any]]] = []
        current_receiver_estimates: dict[int, np.ndarray] = {}
        receiver_failures: dict[int, str] = {}
        # All receiver observations are generated before any action is applied.
        for receiver_id in sorted(active_receivers):
            observed3 = np.asarray(
                receiver_angles(current[receiver_id - 1], anchors_true), dtype=float
            )
            result = estimate_receiver_with_anchors(
                receiver_id,
                observed3,
                known_anchors,
                initial=TARGET_TEMPLATE[receiver_id - 1],
            )
            estimate_position = np.asarray(result["position"], dtype=float)
            estimate_error = float(
                np.linalg.norm(estimate_position - TARGET_TEMPLATE[receiver_id - 1])
            )
            receiver_estimate_errors[receiver_id] = estimate_error
            row = {
                "stage": "dynamic_main",
                "round": round_i,
                "receiver_id": receiver_id,
                "measurement_slot_count": 1,
                "measurement_tx_count": 3,
                "c_hat_x_used": float(c_hat[0]),
                "c_hat_y_used": float(c_hat[1]),
                "angle_error_to_target_rad": float(
                    np.max(
                        np.abs(
                            observed3
                            - receiver_angles(
                                TARGET_TEMPLATE[receiver_id - 1], TARGET_ANCHORS
                            )
                        )
                    )
                ),
                "estimated_x": float(estimate_position[0]),
                "estimated_y": float(estimate_position[1]),
                "estimated_position_error_d": estimate_error,
                "estimator_success": bool(result["success"]),
                "estimator_residual_rad": float(result["max_angle_residual"]),
                "selected_indices": ";".join(map(str, result.get("selected_indices", []))),
                "delta_x": 0.0,
                "delta_y": 0.0,
                "action_applied": False,
                "event": "observe",
                "failure": "",
            }
            rows.append(row)
            pending.append((receiver_id, result))
            current_receiver_estimates[receiver_id] = estimate_position
            if not result["success"]:
                failure = f"receiver_estimation_failure:{result['message']}"
                row["event"] = "failure"
                row["failure"] = failure
                receiver_failures[receiver_id] = failure

        failures.extend(receiver_failures.values())
        # Observable stop is checked before the synchronized move.  A receiver
        # that already meets the budget is held; all remaining actions are still
        # applied together after the same three-slot round.
        for receiver_id in receiver_failures:
            active_receivers.discard(receiver_id)
        receiver_budget_met = {
            receiver_id
            for receiver_id, position in current_receiver_estimates.items()
            if receiver_id in active_receivers
            and np.linalg.norm(position - TARGET_TEMPLATE[receiver_id - 1])
            <= position_budget
        }
        for receiver_id in receiver_budget_met:
            active_receivers.discard(receiver_id)
            row = next(
                item
                for item in reversed(rows)
                if item.get("stage") == "dynamic_main"
                and item.get("round") == round_i
                and item.get("receiver_id") == receiver_id
            )
            row["event"] = "stop_estimated_position_budget"

        calibration_active = c_hat_error > tau_c
        if not calibration_active and not active_receivers:
            round_summaries.append(
                {
                    "round": round_i,
                    "c_hat_error_to_target_d": c_hat_error,
                    "receiver_budget_met_count": len(receiver_budget_met),
                    "calibration_action": False,
                    "receiver_action_count": 0,
                    "event": "stop_after_observable_verification",
                }
            )
            break

        # Apply all synchronized actions only after the complete round is
        # observed.  FY01 uses the current C_hat; no true C is read by a
        # controller and no previous C_hat survives into the next round.
        round_actions = 0
        if calibration_active:
            delta_c = -float(eta_c) * (c_hat - TARGET_ANCHORS[0])
            current[C_INDEX] += delta_c
            round_actions += 1
            actions.append(
                {
                    "stage": "dynamic_calibration",
                    "round": round_i,
                    "receiver_id": 1,
                    "delta_x": float(delta_c[0]),
                    "delta_y": float(delta_c[1]),
                    "true_position_after_x": float(current[C_INDEX, 0]),
                    "true_position_after_y": float(current[C_INDEX, 1]),
                    "source": "current_C_hat",
                }
            )
            row = next(
                item
                for item in reversed(rows)
                if item.get("stage") == "dynamic_calibration" and item.get("round") == round_i
            )
            row["calibration_delta_x"] = float(delta_c[0])
            row["calibration_delta_y"] = float(delta_c[1])
            row["calibration_action_applied"] = True
        else:
            row = next(
                item
                for item in reversed(rows)
                if item.get("stage") == "dynamic_calibration" and item.get("round") == round_i
            )
            row["calibration_delta_x"] = 0.0
            row["calibration_delta_y"] = 0.0
            row["calibration_action_applied"] = False
        for receiver_id, result in pending:
            if receiver_id not in active_receivers:
                continue
            if not result["success"]:
                continue
            estimate_position = current_receiver_estimates[receiver_id]
            delta = -float(eta_r) * (
                estimate_position - TARGET_TEMPLATE[receiver_id - 1]
            )
            current[receiver_id - 1] += delta
            round_actions += 1
            actions.append(
                {
                    "stage": "dynamic_main",
                    "round": round_i,
                    "receiver_id": receiver_id,
                    "delta_x": float(delta[0]),
                    "delta_y": float(delta[1]),
                    "true_position_after_x": float(current[receiver_id - 1, 0]),
                    "true_position_after_y": float(current[receiver_id - 1, 1]),
                    "source": "known_current_anchors",
                }
            )
            row = next(
                item
                for item in reversed(rows)
                if item.get("stage") == "dynamic_main"
                and item.get("round") == round_i
                and item.get("receiver_id") == receiver_id
            )
            row["delta_x"] = float(delta[0])
            row["delta_y"] = float(delta[1])
            row["action_applied"] = True
            row["event"] = "observe_and_move"
        round_summaries.append(
            {
                "round": round_i,
                "c_hat_error_to_target_d": c_hat_error,
                "receiver_budget_met_count": len(receiver_budget_met),
                "calibration_action": calibration_active,
                "receiver_action_count": round_actions - int(calibration_active),
                "event": "observe_and_synchronize_move",
            }
        )
    else:
        failures.append("global_budget_exhausted")

    if not failures and not calibration_active and not active_receivers:
        status = "success"
    else:
        status = "failure"
    summary = {
        "status": status,
        "rounds": len(round_summaries),
        "stop_rule": "c_hat_position_and_receiver_estimated_position_budget",
        "position_budget_d": float(position_budget),
        "tau_c_d": float(tau_c),
        "broadcast_slots": len(round_summaries),
        "measurement_slots": len(round_summaries) * 3,
        "tx_uses": len(round_summaries) * 7,
        "relay_scalar_count": len(round_summaries) * 2,
        "public_reference_scalars": len(round_summaries) * 2,
        "failures": failures,
        "rounds_detail": round_summaries,
        "final_c_hat_error_d": c_hat_error,
    }
    return _score_result(
        case,
        current,
        protocol="dynamic_current_reference",
        eta_c=eta_c,
        eta_r=eta_r,
        tau_c=tau_c,
        stop_rule="c_hat_position_and_receiver_estimated_position_budget",
        measurement_slots=int(summary["measurement_slots"]),
        tx_uses=int(summary["tx_uses"]),
        relay_scalars=int(summary["relay_scalar_count"]),
        rows=rows,
        actions=actions,
        status=status,
        failures=failures,
        main_summary=summary,
        receiver_estimate_budget=position_budget,
        budget_group=budget_group,
        public_reference_scalars=int(summary["relay_scalar_count"]),
    )


def run_comparison(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    rho: float = RHO,
    seeds: tuple[int, ...] = SEEDS,
    tau_values: tuple[float, ...] = TAU_C_VALUES,
    gain_pairs: tuple[tuple[float, float], ...] = GAIN_PAIRS,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, Any]:
    """Run bounded strict, fair-budget staged, and dynamic comparisons."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    case_index = 0

    def add(result: dict[str, Any], *, record_kind: str) -> None:
        nonlocal case_index
        case_index += 1
        seed_value = result["seed"]
        reconstructed = make_initial_state(
            rho,
            None if seed_value == "ideal" else int(seed_value),
        ).positions.copy()
        for action in result["actions"]:
            reconstructed[int(action["receiver_id"]) - 1] += np.asarray(
                [float(action["delta_x"]), float(action["delta_y"])]
            )
        np.testing.assert_allclose(
            reconstructed,
            result["final_positions"],
            atol=1e-12,
            rtol=0.0,
            err_msg=f"action reconstruction failed for case {case_index}",
        )
        case_dir = output_dir / "cases" / f"case_{case_index:03d}_{record_kind}"
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "summary.json", {k: v for k, v in result.items() if k not in {"rows", "actions", "final_positions"}})
        np.savetxt(case_dir / "final_positions.csv", result["final_positions"], delimiter=",")
        _write_csv(case_dir / "observations.csv", result["rows"])
        _write_csv(case_dir / "actions.csv", result["actions"])
        row = {
            k: v
            for k, v in result.items()
            if k not in {"bootstrap", "main", "rows", "actions", "final_positions", "failures"}
        }
        row["record_kind"] = record_kind
        row["failures_json"] = json.dumps(result["failures"], ensure_ascii=False, separators=(",", ":"))
        row["bootstrap_summary_json"] = json.dumps(_norm(result.get("bootstrap", {})), ensure_ascii=False, separators=(",", ":"))
        row["main_summary_json"] = json.dumps(_norm(result.get("main", {})), ensure_ascii=False, separators=(",", ":"))
        row["case_dir"] = str(case_dir)
        rows.append(row)
        details.append({"case_dir": str(case_dir), "protocol": result["protocol"], "record_kind": record_kind})

    # Original strict main-angle contract, one unit-gain reference case per seed.
    for seed in seeds:
        add(
            run_staged_case(rho, seed, 1.0, 1.0, None, max_rounds=max_rounds),
            record_kind="strict_angle",
        )
    for tau_c in tau_values:
        for eta_c, eta_r in gain_pairs:
            for seed in seeds:
                add(
                    run_staged_budget_case(
                        rho,
                        seed,
                        eta_c,
                        eta_r,
                        tau_c,
                        max_rounds=max_rounds,
                    ),
                    record_kind="unallocated_reference_budget",
                )
                add(
                    run_dynamic_case(
                        rho,
                        seed,
                        eta_c,
                        eta_r,
                        tau_c,
                        max_rounds=max_rounds,
                    ),
                    record_kind="unallocated_reference_budget",
                )
    # Formal common-budget group: the receiver estimate gets only the
    # remaining .0045d after tau_C=.005d and a small nonlinear margin.  Both
    # protocols use this same observable receiver threshold; final true Emax
    # is still reported separately and is the only .01d acceptance claim.
    for eta_c, eta_r in gain_pairs:
        for seed in seeds:
            add(
                run_staged_budget_case(
                    rho,
                    seed,
                    eta_c,
                    eta_r,
                    FAIR_TAU_C,
                    max_rounds=max_rounds,
                    position_budget=FAIR_RECEIVER_ESTIMATE_BUDGET,
                    budget_group="fair_shared_position_budget",
                ),
                record_kind="fair_shared_position_budget",
            )
            add(
                run_dynamic_case(
                    rho,
                    seed,
                    eta_c,
                    eta_r,
                    FAIR_TAU_C,
                    max_rounds=max_rounds,
                    position_budget=FAIR_RECEIVER_ESTIMATE_BUDGET,
                    budget_group="fair_shared_position_budget",
                ),
                record_kind="fair_shared_position_budget",
            )
    _write_csv(output_dir / "protocol_comparison.csv", rows)
    summary = {
        "contract": {
            "rho": rho,
            "seeds": list(seeds),
            "tau_c_values_d": list(tau_values),
            "gain_pairs_eta_c_eta_r": [list(pair) for pair in gain_pairs],
            "final_position_budget_d": FINAL_POSITION_BUDGET,
            "fair_shared_budget": {
                "tau_c_d": FAIR_TAU_C,
                "receiver_estimate_budget_d": FAIR_RECEIVER_ESTIMATE_BUDGET,
                "first_order_reference_bound_d": G_MAX_FIRST_ORDER * FAIR_TAU_C,
                "nonlinear_margin_d": 0.0002,
                "budget_formula": "0.01 - 1.040833*0.005 - 0.0002 is conservatively rounded to 0.0045",
            },
            "strict_angle_tolerance_rad": STRICT_ANGLE_TOLERANCE,
            "max_rounds_per_stage": max_rounds,
            "units": "normalised d; no metre scale inferred from angles",
        },
        "accounting": {
            "bootstrap_round": {
                "measurement_slots": 2,
                "transmitter_uses": 4,
                "relay_scalars": 2,
            },
            "fixed_main_broadcast": {"measurement_slots": 1, "transmitter_uses": 3},
            "dynamic_round": {
                "measurement_slots": 3,
                "transmitter_uses": 7,
                "relay_scalars": 2,
                "explanation": "two bottom-angle slots (FY01+one bottom corner each) plus one (1,11,15) broadcast",
            },
            "static_full_protocol": "2*n_C+n_R slots, 4*n_C+3*n_R transmitter uses",
            "dynamic_full_protocol": "3*n_dynamic slots, 7*n_dynamic transmitter uses, 2*n_dynamic public scalars",
            "dynamic_payload_visibility": "the 2 current C_hat coordinates are counted separately as public_reference_scalars; no extra physical slot is invented",
        },
        "run_count": len(rows),
        "success_count": sum(row["status"] == "success" for row in rows),
        "failure_count": sum(row["status"] != "success" for row in rows),
        "success_count_meaning": "online protocol stop only; actual final position budget is counted separately",
        "online_success_count": sum(row["online_status"] == "success" for row in rows),
        "final_position_budget_pass_count": sum(row["final_position_budget_pass"] for row in rows),
        "final_position_budget_fail_count": sum(not row["final_position_budget_pass"] for row in rows),
        "actual_budget_by_group": {
            group: {
                "runs": sum(row["record_kind"] == group for row in rows),
                "passed": sum(row["record_kind"] == group and row["final_position_budget_pass"] for row in rows),
                "failed": sum(row["record_kind"] == group and not row["final_position_budget_pass"] for row in rows),
            }
            for group in ("strict_angle", "unallocated_reference_budget", "fair_shared_position_budget")
        },
        "strict_angle_records": sum(row["record_kind"] == "strict_angle" for row in rows),
        "record_groups": {
            "strict_angle": sum(row["record_kind"] == "strict_angle" for row in rows),
            "unallocated_reference_budget": sum(
                row["record_kind"] == "unallocated_reference_budget" for row in rows
            ),
            "fair_shared_position_budget": sum(
                row["record_kind"] == "fair_shared_position_budget" for row in rows
            ),
        },
        "details": details,
        "decision": {
            "mainline": "staged_fixed_reference_with_finite_tau_C_then_12_parallel_receivers",
            "dynamic_role": "bounded comparison; current-anchor remeasurement pays 7 transmitter uses and 2 public scalars per round",
            "interpretation": "A dynamic synchronized round can match the six-slot strict unit-gain static route in the exact model, but it has no cost advantage there; its value is conditional on avoiding a long finite calibration or on reference motion requiring fresh anchors.",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    args = parser.parse_args()
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    summary = run_comparison(args.output_dir, max_rounds=args.max_rounds)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "run_count": summary["run_count"],
                "online_success_count": summary["online_success_count"],
                "online_failure_count": summary["failure_count"],
                "final_position_budget_pass_count": summary["final_position_budget_pass_count"],
                "final_position_budget_fail_count": summary["final_position_budget_fail_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
