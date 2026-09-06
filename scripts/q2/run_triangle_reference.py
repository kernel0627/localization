"""Run the bounded noiseless Q2 triangle-reference closed loop.

The simulator owns the true 15-point state and applies ideal normalized
coordinate moves.  The estimator receives only angle observations and the
canonical template through ``scripts.q2.triangle_reference``.  This file is
kept as a batch runner so the geometry implementation remains reusable and
the truth/estimation boundary is visible in the recorded tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Permit both ``python -m scripts.q2.run_triangle_reference`` and the direct
# repository entry point ``python scripts/q2/run_triangle_reference.py``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.q2.triangle_reference import (
    ANGLE_RESIDUAL_TOLERANCE,
    RECEIVER_IDS,
    REFERENCE_IDS,
    TARGET_ANCHORS,
    TARGET_TEMPLATE,
    bootstrap_angles,
    estimate_apex,
    estimate_receiver,
    receiver_angles,
    template,
)


DEFAULT_OUTPUT = Path("outputs/q2/triangle_reference")
PERTURBATION_LEVELS = (0.0, 0.01, 0.05, 0.10, 0.20)
SEEDS = (11, 23, 47)
GAINS = (1.0, 0.5)
MAX_ROUNDS = 30
ANGLE_TOLERANCE = float(ANGLE_RESIDUAL_TOLERANCE)
SHAPE_TOLERANCE = 1e-6
A_ID, B_ID, C_ID = 11, 15, 1
A_INDEX, B_INDEX, C_INDEX = A_ID - 1, B_ID - 1, C_ID - 1


@dataclass(frozen=True)
class InitialState:
    """One perturbed, AB-normalized true state and its construction metrics."""

    positions: np.ndarray
    rho: float
    seed: int | None
    actual_spacing: float
    raw_base_length: float
    raw_max_perturbation: float
    normalized_max_deviation: float
    normalized_rms_deviation: float
    mirrored_apex_side: bool


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _normalize_by_ab(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Set A=(0,0), B=(4,0), retaining the orientation of the input."""

    value = np.asarray(points, dtype=float)
    base = value[B_INDEX] - value[A_INDEX]
    length = float(np.linalg.norm(base))
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("perturbed A/B pair is degenerate")
    ex = base / length
    ey = np.array([-ex[1], ex[0]], dtype=float)
    vectors = value - value[A_INDEX]
    coordinates = np.column_stack((vectors @ ex, vectors @ ey))
    normalized = coordinates * (4.0 / length)
    normalized[A_INDEX] = (0.0, 0.0)
    normalized[B_INDEX] = (4.0, 0.0)
    return normalized, length / 4.0


def make_initial_state(rho: float, seed: int | None) -> InitialState:
    """Perturb every point, then let the simulator normalize by its actual AB."""

    if rho < 0 or not np.isfinite(rho):
        raise ValueError("rho must be finite and nonnegative")
    target = template()
    if rho == 0.0:
        raw = target.copy()
        seed_value = None
        perturbation = np.zeros_like(raw)
    else:
        if seed is None:
            raise ValueError("a positive rho requires a fixed seed")
        rng = np.random.default_rng(seed)
        perturbation = rng.normal(size=target.shape)
        row_norms = np.linalg.norm(perturbation, axis=1)
        perturbation /= float(np.max(row_norms))
        perturbation *= float(rho)
        raw = target + perturbation
        seed_value = int(seed)
    normalized, actual_spacing = _normalize_by_ab(raw)
    normalized_delta = normalized - target
    signed_side = float(
        np.cross(
            normalized[B_INDEX] - normalized[A_INDEX],
            normalized[C_INDEX] - normalized[A_INDEX],
        )
    )
    return InitialState(
        positions=normalized,
        rho=float(rho),
        seed=seed_value,
        actual_spacing=float(actual_spacing),
        raw_base_length=float(4.0 * actual_spacing),
        raw_max_perturbation=float(np.max(np.linalg.norm(perturbation, axis=1))),
        normalized_max_deviation=float(np.max(np.linalg.norm(normalized_delta, axis=1))),
        normalized_rms_deviation=float(
            np.sqrt(np.mean(np.sum(normalized_delta**2, axis=1)))
        ),
        mirrored_apex_side=bool(signed_side < 0.0),
    )


def _bootstrap_target_angles() -> np.ndarray:
    return np.asarray(
        [
            # bootstrap_angles accepts a full state; this keeps the target
            # contract in one place and avoids a second angle implementation.
            *bootstrap_angles(TARGET_TEMPLATE),
        ],
        dtype=float,
    )


def _receiver_target_angles(receiver_id: int) -> np.ndarray:
    return receiver_angles(TARGET_TEMPLATE[receiver_id - 1], TARGET_ANCHORS)


def _bootstrap_record(
    *,
    case: InitialState,
    gain: float,
    max_rounds: int,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run bootstrap, returning the post-bootstrap state and audit rows."""

    state = case.positions.copy()
    target_angles = _bootstrap_target_angles()
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    action_count = 0
    relay_scalars = 0
    status = "budget_exhausted"
    stop_round: int | None = None
    failure: str | None = None
    for round_index in range(1, max_rounds + 1):
        observed = np.asarray(bootstrap_angles(state), dtype=float)
        estimate: np.ndarray | None = None
        estimate_error: str | None = None
        try:
            estimate = np.asarray(estimate_apex(observed), dtype=float)
        except (ValueError, FloatingPointError) as error:
            estimate_error = str(error)
        angle_error = float(np.max(np.abs(observed - target_angles)))
        relay_scalars += 2
        row = {
            "rho": case.rho,
            "seed": case.seed if case.seed is not None else "ideal",
            "gain": gain,
            "round": round_index,
            "alpha_rad": float(observed[0]),
            "beta_rad": float(observed[1]),
            "target_alpha_rad": float(target_angles[0]),
            "target_beta_rad": float(target_angles[1]),
            "max_angle_error_rad": angle_error,
            "relay_scalar_count": 2,
            "measurement_slot_count": 2,
            "measurement_tx_count": 4,
            "estimate_x": "",
            "estimate_y": "",
            "action_x": 0.0,
            "action_y": 0.0,
            "action_applied": False,
            "event": "observe",
            "failure": "",
        }
        if estimate is not None:
            row["estimate_x"] = float(estimate[0])
            row["estimate_y"] = float(estimate[1])
        if estimate_error is not None:
            row["failure"] = f"bootstrap_estimation_failure:{estimate_error}"
            rows.append(row)
            failure = row["failure"]
            status = "failure"
            break
        if angle_error <= ANGLE_TOLERANCE:
            row["event"] = "stop_angle_threshold"
            rows.append(row)
            # The online protocol has only the two unsigned angles.  Side or
            # mirror classification is recorded offline from simulator truth
            # below and is never used to select this stop.
            status = "converged"
            stop_round = round_index
            break
        delta = -float(gain) * (estimate - TARGET_TEMPLATE[C_INDEX])
        state[C_INDEX] += delta
        state[A_INDEX] = TARGET_ANCHORS[1]
        state[B_INDEX] = TARGET_ANCHORS[2]
        action_count += 1
        action_row = {
            "rho": case.rho,
            "seed": case.seed if case.seed is not None else "ideal",
            "gain": gain,
            "round": round_index,
            "receiver_id": C_ID,
            "delta_x": float(delta[0]),
            "delta_y": float(delta[1]),
            "true_position_after_x": float(state[C_INDEX, 0]),
            "true_position_after_y": float(state[C_INDEX, 1]),
            "action_number": action_count,
        }
        actions.append(action_row)
        row["action_x"] = float(delta[0])
        row["action_y"] = float(delta[1])
        row["action_applied"] = True
        row["event"] = "observe_and_move"
        rows.append(row)
    else:
        failure = "bootstrap_budget_exhausted"
        status = "failure"
    summary = {
        "status": status,
        "failure": failure,
        "rounds": len(rows),
        "stop_round": stop_round,
        "action_count": action_count,
        "relay_scalar_count": relay_scalars,
        "angle_slots": len(rows) * 2,
        # Each bootstrap observation has two physical slots (a and b), and
        # each slot uses C plus one fixed endpoint: 2 + 2 transmitter uses.
        "tx_uses": len(rows) * 4,
        "final_angle_error_rad": float(np.max(np.abs(bootstrap_angles(state) - target_angles))),
        "final_apex_x": float(state[C_INDEX, 0]),
        "final_apex_y": float(state[C_INDEX, 1]),
    }
    return state, summary, rows, actions


def _main_record(
    *,
    case: InitialState,
    gain: float,
    state: np.ndarray,
    max_rounds: int,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the simultaneous ABC broadcast loop for all 12 receivers."""

    current = state.copy()
    active = set(RECEIVER_IDS)
    records: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    per_receiver_rounds = {identifier: 0 for identifier in RECEIVER_IDS}
    per_receiver_actions = {identifier: 0 for identifier in RECEIVER_IDS}
    per_receiver_status = {identifier: "active" for identifier in RECEIVER_IDS}
    failures: dict[int, str] = {}
    broadcast_slots = 0
    for round_index in range(1, max_rounds + 1):
        if not active:
            break
        broadcast_slots += 1
        observed: dict[int, np.ndarray] = {}
        pending: list[int] = []
        for receiver_id in sorted(active):
            per_receiver_rounds[receiver_id] += 1
            position = current[receiver_id - 1]
            anchors = current[np.asarray(REFERENCE_IDS) - 1]
            angles = np.asarray(receiver_angles(position, anchors), dtype=float)
            target_angles = _receiver_target_angles(receiver_id)
            angle_error = float(np.max(np.abs(angles - target_angles)))
            observed[receiver_id] = angles
            row = {
                "rho": case.rho,
                "seed": case.seed if case.seed is not None else "ideal",
                "gain": gain,
                "round": round_index,
                "receiver_id": receiver_id,
                "angle_01_rad": float(angles[0]),
                "angle_02_rad": float(angles[1]),
                "angle_12_rad": float(angles[2]),
                "target_angle_01_rad": float(target_angles[0]),
                "target_angle_02_rad": float(target_angles[1]),
                "target_angle_12_rad": float(target_angles[2]),
                "max_angle_error_rad": angle_error,
                "measurement_slot": broadcast_slots,
                "measurement_tx_count": 3,
                "estimated_x": "",
                "estimated_y": "",
                "estimator_residual_rad": "",
                "estimator_success": "",
                "nfev": "",
                "delta_x": 0.0,
                "delta_y": 0.0,
                "action_applied": False,
                "event": "observe",
                "failure": "",
            }
            if angle_error <= ANGLE_TOLERANCE:
                row["event"] = "stop_angle_threshold"
                per_receiver_status[receiver_id] = "converged"
                active.remove(receiver_id)
            else:
                pending.append(receiver_id)
            records.append(row)
        for receiver_id in pending:
            if receiver_id not in active:
                continue
            observed3 = observed[receiver_id]
            estimate: dict[str, Any]
            try:
                # This API boundary receives only the measured angles and the
                # template-selected initial point, never current truth.
                estimate = estimate_receiver(
                    receiver_id,
                    observed3,
                    initial=TARGET_TEMPLATE[receiver_id - 1],
                )
            except (ValueError, FloatingPointError) as error:
                estimate = {
                    "position": TARGET_TEMPLATE[receiver_id - 1].copy(),
                    "success": False,
                    "max_angle_residual": float("inf"),
                    "nfev": 0,
                    "message": str(error),
                }
            row = next(
                item
                for item in reversed(records)
                if item["round"] == round_index
                and item["receiver_id"] == receiver_id
            )
            estimate_position = np.asarray(estimate["position"], dtype=float)
            row["estimated_x"] = float(estimate_position[0])
            row["estimated_y"] = float(estimate_position[1])
            row["estimator_residual_rad"] = float(estimate["max_angle_residual"])
            row["estimator_success"] = bool(estimate["success"])
            row["nfev"] = int(estimate["nfev"])
            if not bool(estimate["success"]):
                message = str(estimate.get("message", "estimator failed"))
                failure = f"receiver_estimation_failure:{message}"
                row["event"] = "failure"
                row["failure"] = failure
                failures[receiver_id] = failure
                per_receiver_status[receiver_id] = "failure"
                active.remove(receiver_id)
                continue
            delta = -float(gain) * (
                estimate_position - TARGET_TEMPLATE[receiver_id - 1]
            )
            current[receiver_id - 1] += delta
            per_receiver_actions[receiver_id] += 1
            row["delta_x"] = float(delta[0])
            row["delta_y"] = float(delta[1])
            row["action_applied"] = True
            row["event"] = "observe_and_move"
            actions.append(
                {
                    "rho": case.rho,
                    "seed": case.seed if case.seed is not None else "ideal",
                    "gain": gain,
                    "round": round_index,
                    "receiver_id": receiver_id,
                    "delta_x": float(delta[0]),
                    "delta_y": float(delta[1]),
                    "true_position_after_x": float(current[receiver_id - 1, 0]),
                    "true_position_after_y": float(current[receiver_id - 1, 1]),
                    "action_number": per_receiver_actions[receiver_id],
                }
            )
        # All updates happen after the same broadcast has been consumed.  The
        # next loop iteration is the only online verification of the update.
    for receiver_id in sorted(active):
        per_receiver_status[receiver_id] = "budget_exhausted"
        failures[receiver_id] = "receiver_budget_exhausted"
    status = "converged" if not failures else "failure"
    final_angle_errors = {
        receiver_id: float(
            np.max(
                np.abs(
                    receiver_angles(
                        current[receiver_id - 1],
                        current[np.asarray(REFERENCE_IDS) - 1],
                    )
                    - _receiver_target_angles(receiver_id)
                )
            )
        )
        for receiver_id in RECEIVER_IDS
    }
    summary = {
        "status": status,
        "failure_count": len(failures),
        "failures": {str(key): value for key, value in failures.items()},
        "broadcast_slots": broadcast_slots,
        "measurement_slots": broadcast_slots,
        "angle_rows": len(records) * 3,
        "tx_uses": broadcast_slots * 3,
        "action_count": len(actions),
        "receiver_rounds": per_receiver_rounds,
        "receiver_actions": per_receiver_actions,
        "receiver_status": per_receiver_status,
        "final_angle_error_rad": final_angle_errors,
        "final_max_angle_error_rad": max(final_angle_errors.values()),
    }
    return current, summary, records, actions


def run_case(rho: float, seed: int | None, gain: float, max_rounds: int) -> dict[str, Any]:
    case = make_initial_state(rho, seed)
    bootstrap_state, bootstrap_summary, bootstrap_rows, bootstrap_actions = _bootstrap_record(
        case=case, gain=gain, max_rounds=max_rounds
    )
    if bootstrap_summary["status"] == "converged":
        main_state, main_summary, main_rows, main_actions = _main_record(
            case=case,
            gain=gain,
            state=bootstrap_state,
            max_rounds=max_rounds,
        )
    else:
        # Do not start ABC estimation with an uncalibrated reference triangle.
        # This keeps a bootstrap failure visible instead of masking it with a
        # downstream run that violates the staged protocol.
        main_state = bootstrap_state.copy()
        main_rows = []
        main_actions = []
        main_summary = {
            "status": "skipped_bootstrap_failure",
            "failure_count": 0,
            "failures": {},
            "broadcast_slots": 0,
            "measurement_slots": 0,
            "angle_rows": 0,
            "tx_uses": 0,
            "action_count": 0,
            "receiver_rounds": {},
            "receiver_actions": {},
            "receiver_status": {
                str(identifier): "skipped" for identifier in RECEIVER_IDS
            },
            "final_angle_error_rad": {},
            "final_max_angle_error_rad": float("inf"),
        }
    initial_error = np.linalg.norm(case.positions - TARGET_TEMPLATE, axis=1)
    final_error = np.linalg.norm(main_state - TARGET_TEMPLATE, axis=1)
    shape_error = float(np.max(final_error))
    shape_rms = float(np.sqrt(np.mean(final_error**2)))
    bootstrap_distance = float(
        sum(
            np.linalg.norm([row["delta_x"], row["delta_y"]])
            for row in bootstrap_actions
        )
    )
    main_distance = float(
        sum(
            np.linalg.norm([row["delta_x"], row["delta_y"]])
            for row in main_actions
        )
    )
    online_status = (
        "success"
        if bootstrap_summary["status"] == "converged"
        and main_summary["status"] == "converged"
        and main_summary["final_max_angle_error_rad"] <= ANGLE_TOLERANCE
        else "failure"
    )
    failure_types: list[str] = []
    if bootstrap_summary.get("failure"):
        failure_types.append(str(bootstrap_summary["failure"]))
    if main_summary.get("failures"):
        failure_types.extend(str(value) for value in main_summary["failures"].values())
    if case.mirrored_apex_side:
        failure_types.append("mirror_branch")
    if shape_error > SHAPE_TOLERANCE:
        failure_types.append("shape_error_above_1e-6d")
    tx_uses = int(bootstrap_summary["tx_uses"] + main_summary["tx_uses"])
    measurement_slots = int(
        bootstrap_summary["angle_slots"] + main_summary["measurement_slots"]
    )
    return {
        "rho": case.rho,
        "seed": case.seed if case.seed is not None else "ideal",
        "gain": float(gain),
        "max_rounds": int(max_rounds),
        "online_status": online_status,
        "status": "success" if not failure_types else "failure",
        "failure_types": sorted(set(failure_types)),
        "actual_spacing_d": case.actual_spacing,
        "d_actual": case.actual_spacing,
        "d_actual_ratio": case.actual_spacing,
        "actual_spacing_over_template_d": case.actual_spacing,
        "raw_base_length": case.raw_base_length,
        "raw_max_perturbation_d": case.raw_max_perturbation,
        "initial_max_deviation_d": case.normalized_max_deviation,
        "initial_rms_deviation_d": case.normalized_rms_deviation,
        "initial_max_error_d": float(np.max(initial_error)),
        "initial_rms_error_d": float(np.sqrt(np.mean(initial_error**2))),
        "mirrored_apex_side": case.mirrored_apex_side,
        "bootstrap": bootstrap_summary,
        "main": main_summary,
        "measurement_slots": measurement_slots,
        "tx_uses": tx_uses,
        "bootstrap_relay_scalars": int(bootstrap_summary["relay_scalar_count"]),
        "action_count": int(
            bootstrap_summary["action_count"] + main_summary["action_count"]
        ),
        "bootstrap_cumulative_move_d": bootstrap_distance,
        "main_cumulative_move_d": main_distance,
        "total_cumulative_move_d": bootstrap_distance + main_distance,
        "bootstrap_cumulative_move_raw_d": bootstrap_distance * case.actual_spacing,
        "main_cumulative_move_raw_d": main_distance * case.actual_spacing,
        "total_cumulative_move_raw_d": (bootstrap_distance + main_distance)
        * case.actual_spacing,
        "final_max_position_error_d": shape_error,
        "final_rms_position_error_d": shape_rms,
        "shape_error_pass": bool(shape_error <= SHAPE_TOLERANCE),
        "final_positions": main_state,
        "bootstrap_rows": bootstrap_rows,
        "bootstrap_actions": bootstrap_actions,
        "main_rows": main_rows,
        "main_actions": main_actions,
    }


def _case_seed(rho: float, seed: int) -> int | None:
    return None if rho == 0.0 else seed


def run_batch(
    output_dir: Path,
    *,
    gains: tuple[float, ...] = GAINS,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [(0.0, None)] + [
        (rho, seed) for rho in PERTURBATION_LEVELS[1:] for seed in SEEDS
    ]
    summaries: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_actions: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    main_actions: list[dict[str, Any]] = []
    for gain in gains:
        for rho, seed in cases:
            result = run_case(
                rho,
                _case_seed(rho, seed or 0),
                float(gain),
                max_rounds,
            )
            summary = {key: value for key, value in result.items() if key not in {
                "final_positions", "bootstrap_rows", "bootstrap_actions", "main_rows", "main_actions"
            }}
            summaries.append(summary)
            identity = {
                "rho": rho,
                "seed": result["seed"],
                "gain": gain,
            }
            for row in result["bootstrap_rows"]:
                bootstrap_rows.append({**identity, **row})
            for row in result["bootstrap_actions"]:
                bootstrap_actions.append({**identity, **row})
            for row in result["main_rows"]:
                main_rows.append({**identity, **row})
            for row in result["main_actions"]:
                main_actions.append({**identity, **row})
            case_dir = output_dir / f"gain_{gain:g}" / f"rho_{rho:g}" / f"seed_{result['seed']}"
            case_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(case_dir / "final_positions.csv", result["final_positions"], delimiter=",")
            case_summary = {
                key: value
                for key, value in result.items()
                if key not in {
                    "final_positions",
                    "bootstrap_rows",
                    "bootstrap_actions",
                    "main_rows",
                    "main_actions",
                }
            }
            (case_dir / "summary.json").write_text(
                json.dumps(_json_value(case_summary), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
    _write_csv(output_dir / "bootstrap_observations.csv", bootstrap_rows)
    _write_csv(output_dir / "bootstrap_actions.csv", bootstrap_actions)
    _write_csv(output_dir / "receiver_observations.csv", main_rows)
    _write_csv(output_dir / "receiver_actions.csv", main_actions)
    flat_summaries: list[dict[str, Any]] = []
    for row in summaries:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"bootstrap", "main", "failure_types"}
        }
        flat["bootstrap_status"] = row["bootstrap"]["status"]
        flat["bootstrap_rounds"] = row["bootstrap"]["rounds"]
        flat["bootstrap_actions"] = row["bootstrap"]["action_count"]
        flat["main_status"] = row["main"]["status"]
        flat["main_broadcast_slots"] = row["main"]["broadcast_slots"]
        flat["main_actions"] = row["main"]["action_count"]
        flat["failure_types_json"] = json.dumps(
            row["failure_types"], ensure_ascii=False, separators=(",", ":")
        )
        flat["bootstrap_summary_json"] = json.dumps(
            _json_value(row["bootstrap"]), ensure_ascii=False, separators=(",", ":")
        )
        flat["main_summary_json"] = json.dumps(
            _json_value(row["main"]), ensure_ascii=False, separators=(",", ":")
        )
        flat_summaries.append(flat)
    _write_csv(output_dir / "runs.csv", flat_summaries)
    summary = {
        "protocol": "triangle_reference_bootstrap_then_ABC_parallel_closed_loop",
        "estimator_truth_boundary": "estimator receives observations and canonical template only; simulator owns truth and applies ideal normalized-coordinate moves",
        "reference_ids": list(REFERENCE_IDS),
        "receiver_ids": list(RECEIVER_IDS),
        "target_anchor_coordinates": TARGET_ANCHORS,
        "target_template": TARGET_TEMPLATE,
        "gains": list(gains),
        "perturbation_levels": list(PERTURBATION_LEVELS),
        "seeds": list(SEEDS),
        "case_count_per_gain": len(cases),
        "run_count": len(summaries),
        "max_rounds_per_stage": max_rounds,
        "angle_tolerance_rad": ANGLE_TOLERANCE,
        "shape_tolerance_d": SHAPE_TOLERANCE,
        "measurement_accounting": {
            "bootstrap_angle_slots_per_observation": 2,
            "bootstrap_tx_uses_per_observation": 4,
            "bootstrap_tx_uses_per_angle_slot": 2,
            "main_tx_uses_per_broadcast": 3,
            "bootstrap_relay_scalars_per_observation": 2,
            "main_broadcast_is_shared_by_12_receivers": True,
        },
        "runs": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_value(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=list(GAINS),
        help="gains to run, default: 1.0 0.5",
    )
    args = parser.parse_args()
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    gains = tuple(float(value) for value in args.gains)
    if any(not np.isfinite(value) or value <= 0 or value > 1 for value in gains):
        parser.error("all gains must lie in (0, 1]")
    summary = run_batch(args.output_dir, gains=gains, max_rounds=args.max_rounds)
    success = sum(row["status"] == "success" for row in summary["runs"])
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "run_count": summary["run_count"],
                "success_count": int(success),
                "failure_count": int(summary["run_count"] - success),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
