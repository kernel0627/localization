#!/usr/bin/env python3
"""Reproduce Table 1, transmitter enumeration and two-round adjustment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.q1_2.run_validation import table_positions, write_csv
from scripts.q1_3.joint_localization import (
    ideal_formation,
    jacobian_diagnostics,
    joint_jacobian,
    localize_formation,
    movement_commands,
    observation_layout,
    predict_angles,
    validate_pair,
)
from scripts.q1_3.transmitter_selection import enumerate_designs, select_design

ROOT = Path(__file__).resolve().parents[2]


def formation_metrics(positions: np.ndarray) -> dict:
    errors = np.linalg.norm(positions[1:] - ideal_formation()[1:], axis=1)
    radial = np.linalg.norm(positions[1:], axis=1)
    theta = np.rad2deg(np.arctan2(positions[1:, 1], positions[1:, 0])) % 360
    gaps = (np.roll(theta, -1) - theta) % 360
    return {
        "max_position_error_m": float(errors.max()),
        "rms_position_error_m": float(np.sqrt(np.mean(errors**2))),
        "max_radial_error_m": float(np.max(np.abs(radial - 100))),
        "max_adjacent_gap_error_deg": float(np.max(np.abs(gaps - 40))),
    }


def simulate_adjustment(initial_positions: np.ndarray, configurations: tuple) -> dict:
    """Truth is owned here by the simulator; inference sees only angle data."""
    if len(configurations) != 2:
        raise ValueError("two configurations are required")
    configurations = tuple(validate_pair(p) for p in configurations)
    if not set(configurations[0]).isdisjoint(configurations[1]):
        raise ValueError("two-round adjustment requires disjoint auxiliary pairs")
    actual = np.array(initial_positions, dtype=float, copy=True)
    nominal = ideal_formation()
    if actual.shape != (10, 2) or not np.isfinite(actual).all():
        raise ValueError("initial_positions must be finite with shape (10, 2)")
    if not np.allclose(actual[:2], nominal[:2], rtol=0, atol=1e-12):
        raise ValueError("this Table 1 model requires fixed nominal FY00 and FY01")
    layout = observation_layout(configurations)
    states = [actual.copy()]
    metrics = [{"round": 0, **formation_metrics(actual)}]
    steps, observations, rounds = [], [], []
    for round_number, held in enumerate(configurations, start=1):
        # Both slots sample this same frozen state. The state changes only
        # after the complete joint estimate has been accepted.
        observed = predict_angles(actual, layout)
        result = localize_formation(configurations, observed)
        if not result.success:
            raise RuntimeError(f"round {round_number} fit rejected: {result.message}")
        commands = movement_commands(result.positions, held)
        after = actual + commands
        for index, ((i, j, k), angle) in enumerate(zip(layout, observed)):
            observations.append(
                {
                    "round": round_number,
                    "slot": index // 36 + 1,
                    "receiver_id": int(i),
                    "transmitter_j": int(j),
                    "transmitter_k": int(k),
                    "angle_rad": float(angle),
                }
            )
        for i in range(10):
            steps.append(
                {
                    "round": round_number,
                    "drone_id": i,
                    "role": "transmitter_hold"
                    if i in (0, 1, *held)
                    else "receiver_move",
                    "before_x_m": float(actual[i, 0]),
                    "before_y_m": float(actual[i, 1]),
                    "estimate_x_m": float(result.positions[i, 0]),
                    "estimate_y_m": float(result.positions[i, 1]),
                    "delta_x_m": float(commands[i, 0]),
                    "delta_y_m": float(commands[i, 1]),
                    "travel_m": float(np.linalg.norm(commands[i])),
                    "after_x_m": float(after[i, 0]),
                    "after_y_m": float(after[i, 1]),
                    "error_after_m": float(np.linalg.norm(after[i] - nominal[i])),
                }
            )
        rounds.append(
            {
                "round": round_number,
                "held_auxiliary_ids": list(held),
                "moving_ids": [i for i in range(2, 10) if i not in held],
                "measurement_slots": [list(p) for p in configurations],
                "angle_count": len(observed),
                "nfev": result.nfev,
                "residual_norm_rad": result.residual_norm_rad,
                "max_localization_error_m": float(
                    np.linalg.norm(result.positions[2:] - actual[2:], axis=1).max()
                ),
                **result.diagnostics,
            }
        )
        actual = after
        states.append(actual.copy())
        metrics.append({"round": round_number, **formation_metrics(actual)})
    return {
        "states": states,
        "metrics": metrics,
        "steps": steps,
        "observations": observations,
        "rounds": rounds,
    }


def run_validation(output_dir: Path) -> dict:
    singles, designs = enumerate_designs()
    selected = select_design(designs)
    configurations = (tuple(selected["a"]), tuple(selected["b"]))
    table = table_positions()
    initial = np.array([table[i] for i in range(10)])
    single_state_diagnostics = []
    for single in singles:
        pair = tuple(single["pair"])
        for state_name, points in (("nominal", ideal_formation()), ("table1", initial)):
            single_state_diagnostics.append(
                {
                    "state": state_name,
                    "pair": "-".join(map(str, pair)),
                    **jacobian_diagnostics(
                        joint_jacobian(points, observation_layout((pair,)))
                    ),
                }
            )
    selected_run = simulate_adjustment(initial, configurations)
    baseline_run = simulate_adjustment(initial, ((5, 8), (4, 7)))
    # Score all designs only AFTER selection on nominal geometry. These
    # truth-based checks are validation and never feed back into selection.
    checks = []
    for design in designs:
        configs = (tuple(design["a"]), tuple(design["b"]))
        failure = ""
        try:
            run = simulate_adjustment(initial, configs)
            final_error = run["metrics"][-1]["max_position_error_m"]
            worst_localization = max(
                r["max_localization_error_m"] for r in run["rounds"]
            )
            passed = final_error < 1e-6 and worst_localization < 1e-6
        except (ValueError, RuntimeError) as error:
            final_error, worst_localization, passed = None, None, False
            failure = str(error)
        checks.append(
            {
                "a": "-".join(map(str, configs[0])),
                "b": "-".join(map(str, configs[1])),
                "passed": passed,
                "final_error_m": final_error,
                "max_localization_error_m": worst_localization,
                "failure": failure,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("single_configurations", singles),
        ("transmitter_pairs", designs),
    ):
        flattened = [
            {
                k: "-".join(map(str, v)) if isinstance(v, list) else v
                for k, v in row.items()
            }
            for row in rows
        ]
        write_csv(output_dir / f"{name}.csv", flattened)
    for name, rows in (
        ("adjustment_steps", selected_run["steps"]),
        ("observations", selected_run["observations"]),
        ("error_history", selected_run["metrics"]),
        ("design_validation", checks),
        ("single_state_diagnostics", single_state_diagnostics),
    ):
        write_csv(output_dir / f"{name}.csv", rows)
    positions = []
    nominal = ideal_formation()
    for i in range(10):
        row = {
            "drone_id": i,
            "target_x_m": float(nominal[i, 0]),
            "target_y_m": float(nominal[i, 1]),
        }
        for stage, state in enumerate(selected_run["states"]):
            row.update(
                {
                    f"stage{stage}_x_m": float(state[i, 0]),
                    f"stage{stage}_y_m": float(state[i, 1]),
                }
            )
        positions.append(row)
    write_csv(output_dir / "positions.csv", positions)
    summary = {
        "model_status": "ideal centralized simulation baseline; information-sharing and execution protocol require justification",
        "scheduled_movement_batches": 2,
        "measurement_slots_per_batch": 2,
        "stopping_rule": "fixed two-batch schedule followed by truth-based validation",
        "radius_m": 100,
        "fixed_reference_ids": [0, 1],
        "angle_noise": "none",
        "actuation": "exact relative displacement, gain=1",
        "design_evaluation_state": "nominal formation only",
        "objective": "maximize smallest singular value of all-pair angle Jacobian",
        "single_configuration_count": len(singles),
        "single_rank_counts": {
            str(rank): sum(r["rank"] == rank for r in singles)
            for rank in sorted({r["rank"] for r in singles})
        },
        "single_table1_full_rank_count": sum(
            r["state"] == "table1" and r["rank"] == 16 for r in single_state_diagnostics
        ),
        "disjoint_unordered_design_count": len(designs),
        "joint_full_rank_count": sum(row["rank"] == 16 for row in designs),
        "selected_design": selected,
        "rounds": selected_run["rounds"],
        "error_history": selected_run["metrics"],
        "reference_design": {
            "a": [5, 8],
            "b": [4, 7],
            "error_history": baseline_run["metrics"],
        },
        "exhaustive_table1": {
            "case_count": len(checks),
            "passed": sum(row["passed"] for row in checks),
            "max_final_error_m": max(
                (r["final_error_m"] for r in checks if r["final_error_m"] is not None),
                default=None,
            ),
            "failures": [r for r in checks if not r["passed"]],
        },
        "evidence_scope": "Local identifiability at nominal and fitted states; finite exact-angle Table 1 simulations. Assumes fixed calibrated FY00/FY01, labeled signals, fusion of all receivers' observations, and displacement execution in the reference frame. No global uniqueness or noisy-flight guarantee.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if (
        summary["exhaustive_table1"]["failures"]
        or selected_run["metrics"][-1]["max_position_error_m"] >= 1e-6
        or baseline_run["metrics"][-1]["max_position_error_m"] >= 1e-6
    ):
        raise RuntimeError("adjustment validation failed; see summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "q1_3" / "joint_baseline"
    )
    args = parser.parse_args()
    run_validation(args.output_dir)


if __name__ == "__main__":
    main()
