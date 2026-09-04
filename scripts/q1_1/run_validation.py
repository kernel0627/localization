#!/usr/bin/env python3
"""Run deterministic exact and noisy validation for Question 1(1)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from localization import (
    find_local_candidates,
    fy_position,
    local_observability,
    localize_receiver,
    pairwise_angles,
    polar_to_cartesian,
)


INITIAL_POLAR = {
    0: (0.0, 0.0),
    1: (100.0, 0.0),
    2: (98.0, 40.10),
    3: (112.0, 80.21),
    4: (105.0, 119.75),
    5: (98.0, 159.86),
    6: (112.0, 199.96),
    7: (105.0, 240.07),
    8: (98.0, 280.17),
    9: (112.0, 320.28),
}

REPRESENTATIVE_CONFIGS = (
    (40, (0, 1, 2)),
    (80, (0, 1, 3)),
    (120, (0, 1, 4)),
    (160, (0, 1, 5)),
)


def actual_position(drone_id: int) -> np.ndarray:
    radius, angle = INITIAL_POLAR[drone_id]
    return polar_to_cartesian(radius, angle)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def run_validation(noise_std_deg: float, trials: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    exact_rows: list[dict] = []
    summary_rows: list[dict] = []

    for gap_deg, transmitter_ids in REPRESENTATIVE_CONFIGS:
        anchors = np.vstack([fy_position(drone_id) for drone_id in transmitter_ids])
        receiver_ids = [
            drone_id for drone_id in range(1, 10) if drone_id not in transmitter_ids
        ]
        exact_errors: list[float] = []
        sigma_mins: list[float] = []
        ideal_sigma_mins: list[float] = []
        noisy_errors: list[float] = []
        noisy_failures = 0

        for receiver_id in receiver_ids:
            truth = actual_position(receiver_id)
            observed = pairwise_angles(truth, anchors)
            initial = fy_position(receiver_id)
            result = localize_receiver(anchors, observed, initial)
            error = float(np.linalg.norm(result.position - truth))
            singular_values = local_observability(truth, anchors)
            sigma_min = float(singular_values[-1])
            ideal_sigma_min = float(local_observability(initial, anchors)[-1])
            exact_errors.append(error)
            sigma_mins.append(sigma_min)
            ideal_sigma_mins.append(ideal_sigma_min)
            exact_rows.append(
                {
                    "gap_deg": gap_deg,
                    "transmitters": "-".join(f"FY{x:02d}" for x in transmitter_ids),
                    "receiver": f"FY{receiver_id:02d}",
                    "true_x_m": float(truth[0]),
                    "true_y_m": float(truth[1]),
                    "estimate_x_m": float(result.position[0]),
                    "estimate_y_m": float(result.position[1]),
                    "position_error_m": error,
                    "residual_norm": result.residual_norm,
                    "sigma_min_per_m": sigma_min,
                    "condition_number": result.condition_number,
                }
            )

            for _ in range(trials):
                noisy_observed = np.clip(
                    observed
                    + rng.normal(0.0, np.deg2rad(noise_std_deg), observed.shape),
                    0.0,
                    np.pi,
                )
                noisy_result = localize_receiver(anchors, noisy_observed, initial)
                if not noisy_result.success:
                    noisy_failures += 1
                noisy_errors.append(
                    float(np.linalg.norm(noisy_result.position - truth))
                )

        summary_rows.append(
            {
                "gap_deg": gap_deg,
                "transmitters": "-".join(f"FY{x:02d}" for x in transmitter_ids),
                "receiver_count": len(receiver_ids),
                "exact_max_error_m": max(exact_errors),
                "ideal_worst_sigma_min_per_m": min(ideal_sigma_mins),
                "table_worst_sigma_min_per_m": min(sigma_mins),
                "table_median_sigma_min_per_m": float(np.median(sigma_mins)),
                "noise_std_deg": noise_std_deg,
                "noise_trials_per_receiver": trials,
                "noise_sample_count": len(noisy_errors),
                "noise_median_error_m": float(np.median(noisy_errors)),
                "noise_p95_error_m": percentile(noisy_errors, 95),
                "noise_max_error_m": max(noisy_errors),
                "noise_solver_failures": noisy_failures,
            }
        )

    return exact_rows, summary_rows


def ambiguity_example() -> dict:
    transmitter_ids = (0, 1, 5)
    anchors = np.vstack([fy_position(drone_id) for drone_id in transmitter_ids])
    receiver_id = 3
    truth = actual_position(receiver_id)
    nominal = fy_position(receiver_id)
    observed = pairwise_angles(truth, anchors)
    starts = np.vstack(
        [
            fy_position(drone_id)
            for drone_id in range(1, 10)
            if drone_id not in transmitter_ids
        ]
    )
    candidates = find_local_candidates(anchors, observed, starts)
    candidate_rows = [
        {
            "x_m": float(candidate.position[0]),
            "y_m": float(candidate.position[1]),
            "residual_norm": candidate.residual_norm,
            "distance_to_nominal_m": float(
                np.linalg.norm(candidate.position - nominal)
            ),
        }
        for candidate in candidates
    ]
    selected = min(candidate_rows, key=lambda row: row["distance_to_nominal_m"])
    return {
        "transmitters": "FY00-FY01-FY05",
        "receiver": "FY03",
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "selection_rule": "minimum distance to the known receiver's ideal position",
        "selected_candidate": selected,
        "true_position_m": [float(truth[0]), float(truth[1])],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-std-deg", type=float, default=0.1)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20220904)
    parser.add_argument("--output", type=Path, default=Path("outputs/q1_1"))
    args = parser.parse_args()
    if args.noise_std_deg < 0.0:
        parser.error("--noise-std-deg must be nonnegative")
    if args.trials < 1:
        parser.error("--trials must be positive")

    exact_rows, summary_rows = run_validation(
        args.noise_std_deg, args.trials, args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "exact_cases.csv", exact_rows)
    write_csv(args.output / "noise_summary.csv", summary_rows)
    ambiguity = ambiguity_example()
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "q1_1_localization_validation",
                "seed": args.seed,
                "noise_model": "independent Gaussian perturbation of each pairwise angle",
                "noise_std_deg": args.noise_std_deg,
                "trials_per_receiver": args.trials,
                "configurations": summary_rows,
                "ambiguity_example": ambiguity,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "gap  receivers  exact_max(m)  ideal_sigma_min(1/m)  "
        "table_sigma_min(1/m)  "
        "noise_median(m)  noise_p95(m)  failures"
    )
    for row in summary_rows:
        print(
            f"{row['gap_deg']:>3}  {row['receiver_count']:>9}  "
            f"{row['exact_max_error_m']:>12.3e}  "
            f"{row['ideal_worst_sigma_min_per_m']:>20.3e}  "
            f"{row['table_worst_sigma_min_per_m']:>20.3e}  "
            f"{row['noise_median_error_m']:>15.3f}  "
            f"{row['noise_p95_error_m']:>12.3f}  "
            f"{row['noise_solver_failures']:>8}"
        )
    print("\nUnoriented-angle candidate check:")
    for candidate in ambiguity["candidates"]:
        print(
            f"  ({candidate['x_m']:.4f}, {candidate['y_m']:.4f}), "
            f"residual={candidate['residual_norm']:.2e}, "
            f"distance_to_nominal={candidate['distance_to_nominal_m']:.3f} m"
        )


if __name__ == "__main__":
    main()
