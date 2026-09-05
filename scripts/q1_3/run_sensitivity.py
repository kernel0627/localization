"""Finite gain and nearby-initial-formation checks for local adjustment."""

import json
from pathlib import Path

import numpy as np

from scripts.q1_1.localization import fy_position
from scripts.q1_2.run_validation import table_positions, write_csv
from scripts.q1_3.local_adjustment import LocalSettings
from scripts.q1_3.run_adjustment import simulate_adjustment


def main():
    table = table_positions()
    initial = np.array([table[i] for i in range(10)])
    nominal = np.array([fy_position(i) for i in range(10)])
    cases = [(f"table1_gain_{gain:g}", initial, gain) for gain in (0.25, 0.5, 1.0)]
    rng = np.random.default_rng(20220905)
    cases.append(("nominal", nominal, 0.5))
    for index in range(3):
        positions = nominal.copy()
        positions[2:] += rng.uniform(-5, 5, (8, 2))
        cases.append((f"cartesian_perturbation_{index + 1}", positions, 0.5))
    rows = []
    for name, positions, gain in cases:
        summary = simulate_adjustment(
            positions,
            settings=LocalSettings(gain=gain),
            retain_details=False,
        )["summary"]
        rows.append(
            {
                "case": name,
                "gain": gain,
                "status": summary["status"],
                "epochs": summary["epochs"],
                "measurement_slots": summary["measurement_slots"],
                "failed_local_fits": summary["failed_local_fits"],
                **summary["final_metrics"],
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    output_dir = Path(__file__).resolve().parents[2] / "outputs/q1_3"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "sensitivity.csv", rows)
    if any(
        r["status"] != "quiet_full_cycle" or r["max_position_error_m"] >= 0.01
        for r in rows
    ):
        raise RuntimeError("one or more finite sensitivity cases failed")


if __name__ == "__main__":
    main()
