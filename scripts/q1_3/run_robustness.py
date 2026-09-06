"""Run paired initial-condition experiments without changing main's controller."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts.q1_1.localization import fy_position
from scripts.q1_2.run_validation import table_positions
from scripts.q1_3.local_adjustment import LocalSettings
from scripts.q1_3.run_iterative_reference_baseline import simulate_adjustment
from scripts.q1_3.simulation_noise import SimulationNoise

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = {
    "exact": (0.0, 0.0),
    "bearing_0.001deg": (0.001, 0.0),
    "bearing_0.01deg": (0.01, 0.0),
    "bearing_0.1deg": (0.1, 0.0),
    "actuation_1pct": (0.0, 0.01),
}


def initial_condition(trial):
    if trial == -1:
        table = table_positions()
        return np.array([table[i] for i in range(10)])
    rng = np.random.default_rng(np.random.SeedSequence([20260905, trial, 7]))
    points = np.array([fy_position(i) for i in range(10)])
    radii = 100 + rng.uniform(-12, 12, 8)
    angles = np.deg2rad(np.arange(1, 9) * 40 + rng.uniform(-0.3, 0.3, 8))
    points[2:] = radii[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    return points


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_one(condition, trial, directory, fingerprint):
    dest = (
        Path(directory)
        / condition
        / ("table1" if trial == -1 else f"trial_{trial:03d}")
    )
    summary_path = dest / "summary.json"
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if result["fingerprint"] != fingerprint:
            raise RuntimeError(f"Stale cached run: {dest}")
        return result["evaluation"]
    start = time.monotonic()
    bearing, actuation = CONDITIONS[condition]
    initial = initial_condition(trial)
    noise = SimulationNoise(bearing, actuation, 2026090500 + trial + 1)
    run = simulate_adjustment(initial, noise=noise, retain_details=False)
    summary = run["summary"]
    errors = np.array([h["max_position_error_m"] for h in run["history"]])
    hits = np.flatnonzero(errors < 0.01)
    below = errors < 0.01
    bad = np.flatnonzero(~below)
    sustained = (
        int(bad[-1] + 1)
        if len(bad) and bad[-1] < len(errors) - 1
        else (0 if not len(bad) else None)
    )
    stopped = summary["status"] == "quiet_full_cycle"
    final = summary["final_metrics"]
    evaluation = {
        "condition": condition,
        "trial": trial,
        "bearing_std_deg": bearing,
        "actuation_relative_std": actuation,
        "status": summary["status"],
        "stopped": stopped,
        "final_below_1cm": final["max_position_error_m"] < 0.01,
        "joint_success_1cm": stopped and final["max_position_error_m"] < 0.01,
        "final_below_10cm": final["max_position_error_m"] < 0.1,
        "first_1cm_slot": int(hits[0]) if len(hits) else None,
        "first_1cm_tx": int(hits[0]) * 4 if len(hits) else None,
        "sustained_1cm_from_slot": sustained,
        "max_position_error_m": final["max_position_error_m"],
        "rms_position_error_m": final["rms_position_error_m"],
        "last28_mean_max_error_m": float(errors[-28:].mean()),
        "measurement_slots": summary["measurement_slots"],
        "transmitter_uses": summary["transmitter_uses"],
        "movement_m": summary["total_endpoint_displacement_m"],
        "failed_local_fits": summary["failed_local_fits"],
        "seconds": time.monotonic() - start,
    }
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "slot_metrics.csv", run["history"])
    result = {
        "fingerprint": fingerprint,
        "evaluation": evaluation,
        "initial_positions": initial.tolist(),
        "final_positions": run["final_positions"].tolist(),
        "summary": summary,
    }
    temporary = dest / "summary.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(summary_path)
    return evaluation


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    den = 1 + z * z / count
    center = (p + z * z / (2 * count)) / den
    width = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / den
    return float(max(0, center - width)), float(min(1, center + width))


def aggregate(rows):
    result = []
    for condition in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition and r["trial"] >= 0]
        if not group:
            continue
        n = len(group)
        success = sum(r["joint_success_1cm"] for r in group)
        lo, hi = wilson(success, n)
        row = {
            "condition": condition,
            "runs": n,
            "stopped_count": sum(r["stopped"] for r in group),
            "final_below_1cm_count": sum(r["final_below_1cm"] for r in group),
            "joint_success_count": success,
            "joint_success_rate": success / n,
            "wilson95_low": lo,
            "wilson95_high": hi,
            "final_below_10cm_count": sum(r["final_below_10cm"] for r in group),
            "first_1cm_count": sum(r["first_1cm_slot"] is not None for r in group),
            "fit_failure_runs": sum(r["status"] == "local_fit_failed" for r in group),
            "budget_exhausted_runs": sum(
                r["status"] == "max_epochs_reached" for r in group
            ),
        }
        for metric in [
            "max_position_error_m",
            "rms_position_error_m",
            "last28_mean_max_error_m",
            "measurement_slots",
            "transmitter_uses",
            "movement_m",
        ]:
            vals = [r[metric] for r in group]
            for suffix, q in [("p05", 0.05), ("median", 0.5), ("p95", 0.95)]:
                row[f"{metric}_{suffix}"] = float(np.quantile(vals, q))
        reached = [
            r["first_1cm_slot"] for r in group if r["first_1cm_slot"] is not None
        ]
        row["first_1cm_slot_median_among_hits"] = (
            float(np.median(reached)) if reached else None
        )
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/q1_3/robustness"
    )
    args = parser.parse_args()
    if args.trials < 1 or args.workers < 1:
        parser.error("trials and workers must be positive")
    sources = [
        "local_adjustment.py",
        "simulation_noise.py",
        "run_iterative_reference_baseline.py",
        "run_robustness.py",
    ]
    hashes = {
        s: hashlib.sha256((ROOT / "scripts/q1_3" / s).read_bytes()).hexdigest()
        for s in sources
    }
    fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "trials_per_condition": args.trials,
        "conditions": CONDITIONS,
        "settings": asdict(LocalSettings()),
        "max_epochs": 20,
        "initial_seed": 20260905,
        "noise_seed_base": 2026090500,
        "initial_distribution": "Independent dr U[-12,12] m and dtheta U[-0.3,0.3] deg for FY02..09; calibrated FY00/FY01 fixed. Same initial states and indexed standard noise across conditions.",
        "success": "Protocol stopped within 560 slots AND final Emax < 0.01 m. All failures included.",
        "source_sha256": hashes,
        "fingerprint": fingerprint,
    }
    (args.output_dir / "contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    jobs = [
        (condition, trial)
        for condition in CONDITIONS
        for trial in [-1, *range(args.trials)]
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(run_one, c, t, str(args.output_dir), fingerprint): (c, t)
            for c, t in jobs
        }
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            if len(rows) % 5 == 0 or len(rows) == len(jobs):
                print(
                    f"{len(rows)}/{len(jobs)} completed; {row['condition']} trial {row['trial']}: {row['status']}, Emax={row['max_position_error_m']:.5g} m",
                    flush=True,
                )
    rows.sort(key=lambda r: (list(CONDITIONS).index(r["condition"]), r["trial"]))
    write_csv(args.output_dir / "trials.csv", rows)
    summaries = aggregate(rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
