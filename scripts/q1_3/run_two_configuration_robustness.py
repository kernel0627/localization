"""Run or resume the two-configuration 560-slot robustness contract.

Use ``python -m scripts.q1_3.run_two_configuration_robustness``.  The frozen
gain-0.5 five-condition appendix batch is verified and referenced, never
modified.  Every newly simulated trial has an atomic per-trial cache.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from functools import lru_cache

# Keep worker BLAS libraries single-threaded before importing NumPy/SciPy.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from appendix1.optimize_schedule import simulate
from scripts.q1_2.run_validation import write_csv
from scripts.q1_3.analyze_two_configuration import verify_contract
from scripts.q1_3.local_adjustment import (
    LocalSettings,
    ReceiverController,
    decide_local_adjustment,
)
from scripts.q1_3.run_robustness import initial_condition
from scripts.q1_3.two_configuration_noise import TwoConfigurationNoise


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "appendix1/evaluation_560"
OUTPUT = ROOT / "outputs/q1_3/two_configuration_robustness"
SCHEDULE = ((1, 4, 5), (1, 7, 8))
CONDITIONS = {
    "exact": (0.0, 0.0, 0.0),
    "bearing_0.001deg": (0.001, 0.0, 0.0),
    "bearing_0.01deg": (0.01, 0.0, 0.0),
    "bearing_0.1deg": (0.1, 0.0, 0.0),
    "actuation_1pct": (0.0, 0.01, 0.0),
    "combined_0.001deg_1pct": (0.001, 0.01, 0.0),
    "combined_0.01deg_1pct": (0.01, 0.01, 0.0),
    "link_bias_0.001deg": (0.0, 0.0, 0.001),
}
FROZEN_CONDITIONS = frozenset(tuple(CONDITIONS)[:5])


def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_csv_union(path: Path, rows):
    """Write heterogeneous provenance rows without silently dropping fields."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_path(path: Path):
    """Use a workspace-relative provenance path when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def source_hashes():
    paths = {
        "scripts/q1_3/run_two_configuration_robustness.py": Path(__file__),
        "scripts/q1_3/two_configuration_noise.py": ROOT / "scripts/q1_3/two_configuration_noise.py",
        "appendix1/optimize_schedule.py": ROOT / "appendix1/optimize_schedule.py",
        "scripts/q1_3/local_adjustment.py": ROOT / "scripts/q1_3/local_adjustment.py",
        "scripts/q1_3/run_robustness.py": ROOT / "scripts/q1_3/run_robustness.py",
        "scripts/q1_3/simulation_noise.py": ROOT / "scripts/q1_3/simulation_noise.py",
        "scripts/q1_1/localization.py": ROOT / "scripts/q1_1/localization.py",
        "scripts/q1_2/run_validation.py": ROOT / "scripts/q1_2/run_validation.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def run_directory(output_dir: Path, gain: float, condition: str, trial: int):
    leaf = "table1" if trial == -1 else f"trial_{trial:03d}"
    return output_dir / f"gain_{gain:g}" / condition / leaf


def threshold_metrics(history, threshold):
    errors = np.array([row["max_position_error_m"] for row in history])
    below = errors < threshold
    hits = np.flatnonzero(below)
    first = int(hits[0]) if len(hits) else None
    bad = np.flatnonzero(~below)
    sustained = (
        int(bad[-1] + 1)
        if len(bad) and bad[-1] < len(errors) - 1
        else (0 if not len(bad) else None)
    )
    return {
        "first_slot": first,
        "first_tx": int(history[first]["cumulative_transmitter_uses"]) if first is not None else None,
        "sustained_from_slot": sustained,
        "stays_below_after_first": bool(first is not None and sustained == first),
    }


class DiagnosticController:
    """Delegate to the unchanged controller while recording only local state."""

    def __init__(self, receiver_id, settings, records):
        self.controller = ReceiverController(receiver_id, settings)
        self.receiver_id = receiver_id
        self.settings = settings
        self.records = records

    def decide(self, transmitters, own_angles):
        before_count = self.controller.small_count
        before_holding = self.controller.holding
        decision = self.controller.decide(transmitters, own_angles)
        selected = decision.selected
        ratio = None
        radial_clipped = angular_clipped = False
        if selected is not None:
            ratio = max(
                abs(selected.radial_bias_m) / self.settings.radial_tolerance_m,
                abs(selected.angular_bias_rad) / self.settings.angular_tolerance_rad,
                selected.consistency_rad / self.settings.consistency_tolerance_rad,
            )
            radial_clipped = abs(self.settings.gain * selected.radial_bias_m) > self.settings.max_radial_step_m
            angular_clipped = abs(self.settings.gain * selected.angular_bias_rad) > self.settings.max_angular_step_rad
        self.records.append(
            {
                "receiver_id": self.receiver_id,
                "transmitters": "-".join(map(str, (0, *transmitters))),
                "selected_pair": "-".join(map(str, selected.pair)) if selected else "",
                "status": decision.status,
                "ratio": ratio,
                "small_count_before": before_count,
                "small_count_after": self.controller.small_count,
                "holding_before": before_holding,
                "holding_after": self.controller.holding,
                "estimated_radial_bias_m": selected.radial_bias_m if selected else None,
                "estimated_angular_bias_rad": selected.angular_bias_rad if selected else None,
                "consistency_rad": selected.consistency_rad if selected else None,
                "radial_clipped": radial_clipped,
                "angular_clipped": angular_clipped,
            }
        )
        return decision


class DiagnosticFactory:
    def __init__(self, records):
        self.records = records

    def __call__(self, receiver_id, settings):
        return DiagnosticController(receiver_id, settings, self.records)


@lru_cache(maxsize=1)
def nominal_selected_pairs():
    """Selected local branch at the ideal public geometry, indexed by phase."""
    from scripts.q1_1.localization import fy_position, pairwise_angles

    points = np.array([fy_position(i) for i in range(10)])
    result = {}

    for phase, tx in enumerate(SCHEDULE, 1):
        for receiver in range(2, 10):
            if receiver in tx:
                continue
            decision = decide_local_adjustment(
                receiver, tx, pairwise_angles(points[receiver], points[[0, *tx]])
            )
            if decision.selected is None:
                raise RuntimeError("nominal two-configuration branch has no local fit")
            result[(phase, receiver)] = "-".join(map(str, decision.selected.pair))
    return result


def diagnostic_tail(records, measurement_slots):
    expected = nominal_selected_pairs()
    first = max(1, measurement_slots - 99)
    tail = []
    for record in records:
        if "slot" not in record:
            raise RuntimeError("diagnostic record is missing its simulator slot")
        if record["slot"] >= first:
            value = dict(record)
            value["nominal_selected_pair"] = expected[(1 + (value["slot"] - 1) % 2, value["receiver_id"])]
            value["matches_nominal_selected_pair"] = value["selected_pair"] == value["nominal_selected_pair"]
            tail.append(value)
    return tail


def evaluation_from_summary(*, gain, condition, trial, summary, history, seconds):
    cm, mm = threshold_metrics(history, 0.01), threshold_metrics(history, 0.001)
    final = summary["final_metrics"]
    stopped = summary["status"] == "quiet_full_selected_cycle"
    return {
        "gain": gain,
        "condition": condition,
        "trial": trial,
        "bearing_std_deg": CONDITIONS[condition][0],
        "actuation_relative_std": CONDITIONS[condition][1],
        "link_bias_std_deg": CONDITIONS[condition][2],
        "status": summary["status"],
        "stopped": stopped,
        "budget_exhausted": summary["status"] == "budget_exhausted",
        "fit_failed": summary["status"] == "local_fit_failed",
        "final_below_1cm": final["max_position_error_m"] < 0.01,
        "final_below_1mm": final["max_position_error_m"] < 0.001,
        "joint_success_1cm": stopped and final["max_position_error_m"] < 0.01,
        "joint_success_1mm": stopped and final["max_position_error_m"] < 0.001,
        "first_1cm_slot": cm["first_slot"],
        "first_1cm_tx": cm["first_tx"],
        "sustained_1cm_from_slot": cm["sustained_from_slot"],
        "record_stays_1cm_after_first": cm["stays_below_after_first"],
        "first_1mm_slot": mm["first_slot"],
        "first_1mm_tx": mm["first_tx"],
        "sustained_1mm_from_slot": mm["sustained_from_slot"],
        "record_stays_1mm_after_first": mm["stays_below_after_first"],
        "max_position_error_m": final["max_position_error_m"],
        "rms_position_error_m": final["rms_position_error_m"],
        "measurement_slots": summary["measurement_slots"],
        "transmitter_uses": summary["transmitter_uses"],
        "movement_m": summary["endpoint_m"],
        "failed_local_fits": summary["failed_fits"],
        "seconds": seconds,
    }


def run_new(gain, condition, trial, directory, fingerprint):
    output_dir = Path(directory)
    destination = run_directory(output_dir, gain, condition, trial)
    saved_path = destination / "summary.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise RuntimeError(f"stale cached run: {destination}")
        return saved["evaluation"]
    started = time.monotonic()
    bearing, actuation, bias = CONDITIONS[condition]
    initial = initial_condition(trial)
    noise = TwoConfigurationNoise(
        bearing, actuation, bias, 2026090500 + trial + 1
    )
    diagnostics = []
    summary, history, decisions = simulate(
        initial, SCHEDULE, max_slots=560,
        settings=replace(LocalSettings(), gain=gain), noise=noise,
        controller_factory=DiagnosticFactory(diagnostics),
    )
    # Every two-slot schedule phase has exactly six receivers.  Simulator exit
    # follows the phase only after all six local decisions have been recorded.
    for index, record in enumerate(diagnostics):
        record["slot"] = 1 + index // 6
    tail = diagnostic_tail(diagnostics, summary["measurement_slots"])
    evaluation = evaluation_from_summary(
        gain=gain, condition=condition, trial=trial, summary=summary,
        history=history, seconds=time.monotonic() - started,
    )
    evaluation.update(
        tail100_local_decisions=len(tail),
        tail100_nominal_selected_matches=sum(
            bool(row["matches_nominal_selected_pair"]) for row in tail
        ),
        tail100_nominal_selected_match_rate=(
            sum(bool(row["matches_nominal_selected_pair"]) for row in tail) / len(tail)
            if tail else None
        ),
        tail100_holding_count=sum(bool(row["holding_after"]) for row in tail),
        tail100_radial_clip_count=sum(bool(row["radial_clipped"]) for row in tail),
        tail100_angular_clip_count=sum(bool(row["angular_clipped"]) for row in tail),
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "slot_metrics.csv", history)
    write_csv(destination / "decisions.csv", decisions)
    write_csv(destination / "controller_diagnostics_last100.csv", tail)
    result = {
        "fingerprint": fingerprint,
        "evaluation": evaluation,
        "initial_positions": initial.tolist(),
        "final_positions": summary["final_positions"],
        "summary": summary,
    }
    atomic_text(destination / "summary.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    return evaluation


def bool_or_number(value):
    if value in ("", None):
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def load_frozen_rows(fingerprint):
    archive_contract, checked = verify_contract()
    if archive_contract["fingerprint"] != fingerprint["archive_fingerprint"]:
        raise RuntimeError("frozen appendix contract changed during this run")
    rows = list(csv.DictReader((ARCHIVE / "trials.csv").open(encoding="utf-8")))
    result = []
    for row in rows:
        trial = int(row["trial"])
        condition = row["condition"]
        if condition not in FROZEN_CONDITIONS:
            raise RuntimeError(f"unexpected frozen condition: {condition}")
        directory = run_directory(ARCHIVE, 0.5, condition, trial)
        # Archive layout lacks gain_0.5; resolve its actual evidence directory.
        directory = ARCHIVE / condition / ("table1" if trial == -1 else f"trial_{trial:03d}")
        saved_path = directory / "summary.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != archive_contract["fingerprint"]:
            raise RuntimeError(f"frozen summary fingerprint mismatch: {directory}")
        converted = {key: bool_or_number(value) for key, value in row.items()}
        converted.update(
            gain=0.5,
            budget_exhausted=converted["status"] == "budget_exhausted",
            fit_failed=converted["status"] == "local_fit_failed",
            link_bias_std_deg=0.0,
            source_kind="frozen_appendix1_reuse",
            source_dir=recorded_path(directory),
            summary_path=recorded_path(saved_path),
            cache_sha256=sha256(saved_path),
        )
        result.append(converted)
    return result, checked


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    width = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return float(max(0, center - width)), float(min(1, center + width))


def aggregate(rows):
    output = []
    for gain in (0.5, 1.0):
        for condition in CONDITIONS:
            group = [r for r in rows if r["gain"] == gain and r["condition"] == condition and r["trial"] >= 0]
            if not group:
                continue
            result = {"gain": gain, "condition": condition, "runs": len(group)}
            for label in ("stopped", "final_below_1cm", "final_below_1mm", "joint_success_1cm", "joint_success_1mm"):
                count = sum(bool(r[label]) for r in group)
                lo, hi = wilson(count, len(group))
                result.update({f"{label}_count": count, f"{label}_rate": count / len(group), f"{label}_wilson95_low": lo, f"{label}_wilson95_high": hi})
            result["fit_failure_runs"] = sum(r["fit_failed"] for r in group)
            result["budget_exhausted_runs"] = sum(r["budget_exhausted"] for r in group)
            diagnostics = [r for r in group if r.get("tail100_local_decisions") is not None]
            if diagnostics:
                decision_count = sum(r["tail100_local_decisions"] for r in diagnostics)
                result.update(
                    tail100_diagnostic_runs=len(diagnostics),
                    tail100_local_decisions=decision_count,
                    tail100_nominal_selected_matches=sum(r["tail100_nominal_selected_matches"] for r in diagnostics),
                    tail100_nominal_selected_match_rate=sum(r["tail100_nominal_selected_matches"] for r in diagnostics) / decision_count,
                    tail100_holding_rate=sum(r["tail100_holding_count"] for r in diagnostics) / decision_count,
                    tail100_radial_clip_rate=sum(r["tail100_radial_clip_count"] for r in diagnostics) / decision_count,
                    tail100_angular_clip_rate=sum(r["tail100_angular_clip_count"] for r in diagnostics) / decision_count,
                )
            for metric in ("max_position_error_m", "rms_position_error_m", "measurement_slots", "transmitter_uses", "movement_m"):
                values = np.array([r[metric] for r in group], dtype=float)
                for suffix, q in (("p05", .05), ("median", .5), ("p95", .95)):
                    result[f"{metric}_{suffix}"] = float(np.quantile(values, q))
            for label in ("1cm", "1mm"):
                for kind in ("first", "sustained"):
                    values = [r[f"{kind}_{label}_slot" if kind == "first" else f"sustained_{label}_from_slot"] for r in group]
                    values = [v for v in values if v is not None]
                    result[f"{kind}_{label}_count"] = len(values)
                    result[f"{kind}_{label}_slot_median_among_hits"] = float(np.median(values)) if values else None
            output.append(result)
    return output


def trial_csv_rows(rows, output_dir):
    ordered = []
    for row in rows:
        value = dict(row)
        value.setdefault("source_kind", "new_simulation")
        if value["source_kind"] == "new_simulation":
            directory = run_directory(output_dir, value["gain"], value["condition"], value["trial"])
            summary_path = directory / "summary.json"
            value.update(source_dir=recorded_path(directory), summary_path=recorded_path(summary_path), cache_sha256=sha256(summary_path))
        ordered.append(value)
    return sorted(ordered, key=lambda r: (r["gain"], list(CONDITIONS).index(r["condition"]), r["trial"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--smoke", action="store_true",
        help="verify archive/schema and run two representative new table-1 cases only",
    )
    args = parser.parse_args()
    if args.trials != 100:
        parser.error("the execution contract requires exactly 100 random trials")
    if not 4 <= args.workers <= 6:
        parser.error("the execution contract requires 4 through 6 workers")
    # This checks all 505 archive summary identities plus source hashes before
    # any new simulation is scheduled.
    archive_contract, archive_checked = verify_contract()
    hashes = source_hashes()
    fingerprint = {
        "source_sha256": hashes,
        "archive_contract_sha256": sha256(ARCHIVE / "contract.json"),
        "archive_fingerprint": archive_contract["fingerprint"],
        "fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schedule": [list(slot) for slot in SCHEDULE], "period_slots": 2,
        "settings_by_gain": {str(g): asdict(replace(LocalSettings(), gain=g)) for g in (0.5, 1.0)},
        "max_slots": 560, "trials_per_condition": args.trials,
        "conditions": {name: {"bearing_std_deg": v[0], "actuation_relative_std": v[1], "link_bias_std_deg": v[2]} for name, v in CONDITIONS.items()},
        "initial_seed": 20260905, "noise_seed_base": 2026090500,
        "reused_runs": 505, "new_runs": 1111, "total_runs": 1616,
        "reuse": {"gain": 0.5, "conditions": list(FROZEN_CONDITIONS), "archive": str(ARCHIVE.relative_to(ROOT)), "verified": archive_checked},
        "information_boundary": "The controller uses only current local six angles, IDs and public two-slot schedule, ideal geometry, and its own hold memory. Initial truth, positions, link offsets, and execution errors remain simulator-private. Link offsets are keyed only by public receiver/transmitter IDs and never supplied to the controller.",
        "white_noise": "Four ray directions receive independent per-slot errors before six pair angles are formed. Fixed link offsets use independent receiver-transmitter stream 2; white bearing stream 0 and execution stream 1 retain the old seeded streams.",
        "blas_threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        **fingerprint,
    }
    atomic_text(args.output_dir / "contract.json", json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    atomic_text(args.output_dir / "source_hashes.json", json.dumps(hashes, indent=2) + "\n")
    frozen_rows, _ = load_frozen_rows(fingerprint)
    jobs = [
        (gain, condition, trial)
        for gain in (0.5, 1.0)
        for condition in CONDITIONS
        for trial in [-1, *range(args.trials)]
        if not (gain == 0.5 and condition in FROZEN_CONDITIONS)
    ]
    if args.smoke:
        jobs = [
            (0.5, "combined_0.001deg_1pct", -1),
            (1.0, "link_bias_0.001deg", -1),
        ]
    rows = list(frozen_rows)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_new, *job, str(args.output_dir), fingerprint["fingerprint"]): job for job in jobs}
        for index, future in enumerate(as_completed(pending), 1):
            row = future.result()
            rows.append(row)
            if index % 20 == 0 or index == len(jobs):
                print(f"new {index}/{len(jobs)}: gain={row['gain']:g} {row['condition']} trial={row['trial']} {row['status']} Emax={row['max_position_error_m']:.5g}", flush=True)
    rows = trial_csv_rows(rows, args.output_dir)
    write_csv_union(args.output_dir / "trials.csv", rows)
    summaries = aggregate(rows)
    write_csv_union(args.output_dir / "summary.csv", summaries)
    atomic_text(args.output_dir / "summary.json", json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    manifest = {"complete": not args.smoke and len(rows) == 1616, "smoke": args.smoke, "trial_rows": len(rows), "new_expected": len(jobs), "new_cached_or_run": len(jobs), "reused": len(frozen_rows), "fingerprint": fingerprint["fingerprint"]}
    atomic_text(args.output_dir / "run_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
