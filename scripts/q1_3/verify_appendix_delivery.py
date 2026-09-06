"""Lead-review independent coordinate, cost, event and source checks.

No controller is rerun here. Bootstrap scores are rebuilt from stored position
trajectories. Schedule scores are rebuilt from stored commands and the frozen
simulator-private execution-error stream, independently of its simulation loop.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.q1_3.simulation_noise import SimulationNoise

ROOT = Path(__file__).resolve().parents[2]
TARGET = np.zeros((10, 2))
theta = np.deg2rad(np.arange(9) * 40)
TARGET[1:] = 100 * np.column_stack((np.cos(theta), np.sin(theta)))


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def close(actual, expected, label, tolerance=1e-8):
    if not np.allclose(actual, expected, atol=tolerance, rtol=1e-9):
        raise AssertionError(f"{label}: {actual} != {expected}")


def verify_sources(contract):
    for name, expected in contract["source_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"source changed: {name}")


def events(history, field):
    values = np.array([float(r[field]) for r in history])
    result = {}
    for label, threshold in (("1cm", 0.01), ("1mm", 0.001)):
        hits = np.flatnonzero(values < threshold)
        result[label] = int(hits[0]) if len(hits) else None
    return result


def metrics(points):
    errors = np.linalg.norm(points - TARGET, axis=1)[1:]
    return float(errors.max()), float(np.sqrt(np.mean(errors**2)))


def bootstrap():
    folder = ROOT / "appendix/outputs/fixed_budget_v1"
    contract = json.loads((folder / "contract.json").read_text())
    verify_sources(contract)
    paths = list(folder.glob("*/*/*/summary.json"))
    assert len(paths) == 1010, len(paths)
    result = []
    for path in paths:
        run = json.loads(path.read_text())
        e = run["evaluation"]
        assert run["fingerprint"] == contract["fingerprint"], path
        trajectory = run["trajectory"]
        states = np.array([[r["x_m"], r["y_m"]] for r in trajectory]).reshape(-1, 10, 2)
        history = run["history"]
        assert len(states) == len(history) == e["measurement_slots"] + 1, path
        assert all(
            r["drone_id"] == i % 10 and r["slot"] == i // 10
            for i, r in enumerate(trajectory)
        ), path
        close(states[0], run["initial_positions"], str(path))
        close(states[-1], run["final_positions"], str(path))
        close(
            states[:, :2], np.broadcast_to(TARGET[:2], states[:, :2].shape), str(path)
        )
        movement = 0.0
        uses = 0
        for slot, (points, row) in enumerate(zip(states, history)):
            maximum, rms = metrics(points)
            close(maximum, float(row["formation_max_error_m"]), str(path))
            close(rms, float(row["formation_rms_error_m"]), str(path))
            if slot:
                tx = list(map(int, row["transmitters"].split("-")))
                close(points[tx], states[slot - 1, tx], f"transmitter moved {path}")
                movement += float(
                    np.linalg.norm(points - states[slot - 1], axis=1).sum()
                )
                uses += len(tx)
            close(movement, float(row["cumulative_endpoint_displacement_m"]), str(path))
            assert uses == int(row["cumulative_transmitter_uses"]), path
        close(movement, e["movement_m"], str(path))
        assert uses == e["transmitter_uses"], path
        maximum, rms = metrics(states[-1])
        close(maximum, e["formation_max_error_m"], str(path))
        close(rms, e["formation_rms_error_m"], str(path))
        for label, first in events(history, "formation_max_error_m").items():
            assert first == e[f"first_{label}_slot"], path
        assert e["terminal_below_1cm"] == (maximum < 0.01), path
        result.append(
            {
                "method": e["version"],
                "condition": e["condition"],
                "trial": e["trial"],
                "complete": e["fixed_budget_complete"],
                "max_error_m": maximum,
                "rms_error_m": rms,
                "slots": e["measurement_slots"],
                "transmitter_uses": uses,
            }
        )
    return result


def optimized_main():
    folder = ROOT / "appendix1/evaluation_560"
    contract = json.loads((folder / "contract.json").read_text())
    verify_sources(contract)
    paths = list(folder.glob("*/*/summary.json"))
    assert len(paths) == 505, len(paths)
    result = []
    for index, path in enumerate(paths):
        run = json.loads(path.read_text())
        e = run["evaluation"]
        assert run["fingerprint"] == contract["fingerprint"], path
        points = np.array(run["initial_positions"], dtype=float)
        close(points[:2], TARGET[:2], str(path))
        history = read_csv(path.parent / "slot_metrics.csv")
        decisions = read_csv(path.parent / "decisions.csv")
        noise = SimulationNoise(
            e["bearing_std_deg"],
            e["actuation_relative_std"],
            2026090500 + e["trial"] + 1,
        )
        assert len(history) == e["measurement_slots"] + 1, path
        assert len(decisions) == 6 * e["measurement_slots"], path
        movement = 0.0
        first_quiet_cycle = None
        cycle_quiet = True
        for slot, row in enumerate(history):
            if slot:
                before = points.copy()
                tx = set(map(int, row["transmitters"].split("-")))
                assert tx == ({0, 1, 4, 5} if slot % 2 else {0, 1, 7, 8}), path
                if slot % 2:
                    cycle_quiet = True
                actions = decisions[6 * (slot - 1) : 6 * slot]
                cycle_quiet &= all(a["status"] == "within_tolerance" for a in actions)
                if slot % 2 == 0 and cycle_quiet and first_quiet_cycle is None:
                    first_quiet_cycle = slot
                assert {int(r["receiver_id"]) for r in actions} == set(
                    range(10)
                ) - tx, path
                for action in actions:
                    assert int(action["slot"]) == slot, path
                    receiver = int(action["receiver_id"])
                    dr = float(action["radial_command_m"])
                    da = float(action["angular_command_rad"])
                    if dr or da:
                        dr, da = noise.execute(dr, da, slot, receiver)
                        radius = float(np.linalg.norm(before[receiver])) + dr
                        angle = (
                            float(np.arctan2(before[receiver, 1], before[receiver, 0]))
                            + da
                        )
                        points[receiver] = radius * np.array(
                            [np.cos(angle), np.sin(angle)]
                        )
                movement += float(np.linalg.norm(points - before, axis=1).sum())
                close(points[list(tx)], before[list(tx)], str(path))
            maximum, rms = metrics(points)
            close(maximum, float(row["max_position_error_m"]), str(path))
            close(rms, float(row["rms_position_error_m"]), str(path))
            close(movement, float(row["cumulative_endpoint_m"]), str(path))
            assert int(row["cumulative_transmitter_uses"]) == 4 * slot, path
        close(points, run["final_positions"], str(path))
        close(movement, e["movement_m"], str(path))
        close(maximum, e["max_position_error_m"], str(path))
        close(rms, e["rms_position_error_m"], str(path))
        assert e["transmitter_uses"] == 4 * e["measurement_slots"], path
        assert e["final_below_1cm"] == (maximum < 0.01), path
        if e["stopped"]:
            assert first_quiet_cycle == e["measurement_slots"], path
            assert e["status"] == "quiet_full_selected_cycle", path
        else:
            assert first_quiet_cycle is None, path
            assert e["status"] in {"budget_exhausted", "local_fit_failed"}, path
            if e["status"] == "budget_exhausted":
                assert e["measurement_slots"] == contract["max_slots"], path
        for label, first in events(history, "max_position_error_m").items():
            assert first == e[f"first_{label}_slot"], path
        result.append(
            {
                "method": "appendix1",
                "condition": e["condition"],
                "trial": e["trial"],
                "complete": e["stopped"],
                "max_error_m": maximum,
                "rms_error_m": rms,
                "slots": e["measurement_slots"],
                "transmitter_uses": e["transmitter_uses"],
            }
        )
        if (index + 1) % 100 == 0:
            print(f"Rebuilt {index + 1}/505 schedule trajectories", flush=True)
    return result


def verify_main_pairing():
    """Check paired baseline values against the original baseline records."""
    baseline = ROOT / "outputs/q1_3/robustness"
    contract = json.loads((baseline / "contract.json").read_text())
    original = {
        (r["condition"], int(r["trial"])): r
        for r in read_csv(baseline / "trials.csv")
    }
    paired = read_csv(ROOT / "appendix1/evaluation_560/paired_trials.csv")
    assert len(paired) == len(original) == 505
    assert {(r["condition"], int(r["trial"])) for r in paired} == set(original)
    for row in paired:
        condition, trial = row["condition"], int(row["trial"])
        source = original[condition, trial]
        name = "table1" if trial == -1 else f"trial_{trial:03d}"
        folder = baseline / condition / name
        saved = json.loads((folder / "summary.json").read_text())
        assert saved["fingerprint"] == contract["fingerprint"]
        candidate = json.loads(
            (ROOT / "appendix1/evaluation_560" / condition / name / "summary.json").read_text()
        )
        close(saved["initial_positions"], candidate["initial_positions"], "paired initial")
        for metric in (
            "measurement_slots", "transmitter_uses", "movement_m",
            "max_position_error_m", "rms_position_error_m",
        ):
            close(float(row[f"main_{metric}"]), float(source[metric]), "paired main")
        history = read_csv(folder / "slot_metrics.csv")
        for label, first in events(history, "max_position_error_m").items():
            field = row[f"main_first_{label}_slot"]
            assert (None if field == "" else int(float(field))) == first
    return len(paired)


def main():
    rows = bootstrap() + optimized_main()
    paired_runs = verify_main_pairing()
    summary = []
    for method, condition in sorted({(r["method"], r["condition"]) for r in rows}):
        group = [
            r
            for r in rows
            if (r["method"], r["condition"]) == (method, condition) and r["trial"] >= 0
        ]
        assert sorted(r["trial"] for r in group) == list(range(100))
        summary.append(
            {
                "method": method,
                "condition": condition,
                "random_runs": len(group),
                "completed_count": sum(r["complete"] for r in group),
                "terminal_below_1cm_count": sum(r["max_error_m"] < 0.01 for r in group),
                "median_max_error_m": float(
                    np.median([r["max_error_m"] for r in group])
                ),
                "p95_max_error_m": float(
                    np.quantile([r["max_error_m"] for r in group], 0.95)
                ),
            }
        )
    # Compare the independently rebuilt statistics with both execution reports.
    a_report = json.loads(
        (ROOT / "appendix/outputs/fixed_budget_v1/summary.json").read_text()
    )["random100"]
    b_report = json.loads((ROOT / "appendix1/evaluation_560/summary.json").read_text())
    for row in summary:
        if row["method"] == "appendix1":
            published = next(r for r in b_report if r["condition"] == row["condition"])
            assert published["runs"] == row["random_runs"]
            assert published["stopped_count"] == row["completed_count"]
            assert published["final_below_1cm_count"] == row["terminal_below_1cm_count"]
            close(
                published["max_position_error_m_median"],
                row["median_max_error_m"],
                "schedule median",
            )
            close(
                published["max_position_error_m_p95"],
                row["p95_max_error_m"],
                "schedule p95",
            )
        else:
            published = next(
                r
                for r in a_report
                if r["version"] == row["method"] and r["condition"] == row["condition"]
            )
            assert published["denominator_runs"] == row["random_runs"]
            assert published["fixed_budget_complete_count"] == row["completed_count"]
            assert (
                published["terminal_below_1cm_count"] == row["terminal_below_1cm_count"]
            )
            close(
                published["formation_max_error_m_median"],
                row["median_max_error_m"],
                "bootstrap median",
            )
            close(
                published["formation_max_error_m_p95"],
                row["p95_max_error_m"],
                "bootstrap p95",
            )
    folder = ROOT / "outputs/q1_3/appendix_delivery_review"
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "verified_runs": len(rows),
        "bootstrap_runs": 1010,
        "schedule_runs": 505,
        "independently_verified_main_pairs": paired_runs,
        "checks": "Current source hashes, per-run fingerprints, coordinate metrics with nine-aircraft RMS, roles, fixed references, cumulative motion and transmitter uses, first precision events; schedule positions independently rebuilt from command records and deterministic actuator-error stream.",
        "summary": summary,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (folder / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
