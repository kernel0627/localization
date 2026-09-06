"""Lead review: reconstruct new diagnostic runs and recompute reported metrics."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.q1_2.run_validation import table_positions

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/q1_3/two_configuration_analysis"
ANGLES = np.deg2rad(np.arange(9) * 40)
TARGET = np.vstack(
    (np.zeros(2), 100 * np.column_stack((np.cos(ANGLES), np.sin(ANGLES))))
)


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-8)


def main():
    source_hashes = json.loads((DATA / "source_hashes.json").read_text())
    for name, expected in source_hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    cases = read_csv(DATA / "table1_evaluation.csv")
    assert len(cases) == 7
    costs = read_csv(DATA / "precision_costs.csv")
    states_checked = 0
    for case in cases:
        folder = DATA / case["case"]
        summary = json.loads((folder / "summary.json").read_text())
        history = read_csv(folder / "slot_metrics.csv")
        actions = read_csv(folder / "decisions.csv")
        points = np.array([table_positions()[i] for i in range(10)])
        motion = 0.0
        last_motion = 0
        quiet_cycle = None
        quiet = True
        maximum_errors = []
        for slot, row in enumerate(history):
            if slot:
                before = points.copy()
                transmitters = {0, 1, 4, 5} if slot % 2 else {0, 1, 7, 8}
                assert set(map(int, row["transmitters"].split("-"))) == transmitters
                slot_actions = actions[(slot - 1) * 6 : slot * 6]
                assert len(slot_actions) == 6
                assert {int(a["receiver_id"]) for a in slot_actions} == set(
                    range(10)
                ) - transmitters
                if slot % 2:
                    quiet = True
                quiet &= all(a["status"] == "within_tolerance" for a in slot_actions)
                if not slot % 2 and quiet and quiet_cycle is None:
                    quiet_cycle = slot
                for action in slot_actions:
                    assert int(action["slot"]) == slot
                    receiver = int(action["receiver_id"])
                    dr, da = (
                        float(action["radial_command_m"]),
                        float(action["angular_command_rad"]),
                    )
                    assert abs(dr) <= float(case["max_radial_step_m"]) + 1e-12
                    assert (
                        abs(da)
                        <= np.deg2rad(float(case["max_angular_step_deg"])) + 1e-12
                    )
                    if dr or da:
                        last_motion = slot
                        radius = np.linalg.norm(before[receiver]) + dr
                        theta = (
                            np.arctan2(before[receiver, 1], before[receiver, 0]) + da
                        )
                        points[receiver] = radius * np.array(
                            [np.cos(theta), np.sin(theta)]
                        )
                motion += np.linalg.norm(points - before, axis=1).sum()
                close(points[list(transmitters)], before[list(transmitters)])
            errors = np.linalg.norm(points[1:] - TARGET[1:], axis=1)
            maximum_errors.append(float(errors.max()))
            close(errors.max(), float(row["max_position_error_m"]))
            close(np.sqrt(np.mean(errors**2)), float(row["rms_position_error_m"]))
            close(
                np.max(np.abs(np.linalg.norm(points[1:], axis=1) - 100)),
                float(row["max_radial_error_m"]),
            )
            delta = np.arctan2(points[1:, 1], points[1:, 0]) - ANGLES
            angular = np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))
            close(np.rad2deg(angular), float(row["max_angular_error_deg"]))
            close(100 * angular, float(row["max_tangential_error_m"]))
            close(motion, float(row["cumulative_endpoint_m"]))
            assert int(row["cumulative_transmitter_uses"]) == 4 * slot
            states_checked += 1
        assert (
            last_motion == int(case["last_motion_slot"]) == summary["last_motion_slot"]
        )
        assert (
            quiet_cycle
            == int(case["measurement_slots"])
            == summary["measurement_slots"]
        )
        close(points, summary["final_positions"])
        close(motion, float(case["total_endpoint_displacement_m"]))
        close(maximum_errors[-1], float(case["max_position_error_m"]))
        for event in [r for r in costs if r["case"] == case["case"]]:
            below = np.array(maximum_errors) < float(event["threshold_m"])
            hits = np.flatnonzero(below)
            first = int(hits[0]) if len(hits) else None
            assert first == (int(event["first_slot"]) if event["first_slot"] else None)
            if first is not None:
                assert int(event["transmitter_uses_at_first"]) == first * 4
                close(
                    float(event["endpoint_m_at_first"]),
                    float(history[first]["cumulative_endpoint_m"]),
                )
                assert bool(np.all(below[first:])) == (
                    event["all_recorded_slots_after_first_below"] == "True"
                )
    old = read_csv(ROOT / "appendix1/evaluation_560/exact/table1/slot_metrics.csv")
    new = read_csv(DATA / "gain_0.5/slot_metrics.csv")
    assert len(old) == len(new)
    for a, b in zip(old, new):
        for field in a:
            if field == "transmitters":
                assert a[field] == b[field]
            else:
                close(float(a[field]), float(b[field]))
    gain_rows = read_csv(DATA / "gain_stability.csv")
    for row in gain_rows:
        gain = float(row["gain"])
        a = np.loadtxt(DATA / f"cycle_jacobian_gain_{gain:g}.csv", delimiter=",")
        close(np.max(np.abs(np.linalg.eigvals(a))), float(row["spectral_radius"]))
        close(np.linalg.norm(a, 2), float(row["cycle_operator_norm"]))
        close(np.linalg.norm(a @ a, 2), float(row["two_cycle_operator_norm"]))
    a = np.loadtxt(DATA / "cycle_jacobian_noise_gain_0.5.csv", delimiter=",")
    q = np.loadtxt(DATA / "cycle_noise_per_radian2.csv", delimiter=",")
    p = np.loadtxt(DATA / "stationary_covariance_per_radian2.csv", delimiter=",")
    covariance_residual = float(np.linalg.norm(p - a @ p @ a.T - q) / np.linalg.norm(q))
    assert covariance_residual < 1e-10
    assert np.linalg.eigvalsh(p).min() >= -1e-8
    noise_rows = read_csv(DATA / "noise_validation.csv")
    assert len(noise_rows) == 6
    for row in noise_rows:
        phase = int(row["phase"])
        squared = []
        for trial in range(100):
            history = read_csv(
                ROOT
                / "appendix1/evaluation_560"
                / row["condition"]
                / f"trial_{trial:03d}/slot_metrics.csv"
            )
            slot = 558 + phase
            assert int(history[slot]["slot"]) == slot
            squared.append(float(history[slot]["rms_position_error_m"]) ** 2)
        close(
            np.sqrt(np.mean(squared)),
            float(row["observed_root_expected_rms_squared_m"]),
        )
        if phase == 2:
            sigma = np.deg2rad(
                float(row["condition"].removeprefix("bearing_").removesuffix("deg"))
            )
            close(
                sigma * np.sqrt(np.trace(p) / 9),
                float(row["predicted_root_expected_rms_squared_m"]),
            )
    report = {
        "all_passed": True,
        "table1_parameter_runs": len(cases),
        "reconstructed_states": states_checked,
        "default_matches_frozen_run": True,
        "noise_condition_phase_groups": len(noise_rows),
        "noise_samples_per_group": 100,
        "covariance_relative_residual": covariance_residual,
        "checked_source_files": len(source_hashes),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = ROOT / "outputs/q1_3/two_configuration_review"
    output.mkdir(exist_ok=True, parents=True)
    (output / "analysis_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
