"""Independent nonlinear injection and saved-trajectory audit for robustness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.q1_3.analyze_iterative_reference import error_from_positions
from scripts.q1_3.analyze_two_configuration import SCHEDULE
from scripts.q1_3.local_adjustment import LocalSettings, decide_local_adjustment

ROOT = Path(__file__).resolve().parents[2]
TARGET = np.zeros((10, 2))
THETA = np.deg2rad(np.arange(9) * 40)
TARGET[1:] = 100 * np.column_stack((np.cos(THETA), np.sin(THETA)))


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_close(actual, expected, label, atol=1e-8):
    if not np.allclose(actual, expected, rtol=1e-8, atol=atol):
        raise AssertionError(f"{label}: values disagree")


def nonlinear_bias_cycle(gain, bias):
    """Inject link direction errors directly, independently of noise wrappers."""
    points = TARGET.copy()
    branches = []
    for tx in SCHEDULE:
        before = points.copy()
        ids = (0, *tx)
        for receiver in range(2, 10):
            if receiver in ids:
                continue
            rays = before[list(ids)] - before[receiver]
            directions = np.arctan2(rays[:, 1], rays[:, 0])
            directions += bias[receiver - 2, list(ids)]
            observations = np.array(
                [
                    abs(
                        np.arctan2(
                            np.sin(directions[a] - directions[b]),
                            np.cos(directions[a] - directions[b]),
                        )
                    )
                    for a, b in combinations(range(4), 2)
                ]
            )
            decision = decide_local_adjustment(
                receiver, tx, observations, LocalSettings(gain=gain)
            )
            assert decision.selected is not None
            assert abs(decision.radial_step_m) < 5
            assert abs(decision.angular_step_rad) < np.deg2rad(2)
            branches.append((receiver, decision.selected.pair))
            radius = np.linalg.norm(before[receiver]) + decision.radial_step_m
            angle = np.arctan2(before[receiver, 1], before[receiver, 0])
            angle += decision.angular_step_rad
            points[receiver] = radius * np.array([np.cos(angle), np.sin(angle)])
    return error_from_positions(points), branches


def metrics(points):
    distances = np.linalg.norm(points[1:] - TARGET[1:], axis=1)
    return float(distances.max()), float(np.sqrt(np.mean(distances**2)))


def audit_injections():
    from scripts.q1_3.analyze_two_configuration_robustness import (
        covariance_operator,
        phase_matrices_and_injections,
    )

    rng = np.random.default_rng(60260906)
    rows = []
    for gain in (0.5, 1.0):
        matrices, commands, injections, links = phase_matrices_and_injections(gain)
        loading = matrices[1] @ injections[0] + injections[1]
        _, nominal_branches = nonlinear_bias_cycle(gain, np.zeros((8, 10)))
        for direction_id in range(8):
            direction = rng.normal(size=(8, 10))
            direction /= np.linalg.norm(direction)
            expected = loading @ np.array([direction[r - 2, t] for r, t in links])
            for step in (1e-6, 3e-7):
                plus, plus_branches = nonlinear_bias_cycle(gain, step * direction)
                minus, minus_branches = nonlinear_bias_cycle(gain, -step * direction)
                assert plus_branches == minus_branches == nominal_branches
                measured = (plus - minus) / (2 * step)
                relative = np.linalg.norm(measured - expected) / np.linalg.norm(
                    expected
                )
                assert relative < 1e-4, (gain, direction_id, step, relative)
                rows.append(
                    {
                        "gain": gain,
                        "direction": direction_id,
                        "step_rad": step,
                        "relative_derivative_error": float(relative),
                        "max_absolute_error_m_per_rad": float(
                            np.max(np.abs(measured - expected))
                        ),
                    }
                )
        for matrix, command in zip(matrices, commands):
            factor = rng.normal(size=(16, 16))
            covariance = factor @ factor.T
            operator = covariance_operator(matrix, command, 0.1)
            actual = (operator @ covariance.reshape(-1, order="F")).reshape(
                16, 16, order="F"
            )
            expected = matrix @ covariance @ matrix.T + 0.01 * np.diag(
                np.diag(command @ covariance @ command.T)
            )
            check_close(
                actual, expected, "independent second-moment operator", atol=1e-10
            )
    return {
        "nonlinear_cycle_probes": 2 * len(rows),
        "checks": rows,
        "second_moment_operator_checks": 4,
        "maximum_relative_derivative_error": max(
            r["relative_derivative_error"] for r in rows
        ),
    }


def audit_run(path):
    """Rebuild every state from commands and an independently generated stream."""
    path = Path(path)
    saved = json.loads(path.read_text())
    evaluation = saved["evaluation"]
    history = read_csv(path.parent / "slot_metrics.csv")
    actions = read_csv(path.parent / "decisions.csv")
    points = np.array(saved["initial_positions"], dtype=float)
    initial = points.copy()
    trial = int(evaluation["trial"])
    if trial >= 0:
        initial_rng = np.random.default_rng(
            np.random.SeedSequence([20260905, trial, 7])
        )
        radii = 100 + initial_rng.uniform(-12, 12, 8)
        angles = np.deg2rad(np.arange(1, 9) * 40 + initial_rng.uniform(-0.3, 0.3, 8))
        expected_initial = radii[:, None] * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        check_close(initial[2:], expected_initial, "random initial law and seed")
    count = int(evaluation["measurement_slots"])
    assert len(history) == count + 1 and len(actions) == 6 * count
    check_close(points[:2], TARGET[:2], "calibrated references")
    movement = 0.0
    errors = []
    first_quiet = None
    quiet = True
    seed = 2026090500 + int(evaluation["trial"]) + 1
    sigma = float(evaluation["actuation_relative_std"])
    for slot, row in enumerate(history):
        assert int(row["slot"]) == slot
        if slot:
            before = points.copy()
            transmitters = {0, *SCHEDULE[(slot - 1) % 2]}
            assert set(map(int, row["transmitters"].split("-"))) == transmitters
            decisions = actions[6 * (slot - 1) : 6 * slot]
            assert {int(d["receiver_id"]) for d in decisions} == set(
                range(10)
            ) - transmitters
            if slot % 2:
                quiet = True
            quiet &= all(d["status"] == "within_tolerance" for d in decisions)
            if not slot % 2 and quiet and first_quiet is None:
                first_quiet = slot
            for decision in decisions:
                receiver = int(decision["receiver_id"])
                assert int(decision["slot"]) == slot
                radial = float(decision["radial_command_m"])
                angular = float(decision["angular_command_rad"])
                assert abs(radial) <= 5 + 1e-12
                assert abs(angular) <= np.deg2rad(2) + 1e-12
                if radial or angular:
                    if sigma:
                        rng = np.random.default_rng(
                            np.random.SeedSequence([seed, slot, receiver, 1])
                        )
                        factors = 1 + rng.normal(0, sigma, 2)
                        radial *= factors[0]
                        angular *= factors[1]
                    radius = np.linalg.norm(before[receiver]) + radial
                    angle = (
                        np.arctan2(before[receiver, 1], before[receiver, 0]) + angular
                    )
                    points[receiver] = radius * np.array([np.cos(angle), np.sin(angle)])
            check_close(
                points[list(transmitters)],
                before[list(transmitters)],
                "transmitter immobility",
            )
            movement += np.linalg.norm(points - before, axis=1).sum()
        maximum, rms = metrics(points)
        errors.append(maximum)
        check_close(maximum, float(row["max_position_error_m"]), "geometric maximum")
        check_close(rms, float(row["rms_position_error_m"]), "geometric RMS")
        check_close(
            movement,
            float(row["cumulative_endpoint_m"]),
            "cumulative endpoint movement",
        )
        assert int(row["cumulative_transmitter_uses"]) == 4 * slot
    check_close(points, saved["final_positions"], "final coordinates")
    check_close(movement, evaluation["movement_m"], "movement evaluation")
    check_close(maximum, evaluation["max_position_error_m"], "maximum evaluation")
    check_close(rms, evaluation["rms_position_error_m"], "RMS evaluation")
    assert evaluation["transmitter_uses"] == 4 * count
    assert bool(evaluation["stopped"]) == (first_quiet is not None)
    if first_quiet is not None:
        assert first_quiet == count
    elif evaluation["status"] == "budget_exhausted":
        assert count == 560
    for label, threshold in (("1cm", 0.01), ("1mm", 0.001)):
        good = np.asarray(errors) < threshold
        hits, bad = np.flatnonzero(good), np.flatnonzero(~good)
        first = int(hits[0]) if len(hits) else None
        sustained = (
            int(bad[-1] + 1)
            if len(bad) and bad[-1] < count
            else (0 if not len(bad) else None)
        )
        assert first == evaluation[f"first_{label}_slot"]
        assert sustained == evaluation[f"sustained_{label}_from_slot"]
        assert bool(evaluation[f"final_below_{label}"]) == bool(good[-1])
        assert bool(evaluation[f"joint_success_{label}"]) == bool(
            good[-1] and first_quiet is not None
        )
    diagnostic_path = path.parent / "controller_diagnostics_last100.csv"
    diagnostic_count = 0
    if diagnostic_path.exists():
        remembered = {}
        diagnostic_rows = read_csv(diagnostic_path)
        for diagnostic in diagnostic_rows:
            receiver = int(diagnostic["receiver_id"])
            before = (
                int(diagnostic["small_count_before"]),
                diagnostic["holding_before"] == "True",
            )
            if receiver in remembered:
                assert remembered[receiver] == before
            local_count, holding = before
            if not diagnostic["selected_pair"]:
                assert diagnostic["ratio"] == ""
                assert int(diagnostic["small_count_after"]) == 0
                assert diagnostic["holding_after"] == "False"
                remembered[receiver] = (0, False)
                continue
            ratio = max(
                abs(float(diagnostic["estimated_radial_bias_m"])) / 1e-3,
                abs(float(diagnostic["estimated_angular_bias_rad"])) / 1e-5,
                float(diagnostic["consistency_rad"]) / 1e-5,
            )
            check_close(
                ratio, float(diagnostic["ratio"]), "local normalized tolerance ratio"
            )
            if holding and ratio > 1:
                holding, local_count = False, 0
            if not holding:
                local_count = local_count + 1 if ratio <= 0.5 else 0
                holding = local_count >= 21
            after = (
                int(diagnostic["small_count_after"]),
                diagnostic["holding_after"] == "True",
            )
            assert after == (local_count, holding)
            assert (diagnostic["status"] == "within_tolerance") == holding
            remembered[receiver] = after
        diagnostic_count = len(diagnostic_rows)
        assert diagnostic_count == evaluation["tail100_local_decisions"]
    return {
        "condition": evaluation["condition"],
        "trial": evaluation["trial"],
        "states": len(history),
        "local_diagnostic_decisions": diagnostic_count,
        "source": str(path.relative_to(ROOT)),
        "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "outputs/q1_3/two_configuration_robustness"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/q1_3/two_configuration_robustness_review",
    )
    parser.add_argument("--math-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.math_only:
        result = audit_injections()
        (args.output / "nonlinear_noise_injection_audit.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True
        )
        return
    contract = json.loads((args.data / "contract.json").read_text())
    manifest = json.loads((args.data / "run_manifest.json").read_text())
    assert manifest["complete"] and manifest["trial_rows"] == 1616
    for name, expected in contract["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    index_rows = read_csv(args.data / "trials.csv")
    expected_keys = {
        (g, c, t)
        for g in (0.5, 1.0)
        for c in contract["conditions"]
        for t in range(-1, 100)
    }
    assert len(index_rows) == len(expected_keys) == 1616
    assert {
        (float(r["gain"]), r["condition"], int(float(r["trial"]))) for r in index_rows
    } == expected_keys
    paths = [ROOT / row["summary_path"] for row in index_rows]
    for row, path in zip(index_rows, paths):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["cache_sha256"]
        saved = json.loads(path.read_text())
        assert saved["evaluation"]["condition"] == row["condition"]
        assert saved["evaluation"]["trial"] == int(float(row["trial"]))
        check_close(
            float(row["gain"]), saved["summary"]["settings"]["gain"], "indexed gain"
        )
        for key, value in saved["evaluation"].items():
            assert key in row, (path, key)
            indexed = row[key]
            if value is None:
                assert indexed == "", (path, key)
            elif isinstance(value, bool):
                assert indexed == str(value), (path, key)
            elif isinstance(value, (int, float)):
                check_close(float(indexed), value, f"index field {key}")
            else:
                assert indexed == value, (path, key)
        expected_fingerprint = (
            contract["archive_fingerprint"]
            if row["source_kind"] == "frozen_appendix1_reuse"
            else contract["fingerprint"]
        )
        assert saved["fingerprint"] == expected_fingerprint
    rows = []
    paired_initials = {}
    for index, (path, indexed) in enumerate(zip(paths, index_rows), 1):
        audited = audit_run(path)
        audited["gain"] = float(indexed["gain"])
        key = (audited["condition"], audited["trial"])
        if key in paired_initials:
            assert paired_initials[key] == audited["initial_sha256"]
        paired_initials[key] = audited["initial_sha256"]
        rows.append(audited)
        if index % 100 == 0:
            print(f"independently rebuilt {index}/{len(paths)} runs", flush=True)
    result = {
        "runs": len(rows),
        "states": sum(r["states"] for r in rows),
        "paired_initial_states": len(paired_initials),
        "source_hashes_match": True,
        "summary_hashes_match": True,
        "records": rows,
    }
    (args.output / "trajectory_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
