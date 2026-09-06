"""Formal local and finite-trajectory analysis for the frozen two-slot controller."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from appendix1.optimize_schedule import simulate
from scripts.q1_1.localization import angle_jacobian, pairwise_angles
from scripts.q1_2.run_validation import table_positions, write_csv
from scripts.q1_3.analyze_iterative_reference import (
    DIMENSION,
    cycle_matrix,
    error_from_positions,
    linear_model,
    nominal_positions,
    polar_basis,
    positions_from_error,
    state_slice,
)
from scripts.q1_3.analyze_noise_floor import ray_angle_derivative
from scripts.q1_3.local_adjustment import (
    LocalSettings,
    decide_local_adjustment,
    execute_relative_polar_step,
    public_schedule,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEDULE = ((1, 4, 5), (1, 7, 8))
STATE_ORDER = [f"FY{i:02d}_{c}" for i in range(2, 10) for c in ("dr_m", "R_dtheta_m")]


def source_paths():
    names = [
        "appendix1/optimize_schedule.py",
        "appendix1/run_evaluation.py",
        "scripts/q1_1/localization.py",
        "scripts/q1_2/run_validation.py",
        "scripts/q1_3/analyze_iterative_reference.py",
        "scripts/q1_3/analyze_noise_floor.py",
        "scripts/q1_3/local_adjustment.py",
        "scripts/q1_3/run_iterative_reference_baseline.py",
        "scripts/q1_3/simulation_noise.py",
        "problem/B题.md",
    ]
    return {
        "scripts/q1_3/analyze_two_configuration.py": Path(__file__),
        **{n: ROOT / n for n in names},
    }


def hashes(paths):
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def verify_contract():
    contract = json.loads((ROOT / "appendix1/evaluation_560/contract.json").read_text())
    expected = contract["source_sha256"]
    if hashes({key: ROOT / key for key in expected}) != expected:
        raise RuntimeError("frozen 505 contract source hashes do not match")
    with (ROOT / "appendix1/evaluation_560/trials.csv").open(newline="") as handle:
        trials = list(csv.DictReader(handle))
    if len(trials) != 505:
        raise RuntimeError(f"expected 505 frozen trial rows, got {len(trials)}")
    expected_keys = {
        (condition, trial)
        for condition in contract["conditions"]
        for trial in range(-1, 100)
    }
    if {(row["condition"], int(row["trial"])) for row in trials} != expected_keys:
        raise RuntimeError("frozen trial keys are incomplete or duplicated")
    for row in trials:
        leaf = "table1" if int(row["trial"]) == -1 else f"trial_{int(row['trial']):03d}"
        summary = json.loads(
            (
                ROOT
                / "appendix1/evaluation_560"
                / row["condition"]
                / leaf
                / "summary.json"
            ).read_text()
        )
        evaluation = summary.get("evaluation", {})
        if summary.get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError(f"fingerprint mismatch for {row['condition']}/{leaf}")
        if not {
            "condition",
            "trial",
            "measurement_slots",
            "rms_position_error_m",
        } <= set(evaluation):
            raise RuntimeError(f"schema mismatch for {row['condition']}/{leaf}")
        if evaluation["condition"] != row["condition"] or evaluation["trial"] != int(
            row["trial"]
        ):
            raise RuntimeError(f"wrong trial identity for {row['condition']}/{leaf}")
    return contract, {"trial_rows": 505, "source_hashes_match": True}


def phase(source_phase):
    return 1 if source_phase == public_schedule().index(SCHEDULE[0]) + 1 else 2


def selected_matrices(gain):
    matrices, choices, blocks = linear_model(gain)
    indexes = [public_schedule().index(item) for item in SCHEDULE]
    valid = {index + 1 for index in indexes}
    return (
        [matrices[i] for i in indexes],
        [r for r in choices if r["phase"] in valid],
        [r for r in blocks if r["phase"] in valid],
    )


def active_cycle(error, settings):
    points = positions_from_error(error)
    selected = []
    clipped = failures = 0
    for slot, transmitters in enumerate(SCHEDULE, 1):
        before, after = points.copy(), points.copy()
        for receiver in range(2, 10):
            if receiver in transmitters:
                continue
            decision = decide_local_adjustment(
                receiver,
                transmitters,
                pairwise_angles(before[receiver], before[[0, *transmitters]]),
                settings,
            )
            if decision.selected is None:
                failures += 1
                continue
            selected.append((slot, receiver, decision.selected.pair))
            clipped += int(
                abs(settings.gain * decision.selected.radial_bias_m)
                > settings.max_radial_step_m
                or abs(settings.gain * decision.selected.angular_bias_rad)
                > settings.max_angular_step_rad
            )
            after[receiver] = execute_relative_polar_step(
                before[receiver], decision.radial_step_m, decision.angular_step_rad
            )
        points = after
    return error_from_positions(points), selected, clipped, failures


def fd_checks(out):
    rows = []
    for gain in (0.25, 0.5, 1.0):
        matrices, choices, _ = selected_matrices(gain)
        analytic = cycle_matrix(matrices)
        expected = sorted(
            (
                phase(r["phase"]),
                r["receiver_id"],
                tuple(map(int, r["selected_pair"].split("-"))),
            )
            for r in choices
        )
        for h in (1e-3, 1e-4, 1e-5):
            numeric = np.empty((DIMENSION, DIMENSION))
            switches = clips = failures = 0
            for col in range(DIMENSION):
                values = []
                for sign in (1, -1):
                    value, chosen, nclip, nfail = active_cycle(
                        sign * h * np.eye(DIMENSION)[col], LocalSettings(gain=gain)
                    )
                    values.append(value)
                    switches += sum(a != b for a, b in zip(chosen, expected))
                    clips += nclip
                    failures += nfail
                numeric[:, col] = (values[0] - values[1]) / (2 * h)
            rows.append(
                {
                    "gain": gain,
                    "difference_step_m": h,
                    "absolute_frobenius_error": float(
                        np.linalg.norm(numeric - analytic)
                    ),
                    "relative_frobenius_error": float(
                        np.linalg.norm(numeric - analytic) / np.linalg.norm(analytic)
                    ),
                    "max_entry_error": float(np.max(np.abs(numeric - analytic))),
                    "selection_changes": switches,
                    "clipped_decisions": clips,
                    "failed_fits": failures,
                }
            )
    write_csv(out / "cycle_difference_checks.csv", rows)
    return rows


def blocks(matrix, gain):
    radial = np.arange(0, DIMENSION, 2)
    perm = np.r_[radial, radial + 1]
    values = matrix[np.ix_(perm, perm)]
    component = [
        ("radial", "radial", values[:8, :8]),
        ("tangential", "radial", values[:8, 8:]),
        ("radial", "tangential", values[8:, :8]),
        ("tangential", "tangential", values[8:, 8:]),
    ]
    radial_rows = [
        {
            "gain": gain,
            "input_component": i,
            "output_component": o,
            "operator_norm": float(np.linalg.norm(x, 2)),
            "frobenius_norm": float(np.linalg.norm(x)),
        }
        for i, o, x in component
    ]

    def ids(drones):
        return np.ravel(
            [
                [state_slice(drone).start, state_slice(drone).start + 1]
                for drone in drones
            ]
        )

    core, rest = ids((4, 5, 7, 8)), ids((2, 3, 6, 9))
    expected = (1 - gain) ** 2
    self_error = max(
        np.linalg.norm(matrix[state_slice(d), state_slice(d)] - expected * np.eye(2), 2)
        for d in (2, 3, 6, 9)
    )
    core_rows = [
        {
            "gain": gain,
            "core_ids": "FY04-FY05-FY07-FY08",
            "other_ids": "FY02-FY03-FY06-FY09",
            "core_operator_norm": float(np.linalg.norm(matrix[np.ix_(core, core)], 2)),
            "upper_right_operator_norm": float(
                np.linalg.norm(matrix[np.ix_(core, rest)], 2)
            ),
            "upper_right_max_abs": float(np.max(np.abs(matrix[np.ix_(core, rest)]))),
            "expected_other_self_block_scalar": expected,
            "other_self_block_max_error_from_expected_I": float(self_error),
        }
    ]
    return radial_rows, core_rows


def threshold_rows(history, case):
    rows = []
    for label, threshold in (("2cm", 0.02), ("1cm", 0.01), ("1mm", 0.001)):
        hits = [row for row in history if row["max_position_error_m"] < threshold]
        first = hits[0] if hits else None
        rows.append(
            {
                "case": case,
                "threshold": label,
                "threshold_m": threshold,
                "first_slot": first["slot"] if first else None,
                "transmitter_uses_at_first": first["cumulative_transmitter_uses"]
                if first
                else None,
                "endpoint_m_at_first": first["cumulative_endpoint_m"]
                if first
                else None,
                "all_recorded_slots_after_first_below": bool(first)
                and all(
                    row["max_position_error_m"] < threshold
                    for row in history[first["slot"] :]
                ),
            }
        )
    return rows


def table1(out):
    initial = np.array([table_positions()[i] for i in range(10)])
    default = LocalSettings()
    cases = [
        ("gain_0.25", replace(default, gain=0.25)),
        ("gain_0.5", default),
        ("gain_1", replace(default, gain=1.0)),
        ("radial_limit_2.5m", replace(default, max_radial_step_m=2.5)),
        ("radial_limit_10m", replace(default, max_radial_step_m=10.0)),
        ("angular_limit_1deg", replace(default, max_angular_step_rad=np.deg2rad(1))),
        ("angular_limit_4deg", replace(default, max_angular_step_rad=np.deg2rad(4))),
    ]
    rows, costs = [], []
    for name, settings in cases:
        summary, history, decisions = simulate(
            initial, SCHEDULE, max_slots=560, settings=settings
        )
        for row in history:
            row["max_tangential_error_m"] = float(
                100 * np.deg2rad(row["max_angular_error_deg"])
            )
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / "slot_metrics.csv", history)
        write_csv(folder / "decisions.csv", decisions)
        (folder / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        rows.append(
            {
                "case": name,
                "gain": settings.gain,
                "max_radial_step_m": settings.max_radial_step_m,
                "max_angular_step_deg": float(
                    np.rad2deg(settings.max_angular_step_rad)
                ),
                "status": summary["status"],
                "measurement_slots": summary["measurement_slots"],
                "transmitter_uses": summary["transmitter_uses"],
                "total_endpoint_displacement_m": summary["endpoint_m"],
                "last_motion_slot": summary["last_motion_slot"],
                "post_motion_confirmation_slots": summary[
                    "confirmation_slots_after_last_motion"
                ],
                "failed_fits": summary["failed_fits"],
                "max_position_error_m": summary["final_max_error_m"],
                "rms_position_error_m": summary["final_metrics"][
                    "rms_position_error_m"
                ],
            }
        )
        costs.extend(threshold_rows(history, name))
    write_csv(out / "table1_evaluation.csv", rows)
    write_csv(out / "precision_costs.csv", costs)
    return rows


def noise_analysis(out, matrices, choices, contract):
    nominal = nominal_positions()
    choice = {
        (phase(r["phase"]), r["receiver_id"]): tuple(
            map(int, r["selected_pair"].split("-"))
        )
        for r in choices
    }
    injections, accumulated, cycle = (
        [],
        np.zeros((DIMENSION, DIMENSION)),
        np.eye(DIMENSION),
    )
    for slot, (transmitters, matrix) in enumerate(zip(SCHEDULE, matrices), 1):
        injection = np.zeros((DIMENSION, DIMENSION))
        for receiver in range(2, 10):
            if receiver in transmitters:
                continue
            anchors = nominal[[0, *choice[slot, receiver]]]
            rays = anchors - nominal[receiver]
            response = (
                -0.5
                * np.linalg.pinv(
                    angle_jacobian(nominal[receiver], anchors) @ polar_basis(receiver)
                )
                @ ray_angle_derivative(np.arctan2(rays[:, 1], rays[:, 0]))
            )
            injection[state_slice(receiver), state_slice(receiver)] = (
                response @ response.T
            )
        accumulated = matrix @ accumulated @ matrix.T + injection
        cycle = matrix @ cycle
        injections.append(injection)
    covariance = solve_discrete_lyapunov(cycle, accumulated)
    residual = float(
        np.linalg.norm(covariance - cycle @ covariance @ cycle.T - accumulated)
        / np.linalg.norm(accumulated)
    )
    if np.linalg.eigvalsh(covariance).min() < -1e-9 or residual >= 1e-12:
        raise RuntimeError("covariance PSD or Lyapunov validation failed")
    unit_phase_rows, current = [], covariance.copy()
    for slot, (matrix, injection) in enumerate(zip(matrices, injections), 1):
        current = matrix @ current @ matrix.T + injection
        unit_phase_rows.append(
            {
                "phase": slot,
                "root_expected_rms_squared_m_per_radian": float(
                    np.sqrt(np.trace(current) / 9)
                ),
            }
        )
    closure = float(np.max(np.abs(current - covariance)))
    if closure >= 1e-7:
        raise RuntimeError("two-phase covariance did not close")
    np.savetxt(out / "cycle_jacobian_noise_gain_0.5.csv", cycle, delimiter=",")
    np.savetxt(out / "cycle_noise_per_radian2.csv", accumulated, delimiter=",")
    np.savetxt(out / "stationary_covariance_per_radian2.csv", covariance, delimiter=",")
    phase_rows = [
        {
            "phase": item["phase"],
            "bearing_std_deg": noise,
            "predicted_root_expected_rms_squared_m": float(
                np.deg2rad(noise) * item["root_expected_rms_squared_m_per_radian"]
            ),
        }
        for noise in (0.001, 0.01, 0.1)
        for item in unit_phase_rows
    ]
    write_csv(out / "phase_noise_predictions.csv", phase_rows)
    validation = []
    for noise in (0.001, 0.01, 0.1):
        condition = f"bearing_{noise:g}deg"
        values = {1: [], 2: []}
        for trial in range(100):
            folder = (
                ROOT / "appendix1/evaluation_560" / condition / f"trial_{trial:03d}"
            )
            data = json.loads((folder / "summary.json").read_text())
            value = data["evaluation"]
            if (
                data.get("fingerprint") != contract["fingerprint"]
                or value["condition"] != condition
                or value["trial"] != trial
                or value["measurement_slots"] != 560
            ):
                raise RuntimeError(f"bad frozen noise sample: {folder}")
            with (folder / "slot_metrics.csv").open(newline="") as handle:
                metrics = {int(row["slot"]): row for row in csv.DictReader(handle)}
            for item_phase, slot in ((1, 559), (2, 560)):
                values[item_phase].append(
                    float(metrics[slot]["rms_position_error_m"]) ** 2
                )
            final_rms = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (np.asarray(data["final_positions"])[1:] - nominal[1:])
                            ** 2,
                            axis=1,
                        )
                    )
                )
            )
            if not np.isclose(
                final_rms,
                float(metrics[560]["rms_position_error_m"]),
                atol=1e-12,
                rtol=0,
            ):
                raise RuntimeError(f"geometric RMS mismatch: {folder}")
        for item_phase in (1, 2):
            observed = float(np.sqrt(np.mean(values[item_phase])))
            predicted = next(
                row["predicted_root_expected_rms_squared_m"]
                for row in phase_rows
                if row["bearing_std_deg"] == noise and row["phase"] == item_phase
            )
            validation.append(
                {
                    "condition": condition,
                    "samples": 100,
                    "phase": item_phase,
                    "slot": 558 + item_phase,
                    "statistic": "sqrt(mean(geometric_RMS^2)); FY01-FY09 denominator 9",
                    "observed_root_expected_rms_squared_m": observed,
                    "predicted_root_expected_rms_squared_m": predicted,
                    "ratio_observed_to_predicted": observed / predicted,
                }
            )
    write_csv(out / "noise_validation.csv", validation)
    return {
        "covariance_unit": "m^2 per radian^2",
        "lyapunov_relative_residual": residual,
        "minimum_covariance_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
        "phase_closure_max_abs_error": closure,
        "phase_predictions": phase_rows,
        "validation": validation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--linear-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir or ROOT / "outputs/q1_3" / (
        "two_configuration_linear_only"
        if args.linear_only
        else "two_configuration_analysis"
    )
    out.mkdir(parents=True, exist_ok=True)
    gains, powers, radial, core, selections, reference = [], [], [], [], [], []
    for gain in (0.25, 0.5, 1.0):
        matrices, choices, blocks_now = selected_matrices(gain)
        matrix, eigenvalues = cycle_matrix(matrices), None
        eigenvalues = np.linalg.eigvals(matrix)
        rho = float(np.max(np.abs(eigenvalues)))
        gains.append(
            {
                "gain": gain,
                "spectral_radius": rho,
                "per_slot_spectral_radius": float(np.sqrt(rho)),
                "cycle_operator_norm": float(np.linalg.norm(matrix, 2)),
                "two_cycle_operator_norm": float(np.linalg.norm(matrix @ matrix, 2)),
                "min_relative_selection_gap": min(
                    r["relative_selection_gap"] for r in choices
                ),
            }
        )
        powers.extend(
            {
                "gain": gain,
                "cycles": n,
                "operator_norm": float(
                    np.linalg.norm(np.linalg.matrix_power(matrix, n), 2)
                ),
            }
            for n in range(1, 9)
        )
        a, b = blocks(matrix, gain)
        radial.extend(a)
        core.extend(b)
        if gain == 0.5:
            selections = [
                {**r, "two_configuration_phase": phase(r["phase"])} for r in choices
            ]
            reference = [
                {
                    **r,
                    "source_phase": r["phase"],
                    "two_configuration_phase": phase(r["phase"]),
                }
                for r in blocks_now
            ]
        np.savetxt(out / f"cycle_jacobian_gain_{gain:g}.csv", matrix, delimiter=",")
        write_csv(
            out / f"eigenvalues_gain_{gain:g}.csv",
            [
                {
                    "real": float(x.real),
                    "imaginary": float(x.imag),
                    "modulus": float(abs(x)),
                }
                for x in eigenvalues
            ],
        )
    for name, rows in (
        ("gain_stability.csv", gains),
        ("cycle_power_norms.csv", powers),
        ("nominal_selection.csv", selections),
        ("reference_error_blocks.csv", reference),
        ("radial_tangential_blocks.csv", radial),
        ("core_subsystems.csv", core),
    ):
        write_csv(out / name, rows)
    (out / "source_hashes.json").write_text(
        json.dumps(hashes(source_paths()), ensure_ascii=False, indent=2) + "\n"
    )
    report = {
        "state_order": STATE_ORDER,
        "scope": "Fixed nominal branches; local matrices exclude limiting and hold. Full Table 1 simulations retain limits and 21-observation local hold.",
        "gain_stability": gains,
    }
    if not args.linear_only:
        contract, checked = verify_contract()
        (out / "protocol.json").write_text(
            json.dumps(
                {
                    "schedule": [[0, *x] for x in SCHEDULE],
                    "settings": asdict(LocalSettings()),
                    "state_order": STATE_ORDER,
                    "information_boundary": "current receiver-local angles and own hold memory only; true positions and actions are simulator/offline analysis data",
                    "frozen_505_contract_fingerprint": contract["fingerprint"],
                    "frozen_505_contract_source_hashes": contract["source_sha256"],
                    "contract_validation": checked,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        report["difference_checks"] = fd_checks(out)
        report["table1_evaluation"] = table1(out)
        matrices, choices, _ = selected_matrices(0.5)
        report["noise"] = noise_analysis(out, matrices, choices, contract)
    (out / "analysis_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
