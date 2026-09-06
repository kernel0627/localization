"""Periodic local stochastic model for the frozen two-configuration controller.

The state is ``(dr, R*dtheta)`` in metres for FY02--FY09.  This module is an
offline small-signal analysis: it fixes the nominal pair choices, has no
limiting or hold memory, and never changes the controller or experiment
runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.q1_1.localization import angle_jacobian
from scripts.q1_3.analyze_iterative_reference import (
    DIMENSION,
    nominal_positions,
    polar_basis,
    state_slice,
)
from scripts.q1_3.analyze_noise_floor import ray_angle_derivative
from scripts.q1_3.analyze_two_configuration import SCHEDULE, selected_matrices
from scripts.q1_3.local_adjustment import public_schedule

ROOT = Path(__file__).resolve().parents[2]
STATE_ORDER = [f"FY{i:02d}_{c}" for i in range(2, 10) for c in ("dr_m", "R_dtheta_m")]
PHASE_SLOTS = {1: 559, 2: 560}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _two_phase_choices(gain: float):
    """Return frozen selections keyed by the compact phase 1 or 2."""
    _, choices, _ = selected_matrices(gain)
    source_indexes = [public_schedule().index(item) + 1 for item in SCHEDULE]
    source_to_phase = {source: phase for phase, source in enumerate(source_indexes, 1)}
    result = {}
    for row in choices:
        phase = source_to_phase[row["phase"]]
        result[phase, row["receiver_id"]] = tuple(
            map(int, row["selected_pair"].split("-"))
        )
    return result


def phase_matrices_and_injections(gain: float):
    """Build $M_j$, $U_j=M_j-I$, and raw-ray injection $G_j$.

    Columns of both ``G`` matrices use one common receiver--transmitter link
    order.  It is essential for a fixed link bias, while iid bearing draws use
    a fresh vector in that same coordinate system every slot.
    """
    matrices, _, _ = selected_matrices(gain)
    selected = _two_phase_choices(gain)
    links = sorted(
        {
            (receiver, transmitter)
            for phase in (1, 2)
            for receiver in range(2, 10)
            if receiver not in SCHEDULE[phase - 1]
            for transmitter in (0, *selected[phase, receiver])
        }
    )
    link_index = {link: column for column, link in enumerate(links)}
    nominal = nominal_positions()
    injections = []
    for phase in (1, 2):
        injection = np.zeros((DIMENSION, len(links)))
        for receiver in range(2, 10):
            if receiver in SCHEDULE[phase - 1]:
                continue
            pair = selected[phase, receiver]
            transmitters = (0, *pair)
            anchors = nominal[list(transmitters)]
            rays = anchors - nominal[receiver]
            derivative = ray_angle_derivative(np.arctan2(rays[:, 1], rays[:, 0]))
            local_jacobian = angle_jacobian(nominal[receiver], anchors) @ polar_basis(
                receiver
            )
            # Measurement error enters the fitted bias and the command with
            # the same sign as in the simulator's corrected state update.
            response = -gain * np.linalg.pinv(local_jacobian) @ derivative
            columns = [link_index[receiver, transmitter] for transmitter in transmitters]
            injection[state_slice(receiver), columns] = response
        injections.append(injection)
    return matrices, [matrix - np.eye(DIMENSION) for matrix in matrices], injections, links


def covariance_operator(matrix: np.ndarray, command_matrix: np.ndarray, actuation_std: float):
    r"""Matrix for $C \mapsto MCM^T+s_a^2\operatorname{diag}\!\operatorname{diag}(UCU^T)$.

    It uses column-major vectorisation, so the deterministic component is
    ``kron(M, M)``.  The diagonal correction is formed in one broadcasted
    assignment rather than a four-index loop.
    """
    n = matrix.shape[0]
    operator = np.kron(matrix, matrix)
    diagonal_rows = np.arange(n) * (n + 1)
    diagonal_map = (command_matrix[:, :, None] * command_matrix[:, None, :]).reshape(
        n, n * n, order="F"
    )
    operator[diagonal_rows, :] += actuation_std**2 * diagonal_map
    return operator


def _vec(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float).reshape(-1, order="F")


def _mat(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=float).reshape(DIMENSION, DIMENSION, order="F")


def second_moment_step(
    covariance: np.ndarray,
    matrix: np.ndarray,
    command_matrix: np.ndarray,
    innovation_covariance: np.ndarray,
    actuation_std: float,
) -> np.ndarray:
    """Exact second moment for independent Gaussian ray and execution errors."""
    deterministic = matrix @ covariance @ matrix.T
    multiplicative_state = actuation_std**2 * np.diag(
        np.diag(command_matrix @ covariance @ command_matrix.T)
    )
    innovation = innovation_covariance + actuation_std**2 * np.diag(
        np.diag(innovation_covariance)
    )
    return (deterministic + multiplicative_state + innovation + (deterministic + multiplicative_state + innovation).T) / 2


def periodic_second_moment(
    matrices: list[np.ndarray],
    commands: list[np.ndarray],
    innovations: list[np.ndarray],
    actuation_std: float = 0.0,
):
    """Solve the two-phase stationary covariance and return phase 559/560 moments."""
    operators = [covariance_operator(m, u, actuation_std) for m, u in zip(matrices, commands)]
    effective_q = [
        q + actuation_std**2 * np.diag(np.diag(q)) for q in innovations
    ]
    cycle_operator = operators[1] @ operators[0]
    cycle_q = operators[1] @ _vec(effective_q[0]) + _vec(effective_q[1])
    fixed_point = np.linalg.solve(np.eye(DIMENSION * DIMENSION) - cycle_operator, cycle_q)
    phase_560 = _mat(fixed_point)
    phase_560 = (phase_560 + phase_560.T) / 2
    phase_559 = second_moment_step(
        phase_560, matrices[0], commands[0], innovations[0], actuation_std
    )
    rebuilt_560 = second_moment_step(
        phase_559, matrices[1], commands[1], innovations[1], actuation_std
    )
    residual = np.linalg.norm(rebuilt_560 - phase_560) / max(
        np.linalg.norm(phase_560), np.linalg.norm(cycle_q), 1e-300
    )
    return {
        "operators": operators,
        "cycle_operator": cycle_operator,
        "cycle_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(cycle_operator)))),
        "phase_covariances": {559: phase_559, 560: phase_560},
        "relative_residual": float(residual),
    }


def fixed_link_bias_response(matrices: list[np.ndarray], injections: list[np.ndarray]):
    """Periodic response to one random vector that remains fixed for all slots."""
    cycle = matrices[1] @ matrices[0]
    cycle_loading = matrices[1] @ injections[0] + injections[1]
    loading_560 = np.linalg.solve(np.eye(DIMENSION) - cycle, cycle_loading)
    loading_559 = matrices[0] @ loading_560 + injections[0]
    closure = matrices[1] @ loading_559 + injections[1] - loading_560
    return {
        "cycle": cycle,
        "cycle_loading": cycle_loading,
        "phase_loadings": {559: loading_559, 560: loading_560},
        "closure_relative_residual": float(
            np.linalg.norm(closure) / max(np.linalg.norm(loading_560), 1e-300)
        ),
    }


def rms_from_covariance(covariance: np.ndarray) -> float:
    r"""$\sqrt{E[\mathrm{RMS}^2]}$ for FY01--FY09, retaining denominator nine."""
    return float(np.sqrt(max(0.0, np.trace(covariance) / 9)))


def covariance_row(kind, gain, phase, covariance, residual, **extra):
    return {
        "noise_kind": kind,
        "gain": gain,
        "phase": phase,
        "slot": PHASE_SLOTS[1] if phase == 559 else PHASE_SLOTS[2],
        "statistic": "sqrt(E[geometric_RMS_m^2]); FY01-FY09 denominator 9",
        "root_expected_rms_squared_m": rms_from_covariance(covariance),
        "trace_covariance_m2": float(np.trace(covariance)),
        "minimum_covariance_eigenvalue_m2": float(np.linalg.eigvalsh(covariance).min()),
        "periodic_relative_residual": residual,
        **extra,
    }


def _sample_covariance(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / len(values)


def monte_carlo_checks(gain, matrices, commands, injections, fixed_response, seed=20260906):
    """Independent path simulation of the linear model, including persistent bias."""
    rng = np.random.default_rng(seed + int(1000 * gain))
    paths, cycles = 12000, 80
    sigma_bearing = np.deg2rad(0.01)
    actuation = 0.01
    innovations = [sigma_bearing**2 * g @ g.T for g in injections]
    additive = periodic_second_moment(matrices, commands, innovations)
    combined = periodic_second_moment(matrices, commands, innovations, actuation)
    rows = []
    for kind, actuation_std, theory in (
        ("iid_bearing", 0.0, additive),
        ("combined_bearing_0.01deg_actuation_1pct", actuation, combined),
    ):
        values = np.zeros((paths, DIMENSION))
        for _ in range(cycles):
            noise = rng.normal(scale=sigma_bearing, size=(paths, injections[0].shape[1])) @ injections[0].T
            command = values @ commands[0].T + noise
            values = values @ matrices[0].T + noise
            if actuation_std:
                values += rng.normal(scale=actuation_std, size=values.shape) * command
            phase_559 = values.copy()
            noise = rng.normal(scale=sigma_bearing, size=(paths, injections[1].shape[1])) @ injections[1].T
            command = values @ commands[1].T + noise
            values = values @ matrices[1].T + noise
            if actuation_std:
                values += rng.normal(scale=actuation_std, size=values.shape) * command
        for phase, sample in ((559, phase_559), (560, values)):
            observed = _sample_covariance(sample)
            expected = theory["phase_covariances"][phase]
            rows.append(
                {
                    "check": kind,
                    "gain": gain,
                    "phase": phase,
                    "paths": paths,
                    "cycles": cycles,
                    "sample_root_expected_rms_squared_m": rms_from_covariance(observed),
                    "theory_root_expected_rms_squared_m": rms_from_covariance(expected),
                    "relative_frobenius_covariance_error": float(
                        np.linalg.norm(observed - expected) / np.linalg.norm(expected)
                    ),
                }
            )
    bias = rng.normal(scale=np.deg2rad(0.001), size=(paths, injections[0].shape[1]))
    values = np.zeros((paths, DIMENSION))
    for _ in range(cycles):
        values = values @ matrices[0].T + bias @ injections[0].T
        phase_559 = values.copy()
        values = values @ matrices[1].T + bias @ injections[1].T
    for phase, sample in ((559, phase_559), (560, values)):
        expected = np.deg2rad(0.001) ** 2 * fixed_response["phase_loadings"][phase] @ fixed_response["phase_loadings"][phase].T
        observed = _sample_covariance(sample)
        rows.append(
            {
                "check": "fixed_receiver_tx_link_bias_0.001deg",
                "gain": gain,
                "phase": phase,
                "paths": paths,
                "cycles": cycles,
                "sample_root_expected_rms_squared_m": rms_from_covariance(observed),
                "theory_root_expected_rms_squared_m": rms_from_covariance(expected),
                "relative_frobenius_covariance_error": float(
                    np.linalg.norm(observed - expected) / np.linalg.norm(expected)
                ),
            }
        )
    return rows


def _source_hashes():
    paths = [
        ROOT / "scripts/q1_3/analyze_two_configuration_robustness.py",
        ROOT / "scripts/q1_3/analyze_two_configuration.py",
        ROOT / "scripts/q1_3/analyze_noise_floor.py",
        ROOT / "scripts/q1_3/analyze_iterative_reference.py",
        ROOT / "scripts/q1_3/local_adjustment.py",
        ROOT / "docs/q1_3/双配置轮换鲁棒性执行方案.md",
    ]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/q1_3/two_configuration_robustness_math",
    )
    parser.add_argument("--skip-monte-carlo", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    iid_rows, stress_rows, combined_rows, bias_rows, mc_rows = [], [], [], [], []
    npz = {"state_order": np.array(STATE_ORDER), "phase_slots": np.array([559, 560])}
    for gain in (0.5, 1.0):
        matrices, commands, injections, links = phase_matrices_and_injections(gain)
        npz.update(
            {
                f"M_gain_{gain:g}_phase_{phase}": matrix
                for phase, matrix in enumerate(matrices, 1)
            }
        )
        npz.update(
            {
                f"U_gain_{gain:g}_phase_{phase}": matrix
                for phase, matrix in enumerate(commands, 1)
            }
        )
        npz.update(
            {
                f"G_receiver_tx_gain_{gain:g}_phase_{phase}": matrix
                for phase, matrix in enumerate(injections, 1)
            }
        )
        npz[f"Q_iid_per_radian2_gain_{gain:g}_phase_1"] = injections[0] @ injections[0].T
        npz[f"Q_iid_per_radian2_gain_{gain:g}_phase_2"] = injections[1] @ injections[1].T
        if gain == 0.5:
            npz["link_receiver_ids"] = np.array([receiver for receiver, _ in links])
            npz["link_transmitter_ids"] = np.array([transmitter for _, transmitter in links])
            (output / "receiver_tx_link_columns.csv").write_text(
                "column,receiver_id,transmitter_id\n"
                + "".join(f"{column},{receiver},{transmitter}\n" for column, (receiver, transmitter) in enumerate(links))
            )

        unit = periodic_second_moment(
            matrices, commands, [g @ g.T for g in injections]
        )
        for phase, covariance in unit["phase_covariances"].items():
            np.savetxt(output / f"iid_covariance_gain_{gain:g}_phase_{phase}_per_radian2.csv", covariance, delimiter=",")
        for bearing_std_deg in (0.001, 0.01, 0.1):
            scale = np.deg2rad(bearing_std_deg) ** 2
            for phase, covariance in unit["phase_covariances"].items():
                iid_rows.append(
                    covariance_row(
                        "iid_per_ray_bearing", gain, phase, scale * covariance,
                        unit["relative_residual"], bearing_std_deg=bearing_std_deg,
                        covariance_unit="m^2", source="periodic_unit_per_radian2",
                    )
                )

        for actuation_std in (0.01, 0.05, 0.10):
            zero = periodic_second_moment(
                matrices, commands, [np.zeros((DIMENSION, DIMENSION))] * 2, actuation_std
            )
            stress_rows.append(
                {
                    "gain": gain,
                    "actuation_relative_std": actuation_std,
                    "two_slot_second_moment_spectral_radius": zero["cycle_spectral_radius"],
                    "zero_innovation_stationary_root_expected_rms_squared_m": rms_from_covariance(zero["phase_covariances"][560]),
                    "periodic_relative_residual": zero["relative_residual"],
                    "mean_square_stable": bool(zero["cycle_spectral_radius"] < 1),
                    "scope": "Independent radial/tangential relative-command factors; active frozen branch, no limit or hold",
                }
            )

        for bearing_std_deg in (0.001, 0.01):
            innovation_scale = np.deg2rad(bearing_std_deg) ** 2
            combined = periodic_second_moment(
                matrices, commands, [innovation_scale * g @ g.T for g in injections], 0.01
            )
            for phase, covariance in combined["phase_covariances"].items():
                combined_rows.append(
                    covariance_row(
                        "combined_iid_bearing_and_independent_actuation", gain, phase, covariance,
                        combined["relative_residual"], bearing_std_deg=bearing_std_deg,
                        actuation_relative_std=0.01,
                        two_slot_second_moment_spectral_radius=combined["cycle_spectral_radius"],
                    )
                )
                np.savetxt(
                    output / f"combined_covariance_gain_{gain:g}_bearing_{bearing_std_deg:g}deg_phase_{phase}.csv",
                    covariance, delimiter=","
                )

        response = fixed_link_bias_response(matrices, injections)
        for phase, loading in response["phase_loadings"].items():
            covariance = np.deg2rad(0.001) ** 2 * loading @ loading.T
            bias_rows.append(
                covariance_row(
                    "fixed_receiver_tx_link_bias", gain, phase, covariance,
                    response["closure_relative_residual"], link_bias_std_deg=0.001,
                    response_loading_frobenius_m_per_radian=float(np.linalg.norm(loading)),
                    conditional_mean="E[x | b] = S_phase b; unconditional mean is zero",
                )
            )
            np.savetxt(output / f"fixed_link_bias_loading_gain_{gain:g}_phase_{phase}_m_per_radian.csv", loading, delimiter=",")
            np.savetxt(output / f"fixed_link_bias_covariance_gain_{gain:g}_phase_{phase}_0.001deg.csv", covariance, delimiter=",")
        if not args.skip_monte_carlo:
            mc_rows.extend(monte_carlo_checks(gain, matrices, commands, injections, response))

    np.savez_compressed(output / "phase_matrices_and_injections.npz", **npz)
    write_csv(output / "periodic_covariance.csv", iid_rows)
    write_csv(output / "multiplicative_stability.csv", stress_rows)
    write_csv(output / "combined_stationary_moments.csv", combined_rows)
    write_csv(output / "fixed_link_bias_response.csv", bias_rows)
    if mc_rows:
        write_csv(output / "operator_monte_carlo.csv", mc_rows)
    report = {
        "state_order": STATE_ORDER,
        "phase_convention": "The stationary state before phase 1 is after phase 2. Thus phase 1 maps to slot 559 and phase 2 maps to slot 560.",
        "rms_statistic": "sqrt(E[geometric_RMS_m^2]) = sqrt(trace(P)/9), including FY01 and excluding fixed FY00; denominator remains 9.",
        "scope": "Frozen nominal branch with active control, no limiting, branch switching, or receiver hold. White bearing errors are independent by slot and receiver ray. Fixed receiver--TX link bias has one shared value across all uses of that link in a run.",
        "source_sha256": _source_hashes(),
        "rows": {
            "iid_periodic_covariance": len(iid_rows),
            "multiplicative_stability": len(stress_rows),
            "combined_stationary_moments": len(combined_rows),
            "fixed_link_bias_response": len(bias_rows),
            "operator_monte_carlo": len(mc_rows),
        },
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
