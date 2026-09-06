"""Propagate random-initial-state moments through saved frozen local matrices.

This is deliberately independent from the controller and its nonlinear runner.
It reads the already published two-phase matrices/injections, fixes the active
nominal branch, and has neither command clipping nor receiver hold memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRICES = (
    ROOT / "outputs/q1_3/two_configuration_robustness_math/phase_matrices_and_injections.npz"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/q1_3/two_configuration_robustness_math/random_initial_moments.csv"
)
DIMENSION = 16
RADIUS_M = 100.0
RADIAL_HALF_WIDTH_M = 12.0
ANGLE_HALF_WIDTH_DEG = 0.3
BEARING_STD_DEG = 0.001
ACTUATION_STD = 0.01


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def initial_second_moment() -> np.ndarray:
    """Return $E[x_0x_0^T]$ for independent uniform radial/angle offsets."""
    radial_variance = RADIAL_HALF_WIDTH_M**2 / 3
    angle_half_width_rad = np.deg2rad(ANGLE_HALF_WIDTH_DEG)
    tangential_variance = (RADIUS_M * angle_half_width_rad) ** 2 / 3
    return np.diag(np.tile((radial_variance, tangential_variance), 8))


def second_moment_step(
    moment: np.ndarray,
    matrix: np.ndarray,
    command_matrix: np.ndarray,
    innovation: np.ndarray,
    actuation_std: float,
) -> np.ndarray:
    """Exact local second-moment map for the saved independent-error model."""
    result = matrix @ moment @ matrix.T + innovation
    if actuation_std:
        command_second_moment = command_matrix @ moment @ command_matrix.T + innovation
        result += actuation_std**2 * np.diag(np.diag(command_second_moment))
    return (result + result.T) / 2


def relative_frobenius_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-300)
    )


def initial_geometric_moments() -> dict[str, float]:
    """Analytic and deterministic-quadrature checks for the true initial geometry."""
    angle_half_width_rad = np.deg2rad(ANGLE_HALF_WIDTH_DEG)
    radial_variance = RADIAL_HALF_WIDTH_M**2 / 3
    sinc = np.sin(angle_half_width_rad) / angle_half_width_rad
    per_drone = radial_variance + 2 * RADIUS_M**2 * (1 - sinc)
    expected_rms_squared = 8 * per_drone / 9

    # This is numerical integration only of E[cos(delta)]; the radial term has
    # the exact uniform variance above.  It independently checks the closed form.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    numerical_cosine_mean = 0.5 * np.sum(weights * np.cos(angle_half_width_rad * nodes))
    numerical_per_drone = radial_variance + 2 * RADIUS_M**2 * (1 - numerical_cosine_mean)
    numerical_rms_squared = 8 * numerical_per_drone / 9
    return {
        "angle_half_width_rad": float(angle_half_width_rad),
        "radial_variance_m2": float(radial_variance),
        "tangential_variance_m2": float((RADIUS_M * angle_half_width_rad) ** 2 / 3),
        "linear_initial_expected_rms_squared_m2": float(np.trace(initial_second_moment()) / 9),
        "true_geometric_initial_expected_rms_squared_m2": float(expected_rms_squared),
        "true_geometric_initial_root_expected_rms_squared_m": float(np.sqrt(expected_rms_squared)),
        "quadrature_true_geometric_initial_expected_rms_squared_m2": float(numerical_rms_squared),
        "analytic_quadrature_absolute_error_m2": float(abs(expected_rms_squared - numerical_rms_squared)),
    }


def validate_archive(archive: np.lib.npyio.NpzFile) -> None:
    expected_order = [
        f"FY{i:02d}_{component}"
        for i in range(2, 10)
        for component in ("dr_m", "R_dtheta_m")
    ]
    if archive["state_order"].tolist() != expected_order:
        raise ValueError("saved state order differs from the required 16-dimensional local state")
    for gain in (0.5, 1.0):
        for phase in (1, 2):
            for prefix in ("M", "U", "G_receiver_tx", "Q_iid_per_radian2"):
                key = f"{prefix}_gain_{gain:g}_phase_{phase}"
                if key not in archive:
                    raise ValueError(f"missing saved matrix: {key}")
            if archive[f"M_gain_{gain:g}_phase_{phase}"].shape != (DIMENSION, DIMENSION):
                raise ValueError("saved transition does not have shape 16 by 16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrices", type=Path, default=DEFAULT_MATRICES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.matrices
    output = args.output
    if not source.is_file():
        parser.error(f"saved matrix archive is missing: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    initial = initial_second_moment()
    geometry = initial_geometric_moments()
    rows: list[dict[str, object]] = []
    checks: dict[str, object] = {
        "initial_diagonal_max_absolute_error_m2": float(
            np.max(np.abs(np.diag(initial) - np.tile((48.0, geometry["tangential_variance_m2"]), 8)))
        ),
        "initial_off_diagonal_max_absolute_error_m2": float(
            np.max(np.abs(initial - np.diag(np.diag(initial))))
        ),
        "analytic_quadrature_absolute_error_m2": geometry["analytic_quadrature_absolute_error_m2"],
    }
    minimum_moment_eigenvalue = np.inf
    terminal_stationary_consistency: dict[str, dict[str, dict[str, float]]] = {}
    sigma_bearing_rad = np.deg2rad(BEARING_STD_DEG)

    with np.load(source) as archive:
        validate_archive(archive)
        for gain in (0.5, 1.0):
            matrices = [archive[f"M_gain_{gain:g}_phase_{phase}"].copy() for phase in (1, 2)]
            commands = [archive[f"U_gain_{gain:g}_phase_{phase}"].copy() for phase in (1, 2)]
            innovations = [
                sigma_bearing_rad**2 * archive[f"Q_iid_per_radian2_gain_{gain:g}_phase_{phase}"]
                for phase in (1, 2)
            ]
            moments = {
                "exact_local": initial.copy(),
                "bearing_0.001deg_local": initial.copy(),
                "combined_0.001deg_1pct_local": initial.copy(),
            }
            zero_started_moments = {
                "bearing_0.001deg_local": np.zeros((DIMENSION, DIMENSION)),
                "combined_0.001deg_1pct_local": np.zeros((DIMENSION, DIMENSION)),
            }
            for slot in range(561):
                phase = "initial" if slot == 0 else str(1 if slot % 2 else 2)
                for case, moment in moments.items():
                    minimum_moment_eigenvalue = min(
                        minimum_moment_eigenvalue, float(np.linalg.eigvalsh(moment).min())
                    )
                    rows.append(
                        {
                            "case": case,
                            "gain": gain,
                            "slot": slot,
                            "phase_of_current_state": phase,
                            "statistic": "sqrt(E[RMS_lin_m^2]) = sqrt(trace(P_t)/9); FY01-FY09 denominator 9",
                            "root_expected_rms_lin_squared_m": float(np.sqrt(max(0.0, np.trace(moment) / 9))),
                            "trace_second_moment_m2": float(np.trace(moment)),
                            "minimum_second_moment_eigenvalue_m2": float(np.linalg.eigvalsh(moment).min()),
                            "initial_true_geometric_root_expected_rms_squared_m": (
                                geometry["true_geometric_initial_root_expected_rms_squared_m"] if slot == 0 else ""
                            ),
                        }
                    )
                if slot == 560:
                    break
                phase_index = slot % 2
                moments["exact_local"] = second_moment_step(
                    moments["exact_local"], matrices[phase_index], commands[phase_index], np.zeros((DIMENSION, DIMENSION)), 0.0
                )
                moments["bearing_0.001deg_local"] = second_moment_step(
                    moments["bearing_0.001deg_local"], matrices[phase_index], commands[phase_index], innovations[phase_index], 0.0
                )
                moments["combined_0.001deg_1pct_local"] = second_moment_step(
                    moments["combined_0.001deg_1pct_local"], matrices[phase_index], commands[phase_index], innovations[phase_index], ACTUATION_STD
                )
                zero_started_moments["bearing_0.001deg_local"] = second_moment_step(
                    zero_started_moments["bearing_0.001deg_local"], matrices[phase_index], commands[phase_index], innovations[phase_index], 0.0
                )
                zero_started_moments["combined_0.001deg_1pct_local"] = second_moment_step(
                    zero_started_moments["combined_0.001deg_1pct_local"], matrices[phase_index], commands[phase_index], innovations[phase_index], ACTUATION_STD
                )
            terminal_stationary_consistency[f"gain_{gain:g}"] = {}
            for case, zero_started in zero_started_moments.items():
                after_phase_1 = second_moment_step(
                    zero_started, matrices[0], commands[0], innovations[0],
                    0.0 if case == "bearing_0.001deg_local" else ACTUATION_STD,
                )
                after_cycle = second_moment_step(
                    after_phase_1, matrices[1], commands[1], innovations[1],
                    0.0 if case == "bearing_0.001deg_local" else ACTUATION_STD,
                )
                terminal_stationary_consistency[f"gain_{gain:g}"][case] = {
                    "random_initial_vs_zero_started_phase_560_relative_frobenius": relative_frobenius_error(moments[case], zero_started),
                    "zero_started_phase_560_two_slot_closure_relative_frobenius": relative_frobenius_error(after_cycle, zero_started),
                }

    checks["minimum_second_moment_eigenvalue_m2"] = float(minimum_moment_eigenvalue)
    checks["psd_roundoff_tolerance_m2"] = -1e-12
    checks["terminal_stationary_consistency"] = terminal_stationary_consistency

    write_csv(output, rows)
    sidecar = output.with_name(output.stem + "_sources.json")
    sidecar.write_text(
        json.dumps(
            {
                "source_sha256": {
                    str(source.relative_to(ROOT)): sha256(source),
                    str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
                },
                "state_definition": "x=(dr, 100*dtheta) in metres for FY02--FY09; FY01 is fixed and remains in the RMS denominator 9.",
                "initial_law": "For each moving drone independently, dr~Uniform[-12,12] m and dtheta~Uniform[-0.3,0.3] degree.",
                "initial_second_moment": "E[x0]=0; P0=I_8 kron diag(48, 100^2*(0.3*pi/180)^2/3), ordered from FY02 through FY09.",
                "cases": {
                    "exact_local": "P+=M P M^T",
                    "bearing_0.001deg_local": "P+=M P M^T + sigma_beta^2 Q_iid_per_radian2, with sigma_beta=0.001*pi/180 rad.",
                    "combined_0.001deg_1pct_local": "P+=M P M^T + 0.01^2 diag(diag(U P U^T + sigma_beta^2 Q_iid_per_radian2)) + sigma_beta^2 Q_iid_per_radian2, with sigma_beta=0.001*pi/180 rad.",
                },
                "phase_convention": "Slot 1 applies saved phase 1; odd slots apply phase 1 and even slots phase 2. Slot 0 is the random initial state.",
                "scope": "Frozen active nominal branch only. No clipping, receiver hold, candidate switching, or nonlinear geometry update is represented.",
                "initial_geometry": geometry,
                "checks": checks,
                "rows": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"rows": len(rows), "output": str(output), "sidecar": str(sidecar), "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
