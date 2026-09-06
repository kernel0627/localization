"""Periodic small-noise covariance of the frozen-branch active controller."""

from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from scripts.q1_1.localization import angle_jacobian
from scripts.q1_3.analyze_iterative_reference import (
    linear_model,
    nominal_positions,
    polar_basis,
    state_slice,
)
from scripts.q1_3.local_adjustment import public_schedule
from scripts.q1_3.run_robustness import write_csv

ROOT = Path(__file__).resolve().parents[2]


def ray_angle_derivative(directions):
    """Derivative of unsigned pair angles with respect to signed ray bearings."""
    derivative = []
    for a, b in combinations(range(len(directions)), 2):
        if abs(np.sin(directions[a] - directions[b])) < 1e-12:
            raise ValueError("Unsigned angle derivative is undefined at 0 or pi")
        delta = np.arctan2(
            np.sin(directions[a] - directions[b]), np.cos(directions[a] - directions[b])
        )
        row = np.zeros(len(directions))
        row[a], row[b] = np.sign(delta), -np.sign(delta)
        derivative.append(row)
    return np.array(derivative)


def noise_covariance(gain=0.5):
    q = nominal_positions()
    matrices, choices, _ = linear_model(gain)
    indexed = {
        (r["phase"], r["receiver_id"]): tuple(map(int, r["selected_pair"].split("-")))
        for r in choices
    }
    injections = []
    cycle = np.eye(16)
    accumulated = np.zeros((16, 16))
    for phase, tx in enumerate(public_schedule(), 1):
        injection = np.zeros((16, 16))
        for i in range(2, 10):
            if i in tx:
                continue
            pair = indexed[phase, i]
            anchors = q[[0, *pair]]
            rays = anchors - q[i]
            bearing = np.arctan2(rays[:, 1], rays[:, 0])
            derivative = ray_angle_derivative(bearing)
            j = angle_jacobian(q[i], anchors) @ polar_basis(i)
            response = -gain * np.linalg.pinv(j) @ derivative
            injection[state_slice(i), state_slice(i)] = response @ response.T
        matrix = matrices[phase - 1]
        accumulated = matrix @ accumulated @ matrix.T + injection
        cycle = matrix @ cycle
        injections.append(injection)
    covariance = solve_discrete_lyapunov(cycle, accumulated)
    residual = np.linalg.norm(
        covariance - cycle @ covariance @ cycle.T - accumulated
    ) / np.linalg.norm(accumulated)
    phases = []
    current = covariance.copy()
    for phase, (matrix, injection) in enumerate(zip(matrices, injections), 1):
        current = matrix @ current @ matrix.T + injection
        phases.append(
            {
                "phase": phase,
                "expected_rms_per_radian_m": float(np.sqrt(np.trace(current) / 9)),
            }
        )
    assert np.allclose(current, covariance, rtol=1e-10, atol=1e-7)
    return cycle, accumulated, covariance, phases, float(residual)


def main():
    output = ROOT / "outputs/q1_3/noise_analysis"
    output.mkdir(parents=True, exist_ok=True)
    a, q, p, phases, residual = noise_covariance()
    for name, matrix in [
        ("cycle", a),
        ("cycle_noise_per_radian2", q),
        ("stationary_covariance_per_radian2", p),
    ]:
        np.savetxt(output / f"{name}.csv", matrix, delimiter=",")
    write_csv(output / "phase_rms_gain.csv", phases)
    rows = [
        {
            "bearing_std_deg": sigma,
            "predicted_root_expected_rms_squared_m": float(
                np.deg2rad(sigma) * np.sqrt(np.trace(p) / 9)
            ),
        }
        for sigma in (0.001, 0.01, 0.1)
    ]
    write_csv(output / "predicted_noise_floor.csv", rows)
    report = {
        "gain": 0.5,
        "lyapunov_relative_residual": residual,
        "minimum_covariance_eigenvalue": float(np.linalg.eigvalsh(p).min()),
        "scope": "Frozen nominal reference branches, active feedback, no limiting or hold, infinitesimal independent per-ray Gaussian noise. sqrt(E[RMS^2]) at phase 28, not median RMS or maximum error. Finite-noise switching can invalidate approximation.",
        "predictions": rows,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
