"""Gaussian all-drone precision probabilities from saved phase-560 covariances.

This is a numerical integration of the frozen active-branch local Gaussian
model.  It deliberately does not reinterpret the finite nonlinear trials or
the non-Gaussian multiplicative combined model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATH_OUTPUT = ROOT / "outputs/q1_3/two_configuration_robustness_math"
DIMENSION = 16
BASE_SEED = 2026090617
SAMPLES = 200_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, samples: int):
    """Nominal binomial interval for Monte Carlo integration uncertainty only."""
    z = 1.959963984540054
    probability = successes / samples
    denominator = 1 + z**2 / samples
    center = (probability + z**2 / (2 * samples)) / denominator
    half_width = z * np.sqrt(
        probability * (1 - probability) / samples + z**2 / (4 * samples**2)
    ) / denominator
    return float(max(0, center - half_width)), float(min(1, center + half_width))


def gaussian_root(covariance: np.ndarray) -> np.ndarray:
    """PSD square root; clip only round-off-scale negative eigenvalues."""
    symmetric = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues.min() < -1e-10 * scale:
        raise ValueError("saved covariance is not positive semidefinite")
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))


def union_failure_bound(covariance: np.ndarray, epsilon_m: float) -> float:
    """2-D radial Gaussian tail plus union bound; block correlations are allowed."""
    terms = []
    for drone in range(8):
        block = covariance[2 * drone : 2 * drone + 2, 2 * drone : 2 * drone + 2]
        maximum_variance = max(0.0, float(np.linalg.eigvalsh(block).max()))
        if maximum_variance == 0:
            terms.append(0.0)
        else:
            terms.append(float(np.exp(-epsilon_m**2 / (2 * maximum_variance))))
    return float(min(1.0, sum(terms)))


def integrate_all_drone_probability(
    covariance: np.ndarray,
    thresholds_m: tuple[float, ...],
    samples: int,
    seed_sequence: list[int],
    batch_size: int = 20_000,
):
    """Sample the full correlated 16-D Gaussian, then test the eight 2-D norms."""
    root = gaussian_root(covariance)
    rng = np.random.default_rng(np.random.SeedSequence(seed_sequence))
    successes = np.zeros(len(thresholds_m), dtype=np.int64)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        # One matrix multiply per batch keeps all drone correlations intact.
        state = rng.standard_normal((count, DIMENSION)) @ root.T
        maximum_error = np.sqrt((state.reshape(count, 8, 2) ** 2).sum(axis=2)).max(
            axis=1
        )
        successes += [(maximum_error < threshold).sum() for threshold in thresholds_m]
    return successes


def cases(math_output: Path):
    """Only the requested pure iid bearing cases and fixed-link-bias cases."""
    result = []
    for gain in (0.5, 1.0):
        unit_path = math_output / f"iid_covariance_gain_{gain:g}_phase_560_per_radian2.csv"
        unit = np.loadtxt(unit_path, delimiter=",")
        for bearing_std_deg in (0.001, 0.01, 0.1):
            result.append(
                {
                    "model": "pure_iid_bearing_gaussian",
                    "gain": gain,
                    "bearing_std_deg": bearing_std_deg,
                    "link_bias_std_deg": 0.0,
                    "covariance": np.deg2rad(bearing_std_deg) ** 2 * unit,
                    "covariance_source": unit_path,
                    "covariance_transform": "saved per-radian^2 covariance multiplied by bearing_std_rad^2",
                }
            )
        bias_path = math_output / f"fixed_link_bias_covariance_gain_{gain:g}_phase_560_0.001deg.csv"
        result.append(
            {
                "model": "fixed_receiver_tx_link_bias_gaussian",
                "gain": gain,
                "bearing_std_deg": 0.0,
                "link_bias_std_deg": 0.001,
                "covariance": np.loadtxt(bias_path, delimiter=","),
                "covariance_source": bias_path,
                "covariance_transform": "saved phase-560 fixed-link-bias covariance",
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-output", type=Path, default=MATH_OUTPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    output = args.output or args.math_output / "gaussian_precision_probability.csv"
    thresholds = ((0.01, "1cm"), (0.001, "1mm"))
    rows, source_hashes = [], {"script": sha256(Path(__file__))}
    for case_number, case in enumerate(cases(args.math_output)):
        covariance = case.pop("covariance")
        source = case.pop("covariance_source")
        source_hashes[str(source.relative_to(ROOT))] = sha256(source)
        sequence = [BASE_SEED, case_number, int(round(case["gain"] * 10))]
        counts = integrate_all_drone_probability(
            covariance,
            tuple(threshold for threshold, _ in thresholds),
            args.samples,
            sequence,
        )
        for (threshold, label), successes in zip(thresholds, counts):
            low, high = wilson(int(successes), args.samples)
            rows.append(
                {
                    **case,
                    "phase": 560,
                    "slot": 560,
                    "epsilon_m": threshold,
                    "threshold": label,
                    "statistic": "Pr(max_{FY02..FY09} ||x_i||_2 < epsilon); local linear state, FY01 has zero error",
                    "samples": args.samples,
                    "base_seed": BASE_SEED,
                    "seed_sequence": "-".join(map(str, sequence)),
                    "successful_samples": int(successes),
                    "gaussian_numerical_integration_probability": successes / args.samples,
                    "gaussian_numerical_integration_wilson95_low": low,
                    "gaussian_numerical_integration_wilson95_high": high,
                    "conservative_failure_union_upper_bound": union_failure_bound(
                        covariance, threshold
                    ),
                    "minimum_covariance_eigenvalue_m2": float(
                        np.linalg.eigvalsh((covariance + covariance.T) / 2).min()
                    ),
                    "scope": "Frozen active branch, Gaussian iid bearing or Gaussian run-fixed link bias only; numerical-integration interval is not a 100-trial experimental confidence interval",
                }
            )
    write_csv(output, rows)
    sidecar = output.with_name("gaussian_precision_probability_sources.json")
    sidecar.write_text(
        json.dumps(
            {
                "source_sha256": source_hashes,
                "base_seed": BASE_SEED,
                "samples": args.samples,
                "phase": 560,
                "excluded_model": "Combined multiplicative execution noise: products are non-Gaussian, so this Gaussian probability calculation does not apply.",
                "bound": "Pr(Emax_lin >= epsilon) <= min(1, sum_i exp(-epsilon^2/(2 lambda_max(P_ii)))); the union bound does not require drone blocks to be independent.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"rows": len(rows), "output": str(output), "sidecar": str(sidecar)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
