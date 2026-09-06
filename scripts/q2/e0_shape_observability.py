"""Bounded Q2 geometry E0: local shape observability for labeled angle data.

The probe deliberately stops before inversion, control, and noise experiments.
It constructs the numbered 15-point two-dimensional triangular template, removes
the rows at which the unsigned angle is non-smooth (0 or pi), and evaluates the
analytic Jacobian of the remaining angle rows.  All configurations are static;
the two-configuration scan is bounded and deterministic.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

N_POINTS = 15
DIMENSION = 2
ANGLE_CROSS_TOL = 1e-10
RANK_RELATIVE_THRESHOLDS = (1e-8, 1e-9, 1e-10)
DOUBLE_SCAN_LIMIT = 200
REQUESTED_CONFIGURATIONS = ((1, 11, 15), (2, 7, 10))
PERTURBATION_LEVELS = (1e-3, 1e-2)
PERTURBATION_SEEDS = (11, 23, 47)
FIXED_GENERAL_PERTURBATION_LEVEL = 0.013
FIXED_GENERAL_PERTURBATION_SEED = 20260906


def triangular_template(spacing: float = 1.0) -> np.ndarray:
    """Return the numbered five-layer triangular template, IDs 1 through 15."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    points: list[tuple[float, float]] = []
    for row in range(5):
        for column in range(row + 1):
            points.append(
                (
                    spacing * (column - row / 2.0),
                    -spacing * np.sqrt(3.0) * row / 2.0,
                )
            )
    return np.asarray(points, dtype=float)


def validate_configuration(configuration: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(configuration) != 3 or len(set(configuration)) != 3:
        raise ValueError("a configuration must contain three distinct IDs")
    if any(not isinstance(value, (int, np.integer)) for value in configuration):
        raise ValueError("configuration IDs must be integers")
    if any(value < 1 or value > N_POINTS for value in configuration):
        raise ValueError("configuration IDs must be in 1..15")
    return tuple(sorted(int(value) for value in configuration))


def observation_layout(configuration: tuple[int, int, int]) -> np.ndarray:
    """Return rows (receiver ID, transmitter ID, transmitter ID), one per pair."""
    tx = validate_configuration(configuration)
    rows: list[tuple[int, int, int]] = []
    for receiver in range(1, N_POINTS + 1):
        if receiver not in tx:
            rows.extend((receiver, first, second) for first, second in itertools.combinations(tx, 2))
    return np.asarray(rows, dtype=int)


def _vectors(positions: np.ndarray, layout: np.ndarray):
    points = np.asarray(positions, dtype=float)
    if points.shape != (N_POINTS, DIMENSION) or not np.isfinite(points).all():
        raise ValueError("positions must have shape (15, 2) and contain finite values")
    if layout.ndim != 2 or layout.shape[1] != 3:
        raise ValueError("layout must have shape (rows, 3)")
    receiver, first, second = (layout[:, index] - 1 for index in range(3))
    u = points[first] - points[receiver]
    v = points[second] - points[receiver]
    u2 = np.einsum("ij,ij->i", u, u)
    v2 = np.einsum("ij,ij->i", v, v)
    if np.any((u2 <= 1e-24) | (v2 <= 1e-24)):
        raise ValueError("a receiver coincides with a transmitter")
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = np.einsum("ij,ij->i", u, v)
    normalized_cross = np.abs(cross) / np.sqrt(u2 * v2)
    return receiver, first, second, u, v, u2, v2, cross, dot, normalized_cross


def predict_angles(positions: np.ndarray, layout: np.ndarray) -> np.ndarray:
    """Return unsigned angles in [0, pi] using atan2(|cross|, dot)."""
    *_, cross, dot, _ = _vectors(positions, layout)
    return np.arctan2(np.abs(cross), dot)


def regular_row_mask(positions: np.ndarray, layout: np.ndarray) -> np.ndarray:
    """Rows where the original unsigned-angle map has an ordinary derivative."""
    return _vectors(positions, layout)[-1] > ANGLE_CROSS_TOL


def angle_jacobian(positions: np.ndarray, layout: np.ndarray) -> np.ndarray:
    """Analytic Jacobian of angle rows with respect to all 30 coordinates."""
    receiver, first, second, u, v, u2, v2, cross, _, normalized_cross = _vectors(
        positions, layout
    )
    if np.any(normalized_cross <= ANGLE_CROSS_TOL):
        raise ValueError("unsigned-angle Jacobian is undefined at 0 or pi")
    sign = np.sign(cross)[:, None]
    rotation_u = np.column_stack((-u[:, 1], u[:, 0]))
    rotation_v = np.column_stack((-v[:, 1], v[:, 0]))
    d_first = -sign * rotation_u / u2[:, None]
    d_second = sign * rotation_v / v2[:, None]
    d_receiver = -d_first - d_second
    jacobian = np.zeros((len(layout), N_POINTS * DIMENSION), dtype=float)
    for ids, derivative in (
        (receiver, d_receiver),
        (first, d_first),
        (second, d_second),
    ):
        rows = np.arange(len(layout))
        columns = DIMENSION * ids
        jacobian[rows, columns] += derivative[:, 0]
        jacobian[rows, columns + 1] += derivative[:, 1]
    return jacobian


def finite_difference_jacobian(
    positions: np.ndarray, layout: np.ndarray, step: float
) -> np.ndarray:
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    base = np.asarray(positions, dtype=float)
    result = np.empty((len(layout), N_POINTS * DIMENSION), dtype=float)
    for column in range(N_POINTS * DIMENSION):
        plus = base.copy().reshape(-1)
        minus = base.copy().reshape(-1)
        plus[column] += step
        minus[column] -= step
        result[:, column] = (
            predict_angles(plus.reshape(N_POINTS, DIMENSION), layout)
            - predict_angles(minus.reshape(N_POINTS, DIMENSION), layout)
        ) / (2.0 * step)
    return result


def singular_values_30(jacobian: np.ndarray) -> np.ndarray:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return np.pad(
        singular_values,
        (0, N_POINTS * DIMENSION - len(singular_values)),
        mode="constant",
    )


def ranks_for_thresholds(singular_values: np.ndarray) -> dict[str, int]:
    sigma_max = float(singular_values[0]) if len(singular_values) else 0.0
    return {
        f"rank_{relative:.0e}": int(
            np.count_nonzero(singular_values > sigma_max * relative)
        )
        for relative in RANK_RELATIVE_THRESHOLDS
    }


def gauge_basis(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return four similarity tangent vectors and a 26-column orthogonal complement."""
    points = np.asarray(positions, dtype=float)
    center = points.mean(axis=0)
    centered = points - center
    basis = np.zeros((N_POINTS * DIMENSION, 4), dtype=float)
    basis[0::2, 0] = 1.0  # x translation
    basis[1::2, 1] = 1.0  # y translation
    basis[0::2, 2] = -centered[:, 1]
    basis[1::2, 2] = centered[:, 0]
    basis[0::2, 3] = centered[:, 0]
    basis[1::2, 3] = centered[:, 1]
    q, _ = np.linalg.qr(basis, mode="complete")
    return q[:, :4], q[:, 4:]


def gauge_diagnostics(jacobian: np.ndarray, positions: np.ndarray) -> dict:
    gauge, complement = gauge_basis(positions)
    restricted = jacobian @ complement
    restricted_singular = np.linalg.svd(restricted, compute_uv=False)
    restricted_singular = np.pad(
        restricted_singular,
        (0, 26 - len(restricted_singular)),
        mode="constant",
    )
    return {
        "gauge_residual": float(
            np.linalg.norm(jacobian @ gauge, ord="fro")
            / max(1.0, np.linalg.norm(jacobian, ord="fro"))
        ),
        "shape_singular_values_26": [float(value) for value in restricted_singular],
        "shape_ranks": ranks_for_thresholds(restricted_singular),
        "shape_sigma_min": float(restricted_singular[-1]),
    }


def jacobian_record(jacobian: np.ndarray, positions: np.ndarray) -> dict:
    singular = singular_values_30(jacobian)
    return {
        **ranks_for_thresholds(singular),
        "singular_values_30": [float(value) for value in singular],
        "sigma_26": float(singular[25]),
        "sigma_27": float(singular[26]),
        **gauge_diagnostics(jacobian, positions),
    }


def perturbation(spacing_template: np.ndarray, level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    delta = rng.normal(size=spacing_template.shape)
    norms = np.linalg.norm(delta, axis=1)
    delta /= np.max(norms)
    return spacing_template + level * delta


def derivative_checks(template: np.ndarray) -> dict:
    configuration = REQUESTED_CONFIGURATIONS[0]
    layout = observation_layout(configuration)
    q_mask = regular_row_mask(template, layout)
    q_layout = layout[q_mask]
    q_jacobian = angle_jacobian(template, q_layout)
    generic = perturbation(
        template,
        FIXED_GENERAL_PERTURBATION_LEVEL,
        FIXED_GENERAL_PERTURBATION_SEED,
    )
    generic_mask = regular_row_mask(generic, q_layout)
    # The selected ideal regular rows remain regular at this small fixed perturbation.
    if not generic_mask.all():
        raise RuntimeError("fixed regular rows became degenerate in derivative check")
    records = {}
    for name, state, jacobian in (("ideal", template, q_jacobian), ("perturbed", generic, angle_jacobian(generic, q_layout))):
        step_records = {}
        for step in (1e-5, 1e-6, 1e-7):
            numerical = finite_difference_jacobian(state, q_layout, step)
            difference = jacobian - numerical
            step_records[f"{step:.0e}"] = {
                "relative_frobenius_error": float(
                    np.linalg.norm(difference, ord="fro")
                    / max(1.0, np.linalg.norm(jacobian, ord="fro"))
                ),
                "max_absolute_error": float(np.max(np.abs(difference))),
            }
        records[name] = {
            "configuration": list(configuration),
            "regular_rows": int(len(q_layout)),
            "steps": step_records,
        }
    return records


def invariance_checks(template: np.ndarray, layout: np.ndarray) -> dict:
    angle = predict_angles(template, layout)
    theta = 0.37
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transformed = 1.7 * template @ rotation.T + np.array([0.37, -0.23])
    reflected = template.copy()
    reflected[:, 0] *= -1.0
    reflected += np.array([-0.19, 0.41])
    return {
        "similarity_max_abs_angle_error": float(np.max(np.abs(angle - predict_angles(transformed, layout)))),
        "reflection_max_abs_angle_error": float(np.max(np.abs(angle - predict_angles(reflected, layout)))),
    }


def configuration_row_data(template: np.ndarray, configuration: tuple[int, int, int]) -> dict:
    layout = observation_layout(configuration)
    mask = regular_row_mask(template, layout)
    jacobian = angle_jacobian(template, layout[mask])
    return {
        "configuration": list(configuration),
        "full_rows": int(len(layout)),
        "degenerate_rows": int(np.count_nonzero(~mask)),
        "regular_rows": int(np.count_nonzero(mask)),
        "degenerate_row_indices": [int(index) for index in np.flatnonzero(~mask)],
        "regular_jacobian": jacobian,
        "regular_layout": layout[mask],
        "diagnostics": jacobian_record(jacobian, template),
    }


def configuration_summary_row(data: dict) -> dict:
    diagnostics = data["diagnostics"]
    configuration = data["configuration"]
    return {
        "configuration": "-".join(map(str, configuration)),
        "full_rows": data["full_rows"],
        "degenerate_rows": data["degenerate_rows"],
        "regular_rows": data["regular_rows"],
        "rank_1e-08": diagnostics["rank_1e-08"],
        "rank_1e-09": diagnostics["rank_1e-09"],
        "rank_1e-10": diagnostics["rank_1e-10"],
        "sigma_26": diagnostics["sigma_26"],
        "sigma_27": diagnostics["sigma_27"],
        "shape_rank_1e-08": diagnostics["shape_ranks"]["rank_1e-08"],
        "shape_rank_1e-09": diagnostics["shape_ranks"]["rank_1e-09"],
        "shape_rank_1e-10": diagnostics["shape_ranks"]["rank_1e-10"],
        "gauge_residual": diagnostics["gauge_residual"],
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path, double_limit: int = DOUBLE_SCAN_LIMIT) -> dict:
    if not isinstance(double_limit, int) or not 1 <= double_limit <= DOUBLE_SCAN_LIMIT:
        raise ValueError(f"double_limit must be an integer in 1..{DOUBLE_SCAN_LIMIT}")
    output_dir.mkdir(parents=True, exist_ok=True)
    template = triangular_template()
    all_layout = np.concatenate(
        [observation_layout(configuration) for configuration in itertools.combinations(range(1, 16), 3)]
    )
    single_rows: list[dict] = []
    single_data: dict[tuple[int, int, int], dict] = {}
    for configuration in itertools.combinations(range(1, 16), 3):
        data = configuration_row_data(template, configuration)
        single_data[configuration] = data
        single_rows.append(configuration_summary_row(data))
    write_csv(
        output_dir / "single_configurations.csv",
        single_rows,
        list(single_rows[0].keys()),
    )

    fixed_general_state = perturbation(
        template,
        FIXED_GENERAL_PERTURBATION_LEVEL,
        FIXED_GENERAL_PERTURBATION_SEED,
    )
    single_perturbed_rows: list[dict] = []
    for configuration in itertools.combinations(range(1, 16), 3):
        data = configuration_row_data(fixed_general_state, configuration)
        row = configuration_summary_row(data)
        row["perturbation_level"] = FIXED_GENERAL_PERTURBATION_LEVEL
        row["perturbation_seed"] = FIXED_GENERAL_PERTURBATION_SEED
        single_perturbed_rows.append(row)
    write_csv(
        output_dir / "single_configurations_perturbed.csv",
        single_perturbed_rows,
        list(single_perturbed_rows[0].keys()),
    )

    triple_list = list(itertools.combinations(range(1, 16), 3))
    requested_pair = tuple(sorted(REQUESTED_CONFIGURATIONS))
    candidates = [requested_pair]
    candidates.extend(
        pair for pair in itertools.combinations(triple_list, 2) if pair != requested_pair
    )
    double_rows: list[dict] = []
    double_details: list[dict] = []
    witnesses: list[dict] = []
    for index, (first, second) in enumerate(candidates[:double_limit], start=1):
        jacobian = np.vstack(
            (single_data[first]["regular_jacobian"], single_data[second]["regular_jacobian"])
        )
        diagnostics = jacobian_record(jacobian, template)
        stable_rank = all(diagnostics[f"rank_{relative:.0e}"] == 26 for relative in RANK_RELATIVE_THRESHOLDS)
        row = {
            "scan_index": index,
            "configuration_a": "-".join(map(str, first)),
            "configuration_b": "-".join(map(str, second)),
            "full_rows_a": single_data[first]["full_rows"],
            "full_rows_b": single_data[second]["full_rows"],
            "degenerate_rows_a": single_data[first]["degenerate_rows"],
            "degenerate_rows_b": single_data[second]["degenerate_rows"],
            "regular_rows_a": single_data[first]["regular_rows"],
            "regular_rows_b": single_data[second]["regular_rows"],
            "full_rows_total": single_data[first]["full_rows"]
            + single_data[second]["full_rows"],
            "degenerate_rows_total": single_data[first]["degenerate_rows"]
            + single_data[second]["degenerate_rows"],
            "regular_rows_total": single_data[first]["regular_rows"]
            + single_data[second]["regular_rows"],
            "rank_1e-08": diagnostics["rank_1e-08"],
            "rank_1e-09": diagnostics["rank_1e-09"],
            "rank_1e-10": diagnostics["rank_1e-10"],
            "shape_rank_1e-08": diagnostics["shape_ranks"]["rank_1e-08"],
            "shape_rank_1e-09": diagnostics["shape_ranks"]["rank_1e-09"],
            "shape_rank_1e-10": diagnostics["shape_ranks"]["rank_1e-10"],
            "shape_sigma_min": diagnostics["shape_sigma_min"],
            "gauge_residual": diagnostics["gauge_residual"],
            "stable_rank_26": stable_rank,
        }
        double_rows.append(row)
        detail = {
            **row,
            "singular_values_30": diagnostics["singular_values_30"],
            "shape_singular_values_26": diagnostics["shape_singular_values_26"],
            "degenerate_row_indices_a": single_data[first]["degenerate_row_indices"],
            "degenerate_row_indices_b": single_data[second]["degenerate_row_indices"],
        }
        double_details.append(detail)
        if stable_rank:
            witnesses.append(detail)
            if len(witnesses) >= 3:
                break
    write_csv(
        output_dir / "double_configurations_checked.csv",
        double_rows,
        list(double_rows[0].keys()),
    )

    witness_continuation: list[dict] = []
    for witness in witnesses:
        first = tuple(int(value) for value in witness["configuration_a"].split("-"))
        second = tuple(int(value) for value in witness["configuration_b"].split("-"))
        for level in PERTURBATION_LEVELS:
            for seed in PERTURBATION_SEEDS:
                state = perturbation(template, level, seed)
                jacobian_a = angle_jacobian(state, single_data[first]["regular_layout"])
                jacobian_b = angle_jacobian(state, single_data[second]["regular_layout"])
                diagnostics = jacobian_record(np.vstack((jacobian_a, jacobian_b)), state)
                witness_continuation.append(
                    {
                        "configuration_a": witness["configuration_a"],
                        "configuration_b": witness["configuration_b"],
                        "level": level,
                        "seed": seed,
                        "rank_1e-08": diagnostics["rank_1e-08"],
                        "rank_1e-09": diagnostics["rank_1e-09"],
                        "rank_1e-10": diagnostics["rank_1e-10"],
                        "shape_rank_1e-08": diagnostics["shape_ranks"]["rank_1e-08"],
                        "shape_rank_1e-09": diagnostics["shape_ranks"]["rank_1e-09"],
                        "shape_rank_1e-10": diagnostics["shape_ranks"]["rank_1e-10"],
                        "shape_sigma_min": diagnostics["shape_sigma_min"],
                        "gauge_residual": diagnostics["gauge_residual"],
                    }
                )
    write_csv(
        output_dir / "witness_perturbation_continuation.csv",
        witness_continuation,
        list(witness_continuation[0].keys()) if witness_continuation else ["configuration_a"],
    )

    derivative = derivative_checks(template)
    invariance = invariance_checks(template, all_layout)
    template_center = template.mean(axis=0)
    nearest_distances = [
        np.linalg.norm(template[first] - template[second])
        for first, second in itertools.combinations(range(N_POINTS), 2)
    ]
    nearest = [distance for distance in nearest_distances if distance < 1.01]
    summary = {
        "schema_version": 1,
        "scope": "Q2 E0 geometry only: static labeled angle batches, three transmitters per configuration, centralized angle collection",
        "not_done": ["angle inversion", "feedback control", "noise", "Monte Carlo recovery", "global uniqueness"],
        "template": {
            "point_count": N_POINTS,
            "layers": [1, 2, 3, 4, 5],
            "spacing": 1.0,
            "numbering": "FY01..FY15, row-major by layers",
            "center": [float(value) for value in template_center],
            "min_pair_distance": float(min(nearest_distances)),
            "nearest_neighbor_pair_count": len(nearest),
            "coordinates": [[float(value) for value in row] for row in template],
        },
        "observation_contract": {
            "angle": "atan2(abs(cross(Pj-Pi, Pk-Pi)), dot(Pj-Pi, Pk-Pi))",
            "value_range": "[0, pi]",
            "rows_per_configuration": 36,
            "static_batch": True,
            "labeled_ids": True,
            "angle_collection": "centralized for E0",
            "degeneracy_rule": f"normalized absolute cross <= {ANGLE_CROSS_TOL:.0e} is excluded from ordinary Jacobian rows",
        },
        "free_dimension": {
            "coordinate_columns": 30,
            "similarity_gauge_dimension": 4,
            "shape_upper_bound": 26,
            "reflection": "discrete ambiguity; not subtracted from differential rank",
        },
        "single_configuration_summary": {
            "count": len(single_rows),
            "max_rank_1e-08": max(row["rank_1e-08"] for row in single_rows),
            "max_rank_1e-09": max(row["rank_1e-09"] for row in single_rows),
            "max_rank_1e-10": max(row["rank_1e-10"] for row in single_rows),
            "all_rank_at_most_24": all(row["rank_1e-09"] <= 24 for row in single_rows),
            "regular_row_count_distribution": sorted({row["regular_rows"] for row in single_rows}),
        },
        "single_configuration_perturbed_summary": {
            "count": len(single_perturbed_rows),
            "perturbation_level": FIXED_GENERAL_PERTURBATION_LEVEL,
            "perturbation_seed": FIXED_GENERAL_PERTURBATION_SEED,
            "max_rank_1e-08": max(row["rank_1e-08"] for row in single_perturbed_rows),
            "max_rank_1e-09": max(row["rank_1e-09"] for row in single_perturbed_rows),
            "max_rank_1e-10": max(row["rank_1e-10"] for row in single_perturbed_rows),
            "all_rank_at_most_24": all(
                row["rank_1e-09"] <= 24 for row in single_perturbed_rows
            ),
            "regular_row_count_distribution": sorted(
                {row["regular_rows"] for row in single_perturbed_rows}
            ),
        },
        "double_scan": {
            "requested_pair_first": [list(configuration) for configuration in requested_pair],
            "requested_pair_row_counts": {
                "full_rows": sum(single_data[configuration]["full_rows"] for configuration in requested_pair),
                "degenerate_rows": sum(
                    single_data[configuration]["degenerate_rows"]
                    for configuration in requested_pair
                ),
                "regular_rows": sum(
                    single_data[configuration]["regular_rows"]
                    for configuration in requested_pair
                ),
            },
            "limit": double_limit,
            "checked": len(double_rows),
            "stopped_after_witnesses": len(witnesses) >= 3,
            "witness_count": len(witnesses),
            "witnesses": [
                {
                    "configuration_a": witness["configuration_a"],
                    "configuration_b": witness["configuration_b"],
                    "rank_1e-08": witness["rank_1e-08"],
                    "rank_1e-09": witness["rank_1e-09"],
                    "rank_1e-10": witness["rank_1e-10"],
                    "shape_sigma_min": witness["shape_sigma_min"],
                    "gauge_residual": witness["gauge_residual"],
                }
                for witness in witnesses
            ],
            "checked_details": double_details,
        },
        "derivative_checks": derivative,
        "invariance_checks": invariance,
        "witness_perturbation_continuation": {
            "levels": list(PERTURBATION_LEVELS),
            "seeds": list(PERTURBATION_SEEDS),
            "rows": witness_continuation,
        },
        "interpretation": {
            "single_configuration": "The regular-row probe has at most 24 differential dimensions at the ideal template; this is a design-class cross-check, not a theorem about every non-smooth full-angle configuration.",
            "double_configuration": "A stable rank-26 witness is a local shape-observability sufficient condition after removing translation, rotation, and scale; it does not establish inversion success, global uniqueness, control convergence, or physical-scale recovery.",
            "scale": "d=1 is a coordinate convention; angle data alone do not identify absolute meters.",
            "reflection": "Mirror images preserve unsigned angles and must be handled as a discrete branch in later inversion.",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/q2/e0_shape"),
        help="directory for the bounded E0 outputs",
    )
    parser.add_argument(
        "--double-limit",
        type=int,
        default=DOUBLE_SCAN_LIMIT,
        help=f"maximum double configurations to check (1..{DOUBLE_SCAN_LIMIT})",
    )
    args = parser.parse_args()
    summary = run(args.output_dir, args.double_limit)
    print(
        json.dumps(
            {
                "single_configurations": summary["single_configuration_summary"],
                "single_configurations_perturbed": summary[
                    "single_configuration_perturbed_summary"
                ],
                "double_scan": summary["double_scan"],
                "derivative_checks": summary["derivative_checks"],
                "invariance_checks": summary["invariance_checks"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
