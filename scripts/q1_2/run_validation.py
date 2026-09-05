#!/usr/bin/env python3
"""Deterministic exhaustive validation of all 56 legal (receiver, emitter) pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.q1_1.localization import fy_position, pairwise_angles, polar_to_cartesian
from scripts.q1_2.identification import identify_anonymous_emitter


ROOT = Path(__file__).resolve().parents[2]


def table_positions() -> dict[int, np.ndarray]:
    """Read the receiver test positions directly from the maintained problem."""
    text = (ROOT / "problem" / "B题.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*(\d+)\s*\|\s*\(([\d.]+),\s*([\d.]+)\)\s*\|", text, re.M)
    positions = {int(i): polar_to_cartesian(float(r), float(a)) for i, r, a in rows}
    if set(positions) != set(range(10)):
        raise ValueError("problem Table 1 must contain FY00 through FY09")
    return positions


def scenarios(receiver_id: int, table: dict[int, np.ndarray]):
    yield "ideal", "ideal", fy_position(receiver_id)
    yield "table1", "table1", table[receiver_id]
    # Eight fixed test points around the nominal position, not a flight bound
    # or a probabilistic model. The zero/zero point is covered by 'ideal'.
    for dr in (-5.0, 0.0, 5.0):
        for da in (-1.0, 0.0, 1.0):
            if dr == 0 and da == 0:
                continue
            yield (
                "fixed_offsets",
                f"dr{dr:+g}_da{da:+g}",
                polar_to_cartesian(
                    100 + dr,
                    40 * (receiver_id - 1) + da,
                ),
            )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_validation(output_dir: Path) -> dict:
    table = table_positions()
    cases, hypotheses, candidates, signatures = [], [], [], []
    example = None
    for k in range(2, 10):
        for suite, sample, truth in scenarios(k, table):
            for q in range(2, 10):
                if q == k:
                    continue
                case_id = f"{sample}_k{k:02d}_q{q:02d}"
                observed = pairwise_angles(truth, [fy_position(i) for i in (0, 1, q)])
                # Only k and the observed angles enter the inference function.
                result = identify_anonymous_emitter(k, observed)
                selected = result.selected
                position_error = (
                    float(np.linalg.norm(selected.position - truth))
                    if selected
                    else None
                )
                case = {
                    "case_id": case_id,
                    "suite": suite,
                    "sample": sample,
                    "receiver_id": k,
                    "true_emitter_id": q,
                    "true_x_m": float(truth[0]),
                    "true_y_m": float(truth[1]),
                    "true_displacement_m": float(
                        np.linalg.norm(truth - fy_position(k))
                    ),
                    "alpha_01_deg": float(np.rad2deg(observed[0])),
                    "alpha_0U_deg": float(np.rad2deg(observed[1])),
                    "alpha_1U_deg": float(np.rad2deg(observed[2])),
                    "status": result.status,
                    "selected_emitter_id": selected.emitter_id if selected else None,
                    "identity_correct": selected is not None
                    and selected.emitter_id == q,
                    "estimate_x_m": float(selected.position[0]) if selected else None,
                    "estimate_y_m": float(selected.position[1]) if selected else None,
                    "position_error_m": position_error,
                    "position_correct": position_error is not None
                    and position_error < 1e-6,
                    "selected_residual_norm_rad": selected.residual_norm
                    if selected
                    else None,
                    "selected_distance_m": selected.distance_to_ideal
                    if selected
                    else None,
                    "distance_margin_m": result.distance_margin,
                    "identity_margin_m": result.identity_margin,
                    "continuum_distance_bound_m": result.continuum_distance_bound,
                    "consistent_id_count": sum(
                        bool(h.candidates) for h in result.hypotheses
                    ),
                    "position_candidate_count": sum(
                        len(h.candidates) for h in result.hypotheses
                    ),
                }
                cases.append(case)
                example_hypotheses = []
                for h in result.hypotheses:
                    row = {
                        "case_id": case_id,
                        "suite": suite,
                        "receiver_id": k,
                        "true_emitter_id": q,
                        "assumed_emitter_id": h.emitter_id,
                        "local_status": h.local_status,
                        "local_residual_norm_rad": h.local_residual_norm,
                        "position_candidate_count": len(h.candidates),
                        "nearest_distance_m": min(
                            (c.distance_to_ideal for c in h.candidates), default=None
                        ),
                        "continuum_branch_count": len(h.continuum_distance_bounds),
                        "continuum_distance_bound_m": min(
                            h.continuum_distance_bounds, default=None
                        ),
                        "local_message": h.local_message,
                    }
                    hypotheses.append(row)
                    example_hypotheses.append(row)
                    for index, c in enumerate(h.candidates):
                        candidates.append(
                            {
                                "case_id": case_id,
                                "suite": suite,
                                "receiver_id": k,
                                "true_emitter_id": q,
                                "assumed_emitter_id": h.emitter_id,
                                "candidate_index": index,
                                "source": c.source,
                                "x_m": float(c.position[0]),
                                "y_m": float(c.position[1]),
                                "residual_norm_rad": c.residual_norm,
                                "distance_to_ideal_m": c.distance_to_ideal,
                            }
                        )
                if suite == "table1" and k == 3 and q == 5:
                    example = {"case": case, "hypotheses": example_hypotheses}
                if suite == "ideal":
                    signatures.append(
                        {
                            "receiver_id": k,
                            "emitter_id": q,
                            "alpha_01_deg": float(np.rad2deg(observed[0])),
                            "alpha_0U_deg": float(np.rad2deg(observed[1])),
                            "alpha_1U_deg": float(np.rad2deg(observed[2])),
                        }
                    )
        print(f"Completed FY{k:02d}: 70 cases", flush=True)

    suite_summaries = {}
    for suite in ("ideal", "table1", "fixed_offsets"):
        rows = [r for r in cases if r["suite"] == suite]
        hypothesis_rows = [r for r in hypotheses if r["suite"] == suite]
        errors = [
            r["position_error_m"] for r in rows if r["position_error_m"] is not None
        ]
        margins = [
            r["identity_margin_m"] for r in rows if r["identity_margin_m"] is not None
        ]
        suite_summaries[suite] = {
            "case_count": len(rows),
            "correct_id_count": sum(r["identity_correct"] for r in rows),
            "correct_position_count": sum(r["position_correct"] for r in rows),
            "max_position_error_m": max(errors, default=None),
            "min_identity_margin_m": min(margins, default=None),
            "max_true_displacement_m": max(r["true_displacement_m"] for r in rows),
            "local_geometry_error_count": sum(
                r["local_status"] == "geometry_error" for r in hypothesis_rows
            ),
            "local_not_converged_count": sum(
                r["local_status"] == "not_converged" for r in hypothesis_rows
            ),
            "circle_completed_after_local_error_count": sum(
                r["local_status"] in ("geometry_error", "not_converged")
                and r["position_candidate_count"] > 0
                for r in hypothesis_rows
            ),
            "continuum_branch_count": sum(
                r["continuum_branch_count"] for r in hypothesis_rows
            ),
            "failures": [
                r["case_id"]
                for r in rows
                if not (r["identity_correct"] and r["position_correct"])
            ],
        }
    signature_separations = []
    for k in range(2, 10):
        angle_vectors = [
            np.array([r["alpha_01_deg"], r["alpha_0U_deg"], r["alpha_1U_deg"]])
            for r in signatures
            if r["receiver_id"] == k
        ]
        signature_separations.extend(
            float(np.linalg.norm(a - b))
            for i, a in enumerate(angle_vectors)
            for b in angle_vectors[i + 1 :]
        )
    summary = {
        "radius_m": 100.0,
        "angle_noise": "none",
        "residual_tolerance_rad": 1e-8,
        "position_tolerance_m": 1e-6,
        "distance_tie_tolerance_m": 1e-5,
        "fixed_offsets": {
            "radial_m": [-5, 0, 5],
            "angular_deg": [-1, 0, 1],
            "exclude_zero_zero": True,
        },
        "case_count": len(cases),
        "hypothesis_count": len(hypotheses),
        "candidate_count": len(candidates),
        "min_ideal_signature_separation_deg_l2": min(signature_separations),
        "suites": suite_summaries,
        "example": example,
        "evidence_scope": "Finite deterministic test points; local sufficiency is proved separately. Table 1 supplies receiver positions only; transmitters are ideal. Distance margins are geometric selection margins, not noise bounds.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("cases", cases),
        ("hypotheses", hypotheses),
        ("candidates", candidates),
        ("ideal_signatures", signatures),
    ):
        write_csv(output_dir / f"{name}.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"case_count": len(cases), "suites": suite_summaries}, indent=2),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "q1_2")
    args = parser.parse_args()
    summary = run_validation(args.output_dir)
    if any(suite["failures"] for suite in summary["suites"].values()):
        raise SystemExit("One or more Q1(2) cases failed; see summary.json")


if __name__ == "__main__":
    main()
