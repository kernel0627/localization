"""Audit the information available to each receiver in the joint baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.q1_2.run_validation import table_positions, write_csv
from scripts.q1_3.joint_localization import (
    ideal_formation,
    joint_jacobian,
    observation_layout,
    predict_angles,
)


def alternative_for_receiver(positions, receiver_id, rotation_rad=0.01):
    """Construct a different nearby state with identical receiver-only angles.

    Move the receiver along the circle through FY00, FY01 and itself. Rotate
    all unknown transmitter directions with its direction to FY00; the angle
    to the other fixed reference remains equal on the same circle arc.
    """
    points = np.asarray(positions, dtype=float)
    origin = points[0]
    receiver = points[receiver_id] - origin
    reference = points[1] - origin
    center = np.linalg.solve(
        2 * np.stack((reference, receiver)),
        np.array([reference @ reference, receiver @ receiver]),
    )

    def rotate(vectors, angle):
        cosine, sine = np.cos(angle), np.sin(angle)
        return vectors @ np.array([[cosine, sine], [-sine, cosine]])

    moved = center + rotate(receiver - center, rotation_rad)
    heading_change = np.arctan2(-moved[1], -moved[0]) - np.arctan2(
        -receiver[1], -receiver[0]
    )
    alternative = points.copy()
    alternative[2:] = (
        origin + moved + rotate(points[2:] - points[receiver_id], heading_change)
    )
    return alternative


def audit_information():
    configurations = ((2, 8), (3, 9))
    layout = observation_layout(configurations)
    table = table_positions()
    states = {
        "nominal": ideal_formation(),
        "table1": np.array([table[i] for i in range(10)]),
    }
    rows = []
    for state_name, positions in states.items():
        jacobian = joint_jacobian(positions, layout)
        for receiver_id in range(2, 10):
            mask = layout[:, 0] == receiver_id
            local = jacobian[mask]
            nuisance = np.delete(
                local, [2 * (receiver_id - 2), 2 * (receiver_id - 2) + 1], axis=1
            )
            tolerance = np.linalg.svd(local, compute_uv=False)[0] * 1e-9
            local_rank = int(np.linalg.matrix_rank(local, tol=tolerance))
            nuisance_rank = int(np.linalg.matrix_rank(nuisance, tol=tolerance))
            alternative = alternative_for_receiver(positions, receiver_id)
            angle_difference = predict_angles(
                alternative, layout[mask]
            ) - predict_angles(positions, layout[mask])
            rows.append(
                {
                    "state": state_name,
                    "receiver_id": receiver_id,
                    "own_angle_count": int(mask.sum()),
                    "local_rank": local_rank,
                    "nuisance_rank": nuisance_rank,
                    "receiver_coordinate_constraints": local_rank - nuisance_rank,
                    "alternative_receiver_displacement_m": float(
                        np.linalg.norm(
                            alternative[receiver_id] - positions[receiver_id]
                        )
                    ),
                    "alternative_max_formation_displacement_m": float(
                        np.linalg.norm(alternative - positions, axis=1).max()
                    ),
                    "max_own_angle_difference_rad": float(
                        np.abs(angle_difference).max()
                    ),
                }
            )
    return rows


def main():
    rows = audit_information()
    output_dir = Path(__file__).resolve().parents[2] / "outputs/q1_3/joint_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "receiver_information.csv", rows)
    summary = {
        "case_count": len(rows),
        "receiver_coordinate_constraints": sorted(
            {row["receiver_coordinate_constraints"] for row in rows}
        ),
        "min_alternative_receiver_displacement_m": min(
            row["alternative_receiver_displacement_m"] for row in rows
        ),
        "max_own_angle_difference_rad": max(
            row["max_own_angle_difference_rad"] for row in rows
        ),
        "interpretation": "The current two-configuration baseline requires shared receiver observations. Each receiver's own measurements admit distinct nearby formations with fixed FY00/FY01. The scheduled two movement batches validate exact centralized localization and actuation; they do not establish a receiver-only adjustment protocol.",
    }
    (output_dir / "information_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if any(
        row["max_own_angle_difference_rad"] > 1e-12
        or row["alternative_receiver_displacement_m"] <= 0.1
        for row in rows
    ):
        raise RuntimeError("receiver ambiguity construction failed")


if __name__ == "__main__":
    main()
