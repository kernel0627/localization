"""Independent nonlinear pulse audit of radial/tangential coupling.

The perturbations are offline diagnostic initial states. Neither their true
coordinates nor achieved motion is supplied to any receiver controller.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.q1_1.localization import pairwise_angles
from scripts.q1_3.analyze_iterative_reference import cycle_matrix, linear_model
from scripts.q1_3.local_adjustment import (
    LocalSettings,
    decide_local_adjustment,
    execute_relative_polar_step,
    public_schedule,
)

ROOT = Path(__file__).resolve().parents[2]
ANGLES = np.deg2rad(np.arange(9) * 40)
TARGET = np.vstack(
    (np.zeros(2), 100 * np.column_stack((np.cos(ANGLES), np.sin(ANGLES))))
)


def active_response(drone, component, amplitude, schedule):
    points = TARGET.copy()
    r = 100 + (amplitude if component == "radial" else 0)
    theta = ANGLES[drone - 1] + (amplitude / 100 if component == "tangential" else 0)
    points[drone] = r * np.array([np.cos(theta), np.sin(theta)])
    choices = []
    for circular in schedule:
        before = points.copy()
        for receiver in range(2, 10):
            if receiver in circular:
                continue
            own_angles = pairwise_angles(before[receiver], before[[0, *circular]])
            decision = decide_local_adjustment(
                receiver, circular, own_angles, LocalSettings()
            )
            assert decision.selected is not None
            choices.append((receiver, decision.selected.pair))
            assert abs(decision.radial_step_m) < 5
            assert abs(decision.angular_step_rad) < np.deg2rad(2)
            points[receiver] = execute_relative_polar_step(
                before[receiver], decision.radial_step_m, decision.angular_step_rad
            )
    error = np.empty(16)
    error[::2] = np.linalg.norm(points[2:], axis=1) - 100
    delta = np.arctan2(points[2:, 1], points[2:, 0]) - ANGLES[1:]
    error[1::2] = 100 * np.arctan2(np.sin(delta), np.cos(delta))
    return error, choices


def main():
    all_matrices, _, _ = linear_model(0.5)
    full = public_schedule()
    selected = ((1, 4, 5), (1, 7, 8))
    methods = {"main": full, "two_configuration": selected}
    radial = np.arange(0, 16, 2)
    tangent = radial + 1
    block_rows, pulse_rows, matrices = [], [], {}
    for method, schedule in methods.items():
        a = cycle_matrix([all_matrices[full.index(s)] for s in schedule])
        matrices[method] = a
        for output_name, output in (("radial", radial), ("tangential", tangent)):
            for input_name, indices in (("radial", radial), ("tangential", tangent)):
                block_rows.append(
                    {
                        "method": method,
                        "period_slots": len(schedule),
                        "output_component": output_name,
                        "input_component": input_name,
                        "operator_norm": float(
                            np.linalg.norm(a[np.ix_(output, indices)], 2)
                        ),
                    }
                )
        for drone, component in ((2, "radial"), (4, "tangential")):
            h = 1e-3
            column = 2 * (drone - 2) + int(component == "tangential")
            plus, plus_choices = active_response(drone, component, h, schedule)
            minus, minus_choices = active_response(drone, component, -h, schedule)
            _, nominal_choices = active_response(drone, component, 0, schedule)
            assert plus_choices == minus_choices == nominal_choices
            fd = (plus - minus) / (2 * h)
            difference = float(np.max(np.abs(fd - a[:, column])))
            assert difference < 2e-6, (method, drone, component, difference)
            cross = tangent if component == "radial" else radial
            winner = int(cross[np.argmax(np.abs(plus[cross]))])
            pulse_rows.append(
                {
                    "method": method,
                    "period_slots": len(schedule),
                    "input_drone": drone,
                    "input_component": component,
                    "input_amplitude_m": h,
                    "largest_cross_output_drone": winner // 2 + 2,
                    "largest_cross_output_component": "tangential"
                    if winner % 2
                    else "radial",
                    "signed_cross_output_m": float(plus[winner]),
                    "linear_predicted_cross_output_m": float(h * a[winner, column]),
                    "center_difference_max_error": difference,
                    "selection_changes": 0,
                }
            )
    same_horizon = [
        {
            "method": "main",
            "slots": 28,
            "operator_norm": float(np.linalg.norm(matrices["main"], 2)),
        },
        {
            "method": "two_configuration",
            "slots": 28,
            "operator_norm": float(
                np.linalg.norm(
                    np.linalg.matrix_power(matrices["two_configuration"], 14), 2
                )
            ),
        },
    ]
    output = ROOT / "outputs/q1_3/two_configuration_review"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Offline small-error active feedback, gain 0.5, no hold; block norms at different period lengths cannot be directly ranked. Same-horizon norms apply only locally and do not predict Table 1 cost.",
        "blocks": block_rows,
        "nonlinear_pulses": pulse_rows,
        "same_28_slot_horizon": same_horizon,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "scripts/q1_3/local_adjustment.py",
                ROOT / "scripts/q1_3/analyze_iterative_reference.py",
            )
        },
    }
    (output / "component_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
