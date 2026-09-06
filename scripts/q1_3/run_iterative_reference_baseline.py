#!/usr/bin/env python3
"""Run receiver-local dual-bias adjustment on the problem's Table 1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.q1_1.localization import fy_position, pairwise_angles
from scripts.q1_2.run_validation import table_positions, write_csv
from scripts.q1_3.local_adjustment import (
    LocalSettings,
    ReceiverController,
    execute_relative_polar_step,
    public_schedule,
)

ROOT = Path(__file__).resolve().parents[2]


def formation_metrics(positions):
    nominal = np.array([fy_position(i) for i in range(10)])
    radial = np.linalg.norm(positions[1:], axis=1) - 100
    theta = np.arctan2(positions[1:, 1], positions[1:, 0])
    angular = (theta - np.deg2rad(np.arange(9) * 40) + np.pi) % (2 * np.pi) - np.pi
    gaps = (np.roll(theta, -1) - theta) % (2 * np.pi)
    errors = np.linalg.norm(positions[1:] - nominal[1:], axis=1)
    return {
        "max_position_error_m": float(errors.max()),
        "rms_position_error_m": float(np.sqrt(np.mean(errors**2))),
        "max_radial_error_m": float(np.abs(radial).max()),
        "max_angular_error_deg": float(np.rad2deg(np.abs(angular).max())),
        "max_adjacent_gap_error_deg": float(np.abs(np.rad2deg(gaps) - 40).max()),
    }


def simulate_adjustment(
    initial_positions,
    *,
    settings=LocalSettings(),
    max_epochs=20,
    retain_details=True,
    noise=None,
):
    """Simulate a fixed public schedule and independent local decisions.

    Each slot samples a frozen state and all receiver commands execute together.
    A quiet full schedule period terminates logging. Each local policy keeps
    monitoring its own angles and would reactivate if its references changed.
    """
    if settings.radius_m != 100:
        raise ValueError("Table 1 simulator uses radius 100 m")
    if not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError("max_epochs must be a positive integer")
    points = np.array(initial_positions, dtype=float, copy=True)
    if points.shape != (10, 2) or not np.isfinite(points).all():
        raise ValueError("initial positions must have finite shape (10, 2)")
    nominal = np.array([fy_position(i) for i in range(10)])
    if not np.allclose(points[:2], nominal[:2], rtol=0, atol=1e-12):
        raise ValueError("FY00 and FY01 must equal the calibrated Table 1 references")
    schedule = public_schedule()
    controllers = {i: ReceiverController(i, settings) for i in range(2, 10)}
    history = [
        {
            "slot": 0,
            "epoch": 0,
            "transmitters": "",
            "transmitter_uses": 0,
            "cumulative_transmitter_uses": 0,
            "endpoint_displacement_m": 0.0,
            "cumulative_endpoint_displacement_m": 0.0,
            **formation_metrics(points),
        }
    ]
    epochs = [{"epoch": 0, "slot": 0, **formation_metrics(points)}]
    steps, observations, candidates = [], [], []
    state_rows = [
        {"slot": 0, "drone_id": i, "x_m": float(p[0]), "y_m": float(p[1])}
        for i, p in enumerate(points)
    ]
    status = "max_epochs_reached"
    attempted, nonzero, failed = 0, 0, 0
    active_slots = 0
    transmitter_uses, movement, last_motion_slot = 0, 0.0, 0
    for epoch in range(1, max_epochs + 1):
        quiet_epoch = True
        for phase, circular_tx in enumerate(schedule, start=1):
            slot = (epoch - 1) * len(schedule) + phase
            tx = (0, *circular_tx)
            before, after = points.copy(), points.copy()
            slot_active = False
            for i in range(2, 10):
                if i in tx:
                    continue
                angles = (
                    pairwise_angles(before[i], before[list(tx)])
                    if noise is None
                    else noise.observe(before[i], before[list(tx)], slot, i)
                )
                decision = controllers[i].decide(circular_tx, angles)
                attempted += 1
                if decision.status == "fit_failed":
                    failed += 1
                quiet_epoch &= decision.status == "within_tolerance"
                moved = decision.radial_step_m != 0 or decision.angular_step_rad != 0
                nonzero += int(moved)
                slot_active |= moved
                if moved:
                    radial, angular = decision.radial_step_m, decision.angular_step_rad
                    if noise is not None:
                        radial, angular = noise.execute(radial, angular, slot, i)
                    after[i] = execute_relative_polar_step(before[i], radial, angular)
                if retain_details:
                    chosen = decision.selected
                    steps.append(
                        {
                            "slot": slot,
                            "epoch": epoch,
                            "receiver_id": i,
                            "transmitters": "-".join(map(str, tx)),
                            "selected_pair": "-".join(map(str, chosen.pair))
                            if chosen
                            else "",
                            "status": decision.status,
                            "estimated_radial_bias_m": chosen.radial_bias_m
                            if chosen
                            else None,
                            "estimated_angular_bias_deg": float(
                                np.rad2deg(chosen.angular_bias_rad)
                            )
                            if chosen
                            else None,
                            "local_consistency_rad": chosen.consistency_rad
                            if chosen
                            else None,
                            "radial_step_m": decision.radial_step_m,
                            "angular_step_deg": float(
                                np.rad2deg(decision.angular_step_rad)
                            ),
                            "before_x_m": float(before[i, 0]),
                            "before_y_m": float(before[i, 1]),
                            "after_x_m": float(after[i, 0]),
                            "after_y_m": float(after[i, 1]),
                        }
                    )
                    observations.append(
                        {
                            "slot": slot,
                            "receiver_id": i,
                            "transmitters": "-".join(map(str, tx)),
                            **{
                                f"angle_{j}_rad": float(a) for j, a in enumerate(angles)
                            },
                        }
                    )
                    for candidate in decision.candidates:
                        row = asdict(candidate)
                        row["pair"] = "-".join(map(str, candidate.pair))
                        candidates.append({"slot": slot, "receiver_id": i, **row})
            points = after
            active_slots += int(slot_active)
            if slot_active:
                last_motion_slot = slot
            transmitter_uses += len(tx)
            slot_movement = float(np.linalg.norm(after - before, axis=1).sum())
            movement += slot_movement
            history.append(
                {
                    "slot": slot,
                    "epoch": epoch,
                    "transmitters": "-".join(map(str, tx)),
                    "transmitter_uses": len(tx),
                    "cumulative_transmitter_uses": transmitter_uses,
                    "endpoint_displacement_m": slot_movement,
                    "cumulative_endpoint_displacement_m": movement,
                    **formation_metrics(points),
                }
            )
            if retain_details:
                state_rows.extend(
                    {
                        "slot": slot,
                        "drone_id": i,
                        "x_m": float(p[0]),
                        "y_m": float(p[1]),
                    }
                    for i, p in enumerate(points)
                )
            if failed:
                break
        epochs.append({"epoch": epoch, "slot": slot, **formation_metrics(points)})
        if failed:
            status = "local_fit_failed"
            break
        if quiet_epoch:
            status = "quiet_full_cycle"
            break
    summary = {
        "method": "receiver-local radial/angular bias estimation with public transmitter rotation",
        "status": status,
        "settings": asdict(settings),
        "fixed_reference_ids": [0, 1],
        "schedule_period_slots": len(schedule),
        "epochs": epoch,
        "measurement_slots": slot,
        "transmitter_uses": transmitter_uses,
        "total_endpoint_displacement_m": movement,
        "last_motion_slot": last_motion_slot,
        "post_motion_confirmation_slots": slot - last_motion_slot,
        "post_motion_transmitter_uses": sum(
            row["transmitter_uses"] for row in history if row["slot"] > last_motion_slot
        ),
        "slots_with_motion": active_slots,
        "receiver_decisions": attempted,
        "nonzero_receiver_moves": nonzero,
        "failed_local_fits": failed,
        "local_hold_rule": "enter after 21 consecutive own observations at half tolerance; resume when any local indicator exceeds full tolerance",
        "termination": "a full 28-slot period with every receiver locally within tolerance; actual coordinates are used only for retrospective metrics",
        "initial_metrics": history[0],
        "final_metrics": history[-1],
        "epoch_history": epochs,
        "scope": "Exact angles and ideal relative radial/angular actuation; calibrated FY00/FY01. Each receiver uses its own six angles. Transmitter schedule is public and fixed; pair selection is local. Finite simulations do not prove general convergence.",
    }
    thresholds = []
    if noise is not None:
        summary["simulation_noise"] = asdict(noise)
        summary["scope"] = (
            "Independent Gaussian errors on four local ray directions before six "
            "pair angles are formed; optional independent relative radial/angular "
            "execution errors. Fixed FY00/FY01, original local controller and stop "
            "thresholds. Noise settings are experimental assumptions."
        )
    for label, threshold in (("2cm", 0.02), ("1cm", 0.01), ("1mm", 0.001)):
        hits = [row for row in history if row["max_position_error_m"] < threshold]
        row = hits[0] if hits else None
        thresholds.append(
            {
                "threshold": label,
                "threshold_m": threshold,
                "first_slot": row["slot"] if row else None,
                "transmitter_uses_at_first": row["cumulative_transmitter_uses"]
                if row
                else None,
                "endpoint_displacement_at_first_m": row[
                    "cumulative_endpoint_displacement_m"
                ]
                if row
                else None,
                "all_recorded_slots_after_first_below": all(
                    item["max_position_error_m"] < threshold
                    for item in history[row["slot"] :]
                )
                if row
                else False,
            }
        )
    summary["precision_thresholds"] = thresholds
    return {
        "summary": summary,
        "history": history,
        "epochs": epochs,
        "steps": steps,
        "observations": observations,
        "candidates": candidates,
        "positions": state_rows,
        "final_positions": points,
    }


def run_validation(output_dir, *, settings=LocalSettings(), max_epochs=20):
    table = table_positions()
    initial = np.array([table[i] for i in range(10)])
    run = simulate_adjustment(initial, settings=settings, max_epochs=max_epochs)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("error_history", run["history"]),
        ("epoch_history", run["epochs"]),
        ("adjustment_steps", run["steps"]),
        ("observations", run["observations"]),
        ("local_candidates", run["candidates"]),
        ("positions", run["positions"]),
    ):
        write_csv(output_dir / f"{name}.csv", rows)
    write_csv(
        output_dir / "transmitter_schedule.csv",
        [
            {
                "phase": index,
                "transmitters": "-".join(map(str, (0, *tx))),
                "receivers": "-".join(str(i) for i in range(2, 10) if i not in tx),
            }
            for index, tx in enumerate(public_schedule(), start=1)
        ],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(run["summary"], indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run["summary"], indent=2, ensure_ascii=False))
    if run["summary"]["status"] != "quiet_full_cycle":
        raise RuntimeError(
            "local iteration did not reach its stopping condition; see summary.json"
        )
    return run["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/q1_3/iterative_reference_baseline",
    )
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()
    run_validation(
        args.output_dir,
        settings=LocalSettings(gain=args.gain),
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
