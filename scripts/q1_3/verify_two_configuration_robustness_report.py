"""Independent checks of report rates, quantiles, sources and export hashes."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    with (ROOT / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify():
    raw = read("outputs/q1_3/two_configuration_robustness/trials.csv")
    report = read("outputs/q1_3/two_configuration_robustness_report/random_summary.csv")
    comparisons = 0
    for row in report:
        group = [
            r
            for r in raw
            if r["gain"] == row["gain"]
            and r["condition"] == row["condition"]
            and float(r["trial"]) >= 0
        ]
        assert len(group) == 100
        for event in (
            "stopped",
            "final_below_1cm",
            "final_below_1mm",
            "joint_success_1cm",
            "joint_success_1mm",
        ):
            count = sum(r[event] == "True" for r in group)
            assert count == int(row[event + "_count"])
            p, n, z = count / 100, 100, 1.959963984540054
            center = (p + z * z / (2 * n)) / (1 + z * z / n)
            width = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            np.testing.assert_allclose(
                [
                    float(row[event + "_wilson95_low"]),
                    float(row[event + "_wilson95_high"]),
                ],
                [max(0, center - width), min(1, center + width)],
                atol=1e-14,
            )
            comparisons += 3
        for metric in (
            "max_position_error_m",
            "rms_position_error_m",
            "measurement_slots",
            "transmitter_uses",
            "movement_m",
        ):
            for label, q in (("p05", 0.05), ("median", 0.5), ("p95", 0.95)):
                expected = np.quantile([float(r[metric]) for r in group], q)
                np.testing.assert_allclose(
                    float(row[metric + "_" + label]), expected, atol=1e-12
                )
                comparisons += 1
    baseline = read("outputs/q1_3/robustness/trials.csv")
    three = read(
        "outputs/q1_3/two_configuration_robustness_report/main_common_conditions_three_way_summary.csv"
    )
    for row in three:
        if row["method"] == "main":
            source = baseline
        else:
            gain = float(row["method"].rsplit("_", 1)[-1])
            source = [r for r in raw if float(r["gain"]) == gain]
        group = [
            r
            for r in source
            if r["condition"] == row["condition"] and float(r["trial"]) >= 0
        ]
        assert len(group) == 100
        for event in ("stopped", "final_below_1cm", "joint_success_1cm"):
            assert int(row[event + "_count"]) == sum(r[event] == "True" for r in group)
            comparisons += 1
        for metric in ("max_position_error_m", "measurement_slots", "transmitter_uses"):
            np.testing.assert_allclose(
                float(row[metric + "_median"]),
                np.median([float(r[metric]) for r in group]),
            )
            comparisons += 1
    manifest_path = (
        ROOT / "figures/q1_3/two_configuration_robustness/figure_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    hash_checks = 0
    for field, values in manifest.items():
        if isinstance(values, dict) and "sha256" in field:
            for name, expected in values.items():
                path = ROOT / name
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
                hash_checks += 1
    theory = read(
        "figures/q1_3/two_configuration_robustness/data/theory_empirical_data.csv"
    )
    assert len(theory) == 24
    for row in theory:
        sample = []
        for r in raw:
            if (
                float(r["gain"]) == float(row["gain"])
                and r["condition"] == row["condition"]
                and float(r["trial"]) >= 0
            ):
                path = Path(r["summary_path"]).parent / "slot_metrics.csv"
                state = next(
                    s for s in read(path) if int(s["slot"]) == int(row["phase"])
                )
                sample.append(float(state["rms_position_error_m"]))
        assert len(sample) == 100
        np.testing.assert_allclose(
            float(row["empirical_root_mean_rms_squared_m"]),
            np.sqrt(np.mean(np.square(sample))),
            rtol=1e-13,
        )
    result = {
        "random_groups": len(report),
        "three_way_groups": len(three),
        "rate_interval_quantile_checks": comparisons,
        "artifact_and_math_hash_checks": hash_checks,
        "phase_statistics_rebuilt": len(theory),
        "phase_theories_within_pointwise_empirical_95ci": sum(
            r["theory_within_empirical_bootstrap95"] == "True" for r in theory
        ),
    }
    (
        ROOT / "outputs/q1_3/two_configuration_robustness_review/report_audit.json"
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    verify()
