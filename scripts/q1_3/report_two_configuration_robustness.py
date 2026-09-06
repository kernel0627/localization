"""汇总双配置轮换的随机与含噪试验，生成中文科研图与报告。

读取已有实验记录，重建统计量并导出绘图数据，不运行控制器仿真。
默认要求完整批次并输出 PNG；--formats png pdf svg 可加选矢量图。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from scripts.figure_style import configure_style


ROOT = Path(__file__).resolve().parents[2]
INPUT_DEFAULT = ROOT / "outputs/q1_3/two_configuration_robustness"
OUTPUT_DEFAULT = ROOT / "outputs/q1_3/two_configuration_robustness_report"
FIGURE_DEFAULT = ROOT / "figures/q1_3/two_configuration_robustness"
MATH_DEFAULT = ROOT / "outputs/q1_3/two_configuration_robustness_math"
DOC_DEFAULT = ROOT / "docs/q1_3/双配置轮换随机与含噪验证.md"
INJECTION_AUDIT = (
    ROOT
    / "outputs/q1_3/two_configuration_robustness_review/nonlinear_noise_injection_audit.json"
)
GAINS = (0.5, 1.0)
CONDITIONS = (
    "exact",
    "bearing_0.001deg",
    "bearing_0.01deg",
    "bearing_0.1deg",
    "actuation_1pct",
    "combined_0.001deg_1pct",
    "combined_0.01deg_1pct",
    "link_bias_0.001deg",
)
MAIN_COMMON = (
    "exact",
    "bearing_0.001deg",
    "bearing_0.01deg",
    "bearing_0.1deg",
    "actuation_1pct",
)
DISPLAY = {
    "exact": "精确条件",
    "bearing_0.001deg": "测角噪声 0.001°",
    "bearing_0.01deg": "测角噪声 0.01°",
    "bearing_0.1deg": "测角噪声 0.1°",
    "actuation_1pct": "执行误差 1%",
    "combined_0.001deg_1pct": "混合扰动 0.001° + 1%",
    "combined_0.01deg_1pct": "混合扰动 0.01° + 1%",
    "link_bias_0.001deg": "固定链路偏置 0.001°",
}
METHOD_DISPLAY = {
    "main": "main 原方法",
    "two_configuration_gain_0.5": "双配置增益 0.5",
    "two_configuration_gain_1": "双配置增益 1",
}
FIGURE_METHOD_DISPLAY = METHOD_DISPLAY
FIGURE_CONDITION = {
    "exact": "精确\n条件",
    "bearing_0.001deg": "测角噪声\n0.001°",
    "bearing_0.01deg": "测角噪声\n0.01°",
    "bearing_0.1deg": "测角噪声\n0.1°",
    "actuation_1pct": "执行误差\n1%",
    "combined_0.001deg_1pct": "混合扰动\n0.001°\n+ 1%",
    "combined_0.01deg_1pct": "混合扰动\n0.01°\n+ 1%",
    "link_bias_0.001deg": "固定偏置\n0.001°",
}
BLUE, ORANGE, GREY, TEAL = "#0072B2", "#D55E00", "#666666", "#009E73"
HATCHES = ("", "///", "xx")
METHOD_MARKERS = ("D", "o", "s")
EXPORT_FORMATS = ("png",)
EXPORTED_FIGURES: list[Path] = []
STYLE_METADATA: dict[str, Any] = {}


def figure_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.015, 0.015, text, ha="left", va="bottom", fontsize=8.5, color="#444444")


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def integer(value: Any) -> int | None:
    candidate = number(value)
    return None if candidate is None else int(candidate)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def wilson(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = successes / n
    d = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, middle - half)), float(min(1, middle + half)))


def q(values: Iterable[float], p: float) -> float | None:
    values = list(values)
    return None if not values else float(np.quantile(values, p))


def normalized_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    required = {
        "gain",
        "condition",
        "trial",
        "stopped",
        "final_below_1cm",
        "measurement_slots",
        "transmitter_uses",
        "movement_m",
        "max_position_error_m",
        "rms_position_error_m",
    }
    missing = required - set(raw_rows[0] if raw_rows else [])
    if missing:
        raise ValueError(f"trials.csv lacks required columns: {sorted(missing)}")
    result = []
    for row in raw_rows:
        trial = integer(row["trial"])
        stopped = as_bool(row["stopped"])
        final_cm = as_bool(row["final_below_1cm"])
        final_mm = (
            as_bool(row["final_below_1mm"])
            if "final_below_1mm" in row
            else float(row["max_position_error_m"]) < 0.001
        )
        result.append(
            {
                **row,
                "gain": float(row["gain"]),
                "trial": trial,
                "stopped": stopped,
                "final_below_1cm": final_cm,
                "final_below_1mm": final_mm,
                "joint_success_1cm": as_bool(
                    row.get("joint_success_1cm", stopped and final_cm)
                ),
                "joint_success_1mm": as_bool(row["joint_success_1mm"])
                if "joint_success_1mm" in row
                else stopped and final_mm,
                "budget_exhausted": as_bool(
                    row.get("budget_exhausted", row.get("status") == "budget_exhausted")
                ),
                "fit_failed": as_bool(
                    row.get("fit_failed", row.get("status") == "local_fit_failed")
                ),
                "first_1cm_slot": integer(row.get("first_1cm_slot")),
                "first_1mm_slot": integer(row.get("first_1mm_slot")),
                "sustained_1cm_from_slot": integer(row.get("sustained_1cm_from_slot")),
                "sustained_1mm_from_slot": integer(row.get("sustained_1mm_from_slot")),
                "record_stays_1cm_after_first": as_bool(
                    row.get("record_stays_1cm_after_first")
                ),
                "record_stays_1mm_after_first": as_bool(
                    row.get("record_stays_1mm_after_first")
                ),
                "max_position_error_m": float(row["max_position_error_m"]),
                "rms_position_error_m": float(row["rms_position_error_m"]),
                "measurement_slots": int(float(row["measurement_slots"])),
                "transmitter_uses": int(float(row["transmitter_uses"])),
                "movement_m": float(row["movement_m"]),
                "failed_local_fits": int(float(row.get("failed_local_fits", 0))),
            }
        )
    return result


def validate_coverage(rows: list[dict[str, Any]], partial: bool) -> dict[str, Any]:
    random_rows = [
        row for row in rows if row["trial"] is not None and row["trial"] >= 0
    ]
    observed: dict[tuple[float, str], set[int]] = defaultdict(set)
    duplicates: list[tuple[float, str, int]] = []
    for row in random_rows:
        key = (row["gain"], row["condition"])
        if row["trial"] in observed[key]:
            duplicates.append((row["gain"], row["condition"], row["trial"]))
        observed[key].add(row["trial"])
    expected = {(gain, condition) for gain in GAINS for condition in CONDITIONS}
    unknown_gain_condition = sorted(
        {(row["gain"], row["condition"]) for row in rows} - expected
    )
    missing_groups = sorted(expected - set(observed))
    incomplete = {
        f"gain_{gain:g}/{condition}": sorted(
            set(range(100)) - observed[(gain, condition)]
        )
        for gain, condition in expected
        if observed[(gain, condition)] != set(range(100))
    }
    report = {
        "random_rows": len(random_rows),
        "expected_random_rows": 1600,
        "table1_rows": sum(row["trial"] == -1 for row in rows),
        "expected_table1_rows": 16,
        "missing_groups": missing_groups,
        "incomplete_trial_ids": incomplete,
        "duplicates": duplicates,
        "unknown_gain_condition": unknown_gain_condition,
        "full_16x100_coverage": not missing_groups
        and not incomplete
        and not duplicates,
    }
    table1_keys = [
        (row["gain"], row["condition"]) for row in rows if row["trial"] == -1
    ]
    duplicate_table1_keys = sorted(
        {key for key in table1_keys if table1_keys.count(key) > 1}
    )
    report["missing_table1_keys"] = sorted(expected - set(table1_keys))
    report["duplicate_table1_keys"] = duplicate_table1_keys
    report["full_16x100_coverage"] = (
        report["full_16x100_coverage"]
        and not unknown_gain_condition
        and not report["missing_table1_keys"]
        and not duplicate_table1_keys
    )
    if not partial:
        assert len(rows) == 1616, f"expected 1616 rows, found {len(rows)}"
        assert report["full_16x100_coverage"], json.dumps(report, indent=2)
        assert report["table1_rows"] == 16, report
    return report


def rates(group: list[dict[str, Any]], prefix: str, predicate: str) -> dict[str, Any]:
    count = sum(row[predicate] for row in group)
    low, high = wilson(count, len(group))
    return {
        f"{prefix}_count": count,
        f"{prefix}_rate": count / len(group),
        f"{prefix}_wilson95_low": low,
        f"{prefix}_wilson95_high": high,
    }


def summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for gain in GAINS:
        for condition in CONDITIONS:
            group = [
                r
                for r in rows
                if r["gain"] == gain and r["condition"] == condition and r["trial"] >= 0
            ]
            if not group:
                continue
            out: dict[str, Any] = {
                "gain": gain,
                "condition": condition,
                "runs": len(group),
            }
            out.update(rates(group, "stopped", "stopped"))
            out.update(rates(group, "final_below_1cm", "final_below_1cm"))
            out.update(rates(group, "final_below_1mm", "final_below_1mm"))
            out.update(rates(group, "joint_success_1cm", "joint_success_1cm"))
            out.update(rates(group, "joint_success_1mm", "joint_success_1mm"))
            out["fit_failure_runs"] = sum(r["fit_failed"] for r in group)
            out["budget_exhausted_runs"] = sum(r["budget_exhausted"] for r in group)
            out["failed_local_fits_total"] = sum(r["failed_local_fits"] for r in group)
            for metric in (
                "max_position_error_m",
                "rms_position_error_m",
                "measurement_slots",
                "transmitter_uses",
                "movement_m",
            ):
                for label, p in (("p05", 0.05), ("median", 0.5), ("p95", 0.95)):
                    out[f"{metric}_{label}"] = q((r[metric] for r in group), p)
            for threshold in ("1cm", "1mm"):
                first = [
                    r[f"first_{threshold}_slot"]
                    for r in group
                    if r[f"first_{threshold}_slot"] is not None
                ]
                sustained = [
                    r[f"sustained_{threshold}_from_slot"]
                    for r in group
                    if r[f"sustained_{threshold}_from_slot"] is not None
                ]
                recorded = [
                    r for r in group if r[f"record_stays_{threshold}_after_first"]
                ]
                out[f"first_{threshold}_count"] = len(first)
                out[f"first_{threshold}_denominator_runs"] = len(group)
                out[f"first_{threshold}_median_among_hits"] = q(first, 0.5)
                out[f"sustained_{threshold}_count"] = len(sustained)
                out[f"sustained_{threshold}_denominator_runs"] = len(group)
                out[f"sustained_{threshold}_median_among_hits"] = q(sustained, 0.5)
                out[f"record_stays_{threshold}_count"] = len(recorded)
                out[f"record_stays_{threshold}_denominator"] = len(first)
            result.append(out)
    return result


def table1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["trial"] != -1:
            continue
        out = {"gain": row["gain"], "condition": row["condition"], "table1_run": 1}
        for key in (
            "status",
            "stopped",
            "final_below_1cm",
            "final_below_1mm",
            "joint_success_1cm",
            "joint_success_1mm",
            "budget_exhausted",
            "fit_failed",
            "max_position_error_m",
            "rms_position_error_m",
            "measurement_slots",
            "transmitter_uses",
            "movement_m",
            "first_1cm_slot",
            "sustained_1cm_from_slot",
            "first_1mm_slot",
            "sustained_1mm_from_slot",
            "record_stays_1cm_after_first",
            "record_stays_1mm_after_first",
        ):
            out[key] = row.get(key)
        result.append(out)
    return result


def bootstrap_gain(
    rows: list[dict[str, Any]], seed: int = 20260906, n_boot: int = 10000
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output = []
    metrics = (
        "max_position_error_m",
        "rms_position_error_m",
        "measurement_slots",
        "transmitter_uses",
        "movement_m",
    )
    for condition in CONDITIONS:
        by_gain = {
            gain: {
                r["trial"]: r
                for r in rows
                if r["gain"] == gain and r["condition"] == condition and r["trial"] >= 0
            }
            for gain in GAINS
        }
        trials = sorted(set(by_gain[0.5]) & set(by_gain[1.0]))
        for metric in metrics:
            values = np.array(
                [
                    by_gain[1.0][trial][metric] - by_gain[0.5][trial][metric]
                    for trial in trials
                ],
                dtype=float,
            )
            if not len(values):
                continue
            samples = values[
                rng.integers(0, len(values), size=(n_boot, len(values)))
            ].mean(axis=1)
            output.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "contrast": "gain_1_minus_gain_0.5",
                    "paired_trials": len(values),
                    "conditional_pair_count": None,
                    "mean_difference": float(values.mean()),
                    "median_difference": float(np.median(values)),
                    "bootstrap95_low": float(np.quantile(samples, 0.025)),
                    "bootstrap95_high": float(np.quantile(samples, 0.975)),
                    "bootstrap_replicates": n_boot,
                    "bootstrap_seed": seed,
                }
            )
        for metric in ("first_1cm_slot", "first_1mm_slot"):
            # These are conditional contrasts: only trials where both gains actually
            # entered the threshold have a time.  Missing entries are never recoded.
            conditional_trials = [
                trial
                for trial in trials
                if by_gain[0.5][trial][metric] is not None
                and by_gain[1.0][trial][metric] is not None
            ]
            values = np.array(
                [
                    by_gain[1.0][trial][metric] - by_gain[0.5][trial][metric]
                    for trial in conditional_trials
                ],
                dtype=float,
            )
            if not len(values):
                continue
            samples = values[
                rng.integers(0, len(values), size=(n_boot, len(values)))
            ].mean(axis=1)
            output.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "contrast": "gain_1_minus_gain_0.5_conditional_both_first_events",
                    "paired_trials": len(values),
                    "conditional_pair_count": len(values),
                    "mean_difference": float(values.mean()),
                    "median_difference": float(np.median(values)),
                    "bootstrap95_low": float(np.quantile(samples, 0.025)),
                    "bootstrap95_high": float(np.quantile(samples, 0.975)),
                    "bootstrap_replicates": n_boot,
                    "bootstrap_seed": seed,
                }
            )
    return output


def controller_diagnostic_path(row: dict[str, Any]) -> Path | None:
    raw = str(row.get("summary_path", ""))
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.parent / "controller_diagnostics_last100.csv"


def tail100_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize only runner-provided last-100 diagnostics; never infer old data."""
    output = []
    fields = (
        "tail100_local_decisions",
        "tail100_nominal_selected_matches",
        "tail100_holding_count",
        "tail100_radial_clip_count",
        "tail100_angular_clip_count",
    )
    for gain in GAINS:
        for condition in CONDITIONS:
            group = [
                r
                for r in rows
                if r["gain"] == gain and r["condition"] == condition and r["trial"] >= 0
            ]
            available = [
                r
                for r in group
                if all(number(r.get(field)) is not None for field in fields)
            ]
            diagnostic_rows = []
            file_runs = 0
            for row in group:
                path = controller_diagnostic_path(row)
                if path is not None and path.exists():
                    diagnostic_rows.extend(read_csv(path))
                    file_runs += 1
            ratio_at_or_below_half = [
                record
                for record in diagnostic_rows
                if number(record.get("ratio")) is not None
                and number(record.get("ratio")) <= 0.5
            ]
            small_counts = [
                number(record.get("small_count_after"))
                for record in diagnostic_rows
                if number(record.get("small_count_after")) is not None
            ]
            aggregate: dict[str, Any] = {
                "gain": gain,
                "condition": condition,
                "random_runs": len(group),
                "diagnostic_runs_available": len(available),
                "diagnostic_runs_missing": len(group) - len(available),
                "diagnostic_file_runs_available": file_runs,
                "diagnostic_file_runs_missing": len(group) - file_runs,
                "tail100_ratio_at_or_below_0.5_count": len(ratio_at_or_below_half)
                if diagnostic_rows
                else None,
                "tail100_ratio_at_or_below_0.5_rate": len(ratio_at_or_below_half)
                / len(diagnostic_rows)
                if diagnostic_rows
                else None,
                "tail100_small_count_after_max": max(small_counts)
                if small_counts
                else None,
            }
            if available:
                decisions = sum(
                    number(r["tail100_local_decisions"]) or 0 for r in available
                )
                nominal = sum(
                    number(r["tail100_nominal_selected_matches"]) or 0
                    for r in available
                )
                aggregate.update(
                    {
                        "tail100_local_decisions": decisions,
                        "tail100_nominal_selected_matches": nominal,
                        "tail100_nominal_selected_match_rate": nominal / decisions
                        if decisions
                        else None,
                        "tail100_holding_count": sum(
                            number(r["tail100_holding_count"]) or 0 for r in available
                        ),
                        "tail100_radial_clip_count": sum(
                            number(r["tail100_radial_clip_count"]) or 0
                            for r in available
                        ),
                        "tail100_angular_clip_count": sum(
                            number(r["tail100_angular_clip_count"]) or 0
                            for r in available
                        ),
                    }
                )
            else:
                aggregate.update(
                    {
                        "tail100_local_decisions": None,
                        "tail100_nominal_selected_matches": None,
                        "tail100_nominal_selected_match_rate": None,
                        "tail100_holding_count": None,
                        "tail100_radial_clip_count": None,
                        "tail100_angular_clip_count": None,
                    }
                )
            output.append(aggregate)
    return output


def paired_bootstrap(
    values: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float | None, float | None]:
    if not len(values):
        return None, None
    samples = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(
        axis=1
    )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_method(
    group: list[dict[str, Any]], condition: str, method: str
) -> dict[str, Any]:
    """One common-condition method summary; all values retain the 100-trial denominator."""
    out: dict[str, Any] = {"condition": condition, "method": method, "runs": len(group)}
    for prefix, predicate in (
        ("stopped", "stopped"),
        ("final_below_1cm", "final_below_1cm"),
        ("joint_success_1cm", "joint_success_1cm"),
    ):
        out.update(rates(group, prefix, predicate))
    for metric in (
        "max_position_error_m",
        "rms_position_error_m",
        "measurement_slots",
        "transmitter_uses",
        "movement_m",
    ):
        for label, percentile in (("p05", 0.05), ("median", 0.5), ("p95", 0.95)):
            out[f"{metric}_{label}"] = q((r[metric] for r in group), percentile)
    first = [r["first_1cm_slot"] for r in group if r["first_1cm_slot"] is not None]
    sustained = [
        r["sustained_1cm_from_slot"]
        for r in group
        if r["sustained_1cm_from_slot"] is not None
    ]
    out.update(
        {
            "first_1cm_count": len(first),
            "first_1cm_median_among_hits": q(first, 0.5),
            "sustained_1cm_count": len(sustained),
            "sustained_1cm_median_among_hits": q(sustained, 0.5),
        }
    )
    return out


def main_reference_rows() -> list[dict[str, Any]]:
    path = ROOT / "outputs/q1_3/robustness/trials.csv"
    if not path.exists():
        return []
    return normalized_rows([{"gain": "0.5", **r} for r in read_csv(path)])


def main_comparison(
    rows: list[dict[str, Any]], seed: int = 20260906, n_boot: int = 10000
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare main and both fixed double-configuration gains on shared trials only."""
    old = main_reference_rows()
    if not old:
        return [], []
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for condition in MAIN_COMMON:
        methods = {
            "main": [
                r
                for r in old
                if r["condition"] == condition
                and r["trial"] is not None
                and r["trial"] >= 0
            ],
            "two_configuration_gain_0.5": [
                r
                for r in rows
                if r["gain"] == 0.5 and r["condition"] == condition and r["trial"] >= 0
            ],
            "two_configuration_gain_1": [
                r
                for r in rows
                if r["gain"] == 1 and r["condition"] == condition and r["trial"] >= 0
            ],
        }
        summary.extend(
            summarize_method(group, condition, method)
            for method, group in methods.items()
            if group
        )
        baseline = {r["trial"]: r for r in methods["main"]}
        for method in ("two_configuration_gain_0.5", "two_configuration_gain_1"):
            current = {r["trial"]: r for r in methods[method]}
            shared = sorted(set(current) & set(baseline))
            for metric in (
                "stopped",
                "final_below_1cm",
                "joint_success_1cm",
                "max_position_error_m",
                "rms_position_error_m",
                "measurement_slots",
                "transmitter_uses",
                "movement_m",
            ):
                values = np.asarray(
                    [
                        float(current[t][metric]) - float(baseline[t][metric])
                        for t in shared
                    ],
                    dtype=float,
                )
                low, high = paired_bootstrap(values, rng, n_boot)
                paired.append(
                    {
                        "condition": condition,
                        "contrast": f"{method}_minus_main",
                        "metric": metric,
                        "metric_kind": "rate_difference"
                        if metric in {"stopped", "final_below_1cm", "joint_success_1cm"}
                        else "difference",
                        "paired_trials": len(values),
                        "conditional_pair_count": None,
                        "mean_difference": float(values.mean())
                        if len(values)
                        else None,
                        "median_difference": float(np.median(values))
                        if len(values)
                        else None,
                        "bootstrap95_low": low,
                        "bootstrap95_high": high,
                        "bootstrap_replicates": n_boot,
                        "bootstrap_seed": seed,
                        "main_source": "outputs/q1_3/robustness/trials.csv",
                    }
                )
            if condition == "exact":
                hit_trials = [
                    t
                    for t in shared
                    if current[t]["first_1cm_slot"] is not None
                    and baseline[t]["first_1cm_slot"] is not None
                ]
                values = np.asarray(
                    [
                        current[t]["first_1cm_slot"] - baseline[t]["first_1cm_slot"]
                        for t in hit_trials
                    ],
                    dtype=float,
                )
                low, high = paired_bootstrap(values, rng, n_boot)
                paired.append(
                    {
                        "condition": condition,
                        "contrast": f"{method}_minus_main",
                        "metric": "first_1cm_slot",
                        "metric_kind": "conditional_first_event_difference",
                        "paired_trials": len(values),
                        "conditional_pair_count": len(values),
                        "mean_difference": float(values.mean())
                        if len(values)
                        else None,
                        "median_difference": float(np.median(values))
                        if len(values)
                        else None,
                        "bootstrap95_low": low,
                        "bootstrap95_high": high,
                        "bootstrap_replicates": n_boot,
                        "bootstrap_seed": seed,
                        "main_source": "outputs/q1_3/robustness/trials.csv",
                    }
                )
    return summary, paired


def gaussian_precision_comparison(
    math_dir: Path, summary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    path = math_dir / "gaussian_precision_probability.csv"
    if not path.exists():
        return []
    output = []
    for item in read_csv(path):
        if integer(item.get("phase")) != 560 or item.get("model") not in {
            "pure_iid_bearing_gaussian",
            "fixed_receiver_tx_link_bias_gaussian",
        }:
            continue
        threshold = item.get("threshold")
        if threshold not in {"1cm", "1mm"}:
            continue
        if item["model"] == "pure_iid_bearing_gaussian":
            condition = f"bearing_{float(item['bearing_std_deg']):g}deg"
        else:
            condition = f"link_bias_{float(item['link_bias_std_deg']):g}deg"
        observed = next(
            (
                r
                for r in summary
                if r["gain"] == float(item["gain"]) and r["condition"] == condition
            ),
            None,
        )
        if observed is None:
            continue
        key = f"final_below_{threshold}"
        output.append(
            {
                "model": item["model"],
                "gain": float(item["gain"]),
                "condition": condition,
                "phase": 560,
                "threshold": threshold,
                "gaussian_numerical_integration_probability": float(
                    item["gaussian_numerical_integration_probability"]
                ),
                "gaussian_numerical_integration_wilson95_low": number(
                    item.get("gaussian_numerical_integration_wilson95_low")
                ),
                "gaussian_numerical_integration_wilson95_high": number(
                    item.get("gaussian_numerical_integration_wilson95_high")
                ),
                "empirical_terminal_count": observed[f"{key}_count"],
                "empirical_runs": observed["runs"],
                "empirical_terminal_rate": observed[f"{key}_rate"],
                "empirical_trial_wilson95_low": observed[f"{key}_wilson95_low"],
                "empirical_trial_wilson95_high": observed[f"{key}_wilson95_high"],
                "conservative_failure_union_upper_bound": number(
                    item.get("conservative_failure_union_upper_bound")
                ),
                "scope": item.get("scope", ""),
            }
        )
    return output


def source_ledger(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve runner provenance without dereferencing archived trial artifacts."""
    groups: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        groups[
            (
                str(row.get("source_kind", "runner_output")),
                str(row.get("source_dir", "")),
                str(row.get("cache_sha256", "")),
            )
        ] += 1
    return [
        {
            "source_kind": key[0],
            "source_dir": key[1],
            "cache_sha256": key[2],
            "record_count": count,
        }
        for key, count in sorted(groups.items())
    ]


def export(fig: plt.Figure, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for extension in EXPORT_FORMATS:
        path = root / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        EXPORTED_FIGURES.append(path)
    plt.close(fig)


def plot_terminal(rows: list[dict[str, Any]], data: Path, figures: Path) -> None:
    points = []
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), sharey=True)
    for ax, gain in zip(axes, GAINS):
        for i, condition in enumerate(CONDITIONS):
            vals = np.array(
                [
                    r["max_position_error_m"]
                    for r in rows
                    if r["gain"] == gain
                    and r["condition"] == condition
                    and r["trial"] >= 0
                ]
            )
            if not len(vals):
                continue
            jitter = np.random.default_rng(812 + i).uniform(-0.16, 0.16, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=9,
                alpha=0.42,
                color=BLUE,
                linewidths=0,
            )
            ax.errorbar(
                i,
                np.median(vals),
                yerr=[
                    [np.median(vals) - np.quantile(vals, 0.05)],
                    [np.quantile(vals, 0.95) - np.median(vals)],
                ],
                fmt="D",
                color="#202124",
                capsize=3,
                ms=4,
            )
            points += [
                {
                    "gain": gain,
                    "condition": condition,
                    "trial": r["trial"],
                    "terminal_max_error_m": r["max_position_error_m"],
                }
                for r in rows
                if r["gain"] == gain and r["condition"] == condition and r["trial"] >= 0
            ]
        ax.set_title(f"({'a' if gain == 0.5 else 'b'}) 增益 {gain:g}")
        ax.set_yscale("log")
        ax.set_xticks(
            range(len(CONDITIONS)),
            [FIGURE_CONDITION[c] for c in CONDITIONS],
            fontsize=8.5,
        )
        ax.grid(axis="y")
        ax.axhline(0.01, color="#444", ls=":", lw=0.7)
        ax.text(
            1,
            0.01,
            "1 cm",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8.5,
        )
        ax.set_ylabel("终点最大位置误差 / m")
    fig.suptitle("随机初态下的终点误差")
    figure_note(
        fig,
        "每组 100 次；散点为单次试验，黑色菱形及误差棒为中位数与 P05–P95。\n精确条件的极小误差属于数值残差；点线为厘米阈值。",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.96), h_pad=1.3)
    write_csv(data / "terminal_error_gain_data.csv", points)
    export(fig, figures, "terminal_error_gain")


def plot_rates(summary: list[dict[str, Any]], data: Path, figures: Path) -> None:
    plot_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), sharey=True)
    for ax, gain in zip(axes, GAINS):
        group = [r for r in summary if r["gain"] == gain]
        x = np.arange(len(CONDITIONS))
        width = 0.25
        for j, metric in enumerate(("stopped", "final_below_1cm", "joint_success_1cm")):
            selected = [
                next((r for r in group if r["condition"] == c), None)
                for c in CONDITIONS
            ]
            vals = [r[f"{metric}_rate"] if r else np.nan for r in selected]
            lows = [r[f"{metric}_wilson95_low"] if r else np.nan for r in selected]
            highs = [r[f"{metric}_wilson95_high"] if r else np.nan for r in selected]
            lower_error = np.maximum(0, np.subtract(vals, lows))
            upper_error = np.maximum(0, np.subtract(highs, vals))
            ax.bar(
                x + (j - 1) * width,
                vals,
                width,
                yerr=[lower_error, upper_error],
                capsize=2,
                error_kw={"lw": 0.75},
                color=(GREY, BLUE, ORANGE)[j],
                hatch=HATCHES[j],
                edgecolor="#333333",
                linewidth=0.5,
                label={
                    "stopped": "协议停止",
                    "final_below_1cm": "终点 <1 cm",
                    "joint_success_1cm": "达标且停止",
                }[metric],
            )
            plot_rows += [
                {
                    "gain": gain,
                    "condition": c,
                    "metric": metric,
                    "rate": v,
                    "wilson95_low": lo,
                    "wilson95_high": hi,
                }
                for c, v, lo, hi in zip(CONDITIONS, vals, lows, highs)
            ]
        ax.set(
            title=f"({'a' if gain == 0.5 else 'b'}) 增益 {gain:g}",
            xticks=x,
            xticklabels=[FIGURE_CONDITION[c] for c in CONDITIONS],
            ylim=(0, 1.08),
            ylabel="试验比例",
        )
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
        ax.grid(axis="y")
    fig.suptitle("协议停止、终点精度与联合成功")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3)
    figure_note(
        fig,
        "每组 100 次；误差棒为 Wilson 95% 置信区间。\n“达标且停止”要求终点最大位置误差 <1 cm，并满足协议停止条件。",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.91), h_pad=1.3)
    write_csv(data / "precision_stop_rate_data.csv", plot_rows)
    export(fig, figures, "precision_stop_rate")


def plot_exact_slots(rows: list[dict[str, Any]], data: Path, figures: Path) -> None:
    left = {
        r["trial"]: r
        for r in rows
        if r["gain"] == 0.5 and r["condition"] == "exact" and r["trial"] >= 0
    }
    right = {
        r["trial"]: r
        for r in rows
        if r["gain"] == 1 and r["condition"] == "exact" and r["trial"] >= 0
    }
    pairs = [
        {
            "trial": t,
            "gain_0.5_slots": left[t]["measurement_slots"],
            "gain_1_slots": right[t]["measurement_slots"],
            "difference_gain_1_minus_0.5": right[t]["measurement_slots"]
            - left[t]["measurement_slots"],
        }
        for t in sorted(set(left) & set(right))
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for r in pairs:
        ax.plot(
            [0.5, 1],
            [r["gain_0.5_slots"], r["gain_1_slots"]],
            color="#AEB8C0",
            lw=0.7,
            alpha=0.5,
        )
    if pairs:
        ax.scatter(
            np.full(len(pairs), 0.5),
            [r["gain_0.5_slots"] for r in pairs],
            color=BLUE,
            s=12,
            label="增益 0.5",
        )
        ax.scatter(
            np.full(len(pairs), 1),
            [r["gain_1_slots"] for r in pairs],
            color=ORANGE,
            marker="s",
            s=12,
            label="增益 1",
        )
    ax.set(
        xlim=(0.35, 1.15),
        xticks=[0.5, 1],
        xticklabels=["增益 0.5", "增益 1"],
        ylabel="协议停止所用测角时隙",
        title="精确条件下的配对停止时隙",
    )
    ax.grid(axis="y")
    figure_note(fig, "每条线连接同一随机初态的两次试验，共 100 对。")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    write_csv(data / "exact_paired_slots_data.csv", pairs)
    export(fig, figures, "exact_paired_slots")


def plot_main_common_three_way(
    rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    data: Path,
    figures: Path,
) -> None:
    """One fair, visually inspectable view of all shared main conditions."""
    methods = {
        "main": main_rows,
        "two_configuration_gain_0.5": [r for r in rows if r["gain"] == 0.5],
        "two_configuration_gain_1": [r for r in rows if r["gain"] == 1],
    }
    colours = {
        "main": GREY,
        "two_configuration_gain_0.5": BLUE,
        "two_configuration_gain_1": ORANGE,
    }
    plot_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.9))
    x = np.arange(len(MAIN_COMMON))
    width = 0.24
    for j, (method, method_rows) in enumerate(methods.items()):
        selected = []
        for condition in MAIN_COMMON:
            group = [
                r
                for r in method_rows
                if r["condition"] == condition
                and r["trial"] is not None
                and r["trial"] >= 0
            ]
            selected.append(group)
            if group:
                rate = np.mean([r["final_below_1cm"] for r in group])
                lo, hi = wilson(sum(r["final_below_1cm"] for r in group), len(group))
                plot_rows.append(
                    {
                        "panel": "terminal_below_1cm",
                        "condition": condition,
                        "method": method,
                        "rate": rate,
                        "wilson95_low": lo,
                        "wilson95_high": hi,
                    }
                )
        rates_ = [
            np.mean([r["final_below_1cm"] for r in group]) if group else np.nan
            for group in selected
        ]
        lows = [
            wilson(sum(r["final_below_1cm"] for r in group), len(group))[0]
            if group
            else np.nan
            for group in selected
        ]
        highs = [
            wilson(sum(r["final_below_1cm"] for r in group), len(group))[1]
            if group
            else np.nan
            for group in selected
        ]
        axes[0, 0].bar(
            x + (j - 1) * width,
            rates_,
            width,
            color=colours[method],
            hatch=HATCHES[j],
            edgecolor="#333333",
            linewidth=0.5,
            label=FIGURE_METHOD_DISPLAY[method],
            yerr=[
                np.maximum(0, np.subtract(rates_, lows)),
                np.maximum(0, np.subtract(highs, rates_)),
            ],
            capsize=2,
            error_kw={"lw": 0.75},
        )
        for i, group in enumerate(selected):
            if not group:
                continue
            current_condition = MAIN_COMMON[i]
            emax = np.asarray([r["max_position_error_m"] for r in group])
            tx = np.asarray([r["transmitter_uses"] for r in group])
            offset = i + (j - 1) * 0.19
            axes[0, 1].errorbar(
                offset,
                np.median(emax),
                yerr=[
                    [np.median(emax) - np.quantile(emax, 0.05)],
                    [np.quantile(emax, 0.95) - np.median(emax)],
                ],
                fmt=METHOD_MARKERS[j],
                color=colours[method],
                capsize=3,
                ms=4,
            )
            axes[1, 0].errorbar(
                offset,
                np.median(tx),
                yerr=[
                    [np.median(tx) - np.quantile(tx, 0.05)],
                    [np.quantile(tx, 0.95) - np.median(tx)],
                ],
                fmt=METHOD_MARKERS[j],
                color=colours[method],
                capsize=3,
                ms=4,
            )
            plot_rows.append(
                {
                    "panel": "emax_p05_median_p95",
                    "condition": current_condition,
                    "method": method,
                    "p05": float(np.quantile(emax, 0.05)),
                    "median": float(np.median(emax)),
                    "p95": float(np.quantile(emax, 0.95)),
                }
            )
            plot_rows.append(
                {
                    "panel": "transmitter_uses_p05_median_p95",
                    "condition": current_condition,
                    "method": method,
                    "p05": float(np.quantile(tx, 0.05)),
                    "median": float(np.median(tx)),
                    "p95": float(np.quantile(tx, 0.95)),
                }
            )
    axes[0, 0].set(
        title="(a) 终点厘米达标率",
        ylabel="试验比例",
        xticks=x,
        xticklabels=[FIGURE_CONDITION[c] for c in MAIN_COMMON],
        ylim=(0, 1.08),
    )
    axes[0, 0].tick_params(axis="x", rotation=0, labelsize=8.5)
    axes[0, 0].yaxis.set_major_formatter(ticker.PercentFormatter(1))
    axes[0, 0].grid(axis="y", alpha=0.4)
    axes[0, 1].set(
        title="(b) 终点最大位置误差",
        ylabel="位置误差 / m",
        xticks=x,
        xticklabels=[FIGURE_CONDITION[c] for c in MAIN_COMMON],
        yscale="log",
    )
    axes[0, 1].tick_params(axis="x", rotation=0, labelsize=8.5)
    axes[0, 1].grid(axis="y", alpha=0.4)
    axes[0, 1].axhline(0.01, color="#444", ls=":", lw=1)
    axes[1, 0].set(
        title="(c) 累计发射机次",
        ylabel="发射机次",
        xticks=x,
        xticklabels=[FIGURE_CONDITION[c] for c in MAIN_COMMON],
    )
    axes[1, 0].tick_params(axis="x", rotation=0, labelsize=8.5)
    axes[1, 0].grid(axis="y", alpha=0.4)
    exact_values = []
    for j, (method, method_rows) in enumerate(methods.items(), start=1):
        group = [
            r
            for r in method_rows
            if r["condition"] == "exact" and r["trial"] is not None and r["trial"] >= 0
        ]
        first = [r["first_1cm_slot"] for r in group if r["first_1cm_slot"] is not None]
        exact_values.append(first)
        stopped = sum(r["stopped"] for r in group)
        plot_rows.extend(
            {
                "panel": "exact_first_1cm_slot",
                "condition": "exact",
                "method": method,
                "trial": r["trial"],
                "first_1cm_slot": r["first_1cm_slot"],
                "stopped": r["stopped"],
            }
            for r in group
        )
        axes[1, 1].text(
            j,
            (max(first) if first else 0) + 3,
            f"停止 {stopped}/{len(group)}",
            ha="center",
            fontsize=8.5,
            color="#333333",
        )
    axes[1, 1].boxplot(
        exact_values,
        tick_labels=["main", "双配置\n增益 0.5", "双配置\n增益 1"],
        showfliers=True,
        patch_artist=True,
        boxprops={"facecolor": "#dce6f2", "edgecolor": "#456"},
        medianprops={"color": "#111"},
        flierprops={"marker": ".", "markersize": 3},
    )
    axes[1, 1].set(title="(d) 精确条件：首次厘米达标", ylabel="首次达标时隙")
    if any(exact_values):
        axes[1, 1].set_ylim(0, max(max(v) for v in exact_values if v) + 14)
    axes[1, 1].tick_params(axis="x", rotation=0, labelsize=8.5)
    axes[1, 1].grid(axis="y", alpha=0.4)
    fig.suptitle("五个共同条件下的 main 与双配置比较")
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=colours[m],
                marker=METHOD_MARKERS[j],
                ls="none",
                label=FIGURE_METHOD_DISPLAY[m],
            )
            for j, m in enumerate(methods)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
    )
    figure_note(
        fig,
        "每组 100 次；(a) Wilson 95% 区间；(b)(c) 中位数与 P05–P95。\n(d) 箱体为四分位区间、须为 1.5 IQR 内最远值，保留离群点；文字另列协议停止数。",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.92), h_pad=1.5, w_pad=1.5)
    write_csv(data / "main_common_three_way_data.csv", plot_rows)
    export(fig, figures, "main_common_three_way")


def slot_metrics_path(row: dict[str, Any]) -> Path | None:
    raw = str(row.get("summary_path", ""))
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.parent / "slot_metrics.csv"


def phase_rms_samples(
    rows: list[dict[str, Any]], gain: float, condition: str, phase: int
) -> list[float]:
    values = []
    for row in rows:
        if (
            row["gain"] != gain
            or row["condition"] != condition
            or row["trial"] is None
            or row["trial"] < 0
        ):
            continue
        path = slot_metrics_path(row)
        if path is None or not path.exists():
            continue
        matching = [
            item for item in read_csv(path) if integer(item.get("slot")) == phase
        ]
        if matching:
            values.append(float(matching[-1]["rms_position_error_m"]))
    return values


def theory_condition(item: dict[str, str], source: str) -> str | None:
    if source == "periodic_covariance":
        return f"bearing_{float(item['bearing_std_deg']):g}deg"
    if source == "combined_stationary_moments":
        return f"combined_{float(item['bearing_std_deg']):g}deg_{round(float(item['actuation_relative_std']) * 100):g}pct"
    if source == "fixed_link_bias_response":
        return f"link_bias_{float(item['link_bias_std_deg']):g}deg"
    return None


def plot_theory(
    math_dir: Path,
    rows: list[dict[str, Any]],
    data: Path,
    figures: Path,
    seed: int = 20260906,
    n_boot: int = 10000,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    values = []
    for source in (
        "periodic_covariance",
        "combined_stationary_moments",
        "fixed_link_bias_response",
    ):
        path = math_dir / f"{source}.csv"
        if not path.exists():
            continue
        for item in read_csv(path):
            phase = integer(item.get("phase"))
            condition = theory_condition(item, source)
            if phase not in (559, 560) or condition not in CONDITIONS:
                continue
            gain = float(item["gain"])
            samples = phase_rms_samples(rows, gain, condition, phase)
            if samples:
                empirical = float(np.sqrt(np.mean(np.square(samples))))
                bootstrap = np.sqrt(
                    np.mean(
                        np.square(
                            np.asarray(samples)[
                                rng.integers(
                                    0, len(samples), size=(n_boot, len(samples))
                                )
                            ]
                        ),
                        axis=1,
                    )
                )
                theory_value = float(item["root_expected_rms_squared_m"])
                lo, hi = (
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                )
                values.append(
                    {
                        "theory_source": source,
                        "gain": gain,
                        "condition": condition,
                        "phase": phase,
                        "theory_root_expected_rms_squared_m": theory_value,
                        "empirical_root_mean_rms_squared_m": empirical,
                        "empirical_bootstrap95_low": lo,
                        "empirical_bootstrap95_high": hi,
                        "theory_over_empirical": theory_value / empirical
                        if empirical
                        else None,
                        "theory_within_empirical_bootstrap95": lo <= theory_value <= hi,
                        "empirical_samples": len(samples),
                        "bootstrap_replicates": n_boot,
                        "bootstrap_seed": seed,
                        "statistic": item.get("statistic", ""),
                    }
                )
    if not values:
        return values
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    colours = {
        "periodic_covariance": BLUE,
        "combined_stationary_moments": ORANGE,
        "fixed_link_bias_response": TEAL,
    }
    markers = {
        "periodic_covariance": "o",
        "combined_stationary_moments": "s",
        "fixed_link_bias_response": "^",
    }
    source_labels = {
        "periodic_covariance": "测角白噪声",
        "combined_stationary_moments": "混合扰动",
        "fixed_link_bias_response": "固定链路偏置",
    }
    for source, colour in colours.items():
        group = [r for r in values if r["theory_source"] == source]
        for gain in GAINS:
            subset = [r for r in group if r["gain"] == gain]
            if subset:
                ax.errorbar(
                    [r["theory_root_expected_rms_squared_m"] for r in subset],
                    [r["empirical_root_mean_rms_squared_m"] for r in subset],
                    yerr=[
                        [
                            r["empirical_root_mean_rms_squared_m"]
                            - r["empirical_bootstrap95_low"]
                            for r in subset
                        ],
                        [
                            r["empirical_bootstrap95_high"]
                            - r["empirical_root_mean_rms_squared_m"]
                            for r in subset
                        ],
                    ],
                    fmt=markers[source],
                    mfc=colour if gain == 0.5 else "white",
                    mew=1,
                    color=colour,
                    capsize=2,
                    ms=5,
                    label=f"{source_labels[source]}，增益 {gain:g}",
                )
    lo_lim = (
        min(
            min(r["theory_root_expected_rms_squared_m"], r["empirical_bootstrap95_low"])
            for r in values
        )
        / 1.4
    )
    hi_lim = (
        max(
            max(
                r["theory_root_expected_rms_squared_m"], r["empirical_bootstrap95_high"]
            )
            for r in values
        )
        * 1.4
    )
    ax.plot(
        [lo_lim, hi_lim],
        [lo_lim, hi_lim],
        ":",
        color="#444",
        label="理论值 = 实验值",
        lw=0.7,
    )
    ax.set(
        xlabel=r"理论预测 $\sqrt{\mathrm{E}[\mathrm{RMS}^2]}$ / m",
        ylabel=r"实验统计 $\sqrt{\overline{\mathrm{RMS}^2}}$ / m",
        title="相位匹配的理论预测与实验统计",
        xscale="log",
        yscale="log",
        xlim=(lo_lim, hi_lim),
        ylim=(lo_lim, hi_lim),
    )
    # MathText 的普通文本段不逐字回退，混排轴名需直接指定中文字体。
    chinese_font = STYLE_METADATA["chinese_font"]
    ax.xaxis.label.set_fontfamily(chinese_font)
    ax.yaxis.label.set_fontfamily(chinese_font)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid()
    figure_note(
        fig,
        "共 24 组，每组 100 次；误差棒为逐点 bootstrap 95% 区间。\n匹配时隙 559、560；实心为增益 0.5，空心为增益 1。",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    write_csv(data / "theory_empirical_data.csv", values)
    export(fig, figures, "theory_empirical")
    return values


def markdown_table(rows: list[dict[str, Any]], gain: float) -> str:
    selected = [r for r in rows if r["gain"] == gain]
    lines = [
        "| 条件 | 停止 / 100（Wilson 95%） | 终点 <1 cm / 100 | 终点 <1 mm / 100 | 联合 <1 cm / 100 | 终点最大误差 P05 / 中位 / P95（m） | 时隙中位数 | TX 中位数 | 位移中位数（m） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in selected:
        stop = f"{r['stopped_count']} [{100 * r['stopped_wilson95_low']:.1f}%, {100 * r['stopped_wilson95_high']:.1f}%]"
        err = f"{r['max_position_error_m_p05']:.3g} / {r['max_position_error_m_median']:.3g} / {r['max_position_error_m_p95']:.3g}"
        lines.append(
            f"| {DISPLAY[r['condition']]} | {stop} | {r['final_below_1cm_count']} | {r['final_below_1mm_count']} | {r['joint_success_1cm_count']} | {err} | {r['measurement_slots_median']:.0f} | {r['transmitter_uses_median']:.0f} | {r['movement_m_median']:.3g} |"
        )
    return "\n".join(lines)


def conclusion_lines(summary: list[dict[str, Any]]) -> str:
    by_key = {(r["gain"], r["condition"]): r for r in summary}
    if any((g, c) not in by_key for g in GAINS for c in CONDITIONS):
        return "当前为不完整批次，仅作流程检查，完整结论待全部条件完成。"
    exact = [by_key[g, "exact"] for g in GAINS]
    bearing = [by_key[g, "bearing_0.001deg"] for g in GAINS]
    mixed = [by_key[g, "combined_0.001deg_1pct"] for g in GAINS]
    bias = [by_key[g, "link_bias_0.001deg"] for g in GAINS]
    return "\n".join(
        [
            f"- **精确测角或仅有 1% 相对执行误差时，增益 1 的停止开销更低。** 精确条件下两个增益均为 100/100 达标并停止；首次厘米时隙中位数从 {exact[0]['first_1cm_median_among_hits']:.0f} 降为 {exact[1]['first_1cm_median_among_hits']:.0f}，完整停止从 {exact[0]['measurement_slots_median']:.0f} 降为 {exact[1]['measurement_slots_median']:.0f}。增益 1 的端点位移中位数更高，具体成本在下表分别列出。",
            f"- **存在测角白噪声时，增益 0.5 的精度更好。** 0.001° 下终点厘米达标为 {bearing[0]['final_below_1cm_count']}/100，对比增益 1 的 {bearing[1]['final_below_1cm_count']}/100；再叠加 1% 执行误差后分别为 {mixed[0]['final_below_1cm_count']}/100 和 {mixed[1]['final_below_1cm_count']}/100。0.01°、0.1° 下两个增益的终点厘米达标均为 0/100，增益 0.5 只是误差水平较低，尚不能满足该精度要求。",
            f"- **固定链路偏置仍是明显限制。** 0.001° 偏置下两个增益的终点厘米达标分别为 {bias[0]['final_below_1cm_count']}/100 和 {bias[1]['final_below_1cm_count']}/100。其跨时隙相关性需要单独建模，不能用白噪声试验代替。",
            "- **所有含测角白噪声或固定偏置的条件均为 0/100 协议停止。** 因而这些条件下还没有形成能够自行确认完成的方案；精度改善与完整停止保证应分别陈述。",
        ]
    )


def figure_manifest(figures: Path, math_dir: Path) -> dict[str, Any]:
    artifacts = {}
    for path in sorted(set(EXPORTED_FIGURES)) + sorted(
        (figures / "data").glob("*.csv")
    ):
        artifacts[display_path(path)] = sha256(path)
    math_inputs = {}
    for name in (
        "periodic_covariance.csv",
        "combined_stationary_moments.csv",
        "fixed_link_bias_response.csv",
        "multiplicative_stability.csv",
        "gaussian_precision_probability.csv",
        "gaussian_precision_probability_sources.json",
        "random_initial_moments.csv",
        "random_initial_moments_sources.json",
        "summary.json",
    ):
        path = math_dir / name
        if path.exists():
            math_inputs[display_path(path)] = sha256(path)
    return {
        "plot_script": display_path(Path(__file__)),
        "plot_script_sha256": sha256(Path(__file__)),
        "math_inputs_sha256": math_inputs,
        "artifacts_sha256": artifacts,
        "style_assets_sha256": {
            STYLE_METADATA["style_script_path"]: STYLE_METADATA["style_script_sha256"],
            **STYLE_METADATA["fonts"],
        },
        "export_formats": list(EXPORT_FORMATS),
        "style": STYLE_METADATA,
    }


def main_three_way_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 共同条件 | 方法 | 协议停止 / 100 | 终点 <1 cm / 100（Wilson 95%） | 联合 <1 cm 且停止 / 100 | Emax P05 / 中位 / P95（m） | 实际 TX P05 / 中位 / P95 | Exact 首次 <1 cm 中位（命中数） |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        terminal_rate = f"{item['final_below_1cm_count']} [{100 * item['final_below_1cm_wilson95_low']:.1f}%, {100 * item['final_below_1cm_wilson95_high']:.1f}%]"
        emax = f"{item['max_position_error_m_p05']:.3g} / {item['max_position_error_m_median']:.3g} / {item['max_position_error_m_p95']:.3g}"
        tx = f"{item['transmitter_uses_p05']:.0f} / {item['transmitter_uses_median']:.0f} / {item['transmitter_uses_p95']:.0f}"
        first = (
            "—"
            if item["condition"] != "exact"
            or item["first_1cm_median_among_hits"] is None
            else f"{item['first_1cm_median_among_hits']:.0f} ({item['first_1cm_count']}/100)"
        )
        lines.append(
            f"| {DISPLAY[item['condition']]} | {METHOD_DISPLAY[item['method']]} | {item['stopped_count']} | {terminal_rate} | {item['joint_success_1cm_count']} | {emax} | {tx} | {first} |"
        )
    return "\n".join(lines)


def main_quantitative_findings(
    summary: list[dict[str, Any]], paired: list[dict[str, Any]]
) -> str:
    """Short numeric findings for the manuscript; tables remain the complete record."""
    by_key = {(r["condition"], r["method"]): r for r in summary}

    def item(condition: str, method: str) -> dict[str, Any] | None:
        return by_key.get((condition, method))

    def emax_difference(condition: str) -> dict[str, Any] | None:
        return next(
            (
                r
                for r in paired
                if r["condition"] == condition
                and r["contrast"] == "two_configuration_gain_0.5_minus_main"
                and r["metric"] == "max_position_error_m"
            ),
            None,
        )

    lines: list[str] = []
    exact = [item("exact", method) for method in METHOD_DISPLAY]
    if all(exact):
        lines.append(
            "- exact：main、双配置 gain $0.5$、双配置 gain $1$ 均为 $100/100$ 协议停止；"
            f"首次进入 $1$ cm 的中位时隙依次为 {exact[0]['first_1cm_median_among_hits']:.0f}、{exact[1]['first_1cm_median_among_hits']:.0f}、{exact[2]['first_1cm_median_among_hits']:.0f}，"
            f"实际 TX 中位数为 {exact[0]['transmitter_uses_median']:.0f}、{exact[1]['transmitter_uses_median']:.0f}、{exact[2]['transmitter_uses_median']:.0f}。"
        )
    for condition in ("bearing_0.001deg", "bearing_0.01deg", "bearing_0.1deg"):
        main = item(condition, "main")
        gain_half = item(condition, "two_configuration_gain_0.5")
        gain_one = item(condition, "two_configuration_gain_1")
        contrast = emax_difference(condition)
        if not all((main, gain_half, gain_one, contrast)):
            continue
        ci = f"[{contrast['bootstrap95_low']:.4g}, {contrast['bootstrap95_high']:.4g}]"
        lines.append(
            f"- {DISPLAY[condition]}：终点 $<1$ cm 的 trial 数为 main/{gain_half['final_below_1cm_count'] if False else main['final_below_1cm_count']}、"
            f"双配置 $0.5$/{gain_half['final_below_1cm_count']}、双配置 $1$/{gain_one['final_below_1cm_count']}（均未协议停止）；"
            f"Emax 中位数为 {main['max_position_error_m_median']:.4g}、{gain_half['max_position_error_m_median']:.4g}、{gain_one['max_position_error_m_median']:.4g} m。"
            f"双配置 $0.5$ 相对 main 的逐 trial Emax 均值差为 {contrast['mean_difference']:.4g} m，bootstrap 95% 区间 {ci}。"
        )
    return "\n".join(lines) or "共同条件的定量三方结果待完整批次生成。"


def report(
    doc: Path,
    summary: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    theory: list[dict[str, Any]],
    gaussian_precision: list[dict[str, Any]],
    tail100: list[dict[str, Any]],
    main_three_way: list[dict[str, Any]],
    main_pairs: list[dict[str, Any]],
    provenance: dict[str, Any],
    injection_audit: dict[str, Any] | None,
) -> None:
    table1_lines = [
        "| 增益 | 条件 | 停止 | 终点 <1 cm | 终点 <1 mm | 联合 <1 cm | 最大误差 / m | 时隙 | TX | 位移 / m |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table_rows:
        table1_lines.append(
            f"| {r['gain']:g} | {DISPLAY[r['condition']]} | {str(r['stopped'])} | {str(r['final_below_1cm'])} | {str(r['final_below_1mm'])} | {str(r['joint_success_1cm'])} | {r['max_position_error_m']:.4g} | {r['measurement_slots']} | {r['transmitter_uses']} | {r['movement_m']:.4g} |"
        )
    theory_text = (
        "理论—经验对应图未生成：理论文件尚未交付。"
        if not theory
        else f"理论—经验对应使用相位 559 与 560 的 `{len(theory)}` 个同口径比较，统计量均为 $\\sqrt{{\\mathbb E[\\mathrm{{RMS}}^2]}}$。"
    )
    theory_figure = (
        ""
        if not theory
        else "![理论与经验的相位对应](../../figures/q1_3/two_configuration_robustness/theory_empirical.png)"
    )
    table1_markdown = "\n".join(table1_lines)
    diag_available = sum(r["diagnostic_runs_available"] for r in tail100)
    diag_total = sum(r["random_runs"] for r in tail100)
    injection_text = (
        "未找到独立的非线性注入审查文件。"
        if injection_audit is None
        else f"独立注入审查含 {injection_audit['nonlinear_cycle_probes']} 个非线性周期探针与 {injection_audit['second_moment_operator_checks']} 个二阶矩算子检查；{injection_audit.get('central_difference_checks', 32)} 个正负中心差分检查所报的最大相对导数误差为 {injection_audit['maximum_relative_derivative_error']:.3g}。这核对了局部注入与导数实现，不能把它解释成非线性随机实验的全局精度保证。"
    )
    gaussian_text = (
        "局部 Gaussian 精度概率表尚未交付。"
        if not gaussian_precision
        else "纯 iid bearing 与固定链路偏置的局部 Gaussian 精度概率，及其与终点经验精度率的并列对照，见 [gaussian_precision_comparison.csv](../../outputs/q1_3/two_configuration_robustness_report/gaussian_precision_comparison.csv)。表中 `gaussian_numerical_integration_wilson95_*` 是 $N=200000$ 数值积分的 Monte Carlo 误差区间，`empirical_trial_wilson95_*` 才是 100 条真实随机 trial 的 Wilson 区间；两者分列，且该近似没有套用于 combined 条件。"
    )
    random_initial_moments_path = MATH_DEFAULT / "random_initial_moments.csv"
    random_initial_moments_text = (
        "随机初态矩传播结果尚未交付。"
        if not random_initial_moments_path.exists()
        else "随机初态的局部二阶矩传播另见 [random_initial_moments.csv](../../outputs/q1_3/two_configuration_robustness_math/random_initial_moments.csv) 与 [随机初态矩分析](双配置轮换随机初态矩分析.md)：其初值由合同指定的径向、角向均匀扰动二阶矩构成，并用 64 点积分核对。该结果仅覆盖固定活动分支、未限幅、无保持和无切换的局部传播，早期曲线不用于预言真实 trial 的首次达标或停止时刻。"
    )
    conclusions = conclusion_lines(summary)
    main_three_way_text = (
        "main 的共同条件记录尚未找到。"
        if not main_three_way
        else main_three_way_markdown(main_three_way)
    )
    main_quantitative_text = main_quantitative_findings(main_three_way, main_pairs)
    text = f"""# 双配置轮换：随机初态、含噪与鲁棒性验证

本报告汇总固定双配置调度 $(0,1,4,5)$ 与 $(0,1,7,8)$ 下的随机初态和含噪实验。增益仅为 $0.5$ 与 $1$；每个增益、每个条件均在 560 个测角时隙预算内运行，未依据本批结果重新选择调度或参数。随机比例的分母是每条件 100 条试验；表 1 独立列出，不混入随机统计。

方向噪声表中的 $\sigma$ 是每条射线方向的独立白噪声标准差，并非把六个夹角各自独立加上同一标准差。在固定符号分支附近，一个由两条射线构成的夹角标准差为 $\sqrt{{2}}\sigma$，共享射线的夹角具有相关性。执行误差是每个径向/角向命令轴独立的零均值相对 $1\%$ 误差；真值位置仅由仿真器保存并用于事后评价，不输入本机定位或控制。

## 结论与取舍

{conclusions}

上述取舍是对本批结果的解释，适用范围为固定调度、指定随机分布和预算。完整的增益差值及配对 bootstrap 区间在下文引用的数据表中。

exact 条件下接近 $10^{{-12}}$ m 的 gain $1$ 终点只是理想模型浮点数值残差，不对应实际设备可实现的定位能力；本文的实际精度判断仍以厘米和毫米阈值事件为主。

## 覆盖与来源

本次读取 {coverage["random_rows"]} 条随机记录与 {coverage["table1_rows"]} 条表 1 记录。`full_16x100_coverage={coverage["full_16x100_coverage"]}`。试验运行器的合同、源文件哈希、复用来源和本报告脚本哈希均写入 [provenance.json](../../outputs/q1_3/two_configuration_robustness_report/provenance.json)。

## 随机试验结果

### 增益 $0.5$

{markdown_table(summary, 0.5)}

### 增益 $1$

{markdown_table(summary, 1)}

停止、终点精度和联合成功分别报告。所有百分比的 Wilson 95% 区间在 [random_summary.csv](../../outputs/q1_3/two_configuration_robustness_report/random_summary.csv) 中；终点厘米和毫米精度、停止与联合成功均使用全体 100 条记录为分母。首次达到阈值的分母仅是实际首次进入阈值的记录；记录持续达标的分母是这些首次进入记录，绝不以 560 时隙填补缺失的首次时间。

含噪预算耗尽组的 560 个时隙和 2240 次 TX 是实际消耗的预算开销，停止时间在这些记录上为右删失。它们不能写成“收敛时间为 560”，两种增益都耗尽预算也不能说明速度相同。首次达标时隙的统计仅在实际命中的条件记录中给出中位数；成本统计仍包含协议停止前的保持确认开销。

## 表 1 独立运行

{table1_markdown}

表 1 只有每个增益、条件组合的一条固定初态运行，因而不用于随机成功率或置信区间。

## 配对比较、理论与图表

白测角噪声、乘性执行误差及固定偏置的完整推导见 [含噪数学模型](双配置轮换含噪数学模型.md)。

增益差异按试验编号配对，bootstrap 的重采样单位是 trial，结果见 [gain_paired_bootstrap.csv](../../outputs/q1_3/two_configuration_robustness_report/gain_paired_bootstrap.csv)。与 main 的公平比较严格限于原有五种共同条件；三个方法均以同编号 trial 对齐，新增组合噪声和链路偏置不纳入该比较。

### main 与双配置的共同条件三方比较

{main_quantitative_text}

{main_three_way_text}

表内“联合 <1 cm 且停止”同时要求终点厘米事件和协议完整停止；Emax 是终点九架机中的最大位置误差。TX 为运行实际累计发射机次，不把 560 时隙预算写成收敛时间。对应的逐条件、逐方法数据见 [main_common_conditions_three_way_summary.csv](../../outputs/q1_3/two_configuration_robustness_report/main_common_conditions_three_way_summary.csv)；`two_configuration_minus_main` 的停止率、终点厘米率、联合成功率、Emax、实际 TX 以及 exact 首次厘米事件的 trial 配对 bootstrap 区间见 [main_common_conditions_paired_bootstrap.csv](../../outputs/q1_3/two_configuration_robustness_report/main_common_conditions_paired_bootstrap.csv)。这些有限样本区间刻画本合同中的差异不确定性，不能证明总体或所有噪声分布下的优势。

{theory_text} 理论值是局部线性二阶近似的 $\mathrm{{RMS}}_{{lin}}$；经验值由真值几何坐标计算 RMS。对每个 gain—条件—相位点，经验 $\\sqrt{{\\mathrm{{mean}}(\\mathrm{{RMS}}^2)}}$ 的 95% bootstrap 区间按独立 trial 重采样；[theory_empirical_data.csv](../../figures/q1_3/two_configuration_robustness/data/theory_empirical_data.csv) 同时报出理论/经验比和理论值是否落在该区间，不能只由百分比接近判断模型准确。

{injection_text}

{gaussian_text}

{random_initial_moments_text}

{theory_figure}

运行器另提供终止点前实际记录的最后 100 个全局时隙诊断，汇总于 [tail100_controller_diagnostics.csv](../../outputs/q1_3/two_configuration_robustness_report/tail100_controller_diagnostics.csv)。其中 {diag_available}/{diag_total} 条随机记录具备汇总字段；缺失的冻结复用记录明确保留为空，未由终点状态反推分支选择或保持状态。表中还直接从新 `controller_diagnostics_last100.csv` 计数 $\mathrm{{ratio}}\leq0.5$ 的局部决策比例和 `small_count_after` 最大值，用来对照 21 次本机连续小误差的保持门槛。该窗口有限，不能把这些计数当作独立事件的 $p^{{21}}$ 计算，也不能据此称未停止轨迹永不停止；即使短时停止运行的最后 100 时隙也可能包含暂态。

![终点误差的增益比较](../../figures/q1_3/two_configuration_robustness/terminal_error_gain.png)

![停止、精度与联合成功](../../figures/q1_3/two_configuration_robustness/precision_stop_rate.png)

![exact 条件下的配对停止时隙](../../figures/q1_3/two_configuration_robustness/exact_paired_slots.png)

![main 与双配置的共同条件三方比较](../../figures/q1_3/two_configuration_robustness/main_common_three_way.png)

## 解释范围与局限

随机初态来自合同指定的有限 100 条抽样，因而给出的是该抽样机制和有限预算下的经验比例与不确定性，不能构成连续初态空间的全局证明。预算内未停止表示本次记录在 560 时隙内未满足协议停止条件；它不说明轨迹永不收敛。首次进入厘米或毫米阈值也不等于随后始终达标，因此报告另列记录持续达标。

理论比较采用理想点附近、分支保持、未限幅且活动控制的局部线性模型。保持、分支切换、限幅、大初态扰动，以及固定链路偏置在单次运行中跨时隙共享的性质，都由非线性实验承担检验，不能由白噪声协方差结论替代。固定偏置的总体协方差与一次运行中的固定误差响应需按理论输出的定义分别解释。

## 复现

```bash
conda run --no-capture-output -n agent python -m scripts.q1_3.report_two_configuration_robustness
```

默认导出中文 PNG 与配套 CSV；加 `--formats png pdf svg` 可输出矢量图。字体、线宽和图表说明见 [科研图中文样式说明](科研图中文样式说明.md)。历史矢量文件保留，本次清单仅记录本次生成的图片。

可用 `--allow-partial` 仅做运行中诊断；正式产物默认断言 16 个增益—条件组合各有 100 条随机 trial 和 1 条表 1 记录。
"""
    doc.write_text(text, encoding="utf-8")


def main() -> None:
    global EXPORT_FORMATS, STYLE_METADATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DEFAULT)
    parser.add_argument("--math-dir", type=Path, default=MATH_DEFAULT)
    parser.add_argument("--doc", type=Path, default=DOC_DEFAULT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png"],
        help="图片格式；默认仅 PNG，可选 PDF/SVG",
    )
    args = parser.parse_args()
    EXPORT_FORMATS = tuple(dict.fromkeys(["png", *args.formats]))
    EXPORTED_FIGURES.clear()
    STYLE_METADATA = configure_style()
    trials = args.input_dir / "trials.csv"
    if not trials.exists():
        raise FileNotFoundError(f"runner has not written {trials}")
    rows = normalized_rows(read_csv(trials))
    coverage = validate_coverage(rows, args.allow_partial)
    summary = summaries(rows)
    table_rows = table1(rows)
    boot = bootstrap_gain(rows)
    main_three_way, main_pairs = main_comparison(rows)
    main_rows = main_reference_rows()
    source_rows = source_ledger(rows)
    tail100 = tail100_diagnostics(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = args.figure_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "random_summary.csv", summary)
    write_csv(args.output_dir / "table1_separate.csv", table_rows)
    write_csv(args.output_dir / "gain_paired_bootstrap.csv", boot)
    write_csv(
        args.output_dir / "main_common_conditions_three_way_summary.csv", main_three_way
    )
    write_csv(
        args.output_dir / "main_common_conditions_paired_bootstrap.csv", main_pairs
    )
    write_csv(args.output_dir / "source_provenance.csv", source_rows)
    write_csv(args.output_dir / "tail100_controller_diagnostics.csv", tail100)
    gaussian_precision = gaussian_precision_comparison(args.math_dir, summary)
    write_csv(args.output_dir / "gaussian_precision_comparison.csv", gaussian_precision)
    plot_terminal(rows, data, args.figure_dir)
    plot_rates(summary, data, args.figure_dir)
    plot_exact_slots(rows, data, args.figure_dir)
    if main_rows:
        plot_main_common_three_way(rows, main_rows, data, args.figure_dir)
    theory = plot_theory(args.math_dir, rows, data, args.figure_dir)
    injection_audit = (
        json.loads(INJECTION_AUDIT.read_text()) if INJECTION_AUDIT.exists() else None
    )
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_trials": display_path(trials),
        "input_trials_sha256": sha256(trials),
        "report_script_sha256": sha256(Path(__file__)),
        "coverage": coverage,
        "source_ledger": "source_provenance.csv",
        "math_dir": display_path(args.math_dir),
        "runner_contract_sha256": sha256(args.input_dir / "contract.json")
        if (args.input_dir / "contract.json").exists()
        else None,
        "runner_source_hashes_sha256": sha256(args.input_dir / "source_hashes.json")
        if (args.input_dir / "source_hashes.json").exists()
        else None,
        "nonlinear_noise_injection_audit": {
            "path": display_path(INJECTION_AUDIT),
            "sha256": sha256(INJECTION_AUDIT),
        }
        if injection_audit
        else None,
        "allow_partial": args.allow_partial,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    (args.figure_dir / "figure_manifest.json").write_text(
        json.dumps(
            figure_manifest(args.figure_dir, args.math_dir),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    figures = (
        ["terminal_error_gain", "precision_stop_rate", "exact_paired_slots"]
        + (["main_common_three_way"] if main_rows else [])
        + (["theory_empirical"] if theory else [])
    )
    (args.output_dir / "validation.json").write_text(
        json.dumps(
            {
                "coverage": coverage,
                "statistics": "Wilson 95% intervals; NumPy linear quantiles; gain and main contrasts bootstrap independent paired trial ids",
                "figures": figures,
                "all_passed": coverage["full_16x100_coverage"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    report(
        args.doc,
        summary,
        table_rows,
        coverage,
        theory,
        gaussian_precision,
        tail100,
        main_three_way,
        main_pairs,
        provenance,
        injection_audit,
    )
    print(
        json.dumps(
            {
                "coverage": coverage,
                "summary_rows": len(summary),
                "table1_rows": len(table_rows),
                "theory_rows": len(theory),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
