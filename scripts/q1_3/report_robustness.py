"""核验 main 鲁棒性记录，生成中文统计图；默认 PNG，可选 PDF/SVG。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.q1_1.localization import fy_position
from scripts.q1_3.plot_main_results import plt, BLUE, ORANGE, INK, GRAY
from scripts.figure_style import configure_style
from scripts.q1_3.run_robustness import CONDITIONS, aggregate, write_csv

ROOT = Path(__file__).resolve().parents[2]
EXPORT_FORMATS = ("png",)
EXPORTED_FIGURES: list[Path] = []
LABELS = [
    "无噪声",
    "方向噪声\n0.001°",
    "方向噪声\n0.01°",
    "方向噪声\n0.1°",
    "动作误差\n1%",
]


def export(fig, name):
    out = ROOT / "figures/q1_3"
    out.mkdir(parents=True, exist_ok=True)
    for extension in EXPORT_FORMATS:
        fig.savefig(
            out / f"{name}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.15
        )
        EXPORTED_FIGURES.append(out / f"{name}.{extension}")
    plt.close(fig)


def main():
    global EXPORT_FORMATS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png"],
        help="默认 PNG；可加选 PDF/SVG",
    )
    args = parser.parse_args()
    EXPORT_FORMATS = tuple(dict.fromkeys(["png", *args.formats]))
    EXPORTED_FIGURES.clear()
    style = configure_style()
    directory = ROOT / "outputs/q1_3/robustness"
    contract = json.loads((directory / "contract.json").read_text())
    n = contract["trials_per_condition"]
    for source, sha in contract["source_sha256"].items():
        assert (
            hashlib.sha256((ROOT / "scripts/q1_3" / source).read_bytes()).hexdigest()
            == sha
        ), source
    q = np.array([fy_position(i) for i in range(10)])
    rows = []
    histories = {}
    checks = 0
    for condition in CONDITIONS:
        for trial in [-1, *range(n)]:
            path = (
                directory
                / condition
                / ("table1" if trial == -1 else f"trial_{trial:03d}")
            )
            run = json.loads((path / "summary.json").read_text())
            assert run["fingerprint"] == contract["fingerprint"]
            e = run["evaluation"]
            summary = run["summary"]
            points = np.array(run["final_positions"])
            initial = np.array(run["initial_positions"])
            np.testing.assert_array_equal(points[:2], q[:2])
            errors = np.linalg.norm(points[1:] - q[1:], axis=1)
            assert np.isclose(errors.max(), e["max_position_error_m"], rtol=1e-12)
            assert np.isclose(
                np.sqrt(np.mean(errors**2)), e["rms_position_error_m"], rtol=1e-12
            )
            h = list(csv.DictReader((path / "slot_metrics.csv").open()))
            assert len(h) == e["measurement_slots"] + 1
            assert 4 * e["measurement_slots"] == e["transmitter_uses"]
            assert np.isclose(
                sum(float(r["endpoint_displacement_m"]) for r in h),
                e["movement_m"],
                rtol=1e-12,
            )
            assert e["joint_success_1cm"] == (
                e["status"] == "quiet_full_cycle" and errors.max() < 0.01
            )
            if condition != "exact":
                reference = json.loads(
                    (directory / "exact" / path.name / "summary.json").read_text()
                )
                np.testing.assert_array_equal(initial, reference["initial_positions"])
            assert summary["settings"] == contract["settings"]
            rows.append(e)
            histories[condition, trial] = h
            checks += 1
    rows.sort(key=lambda r: (list(CONDITIONS).index(r["condition"]), r["trial"]))
    summaries = aggregate(rows)
    with (directory / "summary.csv").open() as f:
        recorded = list(csv.DictReader(f))
    assert len(recorded) == len(summaries)
    for actual, saved in zip(summaries, recorded):
        for key, value in actual.items():
            if isinstance(value, (float, int)):
                assert np.isclose(value, float(saved[key]), rtol=1e-12, atol=1e-14), key
    # Match the analytical statistic exactly; no comparison of median Emax with RMS.
    theory = list(
        csv.DictReader(
            (ROOT / "outputs/q1_3/noise_analysis/predicted_noise_floor.csv").open()
        )
    )
    comparisons = []
    for prediction in theory:
        sigma = float(prediction["bearing_std_deg"])
        condition = f"bearing_{sigma:g}deg"
        selected = [
            r
            for r in rows
            if r["condition"] == condition
            and r["trial"] >= 0
            and r["status"] == "max_epochs_reached"
        ]
        observed = float(
            np.sqrt(np.mean([r["rms_position_error_m"] ** 2 for r in selected]))
        )
        predicted = float(prediction["predicted_root_expected_rms_squared_m"])
        comparisons.append(
            {
                "bearing_std_deg": sigma,
                "budget_exhausted_samples": len(selected),
                "predicted_root_expected_rms_squared_m": predicted,
                "observed_root_mean_rms_squared_m": observed,
                "observed_over_predicted": observed / predicted,
            }
        )
    write_csv(directory / "noise_floor_comparison.csv", comparisons)
    fig, (ax, rates) = plt.subplots(1, 2, figsize=(11.8, 4.9))
    rng = np.random.default_rng(8123)
    for i, summary in enumerate(summaries):
        values = [
            r["max_position_error_m"]
            for r in rows
            if r["condition"] == summary["condition"] and r["trial"] >= 0
        ]
        ax.scatter(
            i + rng.uniform(-0.16, 0.16, n),
            values,
            s=8,
            alpha=0.35,
            color=BLUE,
            linewidths=0,
        )
        median = summary["max_position_error_m_median"]
        lo = summary["max_position_error_m_p05"]
        hi = summary["max_position_error_m_p95"]
        ax.errorbar(
            i,
            median,
            yerr=[[median - lo], [hi - median]],
            fmt="D",
            markersize=4,
            color=INK,
            capsize=4,
            lw=1.5,
        )
    ax.set_yscale("log")
    ax.axhline(0.01, color=INK, ls=":", lw=0.9)
    ax.text(4.3, 0.012, "1 cm", ha="right", fontsize=9)
    ax.set(
        xticks=range(5),
        xticklabels=LABELS,
        ylabel="终态最大位置误差 / m（对数轴）",
        title="(a) 全部样本、中央 90% 区间与中位数",
    )
    ax.grid(axis="y", color=GRAY, lw=0.5)
    x = np.arange(5)
    stopped = np.array([s["stopped_count"] / n * 100 for s in summaries])
    accurate = np.array([s["final_below_1cm_count"] / n * 100 for s in summaries])
    bars1 = rates.bar(x - 0.17, stopped, width=0.32, color=BLUE, label="协议停止")
    bars2 = rates.bar(
        x + 0.17,
        accurate,
        width=0.32,
        facecolor="white",
        edgecolor=ORANGE,
        hatch="///",
        label="终态误差 < 1 cm",
    )
    for bars in (bars1, bars2):
        for bar in bars:
            rates.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{bar.get_height():g}",
                ha="center",
                fontsize=8.5,
            )
    rates.set(
        xticks=x,
        xticklabels=LABELS,
        ylim=(0, 118),
        yticks=[0, 25, 50, 75, 100],
        ylabel="比例 / %",
        title="(b) 停止与精度分别统计",
    )
    rates.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8
    )
    rates.grid(axis="y", color=GRAY, lw=0.5)
    rates.set_axisbelow(True)
    fig.suptitle("随机初态、测角噪声与执行误差", x=0.07, y=1.01, ha="left", fontsize=11)
    fig.text(
        0.07,
        0.94,
        f"每档 {n} 个相同随机初态；原控制器与停止阈值；每次最多 560 时隙",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.015,
        "方向噪声加在四条视线后构造六角；动作误差为指令量的相对高斯误差。两类误差分开施加。",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.19, top=0.81, wspace=0.29)
    export(fig, "main_robustness")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True, sharey=True)
    trendrows = []
    for ax, condition, title in zip(
        axes.flat,
        list(CONDITIONS)[:4],
        ["无噪声", "方向噪声 0.001°", "方向噪声 0.01°", "方向噪声 0.1°"],
    ):
        values = np.full((n, 561), np.nan)
        for trial in range(n):
            h = histories[condition, trial]
            observed = np.array([float(r["max_position_error_m"]) for r in h])
            values[trial, : len(observed)] = observed
            row = next(
                r for r in rows if r["condition"] == condition and r["trial"] == trial
            )
            if row["stopped"]:
                values[trial, len(observed) :] = observed[-1]
        bands = np.nanquantile(values, [0.05, 0.5, 0.95], axis=0)
        slot = np.arange(561)
        ax.fill_between(slot, bands[0], bands[2], color=BLUE, alpha=0.16, linewidth=0)
        ax.plot(slot, bands[1], color=BLUE, lw=1.4)
        ax.axhline(0.01, color=INK, ls=":", lw=0.8)
        ax.set(title=title, yscale="log", ylim=(1e-5, 100), xlim=(0, 560))
        ax.grid(axis="y", color=GRAY, lw=0.5)
        for t in slot:
            trendrows.append(
                {
                    "condition": condition,
                    "slot": int(t),
                    "available_runs": int(np.isfinite(values[:, t]).sum()),
                    "max_error_p05_m": bands[0, t],
                    "max_error_median_m": bands[1, t],
                    "max_error_p95_m": bands[2, t],
                }
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("最大位置误差 / m")
    for ax in axes[-1]:
        ax.set_xlabel("累计测角时隙")
    fig.suptitle("含噪调整的误差过程", x=0.08, y=0.99, ha="left", fontsize=11)
    fig.text(
        0.08,
        0.935,
        f"每档 {n} 次；实线为中位数，色带为 P05–P95；虚线为 1 cm",
        fontsize=10,
    )
    fig.text(
        0.08,
        0.025,
        "已停止的无噪声运行按末态位置延伸以便对齐；不增加测角或移动开销。失败后不补造轨迹。",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.12, top=0.85, hspace=0.25, wspace=0.16
    )
    export(fig, "main_noise_trajectories")
    write_csv(directory / "trajectory_quantiles.csv", trendrows)
    (ROOT / "figures/q1_3/main_robustness_figure_manifest.json").write_text(
        json.dumps(
            {
                "plot_script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "style": style,
                "formats": list(EXPORT_FORMATS),
                "artifacts_sha256": {
                    str(path.relative_to(ROOT)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in EXPORTED_FIGURES
                },
                "plot_data_sha256": {
                    str(path.relative_to(ROOT)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in (
                        directory / "noise_floor_comparison.csv",
                        directory / "trajectory_quantiles.csv",
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (directory / "validation.json").write_text(
        json.dumps(
            {
                "verified_runs": checks,
                "random_runs": 5 * n,
                "table1_sentinels": 5,
                "checks": [
                    "source hashes",
                    "paired initial conditions",
                    "fixed references",
                    "final Emax and RMS from coordinates",
                    "per-slot cumulative costs",
                    "all-trial statistics and success definitions",
                ],
                "all_passed": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "verified_runs": checks,
                "summaries": summaries,
                "noise_floor_comparison": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
