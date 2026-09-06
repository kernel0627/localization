"""从既有结果生成 q1_3 正文的两张组合图与一张汇总表，不运行仿真。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/q13-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from scripts.figure_style import configure_style

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/q1_3"
FIGURES = ROOT / "figures/q1_3/paper"
TABLE = ROOT / "solutions/q1_3/实验汇总表.md"
METHODS = ("main", "two_configuration_gain_0.5", "two_configuration_gain_1")
LABELS = ("全配置 η=0.5", "双配置 η=0.5", "双配置 η=1")
COLORS = ("#0072B2", "#D55E00", "#009E73")
LINES = ("-", "--", "-.")
MARKERS = ("o", "s", "^")
INPUTS: dict[Path, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    INPUTS[path] = digest(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def values(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows])


def export(fig: plt.Figure, name: str, output: Path) -> Path:
    path = output / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return path


def histories() -> list[list[dict]]:
    result = [read(DATA / "error_history.csv")]
    for gain in ("0.5", "1"):
        result.append(
            read(DATA / f"two_configuration_analysis/gain_{gain}/slot_metrics.csv")
        )
    for rows in result:
        np.testing.assert_array_equal(values(rows, "slot"), np.arange(len(rows)))
        if not np.isfinite(values(rows, "max_position_error_m")).all():
            raise ValueError("轨迹包含非有限误差")
    np.testing.assert_allclose(
        [float(rows[0]["max_position_error_m"]) for rows in result],
        float(result[0][0]["max_position_error_m"]),
    )
    return result


def main_figure(history: list[dict], output: Path) -> Path:
    rows = read(DATA / "positions.csv")
    states = np.full((len(history), 10, 2), np.nan)
    for row in rows:
        states[int(row["slot"]), int(row["drone_id"])] = [
            float(row["x_m"]),
            float(row["y_m"]),
        ]
    if not np.isfinite(states).all():
        raise ValueError("位置轨迹缺失")
    theta = np.arange(9) * 2 * np.pi / 9
    target = np.column_stack((100 * np.cos(theta), 100 * np.sin(theta)))
    error = np.linalg.norm(states[:, 1:] - target, axis=2).max(axis=1)
    np.testing.assert_allclose(error, values(history, "max_position_error_m"))
    fig, (ax, curve) = plt.subplots(1, 2, figsize=(7.2, 3.55))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.23, wspace=0.38)
    circle = np.linspace(0, 2 * np.pi, 361)
    ax.plot(
        100 * np.cos(circle), 100 * np.sin(circle), color="#888888", lw=0.8, ls="--"
    )
    ax.scatter(
        *states[0, 1:].T, s=26, facecolor="none", edgecolor=COLORS[1], label="初始"
    )
    ax.scatter(*target.T, s=30, marker="x", color="#333333", label="目标")
    ax.scatter(*states[-1, 1:].T, s=10, color=COLORS[0], label="最终", zorder=4)
    ax.scatter([0], [0], marker="+", color="#333333", s=25)
    ax.annotate("00", (0, 0), xytext=(5, 4), textcoords="offset points", fontsize=8)
    for i, point in enumerate(target, 1):
        align = "left" if point[0] > 60 else "right" if point[0] < -60 else "center"
        ax.text(*(1.30 * point), f"{i:02d}", ha=align, va="center", fontsize=8)
    ax.set(
        aspect="equal",
        xlim=(-153, 153),
        ylim=(-153, 153),
        xlabel="x / m",
        ylabel="y / m",
    )
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-100, 0, 100])
    ax.set_title("(a) 初始与最终队形")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.265, 0.09),
        ncol=3,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    slots = values(history, "slot")
    curve.semilogy(slots, error, color=COLORS[0])
    curve.axhline(0.01, color="#777777", ls=":", lw=0.9)
    first = int(np.flatnonzero(error < 0.01)[0])
    curve.scatter([first, slots[-1]], [error[first], error[-1]], s=22, color=COLORS[0])
    curve.annotate(
        f"首次厘米达标：{first}",
        (first, error[first]),
        xytext=(106, 0.05),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#555555"},
    )
    curve.annotate(
        f"停止确认：{int(slots[-1])}",
        (slots[-1], error[-1]),
        xytext=(155, 0.001),
        ha="center",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#555555"},
    )
    curve.text(5, 0.014, "1 cm", fontsize=8, color="#555555")
    curve.set(
        xlim=(0, slots[-1] + 7),
        ylim=(2e-5, 30),
        xlabel="累计测角时隙",
        ylabel="最大位置误差 / m",
    )
    curve.set_title("(b) 全配置轮换调整过程")
    curve.grid(axis="y", which="major")
    fig.text(
        0.08,
        0.015,
        "表 1 初态；精确测角与执行；η=0.5。编号省略 FY，末态在队形尺度下与目标重合。",
        fontsize=8,
    )
    return export(fig, "main_adjustment", output)


def noise_rows() -> list[dict]:
    predictions = read(DATA / "noise_analysis/predicted_noise_floor.csv")
    trials = read(DATA / "robustness/trials.csv")
    result = []
    for row in predictions:
        sigma = float(row["bearing_std_deg"])
        condition = f"bearing_{sigma:g}deg"
        samples = [
            r for r in trials if r["condition"] == condition and int(r["trial"]) >= 0
        ]
        if len(samples) != 100 or any(
            int(r["measurement_slots"]) != 560 for r in samples
        ):
            raise ValueError("main 白噪声比较需要100个第560时隙样本")
        result.append(
            dict(
                method="main",
                sigma_deg=sigma,
                theory_m=float(row["predicted_root_expected_rms_squared_m"]),
                empirical_m=float(
                    np.sqrt(np.mean(values(samples, "rms_position_error_m") ** 2))
                ),
                low_m="",
                high_m="",
                samples=100,
            )
        )
    theory_path = (
        ROOT
        / "figures/q1_3/two_configuration_robustness/data/theory_empirical_data.csv"
    )
    for row in read(theory_path):
        if row["phase"] != "560" or row["theory_source"] != "periodic_covariance":
            continue
        result.append(
            dict(
                method=f"two_configuration_gain_{float(row['gain']):g}",
                sigma_deg=float(
                    row["condition"].removeprefix("bearing_").removesuffix("deg")
                ),
                theory_m=float(row["theory_root_expected_rms_squared_m"]),
                empirical_m=float(row["empirical_root_mean_rms_squared_m"]),
                low_m=float(row["empirical_bootstrap95_low"]),
                high_m=float(row["empirical_bootstrap95_high"]),
                samples=int(row["empirical_samples"]),
            )
        )
    keys = {(r["method"], r["sigma_deg"]) for r in result}
    if len(result) != 9 or keys != {
        (m, s) for m in METHODS for s in (0.001, 0.01, 0.1)
    }:
        raise ValueError("理论/实验比较缺失或重复")
    return result


def comparison_figure(
    history: list[list[dict]], noise: list[dict], output: Path
) -> Path:
    fig, (curve, comparison) = plt.subplots(1, 2, figsize=(7.2, 3.7))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.26, wspace=0.44)
    for rows, label, color, line in zip(history, LABELS, COLORS, LINES):
        curve.semilogy(
            values(rows, "slot"),
            values(rows, "max_position_error_m"),
            label=label,
            color=color,
            ls=line,
        )
        first = int(np.flatnonzero(values(rows, "max_position_error_m") < 0.01)[0])
        curve.scatter(first, 0.01, color=color, s=16, zorder=4)
        curve.annotate(
            str(first),
            (first, 0.01),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=color,
        )
    curve.axhline(0.01, color="#777777", ls=":", lw=0.8)
    curve.set(
        xlim=(0, 203),
        ylim=(1e-13, 30),
        xlabel="累计测角时隙",
        ylabel="最大位置误差 / m",
    )
    curve.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1])
    curve.set_title("(a) 相同初态下的收敛比较")
    curve.grid(axis="y")
    for method, color, line, marker in zip(METHODS, COLORS, LINES, MARKERS):
        selected = sorted(
            [r for r in noise if r["method"] == method], key=lambda r: r["sigma_deg"]
        )
        x = np.array([r["sigma_deg"] for r in selected])
        theory = np.array([r["theory_m"] for r in selected])
        empirical = np.array([r["empirical_m"] for r in selected])
        comparison.loglog(x, theory, color=color, ls=line, lw=1)
        comparison.scatter(
            x,
            empirical,
            facecolors="white",
            edgecolors=color,
            marker=marker,
            s=30,
            zorder=4,
        )
        if method != "main":
            low = np.array([r["low_m"] for r in selected])
            high = np.array([r["high_m"] for r in selected])
            comparison.errorbar(
                x,
                empirical,
                yerr=[empirical - low, high - empirical],
                fmt="none",
                ecolor=color,
                capsize=2,
                lw=0.8,
            )
    comparison.set(
        xlim=(0.0007, 0.145),
        ylim=(0.002, 1.3),
        xlabel="单射线白噪声标准差 / °",
        ylabel="均方 RMS 的平方根 / m",
    )
    comparison.set_xticks([0.001, 0.01, 0.1], ["0.001", "0.01", "0.1"])
    comparison.set_title("(b) 第 560 时隙的噪声响应")
    comparison.grid(axis="y")
    handles = [
        Line2D([0], [0], color=c, ls=ls, marker=m, markerfacecolor="white", label=label)
        for c, ls, m, label in zip(COLORS, LINES, MARKERS, LABELS)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=3,
        columnspacing=1.7,
    )
    fig.text(
        0.09,
        0.095,
        "(a) 标记首次厘米达标；曲线结束于实际停止。极小残差仅代表精确模型的数值结果。",
        fontsize=8,
    )
    fig.text(
        0.09,
        0.045,
        "(b) 线：局部理论；点：100 次试验；双配置误差棒：95% bootstrap 区间。",
        fontsize=8,
    )
    return export(fig, "schedule_noise_comparison", output)


def build_table() -> str:
    common = read(
        DATA
        / "two_configuration_robustness_report/main_common_conditions_three_way_summary.csv"
    )
    extra = read(DATA / "two_configuration_robustness_report/random_summary.csv")
    rows = {(r["condition"], r["method"]): r for r in common}
    for r in extra:
        if r["condition"] == "link_bias_0.001deg":
            rows[r["condition"], f"two_configuration_gain_{float(r['gain']):g}"] = r
    if any(int(r["runs"]) != 100 for r in rows.values()):
        raise ValueError("正文汇总要求每条件100次随机试验")
    specs = [
        (
            "精确条件",
            "首次厘米达标 / 时隙（中位数）",
            "exact",
            "first_1cm_median_among_hits",
            "integer",
        ),
        (
            "精确条件",
            "完整停止 / 时隙（中位数）",
            "exact",
            "measurement_slots_median",
            "integer",
        ),
        (
            "白噪声 0.001°",
            "终点最大误差 / cm（中位数）",
            "bearing_0.001deg",
            "max_position_error_m_median",
            "cm",
        ),
        (
            "白噪声 0.001°",
            "终点厘米达标数 / 100",
            "bearing_0.001deg",
            "final_below_1cm_count",
            "integer",
        ),
        (
            "白噪声 0.01°",
            "终点最大误差 / cm（中位数）",
            "bearing_0.01deg",
            "max_position_error_m_median",
            "cm",
        ),
        (
            "白噪声 0.1°",
            "终点最大误差 / cm（中位数）",
            "bearing_0.1deg",
            "max_position_error_m_median",
            "cm",
        ),
        (
            "相对执行误差 1%",
            "厘米达标且停止数 / 100",
            "actuation_1pct",
            "joint_success_1cm_count",
            "integer",
        ),
        (
            "固定偏置 0.001°",
            "终点最大误差 / cm（中位数）",
            "link_bias_0.001deg",
            "max_position_error_m_median",
            "cm",
        ),
        (
            "固定偏置 0.001°",
            "终点厘米达标数 / 100",
            "link_bias_0.001deg",
            "final_below_1cm_count",
            "integer",
        ),
    ]
    lines = [
        "| 条件 | 统计量 | 全配置 $\\eta=0.5$ | 双配置 $\\eta=0.5$ | 双配置 $\\eta=1$ |",
        "|---|---|---:|---:|---:|",
    ]
    for label, metric, condition, field, format_name in specs:
        cells = []
        for method in METHODS:
            row = rows.get((condition, method))
            if row is None:
                cells.append("—")
                continue
            value = float(row[field])
            cells.append(f"{100 * value:.3g}" if format_name == "cm" else f"{value:g}")
        lines.append("| " + " | ".join([label, metric, *cells]) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    parser.add_argument("--table-path", type=Path, default=TABLE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_path.parent.mkdir(parents=True, exist_ok=True)
    style = configure_style()
    table = build_table()
    args.table_path.write_text(table, encoding="utf-8")
    history = histories()
    noise = noise_rows()
    exports = [
        main_figure(history[0], args.output_dir),
        comparison_figure(history, noise, args.output_dir),
    ]
    noise_path = args.output_dir / "noise_comparison.csv"
    with noise_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(noise[0]))
        writer.writeheader()
        writer.writerows(noise)
    if any(digest(path) != before for path, before in INPUTS.items()):
        raise RuntimeError("输入数据在生成过程中发生变化")
    manifest = {
        "purpose": "q1_3 正文两图一表；仅读取既有实验",
        "script_sha256": digest(Path(__file__)),
        "style": style,
        "inputs_sha256": {str(p.relative_to(ROOT)): h for p, h in INPUTS.items()},
        "exports_sha256": {
            str(p.resolve()): digest(p) for p in [*exports, noise_path, args.table_path]
        },
        "noise_statistic": "sqrt(mean(geometric_RMS^2)), FY01-FY09 denominator 9, slot 560, n=100",
        "main_uncertainty": "main仅画经验点；双配置区间读取既有逐trial bootstrap结果",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "figures": [str(p) for p in exports],
                "table": str(args.table_path),
                "unchanged_inputs": len(INPUTS),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
