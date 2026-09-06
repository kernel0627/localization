"""Export reproducible figures for the frozen two-configuration evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/localization-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure_style import configure_style  # noqa: E402

BLUE, ORANGE, INK, GRAY, PALE = "#3B6FB6", "#C47A36", "#30343B", "#D4D8DE", "#EDF0F4"
CONDITIONS = [
    "exact",
    "bearing_0.001deg",
    "bearing_0.01deg",
    "bearing_0.1deg",
    "actuation_1pct",
]
CONDITION_LABELS = ["精确", "0.001°", "0.01°", "0.1°", "1% 执行"]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"no plot data for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def export(fig, directory: Path, name: str, formats: tuple[str, ...]) -> list[Path]:
    chinese_font = plt.rcParams["font.family"][1]
    for text in fig.findobj(match=Text):
        content = text.get_text()
        if "$" in content and any("\u4e00" <= char <= "\u9fff" for char in content):
            text.set_fontfamily(chinese_font)
    generated = []
    for suffix in formats:
        path = directory / f"{name}.{suffix}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.12,
        )
        generated.append(path)
    plt.close(fig)
    return generated


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_figure_manifest(
    output: Path,
    analysis: Path,
    exports: list[Path],
    style_metadata: dict,
) -> None:
    """Bind the final exports to their plotting source and input data."""
    inputs = [
        ROOT / "scripts/q1_3/plot_two_configuration_results.py",
        ROOT / "scripts/figure_style.py",
        analysis / "gain_0.5/summary.json",
        analysis / "gain_0.5/slot_metrics.csv",
        analysis / "table1_evaluation.csv",
        analysis / "precision_costs.csv",
        analysis / "radial_tangential_blocks.csv",
        analysis / "noise_validation.csv",
        analysis / "phase_noise_predictions.csv",
        ROOT / "outputs/q1_3/main_analysis/gain_0.5/slot_metrics.csv",
        ROOT / "outputs/q1_3/main_analysis/table1_evaluation.csv",
        ROOT / "outputs/q1_3/main_analysis/precision_costs.csv",
        ROOT / "outputs/q1_3/robustness/summary.csv",
        ROOT / "appendix1/evaluation_560/summary.csv",
        ROOT / "outputs/q1_3/two_configuration_review/component_audit.json",
    ]
    data_files = sorted((output / "data").glob("*.csv"))
    manifest = {
        "plotting_script_sha256": sha256(inputs[0]),
        "style": style_metadata,
        "default_case": "gain_0.5",
        "noise_statistic": "FY01–FY09 的几何均方根平方取平均后开方，分母为 9，并在相同相位比较",
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in inputs[1:]
        },
        "plot_data_sha256": {
            str(path.relative_to(output)): sha256(path) for path in data_files
        },
        "exports_sha256": {
            str(path.relative_to(output)): sha256(path) for path in exports
        },
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def number(record: dict[str, str], key: str) -> float:
    return float(record[key])


def plot_formation(two: Path, output: Path, formats: tuple[str, ...]) -> list[Path]:
    summary = json.loads((two / "gain_0.5/summary.json").read_text())
    from scripts.q1_1.localization import fy_position

    initial = np.array(summary["initial_positions"], dtype=float)
    final = np.array(summary["final_positions"], dtype=float)
    target = np.array([fy_position(i) for i in range(10)])
    errors = np.linalg.norm(final[1:] - target[1:], axis=1)
    data = [
        {
            "section": section,
            "drone_id": drone,
            "x_m": points[drone, 0],
            "y_m": points[drone, 1],
        }
        for section, points in (
            ("initial_position", initial),
            ("target_position", target),
            ("final_position", final),
        )
        for drone in range(10)
    ]
    data += [
        {
            "section": "final_position_error",
            "drone_id": drone,
            "error_m": errors[drone - 1],
        }
        for drone in range(1, 10)
    ]
    write_rows(output / "data/table1_formation_final_data.csv", data)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax = axes[0]
    circle = np.linspace(0, 2 * np.pi, 500)
    ax.plot(
        100 * np.cos(circle),
        100 * np.sin(circle),
        "--",
        color="#717782",
        lw=1,
        label="固定目标圆 R = 100 m",
    )
    ax.scatter(
        initial[1:, 0],
        initial[1:, 1],
        marker="o",
        s=52,
        facecolor="none",
        edgecolor=ORANGE,
        linewidth=1.45,
        label="初始位置",
    )
    ax.scatter(
        target[1:, 0],
        target[1:, 1],
        marker="x",
        s=60,
        color=INK,
        linewidth=1.1,
        label="名义目标",
    )
    ax.scatter(
        final[1:, 0], final[1:, 1], marker=".", s=38, color=BLUE, zorder=4, label="终态"
    )
    ax.scatter(target[:2, 0], target[:2, 1], marker="+", s=75, color=INK, zorder=5)
    ax.annotate("FY00", (0, 0), xytext=(7, 5), textcoords="offset points", fontsize=8.5)
    for drone in range(1, 10):
        ax.text(
            *(1.21 * target[drone]),
            f"FY{drone:02d}",
            ha="left"
            if target[drone, 0] > 60
            else "right"
            if target[drone, 0] < -60
            else "center",
            va="center",
            fontsize=8.5,
        )
    ax.set(
        aspect="equal",
        xlim=(-180, 180),
        ylim=(-137, 137),
        xlabel="x / m",
        ylabel="y / m",
        title="(a) 初态、名义目标与终态",
    )
    ax.grid(color=GRAY, lw=0.55)
    ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=2,
    )
    ax = axes[1]
    ax.bar(np.arange(1, 10), errors * 1e6, color=BLUE, width=0.58)
    ax.set(
        xticks=np.arange(1, 10),
        xlabel="圆周无人机编号",
        ylabel="终态位置误差 / μm",
        title="(b) FY01–FY09 的终态误差",
        ylim=(0, max(errors * 1e6) * 1.22),
    )
    ax.grid(axis="y", color=GRAY, lw=0.55)
    ax.set_axisbelow(True)
    for drone, value in enumerate(errors * 1e6, 1):
        ax.text(
            drone,
            value + max(errors * 1e6) * 0.035,
            f"{value:.2f}",
            ha="center",
            fontsize=8.5,
        )
    fig.suptitle(
        "双配置轮换：表 1 队形与终态误差",
        x=0.08,
        y=1.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.94,
        f"增益 0.5；84 时隙、336 发射机次；FY00、FY01 为固定参考。终态最大误差 {errors.max() * 1e6:.3f} μm。",
        fontsize=9,
    )
    fig.text(
        0.08,
        0.025,
        "左图圆为既定 $R=100\,\mathrm{m}$ 名义目标，不由终态重新拟合；右图由保存终态坐标直接计算。",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.33, top=0.78, wspace=0.3)
    return export(fig, output, "two_configuration_table1_formation_final", formats)


def plot_costs(
    main: Path, two: Path, output: Path, formats: tuple[str, ...]
) -> list[Path]:
    histories = {
        "main": rows(main / "gain_0.5/slot_metrics.csv"),
        "two_configuration": rows(two / "gain_0.5/slot_metrics.csv"),
    }
    evaluations = {
        "main": rows(main / "table1_evaluation.csv"),
        "two_configuration": rows(two / "table1_evaluation.csv"),
    }
    precision = {
        "main": rows(main / "precision_costs.csv"),
        "two_configuration": rows(two / "precision_costs.csv"),
    }
    data = []
    for method, history in histories.items():
        cumulative_uses = 0
        for record in history:
            slot = int(record["slot"])
            if "cumulative_transmitter_uses" in record:
                cumulative_uses = int(record["cumulative_transmitter_uses"])
            else:
                cumulative_uses += int(
                    record.get("slot_transmitter_uses", record["transmitter_uses"])
                )
            assert cumulative_uses == 4 * slot, (method, slot, cumulative_uses)
            data.append(
                {
                    "section": "trajectory",
                    "method": method,
                    "slot": slot,
                    "max_position_error_m": float(record["max_position_error_m"]),
                    "transmitter_uses": cumulative_uses,
                    "cumulative_endpoint_m": float(
                        record["cumulative_endpoint_m"]
                        if "cumulative_endpoint_m" in record
                        else record["cumulative_endpoint_displacement_m"]
                    ),
                }
            )
    for method, expected in (("main", 784), ("two_configuration", 336)):
        actual = max(r["transmitter_uses"] for r in data if r["method"] == method)
        assert actual == expected, (method, actual, expected)
    events = []
    for method in histories:
        evaluation = next(r for r in evaluations[method] if r["case"] == "gain_0.5")
        first = {
            r["threshold"]: r for r in precision[method] if r["case"] == "gain_0.5"
        }
        events.extend(
            [
                {
                    "section": "event",
                    "method": method,
                    "event": "首次 1 cm",
                    "slot": int(first["1cm"]["first_slot"]),
                },
                {
                    "section": "event",
                    "method": method,
                    "event": "首次 1 mm",
                    "slot": int(first["1mm"]["first_slot"]),
                },
                {
                    "section": "event",
                    "method": method,
                    "event": "最后动作",
                    "slot": int(evaluation["last_motion_slot"]),
                },
                {
                    "section": "event",
                    "method": method,
                    "event": "协议停止",
                    "slot": int(evaluation["measurement_slots"]),
                },
            ]
        )
    data.extend(events)
    write_rows(output / "data/main_vs_two_precision_cost_data.csv", data)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.1))
    labels = {"main": "全配置轮换（main）", "two_configuration": "双配置轮换"}
    colors = {"main": BLUE, "two_configuration": ORANGE}
    for method in histories:
        trajectory = [
            r for r in data if r["section"] == "trajectory" and r["method"] == method
        ]
        axes[0, 0].semilogy(
            [r["slot"] for r in trajectory],
            [r["max_position_error_m"] for r in trajectory],
            color=colors[method],
            lw=1.3,
            label=labels[method].replace("（main）", ""),
        )
        axes[0, 1].plot(
            [r["slot"] for r in trajectory],
            [r["transmitter_uses"] for r in trajectory],
            color=colors[method],
            lw=1.3,
            label=labels[method].replace("（main）", ""),
        )
        axes[1, 0].plot(
            [r["slot"] for r in trajectory],
            [r["cumulative_endpoint_m"] for r in trajectory],
            color=colors[method],
            lw=1.3,
            label=labels[method].replace("（main）", ""),
        )
        last = next(
            r["slot"]
            for r in events
            if r["method"] == method and r["event"] == "最后动作"
        )
        stop = next(
            r["slot"]
            for r in events
            if r["method"] == method and r["event"] == "协议停止"
        )
        axes[0, 0].axvspan(last, stop, color=colors[method], alpha=0.08)
    for ax, title, ylabel in (
        (axes[0, 0], "(a) 完整表 1 误差轨迹", "最大位置误差 / m（对数轴）"),
        (axes[0, 1], "(b) 累计发射机次", "发射机次"),
        (axes[1, 0], "(c) 累计端点位移", "位移 / m"),
    ):
        ax.set(title=title, xlabel="测角时隙", ylabel=ylabel)
        ax.grid(axis="y", color=GRAY, lw=0.55)
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_axisbelow(True)
    event_names = ["首次 1 cm", "首次 1 mm", "最后动作", "协议停止"]
    x = np.arange(len(event_names))
    for index, method in enumerate(("main", "two_configuration")):
        values = [
            next(
                r["slot"]
                for r in events
                if r["method"] == method and r["event"] == event
            )
            for event in event_names
        ]
        axes[1, 1].bar(
            x + (index - 0.5) * 0.34,
            values,
            0.34,
            color=colors[method],
            label=labels[method],
        )
    axes[1, 1].set(
        title="(d) 精度、动作与停止事件",
        xticks=x,
        xticklabels=event_names,
        ylabel="时隙",
    )
    axes[1, 1].tick_params(axis="x", labelsize=8)
    axes[1, 1].grid(axis="y", color=GRAY, lw=0.55)
    axes[1, 1].set_axisbelow(True)
    axes[1, 1].legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        "相同表 1：全配置与双配置的精度和全过程开销",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.015,
        "两法每时隙均为 4 架发射机。阴影是最后动作后的测量段：全配置为 51 时隙、双配置为 2 时隙；全配置最终完整安静周期为 169–196，不能把停止时隙差全部归因于定位迭代。",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.1, top=0.88, hspace=0.48, wspace=0.28
    )
    return export(fig, output, "main_vs_two_precision_cost", formats)


def plot_components(
    main: Path, two: Path, review: Path, output: Path, formats: tuple[str, ...]
) -> list[Path]:
    histories = {
        "main": rows(main / "gain_0.5/slot_metrics.csv"),
        "two_configuration": rows(two / "gain_0.5/slot_metrics.csv"),
    }
    blocks = rows(two / "radial_tangential_blocks.csv")
    audit = json.loads(review.read_text())
    data = [
        {
            "section": "trajectory",
            "method": method,
            "slot": int(r["slot"]),
            "max_radial_error_m": number(r, "max_radial_error_m"),
            "max_angular_error_deg": number(r, "max_angular_error_deg"),
        }
        for method, history in histories.items()
        for r in history
    ]
    data += [
        {
            "section": "block",
            "gain": r["gain"],
            "input_component": r["input_component"],
            "output_component": r["output_component"],
            "operator_norm": r["operator_norm"],
        }
        for r in blocks
        if r["gain"] == "0.5"
    ]
    data += [{"section": "pulse", **item} for item in audit["nonlinear_pulses"]]
    write_rows(output / "data/radial_tangential_coupling_data.csv", data)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    for method, color, label in (
        ("main", BLUE, "全配置轮换（main）"),
        ("two_configuration", ORANGE, "双配置轮换"),
    ):
        tr = [r for r in data if r["section"] == "trajectory" and r["method"] == method]
        axes[0, 0].semilogy(
            [r["slot"] for r in tr],
            [r["max_radial_error_m"] for r in tr],
            color=color,
            lw=1.3,
            label=label.replace("（main）", ""),
        )
        axes[0, 1].semilogy(
            [r["slot"] for r in tr],
            [r["max_angular_error_deg"] for r in tr],
            color=color,
            lw=1.3,
            label=label.replace("（main）", ""),
        )
    axes[0, 0].set(
        title="(a) 全配置与双配置：最大径向误差",
        xlabel="测角时隙",
        ylabel="m（对数轴）",
    )
    axes[0, 0].grid(axis="y", color=GRAY, lw=0.55)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[0, 1].set(
        title="(b) 全配置与双配置：最大角偏差", xlabel="测角时隙", ylabel="°（对数轴）"
    )
    axes[0, 1].grid(axis="y", color=GRAY, lw=0.55)
    axes[0, 1].legend(frameon=False, fontsize=8.5, loc="upper right")
    for method, color, xytext in (
        ("main", BLUE, (26, 1.2)),
        ("two_configuration", ORANGE, (37, 0.32)),
    ):
        tr = [r for r in data if r["section"] == "trajectory" and r["method"] == method]
        peak = max(tr, key=lambda r: r["max_angular_error_deg"])
        label = f"{('全配置' if method == 'main' else '双配置')}：{peak['max_angular_error_deg']:.3f}° @ {peak['slot']}"
        axes[0, 1].scatter(
            [peak["slot"]], [peak["max_angular_error_deg"]], s=18, color=color, zorder=4
        )
        axes[0, 1].annotate(
            label,
            (peak["slot"], peak["max_angular_error_deg"]),
            xytext=xytext,
            textcoords="data",
            fontsize=8.5,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
        )
    matrix = np.empty((2, 2))
    component = {"radial": 0, "tangential": 1}
    for r in (item for item in data if item["section"] == "block"):
        matrix[component[r["output_component"]], component[r["input_component"]]] = (
            float(r["operator_norm"])
        )
    im = axes[1, 0].imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    axes[1, 0].set(
        title="(c) 双配置两时隙局部块范数",
        xticks=[0, 1],
        xticklabels=["径向输入", "切向输入"],
        yticks=[0, 1],
        yticklabels=["径向输出", "切向输出"],
    )
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 0.55 else INK,
            )
    fig.colorbar(im, ax=axes[1, 0], fraction=0.047, pad=0.04, label="算子范数")
    pulses = [r for r in data if r["section"] == "pulse"]
    method_label = {"main": "全配置 28 时隙", "two_configuration": "双配置 2 时隙"}
    component_label = {"radial": "径向", "tangential": "切向"}
    labels = [
        f"{method_label[r['method']]}\nFY{int(r['input_drone']):02d} {component_label[r['input_component']]}\n→ FY{int(r['largest_cross_output_drone']):02d} {component_label[r['largest_cross_output_component']]}"
        for r in pulses
    ]
    values = [abs(float(r["signed_cross_output_m"])) * 1e6 for r in pulses]
    axes[1, 1].bar(np.arange(len(pulses)), values, color=[BLUE, BLUE, ORANGE, ORANGE])
    axes[1, 1].set(
        title="(d) 1 mm 纯分量扰动的跨分量响应",
        xticks=np.arange(len(pulses)),
        xticklabels=labels,
        ylabel="绝对交叉输出 / μm",
    )
    axes[1, 1].tick_params(axis="x", labelsize=8)
    axes[1, 1].grid(axis="y", color=GRAY, lw=0.55)
    axes[1, 1].set_axisbelow(True)
    fig.suptitle(
        "径向/切向误差演化与交叉传播",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.015,
        "局部块和纯分量脉冲固定名义择优、无保持和无指令限幅；(d) 仅示跨分量响应，不是算法范数排行。全配置为 28 时隙、双配置为 2 时隙，不能按周期长度直接排名。",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.08, right=0.96, bottom=0.1, top=0.88, hspace=0.48, wspace=0.36
    )
    return export(fig, output, "radial_tangential_coupling", formats)


def plot_random_and_noise(
    main: Path, two: Path, output: Path, formats: tuple[str, ...]
) -> list[Path]:
    two_summary = {
        r["condition"]: r for r in rows(ROOT / "appendix1/evaluation_560/summary.csv")
    }
    main_summary = {
        r["condition"]: r for r in rows(ROOT / "outputs/q1_3/robustness/summary.csv")
    }
    noise = rows(two / "noise_validation.csv")
    phases = rows(two / "phase_noise_predictions.csv")
    data = []
    for method, summary in (("main", main_summary), ("two_configuration", two_summary)):
        for condition in CONDITIONS:
            r = summary[condition]
            data.append(
                {
                    "section": "random",
                    "method": method,
                    "condition": condition,
                    "runs": int(r["runs"]),
                    "stopped_count": int(r["stopped_count"]),
                    "final_below_1cm_count": int(r["final_below_1cm_count"]),
                    "rms_p05_m": float(r["rms_position_error_m_p05"]),
                    "rms_median_m": float(r["rms_position_error_m_median"]),
                    "rms_p95_m": float(r["rms_position_error_m_p95"]),
                }
            )
    data += [{"section": "noise_observed", **r} for r in noise] + [
        {"section": "noise_prediction", **r} for r in phases
    ]
    write_rows(output / "data/random_terminal_stop_noise_data.csv", data)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.45))
    for method, color, marker, offset in (
        ("main", BLUE, "o", -0.12),
        ("two_configuration", ORANGE, "s", 0.12),
    ):
        values = [
            next(
                r
                for r in data
                if r["section"] == "random"
                and r["method"] == method
                and r["condition"] == c
            )
            for c in CONDITIONS
        ]
        med = np.array([r["rms_median_m"] for r in values])
        low = np.array([r["rms_p05_m"] for r in values])
        high = np.array([r["rms_p95_m"] for r in values])
        axes[0].errorbar(
            np.arange(5) + offset,
            med,
            yerr=np.vstack((med - low, high - med)),
            fmt=marker,
            color=color,
            capsize=3,
            label=("全配置轮换", "双配置")[method == "two_configuration"],
        )
    axes[0].set(
        yscale="log",
        xticks=np.arange(5),
        xticklabels=CONDITION_LABELS,
        ylabel="终态均方根 / m（第 5、中位、第 95 百分位）",
        title="(a) 500 次随机终态",
    )
    axes[0].grid(axis="y", color=GRAY, lw=0.55)
    axes[0].legend(frameon=False, fontsize=8.5)
    for method, color, offset in (
        ("main", BLUE, -0.18),
        ("two_configuration", ORANGE, 0.18),
    ):
        values = [
            next(
                r
                for r in data
                if r["section"] == "random"
                and r["method"] == method
                and r["condition"] == c
            )
            for c in CONDITIONS
        ]
        axes[1].bar(
            np.arange(5) + offset,
            [100 * r["stopped_count"] / r["runs"] for r in values],
            0.34,
            color=color,
            label=("全配置轮换", "双配置")[method == "two_configuration"],
        )
    axes[1].set(
        ylim=(0, 108),
        xticks=np.arange(5),
        xticklabels=CONDITION_LABELS,
        ylabel="停止率 / %",
        title="(b) 500 次随机停止",
    )
    axes[1].grid(axis="y", color=GRAY, lw=0.55)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8.5)
    phases = [r for r in phases if float(r["bearing_std_deg"]) == 0.001]
    predicted = [float(r["predicted_root_expected_rms_squared_m"]) for r in phases]
    observed = [r for r in noise if r["condition"] == "bearing_0.001deg"]
    axes[2].plot(
        [int(r["phase"]) for r in phases],
        predicted,
        marker="o",
        color=BLUE,
        lw=1.3,
        label="线性理论（相位）",
    )
    axes[2].scatter(
        [int(r["phase"]) for r in observed],
        [float(r["observed_root_expected_rms_squared_m"]) for r in observed],
        marker="s",
        color=ORANGE,
        s=44,
        zorder=3,
        label="100 样本观测",
    )
    axes[2].set(
        xticks=[1, 2],
        xlabel="两时隙相位",
        ylabel="均方根均值的平方根 / m",
        title="(c) 0.001° 噪声：同统计量",
    )
    axes[2].grid(axis="y", color=GRAY, lw=0.55)
    axes[2].legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        "随机终态、停止与噪声理论对照",
        x=0.07,
        y=1.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.02,
        "每个条件 100 个冻结随机初态；噪声观测与理论均为 FY01–FY09（9 架）的均方根均值的平方根，未以中位数替代。",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.21, top=0.78, wspace=0.35)
    return export(fig, output, "random_terminal_stop_noise", formats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ROOT / "outputs/q1_3/two_configuration_analysis",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "figures/q1_3/two_configuration"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png",),
        help="输出格式；默认仅 PNG。",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style_metadata = configure_style()
    formats = tuple(args.formats)
    generated = plot_formation(args.analysis_dir, args.output_dir, formats)
    generated += plot_costs(
        ROOT / "outputs/q1_3/main_analysis", args.analysis_dir, args.output_dir, formats
    )
    generated += plot_components(
        ROOT / "outputs/q1_3/main_analysis",
        args.analysis_dir,
        ROOT / "outputs/q1_3/two_configuration_review/component_audit.json",
        args.output_dir,
        formats,
    )
    generated += plot_random_and_noise(
        ROOT / "outputs/q1_3/robustness", args.analysis_dir, args.output_dir, formats
    )
    write_figure_manifest(args.output_dir, args.analysis_dir, generated, style_metadata)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "figures": 4,
                "exports": [str(path) for path in generated],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
