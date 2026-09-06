"""Export manuscript figures from main's recorded Table 1 trajectory."""

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
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure_style import configure_style  # noqa: E402
from scripts.q1_1.localization import fy_position  # noqa: E402

BLUE, ORANGE, INK, GRAY = "#3B6FB6", "#C47A36", "#30343B", "#D4D8DE"


def export(fig, directory: Path, name: str, formats: tuple[str, ...]) -> list[Path]:
    generated = []
    for extension in formats:
        path = directory / f"{name}.{extension}"
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
    input_dir: Path,
    exports: list[Path],
    style_metadata: dict,
) -> None:
    """Record only this invocation's exports with their data and font provenance."""
    inputs = [
        ROOT / "scripts/q1_3/plot_main_results.py",
        ROOT / "scripts/figure_style.py",
        input_dir / "summary.json",
        input_dir / "positions.csv",
        input_dir / "error_history.csv",
    ]
    manifest = {
        "plotting_script_sha256": sha256(inputs[0]),
        "style": style_metadata,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in inputs[1:]
        },
        "exports_sha256": {
            str(path.relative_to(output)): sha256(path) for path in exports
        },
    }
    (output / "main_results_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/q1_3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/q1_3")
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
    summary = json.loads((args.input_dir / "summary.json").read_text())
    if "public transmitter rotation" not in summary["method"]:
        raise ValueError("Expected main iterative-reference results")
    with (args.input_dir / "positions.csv").open() as stream:
        positions = list(csv.DictReader(stream))
    states = np.full((summary["measurement_slots"] + 1, 10, 2), np.nan)
    for row in positions:
        states[int(row["slot"]), int(row["drone_id"])] = [
            float(row["x_m"]),
            float(row["y_m"]),
        ]
    if not np.isfinite(states).all():
        raise ValueError("Incomplete position history")
    with (args.input_dir / "error_history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    q = np.array([fy_position(i) for i in range(10)])
    e = np.linalg.norm(states[:, 1:] - q[1:], axis=2)
    np.testing.assert_allclose(
        e.max(axis=1), [float(r["max_position_error_m"]) for r in history]
    )
    slot = np.arange(len(history))
    maximum = e.max(axis=1)
    rms = np.sqrt(np.mean(e**2, axis=1))
    fig, (ax, residual) = plt.subplots(
        1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    circle = np.linspace(0, 2 * np.pi, 500)
    ax.plot(
        100 * np.cos(circle),
        100 * np.sin(circle),
        "--",
        color="#717782",
        lw=1,
        label="目标圆 R = 100 m",
    )
    ax.scatter(
        states[0, 1:, 0],
        states[0, 1:, 1],
        marker="o",
        s=56,
        facecolor="none",
        edgecolor=ORANGE,
        linewidth=1.5,
        label="初始位置",
    )
    ax.scatter(
        q[1:, 0], q[1:, 1], marker="x", s=60, color=INK, linewidth=1.1, label="目标位置"
    )
    ax.scatter(
        states[-1, 1:, 0],
        states[-1, 1:, 1],
        marker=".",
        s=36,
        color=BLUE,
        zorder=4,
        label="最终位置",
    )
    ax.scatter([0], [0], marker="+", s=80, color=INK)
    ax.annotate("FY00", (0, 0), xytext=(7, 5), textcoords="offset points", fontsize=8.5)
    for i in range(1, 10):
        label_point = 1.21 * q[i]
        ax.text(*label_point, f"FY{i:02d}", ha="center", va="center", fontsize=8.5)
    ax.set(
        aspect="equal",
        xlim=(-137, 137),
        ylim=(-137, 137),
        xlabel="x / m",
        ylabel="y / m",
        title="(a) 初态、终态与固定目标",
    )
    ax.grid(color=GRAY, lw=0.5, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.90),
        ncol=4,
        fontsize=8.5,
        frameon=False,
    )
    residual.bar(np.arange(1, 10), e[-1] * 1e6, color=BLUE, width=0.58)
    residual.set(
        xticks=np.arange(1, 10),
        xlabel="圆周无人机编号",
        ylabel="终态位置误差 / μm",
        title="(b) 各机终态数值残差",
        ylim=(0, max(e[-1] * 1e6) * 1.22),
    )
    residual.grid(axis="y", color=GRAY, lw=0.6)
    residual.set_axisbelow(True)
    for i, value in enumerate(e[-1] * 1e6, 1):
        residual.text(i, value + 1.1, f"{value:.1f}", ha="center", fontsize=8.5)
    fig.suptitle("表 1 编队调整结果", x=0.08, y=1.02, ha="left", fontsize=10, fontweight="bold")
    fig.text(
        0.08,
        0.95,
        "FY00、FY01 固定；增益 0.5；精确测角与执行；终态为第 196 时隙",
        fontsize=9,
    )
    fig.text(
        0.08,
        0.025,
        "终态与目标在米尺度下重合，右图单独展示数值残差；不代表实际飞行精度。",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.78, wspace=0.3)
    generated = export(fig, args.output_dir, "main_formation", tuple(args.formats))

    fig, (ax, cost) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax.semilogy(slot, maximum, color=BLUE, lw=1.3, label="最大位置误差")
    ax.semilogy(slot, rms, color=ORANGE, lw=1.3, ls="--", label="位置误差均方根")
    for threshold, label in ((0.01, "1 cm"), (0.001, "1 mm")):
        ax.axhline(threshold, color="#717782", lw=0.8, ls=":")
        ax.text(199, threshold, label, fontsize=8.5, va="center")
    ax.set(
        xlim=(0, 218),
        xlabel="累计测角时隙",
        ylabel="位置误差 / m（对数轴）",
        title="(a) 逐时隙误差",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.grid(axis="y", color=GRAY, lw=0.5, which="major")
    last = summary["last_motion_slot"]
    ax.axvspan(last, slot[-1], color="#EDF0F4", zorder=-1)
    ax.text((last + slot[-1]) / 2, 0.12, "停止确认", ha="center", fontsize=8.5)
    for t, label, location in (
        (92, "92：首次低于 1 cm", (27, 0.00065)),
        (145, "145：最后移动", (68, 0.000022)),
        (196, "196：协议停止", (152, 0.00025)),
    ):
        ax.scatter([t], [maximum[t]], s=17, color=INK, zorder=5)
        ax.annotate(
            label,
            (t, maximum[t]),
            xytext=location,
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.7},
        )
    movement = np.r_[
        0, np.cumsum(np.linalg.norm(np.diff(states, axis=0), axis=2).sum(axis=1))
    ]
    cost.plot(slot, movement, color=BLUE, lw=1.3)
    cost.axvspan(last, slot[-1], color="#EDF0F4", zorder=-1)
    cost.set(
        xlim=(0, 196),
        ylim=(0, 285),
        xlabel="累计测角时隙",
        ylabel="累计端点位移 / m",
        title="(b) 移动量与测角开销",
    )
    cost.grid(axis="y", color=GRAY, lw=0.5)
    cost.set_xticks([0, 49, 92, 145, 196])
    top = cost.secondary_xaxis("top", functions=(lambda t: 4 * t, lambda c: c / 4))
    top.set_xlabel("累计发射机次（每时隙 4 架）", fontsize=9.5)
    top.set_xticks([0, 196, 368, 580, 784])
    cost.annotate(
        f"{movement[-1]:.3f} m",
        xy=(196, movement[-1]),
        xytext=(110, 225),
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.7},
    )
    cost.text(151, 60, "51 时隙确认\n204 发射机次\n无新增位移", fontsize=8.5)
    fig.suptitle("误差演化与全过程开销", x=0.08, y=1.02, ha="left", fontsize=10, fontweight="bold")
    fig.text(
        0.08,
        0.95,
        "表 1；九架圆周机；精确观测与执行；保留全部 197 个状态记录",
        fontsize=9,
    )
    fig.text(
        0.08,
        0.02,
        "端点位移按相邻记录位置计算；阴影表示最后移动后的测量与确认，不是实际飞行时间。",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.77, wspace=0.32)
    generated += export(fig, args.output_dir, "main_convergence_cost", tuple(args.formats))
    write_figure_manifest(args.output_dir, args.input_dir, generated, style_metadata)
    print(
        json.dumps(
            {
                "states": len(states),
                "max_final_error_m": float(maximum[-1]),
                "movement_m": float(movement[-1]),
                "output_dir": str(args.output_dir),
                "exports": [str(path) for path in generated],
            }
        )
    )


if __name__ == "__main__":
    main()
