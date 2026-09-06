"""Build the three Q2 paper figures from frozen experiment outputs.

The script only reads the existing Q2 logs and the shared figure style.  It does
not run a localization or formation-adjustment experiment.  The trajectory
shown in ``formation_adjustment`` is reconstructed from the action log by
subtracting each logged action from its post-action position.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/q2-paper-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

# Make ``conda run -n agent python scripts/q2/build_paper_assets.py`` work
# from the repository root as well as ``python -m``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figure_style import configure_style  # noqa: E402
from scripts.q2.q2 import TARGET_TEMPLATE  # noqa: E402


ROOT = _REPO_ROOT
DEFAULT_OUTPUT = ROOT / "figures/q2/paper"
CASE_DIR = (
    ROOT
    / "outputs/q2/protocol_comparison/cases/case_084_fair_shared_position_budget"
)
THRESHOLD_PATH = ROOT / "outputs/q2/reference_residual/threshold_comparison.csv"
METHOD_COLORS = {"reference": "#0072B2", "receiver": "#D55E00", "target": "#555555"}
SEED_COLORS = {11: "#0072B2", 23: "#D55E00", 47: "#009E73"}
RULES = ("angle_1e-8", "position_0.005d", "position_0.01d")
RULE_LABELS = ("严格角度校准", "位置 0.005d", "位置 0.01d")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_figure(fig: plt.Figure, name: str, output: Path) -> list[Path]:
    paths = [output / f"{name}.{suffix}" for suffix in ("png", "svg", "pdf")]
    for path in paths:
        if path.suffix == ".png":
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return paths


def target_rows() -> np.ndarray:
    return np.asarray(TARGET_TEMPLATE, dtype=float).copy()


def reconstruct_formation() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, str]]]:
    """Recover initial, reference-lock, and final states from one action log."""
    action_path = CASE_DIR / "actions.csv"
    final_path = CASE_DIR / "final_positions.csv"
    summary_path = CASE_DIR / "summary.json"
    actions = read_csv(action_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target = target_rows()

    initial = np.full((15, 2), np.nan, dtype=float)
    first_by_id: dict[int, np.ndarray] = {}
    bootstrap_after: np.ndarray | None = None
    bootstrap_round = -1
    for row in actions:
        receiver_id = int(row["receiver_id"])
        after = np.array(
            [float(row["true_position_after_x"]), float(row["true_position_after_y"])],
            dtype=float,
        )
        delta = np.array([float(row["delta_x"]), float(row["delta_y"])], dtype=float)
        before = after - delta
        if receiver_id not in first_by_id:
            first_by_id[receiver_id] = before
        if row["stage"] == "bootstrap" and int(row["round"]) >= bootstrap_round:
            bootstrap_round = int(row["round"])
            bootstrap_after = after

    for receiver_id, position in first_by_id.items():
        initial[receiver_id - 1] = position
    # The fixed bottom anchors have no action rows and are part of the gauge.
    initial[10] = target[10]
    initial[14] = target[14]
    if not np.isfinite(initial).all() or bootstrap_after is None:
        raise ValueError("action log does not contain a complete recoverable state")
    lock = initial.copy()
    lock[0] = bootstrap_after

    final_rows = np.loadtxt(final_path, delimiter=",")
    final = np.asarray(final_rows, dtype=float).reshape(15, 2)
    if not np.isfinite(final).all():
        raise ValueError("final_positions.csv contains non-finite values")

    # Replay every logged action in order and require the archived final state
    # to be exactly the endpoint of that same trajectory.
    replayed = initial.copy()
    for row in actions:
        receiver_id = int(row["receiver_id"]) - 1
        delta = np.array([float(row["delta_x"]), float(row["delta_y"])], dtype=float)
        replayed[receiver_id] += delta
        logged_after = np.array(
            [float(row["true_position_after_x"]), float(row["true_position_after_y"])],
            dtype=float,
        )
        np.testing.assert_allclose(replayed[receiver_id], logged_after, rtol=0, atol=5e-13)
    np.testing.assert_allclose(replayed, final, rtol=0, atol=5e-13)

    initial_errors = np.linalg.norm(initial - target, axis=1)
    lock_errors = np.linalg.norm(lock - target, axis=1)
    final_errors = np.linalg.norm(final - target, axis=1)
    action_distance = sum(
        float(np.linalg.norm([float(row["delta_x"]), float(row["delta_y"])]))
        for row in actions
    )
    summary_stats = {
        "case_dir": str(CASE_DIR.relative_to(ROOT)),
        "rho": float(summary["rho"]),
        "seed": int(summary["seed"]),
        "gain": float(summary["eta_c"]),
        "protocol": summary["protocol"],
        "budget_group": summary["budget_group"],
        "stop_rule": summary["stop_rule"],
        "initial_max_error_d": float(initial_errors.max()),
        "initial_rms_error_d": float(np.sqrt(np.mean(initial_errors**2))),
        "lock_reference_error_d": float(lock_errors[0]),
        "lock_max_error_d": float(lock_errors.max()),
        "lock_rms_error_d": float(np.sqrt(np.mean(lock_errors**2))),
        "final_max_error_d": float(final_errors.max()),
        "final_rms_error_d": float(np.sqrt(np.mean(final_errors**2))),
        "measurement_slots": int(summary["measurement_slots"]),
        "tx_uses": int(summary["tx_uses"]),
        "action_count": len(actions),
        "bootstrap_action_count": int(summary["bootstrap"]["action_count"]),
        "main_action_count": sum(row["stage"] == "main_budget" for row in actions),
        "cumulative_displacement_d": action_distance,
        "summary_cumulative_displacement_d": float(summary["cumulative_displacement_d"]),
        "receiver_threshold_d": float(summary["receiver_estimate_budget_d"]),
        "reference_threshold_d": float(summary["tau_c_d"]),
    }
    np.testing.assert_allclose(
        [summary_stats["initial_max_error_d"], summary_stats["final_max_error_d"]],
        [float(summary["initial_max_error_d"]), float(summary["final_max_position_error_d"])],
        rtol=0,
        atol=5e-12,
    )
    np.testing.assert_allclose(
        summary_stats["cumulative_displacement_d"],
        summary_stats["summary_cumulative_displacement_d"],
        rtol=0,
        atol=5e-12,
    )
    return initial, lock, final, summary_stats, actions


def draw_target_grid(ax: plt.Axes, target: np.ndarray) -> None:
    """Show the canonical five rows as quiet guides in every trajectory panel."""
    for row in range(5):
        start = row * (row + 1) // 2
        points = target[start : start + row + 1]
        ax.plot(points[:, 0], points[:, 1], color="#B8B8B8", lw=0.65, alpha=0.8)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            marker="+",
            s=18,
            color="#808080",
            linewidths=0.65,
            zorder=1,
        )


def formation_figure(
    initial: np.ndarray,
    lock: np.ndarray,
    final: np.ndarray,
    stats: dict[str, Any],
    output: Path,
) -> tuple[list[Path], Path, dict[str, Any]]:
    target = target_rows()
    states = (initial, lock, final)
    titles = ("(a) 初态", "(b) FY01 锁定", "(c) 主阶段末态")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.20), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.82, wspace=0.07)
    xlim = (-0.34, 4.34)
    ylim = (-0.34, 3.82)
    for panel, (ax, state, title) in enumerate(zip(axes, states, titles)):
        draw_target_grid(ax, target)
        receiver = np.ones(15, dtype=bool)
        receiver[[0, 10, 14]] = False
        ax.scatter(
            state[receiver, 0],
            state[receiver, 1],
            s=19,
            color=METHOD_COLORS["receiver"],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            state[[0, 10, 14], 0],
            state[[0, 10, 14], 1],
            s=26,
            color=METHOD_COLORS["reference"],
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
        for index, (x, y) in enumerate(state, 1):
            if index in (11, 15):
                continue
            ax.annotate(
                f"{index:02d}",
                (x, y),
                xytext=(3.0, 2.0),
                textcoords="offset points",
                fontsize=6.0,
                color="#333333",
                zorder=5,
            )
        if panel == 1:
            ax.annotate(
                "锁定",
                state[0],
                xytext=(8, 7),
                textcoords="offset points",
                fontsize=7.0,
                color=METHOD_COLORS["reference"],
                arrowprops={"arrowstyle": "-", "color": METHOD_COLORS["reference"], "lw": 0.7},
            )
        ax.set_title(title, loc="left", fontsize=9.3, pad=8)
        if panel == 0:
            metric = rf"$E_{{\max}}={stats['initial_max_error_d']:.3f}d$"
        elif panel == 2:
            metric = rf"$E_{{\max}}={stats['final_max_error_d']:.4f}d$"
        else:
            metric = f"残差={stats['lock_reference_error_d']:.4f}d"
        ax.text(0.02, 0.99, metric, transform=ax.transAxes, ha="left", va="top", fontsize=9.0, color="#111111")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 2, 4])
        ax.set_yticks([0, 2, 3.46])
        ax.grid(True, lw=0.45, alpha=0.25)
        if panel == 0:
            ax.set_ylabel("y / d")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("x / d")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["receiver"], markeredgecolor="white", markersize=5.5, label="12 架接收机"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["reference"], markeredgecolor="white", markersize=5.5, label="FY01 / FY11 / FY15"),
        Line2D([0], [0], marker="+", color="#808080", markersize=6, lw=0.7, label="目标槽位"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.005), ncol=3, columnspacing=0.8, handletextpad=0.35)
    fig.suptitle("三角编队的调整过程", x=0.075, y=0.985, ha="left", fontsize=10.3, fontweight="bold")
    fig.text(0.57, 0.985, r"$(\rho=0.1,\ \mathrm{seed}=23,\ \eta_C=\eta_R=0.5)$", ha="left", va="top", fontsize=9.0)
    paths = export_figure(fig, "formation_adjustment", output)

    rows: list[dict[str, Any]] = []
    for panel_name, state in zip(("initial", "reference_locked", "final"), states):
        errors = np.linalg.norm(state - target, axis=1)
        for index, ((x, y), error) in enumerate(zip(state, errors), 1):
            rows.append(
                {
                    "panel": panel_name,
                    "drone_id": index,
                    "x_d": f"{x:.15g}",
                    "y_d": f"{y:.15g}",
                    "error_d": f"{error:.15g}",
                }
            )
    data_path = output / "formation_positions.csv"
    write_csv(data_path, rows)
    return paths, data_path, {
        "rows": len(rows),
        "panels": ["initial", "reference_locked", "final"],
        "xlim_d": list(xlim),
        "ylim_d": list(ylim),
        "equal_aspect": True,
        "stats": stats,
    }


def method_overview_figure(output: Path) -> list[Path]:
    """Draw a compact, two-panel protocol schematic with explicit roles."""
    fig, ax = plt.subplots(figsize=(7.4, 3.75))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.055, 0.955, "Q2 两阶段参考建立与并行定位", fontsize=11.4, fontweight="bold", va="top")
    ax.text(0.055, 0.905, "先用两底角校准 FY01，再用三个角点服务十二架接收机。", fontsize=8.4, color="#444444", va="top")

    ax.add_patch(FancyBboxPatch((0.045, 0.16), 0.42, 0.66, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor="#EAF3F8", edgecolor="#72A8C2", lw=1.0))
    ax.add_patch(FancyBboxPatch((0.535, 0.16), 0.42, 0.66, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor="#F8F2EA", edgecolor="#D6A56E", lw=1.0))
    ax.text(0.065, 0.775, "阶段 1 · 校准公共参考", color=METHOD_COLORS["reference"], fontsize=9.6, fontweight="bold")
    ax.text(0.555, 0.775, "阶段 2 · 三参考广播与并行反馈", color="#A45B18", fontsize=9.6, fontweight="bold")

    # Left: actual FY01 triangle, target marker, and bottom-angle arcs.
    a = np.array([0.12, 0.51])
    b = np.array([0.37, 0.51])
    c = np.array([0.245, 0.69])
    c0 = np.array([0.245, 0.64])
    ax.plot([a[0], b[0]], [a[1], b[1]], color="#666666", lw=1.0)
    ax.plot([a[0], c[0]], [a[1], c[1]], color="#1775A8", lw=1.0)
    ax.plot([b[0], c[0]], [b[1], c[1]], color="#1775A8", lw=1.0)
    ax.plot([a[0], c0[0]], [a[1], c0[1]], color="#8FB9CC", lw=0.8, ls="--")
    ax.plot([b[0], c0[0]], [b[1], c0[1]], color="#8FB9CC", lw=0.8, ls="--")
    ax.scatter([a[0], b[0]], [a[1], b[1]], s=43, color=METHOD_COLORS["reference"], zorder=3)
    ax.scatter(*c, s=43, color=METHOD_COLORS["receiver"], zorder=3)
    ax.scatter(*c0, s=53, marker="x", color="#555555", lw=1.2, zorder=3)
    angle_a = float(np.degrees(np.arctan2(c[1] - a[1], c[0] - a[0])))
    angle_b = float(np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0])))
    ax.add_patch(Arc(a, 0.085, 0.085, theta1=0, theta2=angle_a, color="#0072B2", lw=1.0))
    ax.add_patch(Arc(b, 0.085, 0.085, theta1=angle_b, theta2=180, color="#0072B2", lw=1.0))
    ax.text(0.153, 0.545, "α", fontsize=9.2, color="#0072B2", fontweight="bold")
    ax.text(0.337, 0.545, "β", fontsize=9.2, color="#0072B2", fontweight="bold")
    ax.text(a[0], a[1] - 0.045, "FY11", ha="center", fontsize=7.3)
    ax.text(b[0], b[1] - 0.045, "FY15", ha="center", fontsize=7.3)
    ax.text(c[0] + 0.027, c[1] + 0.012, "FY01", fontsize=7.3, color="#A45B18")
    ax.text(c0[0] + 0.027, c0[1] - 0.012, "目标 C0", fontsize=7.0, color="#555555")
    ax.text(0.065, 0.425, "底角弧线给出 α、β；虚线为目标参考。", fontsize=7.0, color="#555555")
    for y, text_value in ((0.355, "槽 1：FY01 / FY15 发射 → FY11 接收，观测 α"), (0.295, "槽 2：FY01 / FY11 发射 → FY15 接收，观测 β")):
        ax.add_patch(Rectangle((0.065, y), 0.38, 0.043, facecolor="white", edgecolor="#72A8C2", lw=0.7))
        ax.text(0.255, y + 0.0215, text_value, ha="center", va="center", fontsize=6.9)
    ax.text(0.065, 0.205, r"$\hat{C}(\alpha,\beta)$", fontsize=7.2, color="#444444", va="center")
    ax.text(0.135, 0.205, "→ 反馈 FY01；满足锁定条件后进入阶段 2。", fontsize=6.9, color="#444444", va="center")

    # Right: full five-layer template, with only one representative ray fan.
    target = target_rows()
    right_points = np.column_stack((0.585 + target[:, 0] / 4.0 * 0.16, 0.335 + target[:, 1] / (2 * np.sqrt(3)) * 0.34))
    for row in range(5):
        start = row * (row + 1) // 2
        group = right_points[start : start + row + 1]
        ax.plot(group[:, 0], group[:, 1], color="#C9A77D", lw=0.55, alpha=0.7)
    ref_ids = np.array([1, 11, 15]) - 1
    receiver_ids = np.array([i for i in range(15) if i not in ref_ids])
    ax.scatter(right_points[receiver_ids, 0], right_points[receiver_ids, 1], s=24, color=METHOD_COLORS["receiver"], zorder=3)
    ax.scatter(right_points[ref_ids, 0], right_points[ref_ids, 1], s=35, color=METHOD_COLORS["reference"], zorder=4)
    for index, label in zip(ref_ids, ("FY01", "FY11", "FY15")):
        p = right_points[index]
        ax.text(p[0] + 0.012, p[1] + (0.028 if label == "FY01" else -0.025), label, fontsize=6.8, color=METHOD_COLORS["reference"])
    representative = right_points[7]
    ax.scatter(*representative, s=58, facecolor="none", edgecolor="#A45B18", lw=1.0, zorder=5)
    ax.text(representative[0] + 0.015, representative[1] + 0.02, "FY08（示例）", fontsize=6.8, color="#A45B18")
    for index in ref_ids:
        ax.add_patch(FancyArrowPatch(right_points[index], representative, arrowstyle="-", lw=0.75, ls="--", color="#A45B18", alpha=0.75))
    ax.text(0.78, 0.68, "三参考广播", fontsize=8.0, color="#A45B18", fontweight="bold")
    ax.text(0.78, 0.645, "FY01 / FY11 / FY15", fontsize=7.0, color="#555555")
    ax.text(0.78, 0.575, "12 架分别定位", fontsize=8.0, color="#A45B18", fontweight="bold")
    ax.text(0.78, 0.54, "同一观测批次 · 同步修正", fontsize=7.0, color="#555555")
    ax.add_patch(Rectangle((0.575, 0.235), 0.37, 0.06, facecolor="white", edgecolor="#D6A56E", lw=0.75))
    ax.text(0.76, 0.265, "广播  →  定位  →  修正  →  复测", ha="center", va="center", fontsize=7.7, color="#A45B18", fontweight="bold")
    ax.scatter([0.59], [0.205], s=22, color=METHOD_COLORS["reference"])
    ax.text(0.605, 0.205, "三参考", va="center", fontsize=6.8)
    ax.scatter([0.70], [0.205], s=22, color=METHOD_COLORS["receiver"])
    ax.text(0.715, 0.205, "十二架接收机", va="center", fontsize=6.8)

    ax.add_patch(FancyArrowPatch((0.47, 0.50), (0.525, 0.50), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#555555"))
    ax.text(0.497, 0.535, "锁定", ha="center", fontsize=7.0, color="#555555")
    return export_figure(fig, "method_overview", output)


def calibration_comparison_figure(output: Path) -> tuple[list[Path], Path, dict[str, Any]]:
    rows = read_csv(THRESHOLD_PATH)
    selected = [
        row
        for row in rows
        if float(row["gain"]) == 0.5 and int(row["seed"]) in (11, 23, 47) and row["calibration_rule"] in RULES
    ]
    selected.sort(key=lambda row: (RULES.index(row["calibration_rule"]), int(row["seed"])))
    if len(selected) != 9:
        raise ValueError(f"expected nine gain=.5 rows, got {len(selected)}")
    selected_path = output / "calibration_rule_comparison.csv"
    # Preserve the original nine columns and row values for traceability.
    write_csv(selected_path, selected)

    fig, (ax_slots, ax_error) = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.27, top=0.82, wspace=0.33)
    x = np.arange(3, dtype=float)
    jitter = {11: -0.12, 23: 0.0, 47: 0.12}
    for seed in (11, 23, 47):
        for index, rule in enumerate(RULES):
            row = next(item for item in selected if int(item["seed"]) == seed and item["calibration_rule"] == rule)
            xpos = index + jitter[seed]
            color = SEED_COLORS[seed]
            ax_slots.scatter(xpos, float(row["complete_slots"]), s=31, color=color, edgecolor="white", linewidth=0.55, zorder=3)
            ax_error.scatter(xpos, float(row["final_max_position_error_d"]), s=31, color=color, edgecolor="white", linewidth=0.55, zorder=3)
    ax_slots.set_title("(a) 完整协议槽数", loc="left", fontsize=9.6)
    ax_error.set_title("(b) 全队最大终点误差", loc="left", fontsize=9.6)
    ax_slots.set_ylabel("$N_{\\mathrm{slot}}$")
    ax_error.set_ylabel(r"$E_{\max}/d$")
    for ax in (ax_slots, ax_error):
        ax.set_xticks(x)
        ax.set_xticklabels(RULE_LABELS, rotation=0)
        ax.set_xlim(-0.42, 2.42)
        ax.grid(axis="y", lw=0.45, alpha=0.35)
    ax_slots.set_ylim(0, 82)
    ax_slots.set_yticks([0, 20, 40, 60, 80])
    ax_error.set_yscale("log")
    ax_error.set_ylim(1e-9, 2e-2)
    ax_error.axhline(0.01, color="#555555", ls="--", lw=0.85)
    ax_error.text(2.39, 0.0104, "0.01d 验收线", ha="right", va="bottom", fontsize=7.0, color="#555555")
    legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor=SEED_COLORS[seed], markeredgecolor="white", markersize=5.5, label=f"seed={seed}") for seed in (11, 23, 47)]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.53, 0.07), ncol=3, columnspacing=1.0, handletextpad=0.35)
    fig.suptitle("不同参考校准规则下的最终精度与测角开销", x=0.10, y=0.98, ha="left", fontsize=10.4, fontweight="bold")
    fig.text(0.72, 0.98, r"$(\eta_C=\eta_R=0.5,\ \rho=0.1)$", ha="left", va="top", fontsize=9.0)
    paths = export_figure(fig, "calibration_rule_comparison", output)
    stats = {
        "rows": len(selected),
        "gain": 0.5,
        "seeds": [11, 23, 47],
        "rules": list(RULES),
        "complete_slots": {rule: [int(r["complete_slots"]) for r in selected if r["calibration_rule"] == rule] for rule in RULES},
        "final_max_error_d": {rule: [float(r["final_max_position_error_d"]) for r in selected if r["calibration_rule"] == rule] for rule in RULES},
        "error_axis": "log10",
        "acceptance_line_d": 0.01,
        "point_encoding": "three seed points per categorical rule; fixed horizontal jitter; no connecting lines",
    }
    return paths, selected_path, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style = configure_style()

    initial, lock, final, formation_stats, actions = reconstruct_formation()
    method_paths = method_overview_figure(args.output_dir)
    formation_paths, formation_data_path, formation_data_stats = formation_figure(initial, lock, final, formation_stats, args.output_dir)
    comparison_paths, comparison_data_path, comparison_stats = calibration_comparison_figure(args.output_dir)

    inputs = [
        CASE_DIR / "actions.csv",
        CASE_DIR / "final_positions.csv",
        CASE_DIR / "summary.json",
        THRESHOLD_PATH,
        ROOT / "scripts/figure_style.py",
        ROOT / "scripts/q2/q2.py",
    ]
    manifest = {
        "purpose": "Q2论文三图；仅读取冻结输出，不重跑定位算法",
        "script_sha256": digest(Path(__file__)),
        "style": style,
        "inputs_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "data_sources": {
            "method_overview": "solutions/q2/README.md §§2–3 and protocol contract; schematic only",
            "formation_adjustment": {
                "case": formation_stats["case_dir"],
                "reconstruction": "initial = first post-action position − logged action; reference_locked = initial with FY01 set to final bootstrap action; final = final_positions.csv",
                "action_rows": len(actions),
                "data_csv": str(formation_data_path.relative_to(ROOT)),
                "stats": formation_data_stats,
            },
            "calibration_rule_comparison": {
                "source": str(THRESHOLD_PATH.relative_to(ROOT)),
                "filter": "gain=0.5, seeds={11,23,47}; calibration_rule in {angle_1e-8, position_0.005d, position_0.01d}",
                "data_csv": str(comparison_data_path.relative_to(ROOT)),
                "stats": comparison_stats,
            },
        },
        "exports_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in [*method_paths, *formation_paths, formation_data_path, *comparison_paths, comparison_data_path]
        },
        "figure_files": {
            "method_overview": [str(path.relative_to(ROOT)) for path in method_paths],
            "formation_adjustment": [str(path.relative_to(ROOT)) for path in formation_paths],
            "calibration_rule_comparison": [str(path.relative_to(ROOT)) for path in comparison_paths],
        },
        "font_and_style": {
            "font_family": style["latin_font"] + " + " + style["chinese_font"],
            "font_size_pt": style["font_size_pt"],
            "dpi": style["dpi"],
            "pdf_fonttype": 42,
            "svg_fonttype": "none",
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": manifest["figure_files"], "data": [str(formation_data_path), str(comparison_data_path)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
