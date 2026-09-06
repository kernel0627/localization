"""科研图共用样式：项目内思源黑体、系统 Arial 与明确的字号层级。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]


def configure_style() -> dict[str, Any]:
    """统一中文字体、最终插图字号和线型；记录实际字体而非假定已安装。"""
    folder = ROOT / "assets/fonts/source-han-sans"
    font_paths = [
        folder / f"SourceHanSansSC-{weight}.otf" for weight in ("Regular", "Bold")
    ]
    for path in font_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"缺少项目字体：{path}；请按 assets/fonts/source-han-sans/sources.json 下载。"
            )
        font_manager.fontManager.addfont(str(path))
    chinese = "Source Han Sans SC"
    available = {font.name for font in font_manager.fontManager.ttflist}
    if "Arial" not in available:
        raise RuntimeError("缺少 Arial 字体；请安装已授权的 Arial 后重新绘图。")
    latin = "Arial"
    plt.rcParams.update(
        {
            "font.family": [latin, chinese],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 9,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.titlesize": 11,
            "figure.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#555555",
            "axes.axisbelow": True,
            "lines.linewidth": 1.3,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.4,
            "hatch.linewidth": 0.6,
            "mathtext.fontset": "stixsans",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return {
        "style_script_path": str(Path(__file__).relative_to(ROOT)),
        "style_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fonts": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in font_paths
        },
        "chinese_font": chinese,
        "latin_font": latin,
        "dpi": 300,
        "font_size_pt": 9,
        "axes_label_size_pt": 9.5,
        "tick_legend_size_pt": 8.5,
        "panel_title_size_pt": 10,
        "main_line_width_pt": 1.3,
        "axes_line_width_pt": 0.8,
    }
