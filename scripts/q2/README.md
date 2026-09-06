# 第二问代码入口

[q2.py](q2.py) 是第二问的自包含实现：几何核心、参考校准与并行闭环、参考残差、几何边界、位置预算、测角噪声与增益、协议对照以及 E0 探针集中在一个文件中。只依赖 Python 标准库、NumPy 和 SciPy；可以把这个文件单独复制到其他目录运行。

模型和结论见[第二问整体方法](../../solutions/q2/README.md)。文件中的估计器只使用角度、模板和协议提供的参考坐标；仿真真值用于生成观测、施加理想位移与离线验收。

## 运行

在仓库根目录使用 Conda `agent`：

```bash
conda run -n agent python scripts/q2/q2.py triangle --output-dir scratch/q2_reproduction/triangle
conda run -n agent python scripts/q2/q2.py protocols --output-dir scratch/q2_reproduction/protocols
conda run -n agent python scripts/q2/q2.py --help
```

单独复制后，将命令中的 `scripts/q2/q2.py` 换成该文件所在路径。每个子命令都支持 `--output-dir`；相对路径按当前工作目录解释。省略子命令时显示帮助。

| 子命令 | 内容 | 默认输出目录 | 其他选项 |
|---|---|---|---|
| `triangle` | 13 初态、两档增益的基础闭环 | `outputs/q2/triangle_reference` | `--gains`、`--max-rounds` |
| `residual` | 参考传播、有限校准和多初值 | `outputs/q2/reference_residual` | — |
| `geometry` | 退化路径、最坏传播方向和几何反例 | `outputs/q2/geometry_boundaries` | — |
| `budget` | 精确底角区间、位置预算和有限次置信检查 | `outputs/q2/calibration_budget` | — |
| `noise` | 线性协方差、30 条非线性轨迹及独立增益 | `outputs/q2/noise_gain` | — |
| `protocols` | 93 例两阶段/同步协议对照 | `outputs/q2/protocol_comparison` | `--max-rounds` |
| `e0` | 静止形状局部可辨识性探针 | `outputs/q2/e0_shape` | `--double-limit` |

例如只运行单位增益基础批次，或查看预算入口：

```bash
conda run -n agent python scripts/q2/q2.py triangle --gains 1 --output-dir scratch/q2_reproduction/unit_gain
conda run -n agent python scripts/q2/q2.py budget --help
```

## 文件中的组织

按 `core → triangle → residual → geometry → budget → noise → protocols → e0` 分区，每个分区都有可搜索的标题。核心和仿真函数保持普通 Python 定义；各分区同名辅助函数及常量带前缀，以保持原来的作用域和参数。

| 用途 | 主要接口 |
|---|---|
| 模板与夹角 | `template`、`bootstrap_angles`、`receiver_angles`、`angle_jacobian` |
| 纯角度定位 | `estimate_apex`、`estimate_receiver` |
| 基础闭环 | `make_initial_state`、`run_case`、`run_batch` |
| 参考残差 | `bootstrap_jacobian`、`propagation_blocks` |
| 校准预算 | `apex_angle_box`、`equal_angle_budget`、`gaussian_angle_half_width` |
| 含噪分析 | `linear_theory`、`nonlinear_fixed_budget`、`run_analysis` |
| 分阶段和同步协议 | `run_staged_budget_case`、`run_dynamic_case`、`run_comparison` |

在 Python 中可导入这些接口。`run_case` 保留可选 `bootstrap_position_tolerance` 参数；默认仍使用原严格角度规则。`protocols` 输出分别统计在线停止和离线位置预算，`noise` 保持固定预算评分口径。

## 核验与维护

单文件曾独立复制到临时目录，在清除 `PYTHONPATH` 的环境中运行全部七个子命令；原模块版另行运行相同批次。467 个产物逐一核对：文本只统一输出目录后完全相同，NPZ 中的数组全部相同。详见[核验记录](../../outputs/q2/standalone/verification.json)。该批次覆盖基础 26 例、残差阈值 27 例、最坏方向 30 个主阶段、含噪 30 条轨迹与协议 93 例。

原模块文件保留为既有实验的对照入口，便携复现统一使用 `q2.py`。独立文件没有仓库路径注入或动态加载包装。回归测试包括隔离命令、核心接口和依赖边界：

```bash
conda run -n agent python -m pytest -q tests/q2
conda run -n agent python -m ruff check scripts/q2/q2.py tests/q2/test_standalone.py
```
