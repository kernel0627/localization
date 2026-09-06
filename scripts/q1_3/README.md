# 问题 1（3）代码入口

正文见 [四机发射与轮换互校的编队调整](../../solutions/q1_3/README.md)，公式、图表与计算结果的对应关系见 [资料索引](../../docs/q1_3/资料索引.md)。从仓库根目录运行以下命令，Python 使用 Conda `agent` 环境。

## 整理正文与图表

```bash
conda run --no-capture-output -n agent python -m scripts.q1_3.build_paper_assets
conda run --no-capture-output -n agent python -m scripts.q1_3.sync_solution
conda run --no-capture-output -n agent python -m scripts.q1_3.sync_solution --check
```

第一条只读取现有 CSV，生成 `figures/q1_3/paper/` 内两张中文 PNG、同口径噪声数据与输入散列清单，以及 `solutions/q1_3/实验汇总表.md`。默认输出为 PNG。字体从项目 `assets/fonts/source-han-sans/` 加载；英文和数字使用 Arial。旧的详细绘图脚本和图片继续保留。

`solutions/q1_3/README.md` 是正文编辑源。表格数据若更新，先把生成表的内容写入正文，再运行同步命令；`sync_solution` 检查表格一致性，只替换第一问总文档的 1.3，保留 1.1、1.2 和末尾来源段。总文档的该节不用手工重复修改。

## 控制器与实验

| 用途 | 入口 | 输出或依赖 |
|---|---|---|
| 本机角度估计、参考选择、修正与保持 | [local_adjustment.py](local_adjustment.py) | 公共控制核心；仅接收本机角度和预设信息 |
| 全配置轮换，表 1 单次运行 | [run_adjustment.py](run_adjustment.py) | 默认写 `outputs/q1_3/`，支持 `--output-dir` |
| 全配置轨迹执行与评价 | [run_iterative_reference_baseline.py](run_iterative_reference_baseline.py) | 同步模拟与离线真值评价 |
| 双配置历史调度与模拟 | [optimize_schedule.py](../../appendix1/optimize_schedule.py) | 本地归档中的既有实现，分析脚本仍通过兼容链接导入 |
| 全配置随机与噪声批次 | [run_robustness.py](run_robustness.py) | `outputs/q1_3/robustness/`，505 条记录 |
| 双配置随机与噪声批次 | [run_two_configuration_robustness.py](run_two_configuration_robustness.py) | `outputs/q1_3/two_configuration_robustness/`，1616 条记录；包含冻结批次复用 |
| 白噪声与相对执行误差 | [simulation_noise.py](simulation_noise.py) | 仿真器私有扰动 |
| 双配置联合噪声与固定链路偏置 | [two_configuration_noise.py](two_configuration_noise.py) | 同一链路偏置在一次运行内保持不变 |

全配置表 1 可以另存运行，以免覆盖已有正文数据：

```bash
conda run --no-capture-output -n agent python -m scripts.q1_3.run_adjustment --output-dir scratch/q1_3_reproduction/main
```

完整随机批次的原入口如下；这是重跑或恢复实验，重组正文时无需执行。

```bash
conda run --no-capture-output -n agent python -m scripts.q1_3.run_robustness --trials 100 --workers 4
conda run --no-capture-output -n agent python -m scripts.q1_3.run_two_configuration_robustness --trials 100 --workers 4
```

双配置仿真与分析通过 `appendix1/optimize_schedule.py` 调用调度模拟器；双配置批次还依赖 `appendix1/evaluation_560/` 内 505 条冻结记录及源文件散列。`appendix1` 指向被 Git 忽略的本地 `scratch` 归档，因此运行这些入口需同时携带归档。正式 `outputs/q1_3/` 内已保存用于写作的表 1、随机实验与数学分析结果。

配置的选择过程见[双配置选择依据](../../docs/q1_3/双配置选择依据.md)。已有的 210 组局部排序与候选轨迹比较表保存于 `outputs/q1_3/two_configuration_selection/`，可直接核对选取 04/05 → 07/08 的依据；两份表是原有筛选结果的副本。

## 数学分析与完整报告

| 分析层次 | 数学实现 | 结果目录（位于 `outputs/q1_3/`） |
|---|---|---|
| 全配置参考误差传播与周期稳定性 | [analyze_iterative_reference.py](analyze_iterative_reference.py) | `main_analysis/` |
| 全配置白噪声平台 | [analyze_noise_floor.py](analyze_noise_floor.py) | `noise_analysis/` |
| 双配置结构、周期稳定性与有限轨迹 | [analyze_two_configuration.py](analyze_two_configuration.py) | `two_configuration_analysis/` |
| 白噪声、固定偏置和乘性误差二阶矩 | [analyze_two_configuration_robustness.py](analyze_two_configuration_robustness.py) | `two_configuration_robustness_math/` |
| 随机初态矩传播 | [analyze_two_configuration_random_initial.py](analyze_two_configuration_random_initial.py) | `two_configuration_robustness_math/` |
| Gaussian 局部全队精度概率 | [analyze_two_configuration_precision_probability.py](analyze_two_configuration_precision_probability.py) | `two_configuration_robustness_math/` |
| 全配置完整汇总与补充图 | [report_robustness.py](report_robustness.py) | `robustness/` |
| 双配置及三方法完整统计 | [report_two_configuration_robustness.py](report_two_configuration_robustness.py) | `two_configuration_robustness_report/` |

这些模块均可用 `conda run --no-capture-output -n agent python -m scripts.q1_3.<模块名>` 运行。先得到轨迹和周期矩阵，再计算噪声数学模型、随机初态矩与概率，最后汇总报告。`analyze_two_configuration` 默认也会运行表 1 参数轨迹；只分析矩阵可用 `--linear-only --output-dir scratch/q1_3_reproduction/two_linear`。详细报告脚本会重写其对应报告，不会更新精简正文。

旧的 `plot_main_results.py`、`plot_two_configuration_results.py` 与两个 `report_*` 继续负责完整补充图；正文两图统一由 `build_paper_assets.py` 输出，不修改这些历史绘图入口的默认行为。

## 复核

```bash
conda run --no-capture-output -n agent python -m scripts.q1_3.verify_two_configuration_robustness_report
conda run --no-capture-output -n agent python -m pytest tests/q1_3
```

前者复核已保存的统计、样本覆盖和相位对应，写入报告审查文件；后者用于控制器或数学实现发生修改时的回归检查。正文重排和纯出图不需要重跑全部随机实验。历史 bootstrap、联合定位基线、方案扫描及阶段性审查的脚本保留原位置，通过资料索引查阅。
