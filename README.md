# Localization

本项目围绕 2022 年高教社杯全国大学生数学建模竞赛 B 题“无人机遂行编队飞行中的纯方位无源定位”，维护题面、分问解答、计算脚本、验证记录和可复现结果。

当前已经完成问题 1（1）的定位基线：根据三架编号与位置已知的发射机，利用点积角度约束和非线性最小二乘恢复接收机位置。问题 1（2）、问题 1（3）和问题 2 尚未开始。

## 从哪里开始

1. 阅读 [`problem/B题.md`](problem/B题.md)，查看完整题面。
2. 阅读 [`solutions/q1_1/README.md`](solutions/q1_1/README.md)，查看问题 1（1）的正式解答。
3. 阅读 [`docs/q1_1/验证记录.md`](docs/q1_1/验证记录.md)，查看数值实验、多解边界和证据范围。

## 目录

```text
.
├── problem/                 # 题面 Markdown 与正文图片
├── solutions/
│   └── q1_1/               # 问题 1（1）正式解答
├── scripts/
│   └── q1_1/               # 定位算法与实验入口
├── tests/
│   └── q1_1/               # 自动测试
├── docs/
│   └── q1_1/               # 推导补充与验证记录
├── outputs/
│   └── q1_1/               # 固定种子的实验结果
├── requirements.txt
└── README.md
```

`工作总账.md` 是本地协作状态文件，已加入 `.gitignore`，不属于正式项目交付物。

## 安装与运行

```bash
python -m pip install -r requirements.txt
python -m pytest
python scripts/q1_1/run_validation.py
```

实验默认使用表 1 数据、$0.1^\circ$ 独立高斯角度噪声、每个接收机 200 次试验和固定随机种子 20220904，结果写入 `outputs/q1_1/`。
