# 科研图中文字体

此目录包含 Adobe 官方 Source Han Sans SC（思源黑体简体中文）2.005 的常规与粗体字重，供 `scripts/figure_style.py` 直接加载，无需安装到系统字体目录。

上游地址、文件版本与 SHA-256 见 [sources.json](sources.json)。字体遵循 [SIL Open Font License](LICENSE.txt)。英文与数字使用系统已有 Arial，该字体不随本目录分发。

绘图时必须显式注册两个 OTF，避免另一台机器或新的 Matplotlib 缓存悄悄换成其他中文字体。
