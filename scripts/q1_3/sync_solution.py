"""将独立 q1_3 正文同步至第一问总文档，保持 1.1、1.2 和尾部来源段不变。"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "solutions/q1_3/README.md"
TARGET = ROOT / "solutions/q1/README.md"
TABLE = ROOT / "solutions/q1_3/实验汇总表.md"


def render_section(source: str) -> str:
    lines = []
    in_fence = False
    for line in source.splitlines():
        if line.startswith(("```", "~~~")):
            in_fence = not in_fence
        if not in_fence:
            if line.startswith("# "):
                title = re.sub(r"^问题\s*1[（(]3[）)]\s*[：:]\s*", "", line[2:])
                line = f"## 1.3 {title}"
            elif match := re.match(r"^(#{2,5}) (\d+(?:\.\d+)*)\.?\s+(.+)$", line):
                level, number, title = match.groups()
                line = f"{level}# 1.3.{number} {title}"
            elif line.startswith("##"):
                line = "#" + line
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="仅检查正文、生成表与总文档是否同步"
    )
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    target = TARGET.read_text(encoding="utf-8")
    table = TABLE.read_text(encoding="utf-8").strip()
    if table not in source:
        raise SystemExit("独立正文的汇总表与生成文件不一致，请先同步表格内容。")
    start = list(re.finditer(r"^## 1\.3 .+$", target, flags=re.MULTILINE))
    end = list(re.finditer(r"^## 证据与实现来源$", target, flags=re.MULTILINE))
    if len(start) != 1 or len(end) != 1 or start[0].start() >= end[0].start():
        raise SystemExit("总文档章节边界不唯一，未写入任何内容。")
    revised = (
        target[: start[0].start()] + render_section(source) + target[end[0].start() :]
    )
    if args.check:
        if revised != target:
            raise SystemExit("总文档 1.3 尚未同步。")
        print("q1_3 正文、实验汇总表与第一问总文档一致。")
    elif revised != target:
        TARGET.write_text(revised, encoding="utf-8")
        print("已同步第一问总文档 1.3；其余章节保持原样。")
    else:
        print("正文已同步，无需修改。")


if __name__ == "__main__":
    main()
