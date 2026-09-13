"""从同学论文 PDF 附录还原源代码（x 坐标法，缩进判定精确）。

为什么不用"数空格"
------------------
按空格数推层级会抖动：同一层级实测 4~7 个空格，层级间隔约 5，在 7 附近两义，
进而出现 "unexpected indent" / "expected an indented block"，且无法机械收敛。
本脚本改用 PDF 文本块的**精确坐标**（文本矩阵 tm[4] 为横坐标）：

    缩进层级 = round( (该行代码块横坐标 − 该页代码最左横坐标) / 单位缩进 )

单位缩进取同一页内各层级横坐标差的最小值。坐标本身无抖动，故层级判定可靠。
"""
from __future__ import annotations

import ast
import collections
import io
import os
import re

import pypdf

PDF = "f:/B_论文_final.pdf"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tjj组代码")

SECTIONS = [
    ("附录A", "q1.py"),
    ("B.1", "q2.py"),
    ("B.2", "q2_decision.py"),
    ("B.3", "q2_evaluation.py"),
    ("B.4", "q2_evidence.py"),
    ("附录C", "q3.py"),
    ("附录D", "q4.py"),
]
HEAD_RE = re.compile(r"^\s*(附录\s*[A-D]|B\.[1-9]|C\.[1-9]|D\.[1-9])")
NUM_RE = re.compile(r"^\s*(\d{1,4})\s*$")


def page_rows(page):
    """把一页拆成"行"：每行 = [(x, text), ...]，按 y 归并、按 x 排序。"""
    chunks = []

    def visitor(text, cm, tm, font_dict, font_size):
        if text.strip():
            chunks.append((round(tm[5], 0), round(tm[4], 2), text))

    page.extract_text(visitor_text=visitor)
    grouped = collections.defaultdict(list)
    for y, x, t in chunks:
        grouped[y].append((x, t))
    return [(y, sorted(grouped[y])) for y in sorted(grouped, reverse=True)]


def rows_to_items(rows):
    """每行拆成 (行号或None, 代码首块横坐标或None, 该行文本)。"""
    items = []
    for _, chunks in rows:
        if not chunks:
            continue
        head = chunks[0][1].strip()
        if not head:
            continue
        num, body = None, []
        for x, text in chunks:
            if num is None and NUM_RE.match(text):
                num = int(NUM_RE.match(text).group(1))
                continue
            body.append((x, text))
        if not body:
            items.append((None, None, ""))
            continue
        items.append((num, body[0][0], "".join(t for _, t in body)))
    return items


def calibrate(reader):
    """全局标定代码列的左边界 x0 与一级缩进的宽度 unit（点）。

    实测所有代码行的横坐标恰为 x0 + k·12.15，k 为缩进层级。必须**全局标定**：
      * x0 取全部页的最小代码横坐标——若逐页取页内最小值，某些"整页都在块内"
        的页面会把 x0 抬高一级，导致该页所有行的层级整体下移一级；
      * unit 取全局最小正差——若逐页估计，某页若无一级缩进行，其页内最小差
        等于 2 个单位，会把整页层级减半。
    """
    xs = []
    for page_index in range(33, len(reader.pages)):
        items = rows_to_items(page_rows(reader.pages[page_index]))
        xs += [x for num, x, txt in items
               if num is not None and x is not None and txt.strip()]
    if not xs:
        return 0.0, 12.15
    x0 = min(xs)
    positive = sorted(round(x - x0, 2) for x in xs if x - x0 > 0.05)
    unit = positive[0] if positive else 12.15
    return x0, unit


def assign_levels(items, x0, unit):
    out = []
    for num, x, txt in items:
        if x is None or num is None:
            out.append((num, 0, txt))
            continue
        level = int(round((x - x0) / unit)) if x > x0 else 0
        out.append((num, level, txt))
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    reader = pypdf.PdfReader(PDF)
    unit = calibrate(reader)
    x0, unit = unit
    print("标定：代码列左边界 x0 = %.2f pt，一级缩进 = %.2f pt" % (x0, unit))
    bucket = collections.defaultdict(list)
    current = None
    for page_index in range(33, len(reader.pages)):
        page = reader.pages[page_index]
        for num, level, txt in assign_levels(rows_to_items(page_rows(page)),
                                             x0, unit):
            stripped = txt.strip()
            if not stripped:
                continue
            if "题程序附录" in stripped:
                continue
            if HEAD_RE.match(stripped):
                key = stripped.replace(" ", "")
                for prefix, name in SECTIONS:
                    if key.startswith(prefix):
                        current = name
                        break
                continue
            if current is None:
                continue
            bucket[current].append(" " * (4 * level) + stripped)

    print("%-20s %7s %7s %s" % ("文件", "行数", "字符", "语法校验"))
    for _, name in SECTIONS:
        lines = bucket.get(name, [])
        if not lines:
            print("%-20s %7d %7s %s" % (name, 0, "-", "未提取到"))
            continue
        src = "\n".join(
            l.replace(chr(0xFFE8), "|").replace(chr(0x2011), "-")
            for l in lines) + "\n"
        io.open(os.path.join(OUT, name), "w", encoding="utf-8").write(src)
        try:
            ast.parse(src)
            note = "OK"
        except SyntaxError as exc:
            note = "line %d: %s" % (exc.lineno, exc.msg)
        print("%-20s %7d %7d %s" % (name, len(src.splitlines()), len(src), note))


if __name__ == "__main__":
    main()
