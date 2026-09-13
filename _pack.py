"""打包支撑材料 zip 到 附件/ 目录。

按竞赛要求与最新指示组织内容：
  包含：全部可运行源程序（论文生成代码除外）、演练/模拟的日志、正式测试的加密日志、图表。
  排除：README.md 及一切说明性 txt 文件、__pycache__ 等缓存、赛题原始材料、
        论文生成代码（make_paper.py 与本机临时脚本）。
"""
import io
import os
import shutil
import zipfile

ROOT = os.path.abspath(".")
OUT_DIR = os.path.join(ROOT, "附件")
ZIP = os.path.join(OUT_DIR, "支撑材料.zip")
SIM_DATA = (r"F:\BaiduNetdiskDownload\CUMCM2026B\Jammers-simulator-win64"
            r"\Jammers-simulator\JammersSimulatorData\behavior-logs")

PAPER_GEN = {"make_paper.py", "_fix_paper.py"}
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
# 只保留这些"代码/数据/图表"目录；不含任何说明性 txt
INCLUDE_DIRS = ("src", "scripts", "tests", "results", "logs")
INCLUDE_FILES = ("requirements.txt", "pytest.ini")

os.makedirs(OUT_DIR, exist_ok=True)
members = []


def add(arc, path):
    members.append((arc, path))


for sub in INCLUDE_DIRS:
    base = os.path.join(ROOT, sub)
    if not os.path.isdir(base):
        continue
    for cur, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files):
            if name in PAPER_GEN:
                continue
            if sub == "src" and name.startswith("_"):
                continue
            # 说明性文本一律不收（只收代码 / json / png / jsonl）
            if os.path.splitext(name)[1].lower() in (".txt", ".md", ".rst"):
                continue
            add("支撑材料/" + os.path.relpath(os.path.join(cur, name), ROOT),
                os.path.join(cur, name))

for name in INCLUDE_FILES:
    path = os.path.join(ROOT, name)
    if os.path.exists(path):
        add("支撑材料/" + name, path)

# AI 工具使用详情（竞赛要求）
ai_pdf = os.path.join(OUT_DIR, "AI工具使用详情.pdf")
if os.path.exists(ai_pdf):
    add("支撑材料/AI工具使用详情.pdf", ai_pdf)

# 加密行为日志：正式测试（必交）+ 演练测试
n_formal = n_practice = 0
if os.path.isdir(SIM_DATA):
    for name in sorted(os.listdir(SIM_DATA)):
        if not name.endswith(".jlog"):
            continue
        src = os.path.join(SIM_DATA, name)
        if name.startswith("formal-"):
            add("支撑材料/正式测试加密日志/" + name, src)
            n_formal += 1
        elif name.startswith("practice-"):
            add("支撑材料/演练测试加密日志/" + name, src)
            n_practice += 1

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for arc, path in members:
        zf.write(path, arc)

print("已生成", ZIP)
print("  代码/数据/图表条目 %d；正式测试加密日志 %d 个；演练加密日志 %d 个"
      % (len(members) - n_formal - n_practice, n_formal, n_practice))
print("  总条目 %d，大小 %.2f MB（上限 20 MB）"
      % (len(members), os.path.getsize(ZIP) / 1024 / 1024))
