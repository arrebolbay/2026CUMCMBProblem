"""临时脚本：统一缩小插图至 10 cm，并测量摘要页余量。"""
import ast
import io
import os
import re

import win32com.client as win32

P = "src/make_paper.py"
s = io.open(P, encoding="utf-8").read()
for name in ("fig1_problem1", "fig2_problem2", "fig3_problem3",
             "fig3b_problem3_scheme", "fig4_problem4"):
    pat = re.compile(r'("' + name + r'\.png"\), )[0-9.]+\)')
    s, n = pat.subn(lambda m: m.group(1) + "10.0)", s)
    print("插图 %-24s -> 10.0 cm（命中 %d）" % (name, n))
io.open(P, "w", encoding="utf-8").write(s)
ast.parse(s)
print("语法 OK")

# 测量摘要页：关键词段落在第几页、结束位置的纵向坐标（页高约 698 pt 可排文本）
w = win32.gencache.EnsureDispatch("Word.Application")
w.Visible = False
w.DisplayAlerts = False
try:
    d = w.Documents.Open(os.path.abspath("附件/论文.docx"), ReadOnly=True)
    d.Repaginate()
    for p in d.Paragraphs:
        t = p.Range.Text.strip()
        if t.startswith("关键词") or t.startswith("一、问题重述"):
            page = d.Range(0, p.Range.Start).Information(3)
            bottom = p.Range.Information(6) + 12.0
            print("  %-12s 页码=%d  段落底部 y=%.0f pt" % (t[:8], page, bottom))
    d.Close(SaveChanges=0)
finally:
    w.Quit()
