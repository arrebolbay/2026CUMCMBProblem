"""临时修复脚本：把三、四章从 build_abstract 移到 build_body 正确位置。"""
import ast
import io

P = "src/make_paper.py"
L = io.open(P, encoding="utf-8").read().splitlines()
assert "三、模型假设" in L[283], L[283]
assert "表 1" in L[327], L[327]
assert "kw = doc.add_paragraph()" in L[330], L[330]
block = L[283:328]                      # 三、模型假设 + 四、符号说明
rest = L[:283] + [""] + L[328:329] + L[334:]
ins = None
for i, line in enumerate(rest):
    if "共同抬高了时间下限" in line:
        ins = i + 1
assert ins, "insert point not found"
out = rest[:ins] + [""] + block + rest[ins:]
text = "\n".join(out)
io.open(P, "w", encoding="utf-8").write(text)
try:
    ast.parse(text)
    print("语法 OK, lines", len(out))
except SyntaxError as exc:
    print("ERR", exc.lineno, exc.msg)
for i, line in enumerate(out[274:292], 275):
    print(i, ":", line)
