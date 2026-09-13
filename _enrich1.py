"""临时脚本：补入问题一/问题二插图并顺移图号（丰富论文内容）。"""
import ast
import io

P = "src/make_paper.py"
s = io.open(P, encoding="utf-8").read()

# 1) 原图号顺移（倒序，避免冲突）
for a, b in (("图 3  问题四：", "图 5  问题四："),
             ("图 2  问题三实际运行方案", "图 4  问题三实际运行方案"),
             ("图 1  7 点覆盖网", "图 3  7 点覆盖网")):
    assert s.count(a) == 1, ("重编号失败", a)
    s = s.replace(a, b)
    print("重编号 OK:", b[:20])

# 2) 问题一（三）末尾插入 fig1
a1 = '    h2(doc, "（四）问题一的结论")'
ins1 = ('    add_figure(doc, os.path.join(ROOT, "results", "figs", '
        '"fig1_problem1.png"), 11.5)\n'
        '    caption(doc, "图 1  问题一：两站交会的定位区域、直径圆与最小覆盖圆'
        '（右为锐角三角形反例——直径圆不覆盖定位区域）")\n')
assert s.count(a1) == 1
s = s.replace(a1, ins1 + a1)
print("插入 图1 OK")

# 3) 问题二（四）末尾插入 fig2
a2 = '    h1(doc, "七、问题三'
ins2 = ('    add_figure(doc, os.path.join(ROOT, "results", "figs", '
        '"fig2_problem2.png"), 11.5)\n'
        '    caption(doc, "图 2  问题二：定位区域面积随基线 b 的变化'
        '（★ 为最优点 b* = d₁、θ* = 45°）与蝶形候选区")\n\n')
assert s.count(a2) == 1
s = s.replace(a2, ins2 + a2)
print("插入 图2 OK")

io.open(P, "w", encoding="utf-8").write(s)
ast.parse(s)
print("语法 OK")
