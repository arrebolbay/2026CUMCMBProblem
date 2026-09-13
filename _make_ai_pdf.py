"""临时脚本：生成《AI工具使用详情.pdf》（竞赛要求的支撑材料之一）。

注意：本脚本属于论文/材料生成代码，不进入论文附录与支撑材料正文清单。
"""
import os

import win32com.client as win32
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HEI, SONG, TNR = "黑体", "宋体", "Times New Roman"
OUT_DIR = os.path.abspath("附件")
DOCX = os.path.join(OUT_DIR, "AI工具使用详情.docx")
PDF = os.path.join(OUT_DIR, "AI工具使用详情.pdf")


def font(run, size, cn, bold=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = TNR
    rpr = run._element.get_or_add_rPr()
    rf = OxmlElement("w:rFonts")
    rf.set(qn("w:ascii"), TNR)
    rf.set(qn("w:hAnsi"), TNR)
    rf.set(qn("w:eastAsia"), cn)
    rpr.append(rf)


def para(doc, text, size=12.0, cn=SONG, bold=False, indent=True, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.first_line_indent = Pt(size * (2 if indent else 0))
    if align:
        p.paragraph_format.alignment = align
    if text:
        font(p.add_run(text), size, cn, bold)
    return p


def head(doc, text, size=14.0):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    font(p.add_run(text), size, HEI)
    return p


def table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        font(p.add_run(h), 10.5, SONG, True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            font(p.add_run(v), 10.5, SONG)
    return t


os.makedirs(OUT_DIR, exist_ok=True)
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.54)
sec.left_margin = sec.right_margin = Cm(3.17)

p = para(doc, "AI 工具使用详情", 16.0, HEI, align=WD_ALIGN_PARAGRAPH.CENTER,
         indent=False)
p.paragraph_format.space_after = Pt(10)

head(doc, "一、所用 AI 工具名称、版本或型号")
para(doc, "本参赛队在竞赛过程中使用了下列 AI 工具：")
table(doc, ["序号", "工具名称", "版本 / 型号"],
      [["1", "DeepSeek", "DeepSeek-V4.1-flash"],
       ["2", "DeepSeek", "DeepSeek-V4-pro"],
       ["3", "OpenAI ChatGPT", "ChatGPT-5.6-sol"],
       ["4", "Moonshot Kimi", "Kimi-k3"],
       ["5", "Anthropic Claude", "Claude-Opus-5"]])

head(doc, "二、具体使用目的和环节")
for text in (
    "**1. 建模思路的辅助梳理**：在问题一至问题四的建模阶段，用于检索相关领域的常见做法"
    "（如交会定位的精度度量、圆覆盖的最少布点数、局部搜索的邻域算子），"
    "以及请其协助检查我们已写出模型的逻辑一致性。",
    "**2. 代码调试与辅助编写**：用于排查程序中的偶发缺陷、检查边界条件、"
    "以及辅助撰写重复性较强的样板代码（如批量实验驱动、结果统计脚本）。",
    "**3. 论文语言润色**：用于对已成稿段落做语句通顺化与术语统一。",
):
    para(doc, text.replace("**", ""))

head(doc, "三、主要提示方式与使用过程说明")
para(doc, "本队使用 AI 工具时的提示方式有以下共同特点："
          "（1）**先给出充分上下文**（问题背景、模拟器规则、已有的公式或代码片段）；"
          "（2）**明确要求给出推导过程、依据或出处**，不接受只有结论的回答；"
          "（3）**要求给出可验证的判据或反驳条件**，以便我队自行实验检验；"
          "（4）对关键数值一律要求“给出可复现的计算方式”，再由我队独立复算。")
para(doc, "**典型交互示例一（建模辅助，问题三的布点数下界）**")
para(doc, "提问要点：“用有效半径 1000 m 的检测圆覆盖半径 1800 m 的圆形区域，"
          "最少需要几个点？请给出依据或出处。”")
para(doc, "AI 回答大意：指出该比值 1.8 处于圆覆盖问题的临界区间，"
          "给出若干常数（如 n = 6 与 n = 7 对应的最优半径比）与相关文献线索。")
para(doc, "我队的处理：不直接采用该结论，而是**自行做了数值优化**"
          "（6 点最优布局的最坏覆盖半径 1015.06 m > 1000 m 不可行；"
          "7 点最优布局 913.86 m 可行），并**解析推导**了运行布点的最坏距离"
          "√(1800²+1000²−2·1800·1000·cos(π/7)) = 998.25 m，"
          "与高密网格数值复核（998.2434 m）相互印证后才写入论文。")
para(doc, "**典型交互示例二（代码调试，偶发漏清缺陷）**")
para(doc, "提问要点：描述“300 例中偶发 3~13 例未完全清除”的现象，"
          "并给出覆盖判据函数与剪枝函数的代码片段，请其分析可能原因。")
para(doc, "AI 回答大意：提出两个可疑方向——"
          "（1）覆盖判据的边界采样数可能被误传为极小值，导致最坏距离被低估；"
          "（2）剪枝时按索引删除元素可能造成后续元素前移、主循环跳过某个覆盖点。")
para(doc, "我队的处理：**逐条构造最小复现案例验证**，确认两处缺陷真实存在后修复"
          "（边界采样改为 14400 点；剪枝候选限定为“严格未来”的点），"
          "并补充“探测步长必须包含不超过清除半径量级的小值”这一通用安全性修复。"
          "修复后 300 例复测零未清，80 项单元测试全部通过。")
para(doc, "**典型交互示例三（语言润色）**")
para(doc, "提问要点：给出已成稿的技术段落，要求“在不改变任何技术含义、"
          "不新增或删除数据的前提下使表述更简洁规范”。")

head(doc, "四、对 AI 输出的采纳、人工修改和核验情况")
para(doc, "**总体原则**：AI 输出仅作“线索”或“候选”，一律不直接采用；"
          "凡进入论文的结论、公式、参数与数据，都必须经过我队**独立的数学推导**"
          "或**可复现的实验验证**。具体如下：")
for text in (
    "**1. 建模辅助环节**：AI 提供的文献线索与常数，均由我队独立实现数值优化、"
    "高密采样与解析推导三重交叉验证后才采用；"
    "AI 建议的若干优化思路（例如“预测性牵引”“区域感知插入”“滚动重优化”等）"
    "经我队实测被证伪（分别使清除次数暴涨至 238~535 次、或结果更差），"
    "已明确弃用并在论文第九章如实记录。",
    "**2. 代码调试环节**：AI 给出的排查方向均需先用最小复现案例确认，"
    "确认后才修改代码；每次修改后都必须通过全部 80 项单元测试与 300 例离线复测，"
    "方可保留。",
    "**3. 语言润色环节**：仅调整措辞与术语，**不改动任何数据、公式与结论**；"
    "论文中出现的全部数值（清除比例 1.000、平均定位清除时间、行程、下界等）"
    "均来自我队自行运行的程序，可通过支撑材料中的源代码复现。",
):
    para(doc, text.replace("**", ""))

para(doc, "**需要说明的是**：本文的核心模型（交会定位区域的凸多边形刻画与直径算法、"
          "定向覆盖的角隙充要判据、覆盖—清除一体化调度策略）、"
          "全部算法实现、参数取值与实验设计，均由参赛队自主完成；"
          "AI 工具未参与上述核心内容的原创设计。本参赛队对所提交论文与支撑材料的"
          "全部内容负责。")

doc.save(DOCX)
print("已生成", DOCX)

word = win32.gencache.EnsureDispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = False
try:
    d = word.Documents.Open(os.path.abspath(DOCX))
    d.SaveAs2(os.path.abspath(PDF), FileFormat=17)
    d.Close(SaveChanges=0)
    print("已生成", PDF, "%.0f KB" % (os.path.getsize(PDF) / 1024))
finally:
    word.Quit()
