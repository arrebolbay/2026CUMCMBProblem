"""临时脚本：用 Word 统计论文页数并导出 PDF（不属于论文生成代码）。"""
import os

import win32com.client as win32

DOCX = os.path.abspath(os.path.join("附件", "论文.docx"))
PDF = os.path.abspath(os.path.join("附件", "论文.pdf"))

word = win32.gencache.EnsureDispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = False
try:
    doc = word.Documents.Open(DOCX, ReadOnly=False)
    doc.Repaginate()
    total = doc.ComputeStatistics(2)          # wdStatisticPages
    words = doc.ComputeStatistics(0)          # wdStatisticWords
    chars = doc.ComputeStatistics(3)          # wdStatisticCharacters
    print("总页数 =", total)
    print("字数 =", words, " 字符数 =", chars)

    # 定位关键节点的页码
    def page_of(text):
        rng = doc.Content
        find = rng.Find
        find.ClearFormatting()
        found = find.Execute(FindText=text, Forward=True, Wrap=0)
        if not found:
            return None
        return doc.Range(0, rng.Start).Information(3)   # wdActiveEndPageNumber

    for label in ("摘  要", "一、问题重述", "十一、模型的检验、评价与推广",
                  "AI 工具使用声明", "参考文献", "附录 A", "附录 B"):
        print(f"    “{label}” 起始页 = {page_of(label)}")

    # 导出 PDF
    doc.SaveAs2(PDF, FileFormat=17)
    print("已导出 PDF：", PDF, "%.2f MB" % (os.path.getsize(PDF) / 1024 / 1024))
    doc.Close(SaveChanges=0)
finally:
    word.Quit()
