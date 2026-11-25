# -*- coding: utf-8 -*-
"""生成《轮组冬招题：三维点云的识别》Word 文档。

运行：
    python docs/make_doc.py

输出：
    在项目根目录生成 轮组冬招题_三维点云的识别.docx
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent          # 数据与代码资料/
OUT = ROOT.parent / "轮组冬招题_三维点云的识别.docx"     # 输出到项目根目录

BODY_CN, BODY_EN = "宋体", "Times New Roman"
HEAD_CN, HEAD_EN = "黑体", "Arial"
MONO = "Consolas"
ACCENT = RGBColor(0x1F, 0x3B, 0x63)


# --------------------------------------------------------------------------- #
# 基础排版工具
# --------------------------------------------------------------------------- #
def set_run(run, size=10.5, bold=False, cn=BODY_CN, en=BODY_EN, color=None, italic=False):
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), en)
    rFonts.set(qn("w:hAnsi"), en)
    rFonts.set(qn("w:eastAsia"), cn)
    return run


def body(doc, text="", size=10.5, bold=False, first_indent=True, space_after=4,
         align=None, color=None, line=1.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    if first_indent:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    if text:
        set_run(p.add_run(text), size=size, bold=bold, color=color)
    return p


def rich(doc, parts, size=10.5, first_indent=True, space_after=4, line=1.5):
    """parts = [(文本, 是否加粗), ...]"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    if first_indent:
        pf.first_line_indent = Pt(size * 2)
    for text, bold in parts:
        set_run(p.add_run(text), size=size, bold=bold)
    return p


def bullet(doc, text, size=10.5, bold_prefix=None, level=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.4
    pf.space_before = Pt(0)
    pf.space_after = Pt(2)
    pf.left_indent = Pt(18 + 18 * level)
    pf.first_line_indent = Pt(-9)
    set_run(p.add_run("· "), size=size)
    if bold_prefix:
        set_run(p.add_run(bold_prefix), size=size, bold=True)
    set_run(p.add_run(text), size=size)
    return p


def numbered(doc, idx, text, size=10.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.45
    pf.space_before = Pt(0)
    pf.space_after = Pt(3)
    pf.left_indent = Pt(24)
    pf.first_line_indent = Pt(-24)
    set_run(p.add_run(f"{idx}. "), size=size, bold=True)
    set_run(p.add_run(text), size=size)
    return p


def heading(doc, text, level=1):
    cfg = {
        1: (16, 14, 8),
        2: (13, 10, 5),
        3: (11.5, 8, 4),
    }[level]
    size, before, after = cfg
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = True
    pf.line_spacing = 1.25
    set_run(p.add_run(text), size=size, bold=True, cn=HEAD_CN, en=HEAD_EN, color=ACCENT)
    return p


def shade(paragraph_or_cell, fill="F4F6F9"):
    el = paragraph_or_cell._element
    pr = el.find(qn("w:pPr")) if el.tag.endswith("}p") else el.find(qn("w:tcPr"))
    if pr is None:
        pr = OxmlElement("w:pPr") if el.tag.endswith("}p") else OxmlElement("w:tcPr")
        el.insert(0, pr)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pr.append(shd)


def para_border(p, color="C9D2DE", sz=6):
    pPr = p._element.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    for side in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "4")
        e.set(qn("w:color"), color)
        pbdr.append(e)
    pPr.append(pbdr)


def code_block(doc, text, size=8.8):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.15
    pf.space_before = Pt(3)
    pf.space_after = Pt(6)
    pf.left_indent = Pt(10)
    pf.right_indent = Pt(6)
    lines = text.rstrip("\n").split("\n")
    for i, line in enumerate(lines):
        if i:
            p.add_run().add_break()
        set_run(p.add_run(line if line else " "), size=size, cn=MONO, en=MONO)
    shade(p)
    para_border(p)
    return p


def note_box(doc, title, text, fill="FFF8E6"):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.45
    pf.space_before = Pt(6)
    pf.space_after = Pt(8)
    pf.left_indent = Pt(10)
    pf.right_indent = Pt(6)
    pf.first_line_indent = Pt(0)
    set_run(p.add_run(f"{title}  "), size=10, bold=True, cn=HEAD_CN, en=HEAD_EN)
    set_run(p.add_run(text), size=10)
    shade(p, fill)
    para_border(p, color="E6C97A")
    return p


def table(doc, header, rows, widths=None, size=9.5, header_fill="DCE4F0"):
    # 表格前面那句"下面这张表……"要跟着表格走，否则会出现"引导语在页尾、表格在下一页"
    if doc.paragraphs:
        doc.paragraphs[-1].paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 表头在跨页时重复，避免第二页出现没有标题的续行
    trPr = t.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        set_run(p.add_run(h), size=size, bold=True, cn=HEAD_CN, en=HEAD_EN)
        shade(hdr[i], header_fill)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.25
            txt = str(v)
            # 纯 ASCII 且很短的单元格（id、数值、参数名）用等宽字体，读起来更像代码
            mono = txt.isascii() and len(txt) <= 24 and not txt.endswith("。")
            set_run(p.add_run(txt), size=size,
                    cn=MONO if mono else BODY_CN, en=MONO if mono else BODY_EN)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    # cantSplit：任何一行都不许被页边界劈成两半（否则下一页会出现一条空白行）。
    # 这一步与分页位置无关，总是对的，所以所有行都加。
    for r in t.rows:
        trPr2 = r._tr.get_or_add_trPr()
        trPr2.append(OxmlElement("w:cantSplit"))
    # keepNext 只加在最后一行与"表后空行"上：让表题不至于被挤到下一页变成孤零零一行。
    # 注意不能给所有行都加 keepNext，否则整张长表会被硬推到下一页，在本页留下大片空白。
    last = t.rows[-1]
    for c in last.cells:
        for p in c.paragraphs:
            p.paragraph_format.keep_with_next = True
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.keep_with_next = True
    return t


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(8)
    set_run(p.add_run(text), size=9, cn=HEAD_CN, en=HEAD_EN)
    return p


def hrule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    pPr = p._element.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    e = OxmlElement("w:bottom")
    e.set(qn("w:val"), "single")
    e.set(qn("w:sz"), "8")
    e.set(qn("w:space"), "1")
    e.set(qn("w:color"), "1F3B63")
    pbdr.append(e)
    pPr.append(pbdr)
    return p


# --------------------------------------------------------------------------- #
# 文档
# --------------------------------------------------------------------------- #
def build():
    doc = Document()

    # 页面
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = Cm(2.4)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.6)
    sec.right_margin = Cm(2.4)

    # 默认样式
    style = doc.styles["Normal"]
    style.font.name = BODY_EN
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_CN)

    # ---------------- 封面 ----------------
    for _ in range(3):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run(p.add_run("轮组冬招题"), size=26, bold=True, cn=HEAD_CN, en=HEAD_EN, color=ACCENT)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    set_run(p.add_run("三维点云的识别"), size=34, bold=True, cn=HEAD_CN, en=HEAD_EN, color=ACCENT)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(16)
    set_run(p.add_run("面向稀疏、缺失与噪声的道路三维点云语义分割"),
            size=13, cn=HEAD_CN, en=HEAD_EN)

    for _ in range(6):
        doc.add_paragraph()

    info = [
        ("适用对象", "具备基础 Python 编程能力的大一学生"),
        ("建议周期", "4—5 周"),
        ("组队方式", "仅由 1 人独立完成"),
        ("课程数据", "SemanticPOSS（北京大学，2988 帧激光雷达扫描）"),
        ("允许条件", "允许使用 AI 辅助学习、编程和调试（见第九章）"),
        ("文档版本", "v1.0"),
    ]
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for k, v in info:
        cells = t.add_row().cells
        cells[0].width = Cm(3.4)
        cells[1].width = Cm(10.6)
        p0 = cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p0.paragraph_format.space_after = Pt(6)
        set_run(p0.add_run(k + "："), size=11, bold=True, cn=HEAD_CN, en=HEAD_EN)
        p1 = cells[1].paragraphs[0]
        p1.paragraph_format.space_after = Pt(6)
        set_run(p1.add_run(v), size=11)

    doc.add_page_break()

    # ---------------- 目录 ----------------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run(p.add_run("目  录"), size=18, bold=True, cn=HEAD_CN, en=HEAD_EN, color=ACCENT)
    doc.add_paragraph()

    toc = [
        ("一、任务目标", ""),
        ("二、课程数据与标注口径", "数据来源、五类映射、划分与评测口径"),
        ("三、基线模型要求", "PointNet 四段主干与四个必答问题"),
        ("四、鲁棒性挑战", "稀疏、缺失、噪声与离群点"),
        ("五、论文检索与模型改进", "注意力模块的六项说明要求"),
        ("六、实验要求", "三组必做实验与三项必报结果"),
        ("七、交付内容", "五项提交材料"),
        ("八、答辩要求", "现场需要解释的内容"),
        ("九、AI 的使用", ""),
        ("附录 A  评分标准", "六项评分维度与分值"),
        ("附录 B  进阶探索（非必做）", "更多消融、成本报告、可视化"),
        ("附录 C  代码资料与运行说明", "roadseg 参考实现、环境与命令"),
    ]
    for name, desc in toc:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.6
        pf.space_after = Pt(2)
        pf.left_indent = Pt(24)
        pf.first_line_indent = Pt(-12)
        set_run(p.add_run(name), size=11)
        if desc:
            set_run(p.add_run(f"    {desc}"), size=9, color=RGBColor(0x77, 0x7F, 0x8C))

    doc.add_page_break()

    # ======================= 一、任务目标 =======================
    heading(doc, "一、任务目标", 1)

    body(doc, "自动驾驶车辆会用到传感器采集到的三维点云来理解周围的环境，判断每一个点属于地面、"
              "车辆、行人还是建筑；而在实际采集的过程中，点云的密度会降低，局部会出现缺失，"
              "坐标上也会带上噪声，这些情况都会影响模型的预测结果。")

    body(doc, "本题要求你从 PointNet 基线出发，完成道路场景里逐点的语义分类；再通过检索论文和"
              "改进模型，把模型在退化点云条件下的鲁棒性提高上去。")

    note_box(doc, "注意",
             "PointNet 与 Transformer 属于两种不同的模型架构。本题要求你先把 PointNet 处理点集的方式"
             "弄清楚，到了改进阶段再研究注意力机制能怎样跟点云的空间结构结合起来。"
             "PointNet 原始的结构可以参阅《PointNet: Deep Learning on Point Sets for 3D "
             "Classification and Segmentation》这篇文章。")

    body(doc, "假设一个道路场景里含有 N 个点，可以记作 P，其中每一个元素为一个点的坐标。")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    set_run(p.add_run("P = { (x_i, y_i, z_i) }，i = 1, 2, …, N"), size=12, cn=MONO, en=MONO)

    body(doc, "模型要为每一个输入点预测出一个语义类别。")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    set_run(p.add_run("f(P) = ( ŷ₁, ŷ₂, …, ŷ_N )"), size=12, cn=MONO, en=MONO)

    body(doc, "课程数据由出题方统一提供，分为训练集、验证集和测试集；目标类别为地面、车辆、行人、"
              "建筑、植被五类，其余类别按照统一规则忽略。合并成五类的原因，以及每一类对应数据集里"
              "哪一个原始标签，都写在第二章里。")

    body(doc, "整个项目做完，你需要完成五件事。", first_indent=False, space_after=3)
    for i, s in enumerate([
        "查询相关资料，搭建并训练一个 PointNet 模型。",
        "分析原始的 PointNet 在点云退化条件下的失效情况。",
        "自主检索论文，设计并实现包含注意力机制的改进模型。",
        "通过对照实验、消融实验和可视化分析验证改进。",
        "提交可复现的项目，并完成个人答辩。",
    ], 1):
        numbered(doc, i, s)

    # ======================= 二、课程数据 =======================
    heading(doc, "二、课程数据与标注口径", 1)

    body(doc, "所有实验都要用到同一份数据划分和同一套评测口径，结果之间才能横向比较；"
              "出题方提供的数据、标签的对齐办法、划分的方式和评测的细节，都写在第二章里。")

    heading(doc, "2.1  数据来源", 2)
    body(doc, "课程统一使用 SemanticPOSS 数据集（Yancheng Pan 等，北京大学，arXiv:2002.09147，"
              "投稿于 IEEE Intelligent Vehicles Symposium 2020）。这是一套校园道路场景的激光雷达"
              "逐点标注数据，采集设备为禾赛 Pandora 40 线激光雷达，垂直分辨率为 0.33°，量程为 200 m，"
              "车上还配了 RTK 定位和惯性单元。数据采自北京大学的校园，全长约 1.5 km、时长约 5 分钟，"
              "里面包含了大量行人、骑行者和车辆，跟本题设定的“道路场景语义分割”是吻合的。")

    table(doc,
          ["项目", "内容"],
          [
              ["名称", "SemanticPOSS"],
              ["规模", "2988 帧扫描，约 2.16 亿点，逐点标注并附带实例标注"],
              ["传感器", "Hesai Pandora 40 线 LiDAR，0.33° 垂直分辨率，200 m"],
              ["场景", "北京大学校园，约 5 分钟、约 1.5 km"],
              ["格式", "与 SemanticKITTI 一致，velodyne/*.bin 配 labels/*.label"],
              ["许可", "CC BY-NC-SA 3.0，仅限非商业的教学与科研使用"],
              ["下载", "http://www.poss.pku.edu.cn/semanticposs.html"],
          ],
          widths=[3.6, 11.6])
    caption(doc, "表 2-1  SemanticPOSS 数据集概况")

    body(doc, "数据文件的格式跟 SemanticKITTI 完全一致，你不需要再去找额外的格式转换工具。")
    bullet(doc, "坐标为 float32，每行 4 个数 (x, y, z, intensity)，单位米；", bold_prefix="velodyne/000000.bin：")
    bullet(doc, "为 uint32，低 16 位为语义标签，高 16 位为实例编号；", bold_prefix="labels/000000.label：")
    bullet(doc, "共 6 个序列（00—05），每个序列约 500 帧。", bold_prefix="序列组织：")

    heading(doc, "2.2  为什么是 5 类而不是 6 类", 2)
    body(doc, "题目初稿里写的是“道路、人行道、车辆、行人、建筑、植被”这六类；在核对 SemanticPOSS 的"
              "标注体系之后发现，数据集里只有 ground 一个地面语义类，它并不区分车道和人行道，"
              "所以“道路”和“人行道”在数据里根本不存在。")

    body(doc, "如果照搬六类，会得到两个后果，一是“人行道”永远没有样本，它的 IoU 恒为 0 或者"
              "干脆算不出来，只会把 mIoU 白白拉低，一点信息量都没有；二是学生为了让这个类别“有数字”，"
              "凭颜色或者高度去猜出一条人行道，这样做既不可复现，也跟标注的事实不符。")

    body(doc, "因此课程数据统一合并成五类；合并的原因出在数据集自身，跟降低难度没有关系，"
              "报告里应当把这句话写清楚。")

    table(doc,
          ["课程类别 id", "课程类别", "英文名", "SemanticPOSS 原始标签"],
          [
              ["0", "地面", "ground", "22 ground"],
              ["1", "车辆", "car", "7 car"],
              ["2", "行人", "pedestrian", "4 person、5 persons"],
              ["3", "建筑", "building", "15 building"],
              ["4", "植被", "vegetation", "9 plants"],
              ["—", "（忽略）", "ignore", "0、1、2、3、6、8、10、11、12、13、14、16、17、18、19、20、21"],
          ],
          widths=[2.6, 2.6, 3.0, 7.4])
    caption(doc, "表 2-2  课程五类与 SemanticPOSS 原始标签的映射（默认口径）")

    body(doc, "被忽略的原始标签依次是 0 unlabeled、1 not know、6 rider、8 trunk、"
              "10/11/12 traffic sign（三个子类）、13 pole、14 trashcan、16 cone-stone、17 fence、21 bike，"
              "另外 2、3、18、19、20 这五个是 SemanticPOSS 标注表里没有定义的占位 id。"
              "忽略它们的理由分三类。标注自身的保留位（0、1 与 2/3/18/19/20 占位符）；"
              "题目没有要求的细类（traffic sign、trashcan、cone）；"
              "以及会跟目标类发生语义重叠的类（rider 同时是人和车，trunk 和 plants 属于同一棵树）。")

    body(doc, "凡是没在表 2-2 的“SemanticPOSS 原始标签”列里出现的 id，包括 23—31 这类未定义的区间，"
              "都一律映射成 255；实现映射表的时候请直接查表，roadseg/labels.py 里的 DEFAULT_LUT "
              "长度为 256，不要用 if/else 一个个去判断，那样很容易把未定义的 id 漏掉。")

    note_box(doc, "口径可选",
             "参考实现还提供了一个宽口径的映射，就是 roadseg/labels.py 里的 ALT_RULES，"
             "车辆额外并入 bike（21），行人额外并入 rider（6），植被额外并入 trunk（8）；"
             "两种口径都合法，一旦选定就要全程一致，并在报告里说明。默认口径更保守，建议采用。")

    heading(doc, "2.3  数据划分", 2)
    body(doc, "SemanticPOSS 官方只给出“训练部分”和“测试部分”，论文把数据按车辆位置切成 6 份、"
              "每份 500 帧，并指定 part 3 为测试集，其余为训练，官方没有单独的验证集。")

    body(doc, "本题要求“不得使用测试集选择参数”，因此出题方固定从训练部分里切出一个验证集，"
              "划分方式写在表 2-3 里，所有同学必须完全相同。")

    table(doc,
          ["划分", "序列", "用途"],
          [
              ["train", "00、01、02、05", "训练，可做数据增强"],
              ["val", "04", "选模型、调超参（每 5 个 epoch 评估一次）"],
              ["test", "03", "最终评测，只允许用一次"],
          ],
          widths=[2.4, 4.6, 8.4])
    caption(doc, "表 2-3  课程固定划分")

    body(doc, "验证集和测试集的用法必须分开，验证集可以反复看、反复调，测试集只能看一次；"
              "如果看到测试集的结果再回头改动超参，全部结论都会失效，评分时会重点检查这种情况。")

    body(doc, "论文给出 6 个序列共 2988 帧，每序列约 500 帧，实际略有出入。表 2-3 的六个序列已经覆盖"
              "全部数据，没有留下未分配的序列，其中 train 约 2000 帧、val 约 500 帧、test 约 500 帧；"
              "如果只想做小规模验证，可以用 train.py 的 --limit-frames 参数截取部分训练帧，"
              "不过报告里的最终数字必须在完整划分上得到。")

    heading(doc, "2.4  下发的数据目录", 2)
    body(doc, "出题方提供预处理后的课程数据，由原始的 .bin 和 .label 转成 .npz，标签映射与划分都已经做好，"
              "目录里每一项的含义在代码块的注释里标出来了。")
    code_block(doc, """data/course/
├── frames/              # 每帧一个 npz
│   ├── 00_000000.npz    # points (N,3) float32 / labels (N,) uint8 / intensity (N,) float32
│   └── ...
├── manifest.csv         # 每帧点数、有效点数、五类占比
├── label_stats.json     # 全库类别比例、映射规则、序列与划分
└── splits/
    ├── train.txt        # 每行一个 frame_id
    ├── val.txt
    └── test.txt""")

    body(doc, "如果你拿到的是原始压缩包 SemanticPOSS_dataset.zip，大小约 2.41 GB，"
              "可以用仓库里的预处理脚本自行转换，转换出来的目录结构跟上面完全一致。")
    code_block(doc, """# 先抽查 20 帧，确认标签映射正确
python tools/prepare_data.py --raw D:/SemanticPOSS/dataset/sequences \\
    --out data/course --check

# 全量转换
python tools/prepare_data.py --raw D:/SemanticPOSS/dataset/sequences \\
    --out data/course

# 采用宽口径映射（见上文“口径可选”）
python tools/prepare_data.py --raw D:/SemanticPOSS/dataset/sequences \\
    --out data/course --alt-mapping""")

    heading(doc, "2.5  评测口径", 2)
    body(doc, "所有人的数字要能直接横向比较，评测就必须满足五个条件；参考实现已经把这几条写死在 "
              "roadseg/metrics.py 和 roadseg/degradation.py 里，你直接调用即可。")
    for i, s in enumerate([
        "只统计保留下来的原始点，被删除的点、插入的离群点都不进入混淆矩阵。",
        "每帧的采样点数与采样方式固定，默认等间隔 stride 采样到 8192 点，两个模型完全相同；"
        "若某档退化后剩余点数不足 8192，则该帧直接使用全部剩余点，不上采样、不重复采样；"
        "此时应在报告里一并记录实际参与评测的点数。",
        "IoU 按类计算，分母为 TP + FP + FN；mIoU 为参与评测的类别 IoU 的算术平均。",
        "忽略类（ignore_index=255）不参与任何指标计算，也不计入 mIoU 的分母个数。",
        "测试集只评估一次；所有退化档位的随机种子固定，保证两次运行的结果可复现。",
    ], 1):
        numbered(doc, i, s)

    body(doc, "噪声强度的约定统一成“每轴独立高斯噪声的标准差 σ”，单位厘米，即 sigma=5 表示 "
              "x、y、z 三轴各自叠加 N(0, 5cm²)；它不是方差，也不是三轴合成之后的模长；"
              "参考实现 roadseg/degradation.py 的 NoiseSpec(sigma=...) 跟这条约定一致。")

    body(doc, "哪些点算数这件事需要说清楚，施加“删除 60° 视角扇区”这种退化之后，剩下来的点的"
              "真实标签依然是已知的，模型也依然要为它们输出预测，所以这些点全部计入指标；"
              "恢复被删掉的点和本题要求的内容属于两件不同的事，本题并不要求做点云补全。")

    # ======================= 三、基线模型要求 =======================
    doc.add_page_break()
    heading(doc, "三、基线模型要求", 1)

    body(doc, "你要用 PyTorch 搭出一个 PointNet 风格的语义分割网络，网络至少要包含四个部分。")
    for s in [
        "共享的逐点特征提取模块；",
        "对称的全局特征聚合操作；",
        "局部点特征与全局特征的融合；",
        "为每个点输出类别的预测模块。",
    ]:
        bullet(doc, s)

    body(doc, "参考实现能不能用，边界可以划成三条线。")
    for s in [
        "四段主干必须由你自己写出，并能在答辩时逐行解释；允许参考 torch.nn 的 Conv1d、BatchNorm1d、"
        "MaxPool1d、ReLU、Dropout 等基础层，也允许参考任何论文里的公式；",
        "不建议直接 import 参考实现的 PointNetSeg 当作自己的基线交上来；若确实用了，"
        "必须在报告里明确标注哪些行是自己写的、参考实现改了哪几处，并在答辩时能解释被改动的部分；"
        "没有标注就直接复用，按“不是自己写的”处理；",
        "改进部分（第五章）允许借鉴、修改、替换参考实现的 rpa.py，但必须说清改了哪里、为什么改。",
    ]:
        bullet(doc, s)

    body(doc, "基线网络的四段主干要由你亲手写出来，改进模块则可以从参考实现或者论文里借鉴，"
              "只是改了什么、为什么改都要讲得清楚。")

    body(doc, "你要结合自己的代码回答四个问题，这四个问题同时也是答辩的必问内容，"
              "答不出来会直接影响“基线搭建与原理理解”的得分。", space_after=5)
    for i, s in enumerate([
        "为什么点云不能简单当作有固定顺序的序列？",
        "模型如何处理输入点的顺序变化？",
        "全局特征怎样参与每个点的分类？",
        "仅靠逐点特征和全局特征，可能遗漏哪些信息？",
    ], 1):
        numbered(doc, i, s)

    note_box(doc, "提示",
             "第四问会牵出后面几乎所有的改进工作，回答它的时候一定要拿实验现象说话，"
             "只写一句“缺少局部信息”是不够的。你可以把小物体单独切出来看预测，"
             "可以看稀疏条件下的类别混淆矩阵，也可以看一个点被误判时它的邻居都是什么类别。"
             "先看到现象、再提出假设、最后设计模块，这条路走下来的得分会明显高于直接抄一个注意力模块。")

    # ======================= 四、鲁棒性挑战 =======================
    heading(doc, "四、鲁棒性挑战", 1)

    body(doc, "除了原始的测试数据之外，模型还要接受四类测试。")

    table(doc,
          ["挑战", "测试方式", "关注问题"],
          [
              ["点云稀疏", "保留原点云的 50%、25%、12.5%", "点数减少以后，模型能否保持分割能力？"],
              ["局部缺失", "删除以原点为中心 6 m × 6 m × 3 m 长方体，或正前方 60°（±30°）扇区内的点",
               "局部结构不完整时，剩余的点能否被正确分类？"],
              ["坐标噪声", "对每轴独立叠加标准差 σ = 2 / 5 / 10 cm 的高斯噪声",
               "模型对测量误差是否敏感？"],
              ["离群点干扰", "在已有点的局部邻域内（距某个原始点 0.5—2 m）插入 64 / 256 个随机方向的异常点",
               "异常点是否会干扰其他点的预测？"],
          ],
          widths=[2.6, 6.6, 6.0])
    caption(doc, "表 4-1  四类退化挑战")

    body(doc, "公开的验证工具给出了基本的扰动设置，也就是参考实现 roadseg/degradation.py 里的 14 个档位；"
              "隐藏测试则会用到没有公开的扰动参数和一部分组合条件，例如“稀疏 + 坐标噪声”。")

    body(doc, "测试只评价保留下来的原始点，新增的离群点不计入语义指标，但会作为模型输入参与前向计算，"
              "因此可能干扰其他点的预测；本题不要求恢复被删除的点，也不要求完成点云补全。")

    note_box(doc, "隐藏测试怎么算分",
             "隐藏测试由出题方用你提交的代码与权重统一跑，你本人不参与，也拿不到隐藏档位的具体参数。"
             "评分只使用两种来源的数字，一是你报告里自己跑出的公开档位结果，二是出题方用你的代码"
             "复跑同一份权重得到的隐藏档位结果。两者会互相核对，如果同一份权重下你的公开数字跟"
             "复跑结果差得很明显，说明评测流程（采样点数、归一化、seed）没有按第 2.5 节执行，"
             "此时以复跑的结果为准。所以正确的做法是让 evaluate.py --ckpt 权重 一键复现，"
             "把精力花在猜隐藏参数上没有意义。")

    note_box(doc, "关于隐藏测试",
             "隐藏测试要检验的是你的改进有没有只在某一个有利的档位上才有效，猜参数本身并不在考察范围之内；"
             "如果在 5 cm 噪声下有提升、在 10 cm 噪声下反而下降，说明改进只是把误差挪了个位置，"
             "这种情况如实报告反而比藏起来更划算。")

    # ======================= 五、论文检索与改进 =======================
    heading(doc, "五、论文检索与模型改进", 1)

    body(doc, "PointNet 原始的结构里还留着一些问题，你要检索相关的论文，对原始模型做出改进。")

    body(doc, "改进后的模型里至少要有一个把空间邻域或者相对位置结合进来的注意力模块，并且要说清楚六件事。")
    for s in [
        "基线存在什么问题，有什么实验现象作为依据？",
        "所选论文提出了什么方法？",
        "你借鉴了哪些设计，又进行了哪些调整？",
        "注意力中的 Q、K、V 来自哪些特征？",
        "空间位置或者邻域关系如何参与计算？",
        "改进可能增加哪些计算成本，又可能在哪些条件下失效？",
    ]:
        bullet(doc, s)

    body(doc, "可以复现或者简化已有的方法，不要求提出全新的算法；仅仅调整学习率、训练轮数、网络宽度，"
              "或者只增加数据增强，都不满足本部分的要求。")

    body(doc, "检索记录要能核查，写清楚检索用的平台、关键词、时间，以及每一篇论文的完整引用信息；"
              "至少要检索三篇，其中挑一篇做重点阅读。")

    table(doc,
          ["方向", "可参考的公开工作"],
          [
              ["局部几何建模", "PointNet++（层次化邻域特征）、DGCNN（EdgeConv 动态图）"],
              ["密度适应", "RandLA-Net（随机采样加局部特征聚合）、Point Transformer（局部注意力）"],
              ["鲁棒特征聚合", "PointNet++ 的 max pooling 与注意力池化对比、KPConv"],
              ["点云注意力", "Point Transformer、PCT、Stratified Transformer"],
          ],
          widths=[3.6, 11.6])
    caption(doc, "表 5-1  检索方向的参考起点（可以自行扩展，不限于此表）")

    # ======================= 六、实验要求 =======================
    heading(doc, "六、实验要求", 1)

    body(doc, "整个项目里要跑三组实验。")
    table(doc,
          ["实验", "目的", "是否必做"],
          [
              ["PointNet 基线", "建立性能参照", "必做"],
              ["完整改进模型", "检验整体效果", "必做"],
              ["核心模块消融", "证明提升来自该模块，跟参数变多无关", "必做（至少一组）"],
          ],
          widths=[4.4, 7.6, 3.2])
    caption(doc, "表 6-1  必做实验")

    body(doc, "第三组消融是“能不能把结论说清楚”的关键证据，所以列为必做。做法是只改一个变量，"
              "让参数量尽量接近，并且让结论能被重复出来。比如你认定相对位置编码有效，"
              "就要把加入和移除该编码的结果放在一起比较。认定局部注意力有效，"
              "就要在完全相同的邻域上改用 max 池化再比一次。如果改进版本比基线多了 40% 的参数，"
              "性能提升就没法归到“注意力”这项设计上。附录 B.1 里还有更多消融组合，属于选做。")

    body(doc, "报告里要给出三项结果。")
    for s in [
        "原始测试集以及各退化条件下的 mIoU；",
        "各类别的 IoU，尤其要分析表现较差的类别；",
        "点云保留比例与模型性能之间的关系曲线；",
    ]:
        bullet(doc, s)

    body(doc, "模型没有取得提升的时候也要如实报告，能够通过实验解释失败的原因，同样属于有效成果。")

    body(doc, "三组实验必须使用统一的数据划分、输入点数规则与训练预算，包括 epoch 数、batch size 和优化器；"
              "训练数据可以做增强，但三组要一致，否则对比不成立。")

    note_box(doc, "最低可交付路径",
             "表 6-1 里的三组实验是硬指标，14 个退化档位则没有要求全部由你亲自训练得到。"
             "算力或者时间不够的时候，可以按这套办法交付并在报告里说明。"
             "clean、sparse_25、noise_5cm 三档要自己跑出结果，其余档位可以直接调用参考实现里的 "
             "evaluate.py 在同一份权重上算出来，不需要重新训练，单帧推理就够，"
             "也可以明确标注成“未评测”并说明原因。用“未评测”冒充“评测过”属于学术不端，"
             "如实标注缺项不会扣分。")

    # ======================= 七、交付内容 =======================
    heading(doc, "七、交付内容", 1)

    body(doc, "需要提交五份材料。", first_indent=False, space_after=3)
    items = [
        ("源代码与运行说明", "包含环境配置、训练和评估命令，以及模型权重。"),
        ("论文阅读记录", "提供可核查的引用，说明检索过程与选文理由。"),
        ("项目报告", "包含模型结构、问题假设、改进方案、对照实验、消融实验和局限分析。"),
        ("可视化结果", "展示原始点云、真实标签、模型预测及典型错误。"),
        ("AI 使用说明", "说明 AI 参与的环节，以及哪些结论经过了人工核查或实验验证。"),
    ]
    for i, (name, desc) in enumerate(items, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.45
        pf.left_indent = Pt(24)
        pf.first_line_indent = Pt(-24)
        pf.space_after = Pt(3)
        set_run(p.add_run(f"{i}. "), size=10.5, bold=True)
        set_run(p.add_run(name + "："), size=10.5, bold=True)
        set_run(p.add_run(desc), size=10.5)

    # ======================= 八、答辩要求 =======================
    heading(doc, "八、答辩要求", 1)

    body(doc, "答辩的时候你要解释自己提交的代码和实验，考官会问到四个方面的内容。")
    for s in [
        "PointNet 的逐点特征、对称聚合与分割流程；",
        "改进模型的注意力计算和张量形状；",
        "论文方法与本项目实现之间的联系；",
        "对照实验是否支持项目结论。",
    ]:
        bullet(doc, s)

    body(doc, "答辩过程中可能还会让你现场关掉某个模块再解释现象，让你逐行解释一段代码，"
              "或者判断某一项实验设置是否合理。")

    note_box(doc, "准备建议",
             "把“张量形状”写在纸上练一遍，输入 (B, N, 3) 经过每一层之后会变成什么形状，"
             "Q、K、V 各自是几维，注意力矩阵是 (B, N, k) 还是 (B, N, N)；"
             "能一口气把这些形状写对的人，多半也真的把这个模块弄懂了。")

    # ======================= 九、AI 的使用 =======================
    heading(doc, "九、AI 的使用", 1)

    body(doc, "跟 AI 协作的能力确实很重要，不过我们不希望看到你把整件事都交给 AI，"
              "也不希望看到你只写一句愿望就等着代码自己跑通；在这个项目里真正学到东西才是最宝贵的。")

    body(doc, "交付材料里必须有一份“AI 使用说明”，写清楚哪些环节用了 AI，比如查论文、写数据加载、"
              "调参、改 bug、润色报告，也写清楚哪些结论是你自己通过实验验证或者人工核查过的。")
    body(doc, "答辩时可能会针对 AI 生成的代码提问；如果无法解释提交代码的关键部分，"
              "该部分将不被认定为你的工作量，相应评分项按缺失处理；"
              "如实说明“这段是 AI 写的、我还解释不了”，远好于把 AI 的输出当成自己的成果。")

    # ======================= 附录 A =======================
    doc.add_page_break()
    heading(doc, "附录 A  评分标准", 1)
    body(doc, "评分表对应正文第一至第九章与附录 C 的要求，六个维度合计 100 分。")

    table(doc,
          ["评价项目", "分值", "主要依据"],
          [
              ["基线搭建与原理理解", "15", "实现正确，能解释点集处理和逐点预测过程"],
              ["论文理解与模型改进", "10", "方法理解准确，改进针对明确问题，结构设计合理"],
              ["原始测试集表现", "15", "统一测试结果、各类别表现"],
              ["鲁棒性表现与实验分析", "30", "退化条件表现、消融证据、失败原因分析"],
              ["工程实现与可复现性", "15", "代码、配置、日志、一键复现与运行说明完整"],
              ["答辩", "15", "能独立解释、诊断和判断"],
          ],
          widths=[4.6, 1.6, 9.0])
    caption(doc, "表 A-1  评分维度与分值")

    body(doc, "成绩会同时考虑绝对性能、相对基线的变化和实验的可信度；鲁棒性上的提升不能只依赖"
              "某一个有利的测试条件，也需要报告它对原始数据表现和计算成本的影响。")

    # ======================= 附录 B =======================
    heading(doc, "附录 B  进阶探索（非必做）", 1)
    body(doc, "附录 B 里的内容不在基础达标要求之内，把必做内容做完之后可以继续做，不做也不扣分；"
              "额外堆叠模块或者增加训练资源本身不构成加分依据；"
              "第六章表 6-1 的第三组实验（至少一组核心模块消融）为必做，本附录只列更多可选组合。")

    heading(doc, "B.1  更多消融组合（选做，必做部分见第六章）", 2)
    body(doc, "消融要对应你自己的核心主张；如果认为相对位置编码有效，就应当比较加入和移除该编码的结果；"
              "如果认为局部注意力有效，就应当设计一个合理的替代聚合方式再做比较，"
              "例如在完全相同的邻域上改用 max 池化。")
    body(doc, "理想的消融有三个特征，只改一个变量、参数量尽量接近、结论能被重复；"
              "如果改进版本比基线多了 40% 参数，那么性能提升就无法归因到“注意力”这项设计上。")
    body(doc, "可选的进一步组合包括块数（1 块与 2 块）、注意力头数、邻域大小 k、"
              "全局池化方式（max 与注意力池化）、是否使用 RBF 相对位置编码；"
              "参考实现 configs/ 下已经备好 rpa_no_rpe、rpa_local_max、rpa_1block、rpa_attnpool 四组配置。")

    heading(doc, "B.2  计算成本报告", 2)
    body(doc, "报告里要给出参数量、相同硬件与相同输入规模下的单帧推理时间和显存占用；"
              "注意力模块的显存开销为 O(N·k)，N 取 8192、k 取 16 时还算可以接受，"
              "一旦改成全连接注意力，复杂度变成 O(N²)，显存会立刻被占满；"
              "这个现象本身就值得在报告里讨论。")

    heading(doc, "B.3  可视化", 2)
    body(doc, "至少提供三个成功案例与三个失败案例的预测可视化，每张图都包含原始点云、真实标签、"
              "模型预测，以及你对该错误的归因；失败案例比成功案例更能体现理解深度。")

    heading(doc, "B.4  扩展方向", 2)
    for s in [
        "跨场景泛化，在校园数据上训练，看能否迁移到其他的道路场景；",
        "多帧点云融合，把相邻帧叠加起来，观察稀疏条件下的改善与代价；",
        "更复杂的退化组合，例如“稀疏 + 扇区缺失 + 噪声”三重叠加。",
    ]:
        bullet(doc, s)
    body(doc, "扩展工作同样需要提供清晰的证据，只写一句“我试过了”并不够。")

    # ======================= 附录 C =======================
    doc.add_page_break()
    heading(doc, "附录 C  代码资料与运行说明", 1)

    body(doc, "随题下发的代码目录 数据与代码资料/ 里包含一个参考实现 roadseg，"
              "它实现了本题要求的全部基础组件，标签映射、退化模拟、指标计算、PointNet 基线、"
              "带相对位置注意力的改进模型、训练与评估脚本都在里面。")

    note_box(doc, "使用边界",
             "这份代码是出题方的参考实现，用途是让你有一个正确、可运行、口径统一的起点，"
             "避免把时间花在“标签到底怎么对”“mIoU 到底怎么算”这类跟题目能力点无关的事情上。"
             "你可以直接使用、修改、替换其中的任何部分。基线部分（roadseg/models/pointnet.py）"
             "与改进部分（roadseg/models/rpa.py）有不同的要求。基线四段主干必须自己写，"
             "见第三章的边界说明；改进模块可以借鉴、修改、替换 rpa.py，但要讲清改了哪里。")

    heading(doc, "C.1  目录结构", 2)
    code_block(doc, """数据与代码资料/
├── train.py                    训练（配置驱动，支持 --resume 断点续训）
├── evaluate.py                 在原始 + 14 个退化档位上评估
├── visualize.py                预测可视化（真值 / 预测 / 错误三联图）
├── requirements.txt
├── README.md                   环境、数据、命令、取舍的完整说明
├── configs/
│   ├── pointnet.yaml           基线
│   ├── rpa_full.yaml           完整改进模型
│   ├── rpa_no_rpe.yaml         消融：去掉相对位置编码
│   ├── rpa_local_max.yaml      消融：注意力换成 max 池化
│   ├── rpa_1block.yaml         消融：只留一个注意力块
│   └── rpa_attnpool.yaml       消融：全局池化换成注意力池化
├── docs/
│   └── make_doc.py             本题面（.docx）的生成脚本
├── roadseg/
│   ├── labels.py               五类定义、原始标签 → 课程标签的映射
│   ├── datasets.py             读 npz、空间范围过滤、定长采样、manifest 列定义
│   ├── sampling.py             随机 / 等间隔 / FPS / 体素 / 分层采样
│   ├── degradation.py          四类退化 + 14 个标准档位
│   ├── metrics.py              混淆矩阵、IoU、mIoU、OA、FW-IoU
│   ├── utils.py                种子、归一化、kNN、日志
│   ├── registry.py             模型注册表
│   ├── evaluation.py           退化评估循环（train.py 与 evaluate.py 共用）
│   └── models/
│       ├── pointnet.py         PointNet 基线（四段主干）
│       └── rpa.py              相对位置注意力改进模型 + 6 个消融配置
├── tools/
│   ├── prepare_data.py         原始 .bin/.label → npz + manifest + splits
│   └── make_dummy_data.py      生成合成数据，用于跑通整条链路
└── tests/
    ├── conftest.py             把仓库根加入 sys.path
    ├── test_models.py          形状、交换不变性、RPE 是否真的生效（需要 torch）
    └── test_pipeline.py        标签映射、退化、采样、指标、manifest（纯 numpy）""")

    body(doc, "train.py 与 evaluate.py 共用 roadseg/evaluation.py，所以训练日志里的 val mIoU 和 "
              "evaluate.py 给出的数字属于同一套口径，退化种子相同（base 加帧序号），"
              "计数方式也相同（按帧计，不按 batch 计）；不然训练时看到的趋势会和最终报告里的数字对不上。")

    heading(doc, "C.2  环境", 2)
    code_block(doc, """python -m venv .venv
.venv\\Scripts\\activate            # Windows
pip install -r requirements.txt

# 无 GPU 时安装 CPU 版 torch（体积小很多）
pip install torch --index-url https://download.pytorch.org/whl/cpu""")
    body(doc, "依赖只有 torch、numpy、PyYAML 和 pytest，matplotlib 仅在运行 visualize.py 画预测图时需要；"
              "不依赖 scipy，也不依赖任何点云专用库，因此在普通的笔记本上也能跑。")

    heading(doc, "C.3  三条命令", 2)
    body(doc, "先用合成数据花半分钟确认环境没有问题，再拿真实数据跑。")
    code_block(doc, """# 1) 合成数据自检：不下载任何东西，只验证代码能跑通
python tools/make_dummy_data.py --out data/dummy --frames 15

python train.py --config configs/pointnet.yaml \\
    --data-root data/dummy --epochs 2 --limit-frames 6

python evaluate.py --ckpt runs/pointnet_seed0/best.pth \\
    --data-root data/dummy --max-frames 2

# 2) 真实数据：先只抽查 20 帧，确认标签映射正确
python tools/prepare_data.py --raw D:/SemanticPOSS/dataset/sequences \\
    --out data/course --check

# 3) 正式训练 + 全档位评估
python train.py --config configs/pointnet.yaml
python evaluate.py --ckpt runs/pointnet_seed0/best.pth \\
    --split test --save-json results/pointnet_test.json""")

    body(doc, "评估脚本自带 14 个退化档位，可以用 --list-suites 查看，用 --suite 只跑其中几个。")
    code_block(doc, """python evaluate.py --ckpt runs/pointnet_seed0/best.pth --list-suites

python evaluate.py --ckpt runs/pointnet_seed0/best.pth \\
    --suite clean,sparse_50,sparse_25,sparse_12.5""")

    body(doc, "visualize.py 会画出“真实标签 / 模型预测 / 错误点”的三联图，并写出一个按错误率排序的 "
              "index.csv，方便你挑出成功案例与失败案例。")
    code_block(doc, """python visualize.py --ckpt runs/rpa_full_seed0/best.pth --num 6

python visualize.py --ckpt runs/pointnet_seed0/best.pth \\
    --frames 03_000100,03_000233 --suite sparse_25+noise_5cm""")

    heading(doc, "C.4  14 个退化档位", 2)
    table(doc,
          ["档位", "含义"],
          [
              ["clean", "原始测试数据，不做任何扰动"],
              ["sparse_50 / sparse_25 / sparse_12.5", "保留 50% / 25% / 12.5% 的点"],
              ["sector_60deg", "删除正前方 60°（方位角 −30°—+30°）扇区内的所有点"],
              ["box_6m", "删除以原点为中心、6 m × 6 m × 3 m 长方体内的一切点"],
              ["noise_2cm / noise_5cm / noise_10cm", "每轴独立叠加 N(0, σ²) 高斯噪声，σ = 2 / 5 / 10 cm"],
              ["outlier_64 / outlier_256", "在已有点的局部邻域内（距某个原始点 0.5—2 m）插入 64 / 256 个随机方向的异常点"],
              ["sparse_25+noise_5cm", "组合，保留 25% 的点并叠加 5 cm 噪声"],
              ["sparse_12.5+noise_10cm", "组合，保留 12.5% 的点并叠加 10 cm 噪声"],
              ["sector+outlier", "组合，删除扇区并插入离群点"],
          ],
          widths=[5.6, 9.6])
    caption(doc, "表 C-1  参考实现内置的 14 个退化档位")

    body(doc, "这 14 档对应了正文第四章的四类挑战，另外还加了组合条件；真正评分时用的隐藏参数"
              "跟这 14 档不完全一样，但性质一致，所以把超参针对某一档调到最好并没有多少意义，"
              "让整条曲线都更稳才是目标。")

    heading(doc, "C.5  改进模型的四段主干对照", 2)
    body(doc, "题目要求基线必须具备四段主干，参考实现里的对应关系整理在表 C-2 中，"
              "你可以拿它来检查自己的实现有没有漏掉哪一段。")

    table(doc,
          ["题目要求", "参考实现中的位置", "张量形状"],
          [
              ["共享的逐点特征提取", "PointNetSeg.local_mlp\n（Conv1d(k=1) 实现的共享 MLP）", "(B, 3, N) → (B, 64, N)"],
              ["对称的全局特征聚合", "PointNetSeg.global_mlp + 沿 N 取 max", "(B, 64, N) → (B, 1024)"],
              ["局部与全局特征融合", "沿通道拼接 local 与 broadcast 后的 global", "(B, 1088, N)"],
              ["逐点类别预测", "seg_head（三层 Conv1d(k=1)）", "(B, 1088, N) → (B, 5, N)"],
          ],
          widths=[3.8, 6.2, 5.2])
    caption(doc, "表 C-2  四段主干与张量形状")

    body(doc, "改进模型（rpa）只在这四段之间的一个位置插进了注意力块，位置在“逐点特征”之后、"
              "“全局聚合”之前，对每个点的 k 个近邻做多头注意力，其余四段保持不变，"
              "因此性能上一旦有变化就能归到注意力模块身上，可以排除“换了个更大的网络”这种解释。"
              "所有消融版本的参数量都落在 1.30 M 到 1.49 M 之间，量级是相同的。")

    heading(doc, "C.6  已知取舍", 2)
    body(doc, "参考实现里有几处刻意的工程取舍，读代码的时候容易误解，事实层面的说明单独列出。")
    for s in [
        "不做尺度归一化，只按 xy 均值做平移，z 不平移；因为退化测试扰动的是物理坐标，"
        "噪声在厘米级、扇区在米级，一旦做了缩放，2 cm 的噪声就变成无量纲的 0.05，"
        "跟题目想考的物理含义对不上。",
        "训练时不做“局部挖空”和“离群点”增强；这两类增强属于改进方向，如果加进基线，"
        "基线就不再公平。",
        "kNN 用暴力分块计算，不引入 scipy；实测（纯 numpy，k=16）在 n=4096 时约 0.42 s、"
        "n=8192 时约 1.49 s、n=20000 时约 8.6 s；因此 rpa 系列在 CPU 上训练很慢，"
        "建议先用小点数（比如 4096）和少量帧把流程跑通，再决定是否上 GPU 或者减少 epoch。",
        "注意力块只在 k 个近邻上计算，复杂度为 O(N·k)，全连接注意力并没有被使用；"
        "实现上的细节请你自己去读 roadseg/models/rpa.py，并在答辩时讲出来。",
    ]:
        bullet(doc, s)

    heading(doc, "C.7  自检清单", 2)
    body(doc, "提交之前，可以照着表 C-3 检查一遍。")
    table(doc,
          ["检查项", "通过标准"],
          [
              ["环境能装起来", "pip install -r requirements.txt 后没有报错（含 pytest）"],
              ["单元测试通过", "python -m pytest tests -q 全部通过"],
              ["合成数据能跑通", "make_dummy_data → train → evaluate 三条命令全部成功"],
              ["标签映射正确", "train.txt 里的帧经过映射后，五类占比之和接近 1，且地面占多数"],
              ["评测口径一致", "两个模型使用同一个 --num-points、同一个 --split、同一个 seed"],
              ["测试集只用一次", "报告里的测试集数字来自最终那次 evaluate"],
              ["日志与配置齐全", "runs/ 下每份实验都有 config.json 与 train.log.jsonl"],
              ["权重可复现", "给定 config.json 与 seed，重新训练能得到相近的指标"],
          ],
          widths=[4.6, 10.6])
    caption(doc, "表 C-3  提交前自检清单")

    hrule(doc)
    body(doc, "如有疑问，请在提交之前联系出题方确认，不要自行猜测口径。",
         first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    body(doc, "数据与代码资料的引用方式为，本项目使用 SemanticPOSS 数据集"
              "（Pan et al., arXiv:2002.09147），遵循 CC BY-NC-SA 3.0 许可，仅用于非商业教学。",
         first_indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=9)

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    try:
        doc.save(str(out))
    except PermissionError:
        # 目标文件正被 Word 打开锁住：换一个名字保存，不让整轮生成白费
        out = out.with_name(out.stem + "_new" + out.suffix)
        doc.save(str(out))
        print("目标文件被占用（可能正在 Word 中打开），已改存到备用文件名。")
    print(f"已生成：{out}")
    print(f"大小：{out.stat().st_size / 1024:.0f} KB")
    return out


if __name__ == "__main__":
    build()
