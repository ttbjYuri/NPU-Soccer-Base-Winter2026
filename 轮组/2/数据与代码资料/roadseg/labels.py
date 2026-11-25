"""类别定义与 SemanticPOSS → 课程 5 类的统一映射规则。

这是全课程**唯一**的标签口径来源：数据转换、评测、可视化配色全部从这里读取，
任何地方都不要再手写类别名、颜色或原始标签 id。

为什么是 5 类而不是 6 类
------------------------
题目原本要求"道路、人行道、车辆、行人、建筑、植被"六类。但 SemanticPOSS
的标注体系里**没有独立的 road / sidewalk**：整个可行驶、可步行的硬化地面
只有一个类别 ``ground``（id 22），标注时没有区分车道与人行道。
若强行造一个"人行道"类，该类在所有划分里点数恒为 0，mIoU 会按 nan 剔除，
等于白占一个类目，还会让学生误以为数据有问题。

因此课程统一采用 **5 类**：

    0 地面 ground / 1 车辆 car / 2 行人 pedestrian / 3 建筑 building / 4 植被 vegetation

并在题目与数据说明里显式写明这次合并——这是数据集本身的限制，不是简化。

原始标签体系
------------
SemanticPOSS（Pan et al., 2020, arXiv:2002.09147，北京大学校园内采集）沿用
SemanticKITTI 的 ``*.label`` 格式：uint32，**低 16 位为语义标签、高 16 位为
实例编号**（实例只对 people / car / rider 三类标注）。语义标签 id 并不连续，
官方发布的 id → 名称如下：

===== ===================================== ====================================
id     名称                                  课程用途
===== ===================================== ====================================
0      unlabeled                             忽略
1      not know                              忽略
4      person（1 人）                         → 行人
5      persons（多人/人群）                   → 行人
6      rider                                  忽略（默认；见 ALT_RULES）
7      car                                    → 车辆
8      trunk（树干）                          忽略（默认；见 ALT_RULES）
9      plants（植被）                         → 植被
10/11/12 traffic sign（立牌/挂牌/大型挂牌）    忽略
13     pole（杆）                             忽略
14     trashcan（垃圾桶）                     忽略
15     building                               → 建筑
16     cone / stone（锥桶、石块）              忽略
17     fence（围栏）                          忽略
21     bike（自行车）                         忽略（默认；见 ALT_RULES）
22     ground（地面）                         → 地面
===== ===================================== ====================================

注意 id 2、3 与 23 以上在本数据集中**不存在**；``1`` 与 ``18/19/20`` 只是
"错标/未知"占位，不是真实类别。任何从 SemanticKITTI 抄来的标签映射
（例如 road=40、sidewalk=48、vegetation=70）在这份数据上都是**错的**。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IGNORE: int = 255
"""合并后的忽略标签。"""


# --------------------------------------------------------------------------- #
# 课程类别
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassInfo:
    index: int        # 训练与评测使用的类别编号 0..C-1
    name: str         # 中文名（报告、图表用）
    short: str        # 英文短名（文件名、日志用）
    color: tuple      # RGB 可视化配色


CLASSES: tuple[ClassInfo, ...] = (
    ClassInfo(0, "地面",  "ground",     (104, 104, 104)),
    ClassInfo(1, "车辆",  "car",        (60, 120, 200)),
    ClassInfo(2, "行人",  "pedestrian", (200, 60, 60)),
    ClassInfo(3, "建筑",  "building",   (150, 100, 60)),
    ClassInfo(4, "植被",  "vegetation", (70, 150, 80)),
)

NUM_CLASSES: int = len(CLASSES)
CLASS_NAMES: list[str] = [c.name for c in CLASSES]
CLASS_SHORTS: list[str] = [c.short for c in CLASSES]
CLASS_COLORS: np.ndarray = np.array([c.color for c in CLASSES], dtype=np.uint8)

IGNORE_COLOR: tuple = (200, 200, 200)
"""忽略点的可视化配色（浅灰）。"""


# --------------------------------------------------------------------------- #
# 原始标签表
# --------------------------------------------------------------------------- #
#: SemanticPOSS 官方 14 类的语义 id → 名称。用于反查、生成文档表格。
RAW_LABELS: dict[int, str] = {
    0: "unlabeled",
    4: "person",
    5: "persons",
    6: "rider",
    7: "car",
    8: "trunk",
    9: "plants",
    10: "traffic sign 1",
    11: "traffic sign 2",
    12: "traffic sign 3",
    13: "pole",
    14: "trashcan",
    15: "building",
    16: "cone/stone",
    17: "fence",
    21: "bike",
    22: "ground",
}

#: 占位/无效 id：存在但不对应任何类别，一律忽略。
INVALID_RAW_IDS: tuple[int, ...] = (1, 2, 3, 18, 19, 20)


@dataclass(frozen=True)
class MappingRule:
    """一条映射规则：把一组原始标签并入某一个课程类别。"""

    target: int | None
    """课程类别编号；``None`` 表示忽略。"""

    raw_ids: tuple[int, ...]
    """参与这条规则的原始标签。"""

    reason: str
    """写进数据说明文档的理由，必须让做题人看懂"为什么这么并/为什么忽略"。"""


#: 默认映射（方案 A：语义最保守，只并"同一个东西"）。
DEFAULT_RULES: tuple[MappingRule, ...] = (
    MappingRule(0, (22,),
                "ground → 地面。SemanticPOSS 把车道、人行道、广场等所有硬化地面"
                "合并为一个类，无法再细分。"),
    MappingRule(1, (7,),
                "car → 车辆。不含 bike，避免机动车与非机动车混为一类。"),
    MappingRule(2, (4, 5),
                "person + persons → 行人。标注时把单人与人群分成两个 id，"
                "语义上都是行人，必须合并。"),
    MappingRule(3, (15,),
                "building → 建筑。"),
    MappingRule(4, (9,),
                "plants → 植被。不含 trunk（树干单列会让树冠与树干被拆成两类，"
                "而本任务只要求「植被」这一个粗类）。"),
    MappingRule(None, (0, 1, 6, 8, 10, 11, 12, 13, 14, 16, 17, 21),
                "unlabeled / not know / rider / trunk / 三类交通标志 / pole / "
                "trashcan / cone-stone / fence / bike → 忽略；本任务只评测五类，"
                "其余不计入指标（与 SemanticPOSS 论文 11 类评测的精神一致）。"),
)

#: 备选映射（方案 B：把样本量少的类补厚，但语义变粗）。
#: 若希望"行人""车辆""植被"的点数更充足，可切换到这一套并在报告中说明。
ALT_RULES: tuple[MappingRule, ...] = (
    MappingRule(0, (22,), "ground → 地面。"),
    MappingRule(1, (7, 21), "car + bike → 车辆（含两轮机动车/非机动车）。"),
    MappingRule(2, (4, 5, 6), "person + persons + rider → 行人（含骑行者）。"),
    MappingRule(3, (15,), "building → 建筑。"),
    MappingRule(4, (8, 9), "trunk + plants → 植被（树干与树冠合并）。"),
    MappingRule(None, (0, 1, 10, 11, 12, 13, 14, 16, 17),
                "unlabeled / not know / 交通标志 / pole / trashcan / "
                "cone-stone / fence → 忽略。"),
)


def build_lut(rules: tuple[MappingRule, ...] = DEFAULT_RULES) -> np.ndarray:
    """构造 ``lut[256]`` 查表：``lut[raw_label] = 课程类别 或 255``。

    用查表而不是循环，是为了让 2 亿个点也能一次 ``lut[raw & 0xFFFF]`` 转完。
    未在规则里出现的 id 一律落在 ``IGNORE``（255），这是安全默认值：
    数据集里冒出来的未知 id 不会被误当成某一类。
    """
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for rule in rules:
        target = IGNORE if rule.target is None else rule.target
        for raw_id in rule.raw_ids:
            if not 0 <= raw_id < 256:
                raise ValueError(f"原始标签越界：{raw_id}")
            lut[raw_id] = target
    return lut


DEFAULT_LUT: np.ndarray = build_lut(DEFAULT_RULES)


def remap_labels(raw: np.ndarray, lut: np.ndarray = DEFAULT_LUT) -> np.ndarray:
    """把 SemanticKITTI 格式的 uint32 标签转成课程标签。

    Parameters
    ----------
    raw : np.ndarray
        ``*.label`` 文件读出的 uint32 数组（低 16 位语义、高 16 位实例）。
    lut : np.ndarray
        256 长度 uint8 查表。

    Returns
    -------
    np.ndarray
        int64 数组，取值 0..4 或 255。
    """
    raw = np.asarray(raw)
    semantic = (raw.astype(np.uint32) & 0xFFFF).astype(np.int64)
    semantic = np.clip(semantic, 0, 255)      # >255 的异常值按未知处理
    return lut[semantic].astype(np.int64)


def describe_mapping(rules: tuple[MappingRule, ...] = DEFAULT_RULES) -> list[tuple[str, str, str]]:
    """返回 ``(课程类别, 原始标签, 理由)`` 三元组，用于生成文档表格。"""
    rows = []
    for rule in rules:
        target = "（忽略）" if rule.target is None else CLASS_NAMES[rule.target]
        raws = "、".join(f"{i} {RAW_LABELS.get(i, '?')}" for i in rule.raw_ids)
        rows.append((target, raws, rule.reason))
    return rows


def class_distribution(labels: np.ndarray) -> dict[str, float]:
    """统计各类别占比（忽略 255），用于数据说明与数据校验。"""
    labels = np.asarray(labels).reshape(-1)
    valid = labels != IGNORE
    total = int(valid.sum())
    out: dict[str, float] = {}
    for c in CLASSES:
        n = int(((labels == c.index) & valid).sum())
        out[c.name] = n / total if total else 0.0
    return out
