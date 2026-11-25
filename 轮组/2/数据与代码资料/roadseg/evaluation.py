"""统一的评估流程：``train.py`` 与 ``evaluate.py`` 共用同一份实现。

为什么单独抽出来
----------------
早期版本在 ``train.py`` 里另写了一份 ``evaluate``，与 ``evaluate.py`` 有三处不一致：

1. 退化随机种子不同（``1000 + batch_index`` vs ``10_000 + frame_index``）；
2. ``max_frames`` 的计数口径不同（按 batch 计 vs 按帧计）；
3. 采样顺序不同（边退化边取 batch 内样本 vs 先退化再 stride 采样）。

结果是同一个 ``best.pth`` 在训练日志里打印的 val mIoU，和 ``evaluate.py`` 算出来的
对不上，而两者都自称"按 2.5 节的口径"。这类不一致不会报错，只会让人无法复现数字。
因此这里只保留一份实现，两个入口都调用它。

评测口径（与题面 2.5 节一一对应）
--------------------------------
* 退化施加在**原始物理坐标**上（先退化，后归一化），否则"删除 6 m 立方体"
  对应的物理尺寸会错；
* 每帧先退化，再等间隔 stride 采样到 ``num_points`` 点；不足则用全部剩余点，
  不做上采样、不做重复采样；
* 只统计 ``keep`` 为真的原始点（被删的点和插入的离群点都不进混淆矩阵）；
* 每帧的退化种子固定为 ``seed_base + 帧序号``，与 batch size 无关。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .datasets import CourseDataset, SamplingScope
from .degradation import describe_suite, degrade, standard_suite
from .labels import CLASS_NAMES, IGNORE, NUM_CLASSES
from .metrics import ConfusionMatrixAccumulator, top_confusions
from .sampling import stride_indices
from .utils import normalize_points

DEFAULT_SEED_BASE = 10_000
DEFAULT_NUM_POINTS = 8192


def frame_seed(seed_base: int, frame_index: int) -> int:
    """某一帧在某次评估中的退化种子。只依赖帧序号，与 batch 划分无关。"""
    return int(seed_base) + int(frame_index)


@torch.no_grad()
def evaluate_suites(
    model: nn.Module,
    data_root: str | Path,
    frame_ids: list[str],
    device: torch.device | str = "cpu",
    suites: list[tuple[str, list]] | None = None,
    num_points: int = DEFAULT_NUM_POINTS,
    max_frames: int = 0,
    seed_base: int = DEFAULT_SEED_BASE,
    scope: SamplingScope | None = None,
    verbose: bool = False,
) -> dict[str, dict]:
    """在若干退化档位上评估，返回 ``{档位名: {...}}``。

    Parameters
    ----------
    suites:
        形如 ``[("clean", []), ("sparse_25", [SparsitySpec(0.25)])]``。
        传 ``None`` 时只评 clean 一档。可用 ``standard_suite()`` 取全部 14 档。
    num_points:
        每帧送进模型的最大点数。退化后剩余点数不足此值时用全部剩余点。
    max_frames:
        只评前多少**帧**（不是 batch）。0 表示全部。
    seed_base:
        退化种子基数，默认 10000。同一帧在不同档位共用同一个种子，
        保证"档位之间只差退化类型"。
    """
    model.eval()
    device = torch.device(device)
    suites = suites or [("clean", [])]

    n_used = len(frame_ids)
    if max_frames and max_frames > 0:
        n_used = min(n_used, int(max_frames))
    if n_used == 0:
        raise ValueError("没有可评估的帧：frame_ids 为空")

    # num_points=None 表示读整帧，采样与退化都在本函数内做，
    # 避免 CourseDataset 先降采样一次、这里再降采样一次的口径错位。
    dataset = CourseDataset(
        data_root, list(frame_ids)[:n_used], num_points=None, scope=scope, training=False
    )

    # 循环顺序是「帧在外、档位在内」：每帧的 npz 只读一次，14 个档位复用同一份
    # 原始坐标。反过来的写法（档位在外）会让 500 帧测试集被读 7000 次，
    # 在机械盘上比前向计算还慢。
    accs = {name: ConfusionMatrixAccumulator(NUM_CLASSES, CLASS_NAMES, ignore_index=IGNORE)
            for name, _ in suites}
    kept_points = {name: 0 for name, _ in suites}

    for i in range(n_used):
        sample = dataset[i]
        pts0 = sample["points"]
        lab0 = sample["labels"]

        for name, specs in suites:
            pts, lab = pts0, lab0
            if specs:
                deg = degrade(pts0, lab0, specs, seed=frame_seed(seed_base, i))
                pts, lab, keep = deg.points, deg.labels, deg.keep
            else:
                keep = np.ones(pts0.shape[0], dtype=bool)

            # 退化之后再采样：先采样会让"删除扇区"之类的空间操作作用在稀疏点上。
            if num_points and pts.shape[0] > num_points:
                sel = stride_indices(pts.shape[0], num_points)
                pts, lab, keep = pts[sel], lab[sel], keep[sel]

            pts_n, _ = normalize_points(pts)
            x = torch.from_numpy(np.ascontiguousarray(pts_n, dtype=np.float32))[None].to(device)
            logits = model(x)
            pred = logits.argmax(-1)[0].cpu().numpy()

            accs[name].update(lab[keep], pred[keep])
            kept_points[name] += int(keep.sum())

    results: dict[str, dict] = {}
    for name, _ in suites:
        cm = accs[name].result()
        results[name] = {
            "miou": round(cm.miou(), 4),
            "oa": round(cm.oa(), 4),
            "fwiou": round(cm.fwiou(), 4),
            "per_class_iou": {
                c: (None if np.isnan(v) else round(float(v), 4))
                for c, v in zip(CLASS_NAMES, cm.iou())
            },
            "top_confusions": top_confusions(cm, k=5),
            "num_points": int(kept_points[name]),
            "num_frames": n_used,
        }
        if verbose:
            print(f"  {name:<26s} mIoU={results[name]['miou']:.4f}  "
                  f"OA={results[name]['oa']:.4f}  点数={kept_points[name]}")

    return results


def resolve_suites(names: list[str] | str | None, all_suites: dict | None = None) -> list[tuple[str, list]]:
    """把 ``--suite a,b`` 这样的名字列表解析为实际的档位定义。

    ``names`` 为空时返回全部档位；名字非法时抛 ``ValueError`` 并列出可用档位，
    而不是静默跳过。``names`` 允许直接传字符串（argparse 给的 ``--suite`` 就是
    字符串）——传字符串时会先包成列表，否则逐个字符地迭代，
    ``clean`` 会变成 ``['c','l','e','a','n']`` 然后报"未知档位"。
    """
    all_suites = all_suites or standard_suite()
    if not names:
        return [(k, v) for k, v in all_suites.items()]
    if isinstance(names, str):
        names = [names]

    wanted: list[str] = []
    for raw in names:
        wanted.extend([p.strip() for p in str(raw).split(",") if p.strip()])
    if not wanted:
        return [(k, v) for k, v in all_suites.items()]

    unknown = [n for n in wanted if n not in all_suites]
    if unknown:
        raise ValueError(
            f"未知的退化档位：{unknown}\n可用档位：\n{format_suite_table(all_suites)}"
        )
    return [(n, all_suites[n]) for n in wanted]


def format_suite_table(all_suites: dict | None = None) -> str:
    """把 ``standard_suite()`` 排成可读的表格，供 ``--list-suites`` 使用。"""
    all_suites = all_suites or standard_suite()
    rows = describe_suite()
    width = max((len(n) for n, _ in rows), default=10)
    lines = [f"共 {len(all_suites)} 个退化档位：", ""]
    for name, desc in rows:
        lines.append(f"  {name:<{width}s}  {desc}")
    return "\n".join(lines)
