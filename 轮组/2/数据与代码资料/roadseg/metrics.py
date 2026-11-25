"""逐点语义分割的评价指标（只依赖 numpy）。

统一口径
--------
* 混淆矩阵 ``cm[g, p]``：行是真实类别，列是预测类别。
* ``IoU_c = TP_c / (TP_c + FP_c + FN_c)``，其中 ``FP_c`` 只统计"其他类被判成 c"，
  ``FN_c`` 只统计"c 被判成其他类"，与题目附录 C 的公式一致。
* 某类在测试集中出现次数为 0 时，该类的 IoU 记为 ``nan`` 并从 mIoU 中剔除
  （不参与平均，也不按 0 计入），这是常用做法，需在报告中显式说明。
* 只传入 ``keep=True`` 的点；离群点、被删除的点都不进入混淆矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfusionMatrix:
    cm: np.ndarray                       # (C, C) int64
    class_names: list[str]
    ignore_index: int = 255

    @property
    def num_classes(self) -> int:
        return self.cm.shape[0]

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_pairs(
        cls,
        gt: np.ndarray,
        pred: np.ndarray,
        class_names: list[str],
        ignore_index: int = 255,
    ) -> "ConfusionMatrix":
        """由一对 (真值, 预测) 直接建混淆矩阵，忽略 ``ignore_index`` 与越界值。

        这是把"口径"写死在库里的最小单元：只要都走这个函数，就不会出现
        "某人的脚本忘了过滤 255"这种低级但致命的差异。手算核对也方便。
        """
        gt = np.asarray(gt).reshape(-1)
        pred = np.asarray(pred).reshape(-1)
        if gt.shape != pred.shape:
            raise ValueError(f"gt 与 pred 长度不一致：{gt.shape} vs {pred.shape}")
        c = len(class_names)
        valid = (gt >= 0) & (gt < c) & (gt != ignore_index)
        flat = gt[valid].astype(np.int64) * c + pred[valid].astype(np.int64)
        cm = np.bincount(flat, minlength=c * c).reshape(c, c).astype(np.int64)
        return cls(cm=cm, class_names=list(class_names), ignore_index=ignore_index)

    # -- 指标 ---------------------------------------------------------------
    def tp(self) -> np.ndarray:
        return np.diag(self.cm).astype(np.float64)

    def fp(self) -> np.ndarray:
        return self.cm.sum(axis=0).astype(np.float64) - self.tp()

    def fn(self) -> np.ndarray:
        return self.cm.sum(axis=1).astype(np.float64) - self.tp()

    def iou(self) -> np.ndarray:
        tp, fp, fn = self.tp(), self.fp(), self.fn()
        denom = tp + fp + fn
        out = np.full(self.num_classes, np.nan, dtype=np.float64)
        valid = denom > 0
        out[valid] = tp[valid] / denom[valid]
        return out

    def acc(self) -> np.ndarray:
        tp, fn = self.tp(), self.fn()
        denom = tp + fn
        out = np.full(self.num_classes, np.nan, dtype=np.float64)
        valid = denom > 0
        out[valid] = tp[valid] / denom[valid]
        return out

    def miou(self) -> float:
        v = self.iou()
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else float("nan")

    def oa(self) -> float:
        total = self.cm.sum()
        return float(np.diag(self.cm).sum() / total) if total else float("nan")

    def fwiou(self) -> float:
        v = self.iou()
        freq = self.cm.sum(axis=1).astype(np.float64)
        freq = freq / max(freq.sum(), 1.0)
        mask = ~np.isnan(v)
        if not mask.any():
            return float("nan")
        return float((freq[mask] * v[mask]).sum() / freq[mask].sum())

    # -- 输出 ---------------------------------------------------------------
    def per_class_table(self) -> list[dict]:
        iou, acc = self.iou(), self.acc()
        return [
            {
                "class_id": i,
                "class_name": self.class_names[i],
                "gt_points": int(self.cm[i].sum()),
                "pred_points": int(self.cm[:, i].sum()),
                "iou": None if np.isnan(iou[i]) else round(float(iou[i]), 4),
                "acc": None if np.isnan(acc[i]) else round(float(acc[i]), 4),
            }
            for i in range(self.num_classes)
        ]

    def as_dict(self) -> dict:
        return {
            "mIoU": round(self.miou(), 4),
            "OA": round(self.oa(), 4),
            "FWIoU": round(self.fwiou(), 4),
            "num_points": int(self.cm.sum()),
            "per_class": self.per_class_table(),
        }

    def to_markdown(self) -> str:
        rows = ["| 类别 | GT 点数 | 预测点数 | IoU | Acc |", "|---|---:|---:|---:|---:|"]
        for r in self.per_class_table():
            iou = "—" if r["iou"] is None else f"{r['iou']:.4f}"
            acc = "—" if r["acc"] is None else f"{r['acc']:.4f}"
            rows.append(
                f"| {r['class_name']} | {r['gt_points']} | {r['pred_points']} | {iou} | {acc} |"
            )
        rows.append(f"| **mIoU** | | | **{self.miou():.4f}** | |")
        return "\n".join(rows)

    def to_csv(self) -> str:
        lines = ["class_id,class_name,gt_points,pred_points,iou,acc"]
        for r in self.per_class_table():
            iou = "" if r["iou"] is None else f"{r['iou']:.6f}"
            acc = "" if r["acc"] is None else f"{r['acc']:.6f}"
            lines.append(
                f"{r['class_id']},{r['class_name']},{r['gt_points']},{r['pred_points']},{iou},{acc}"
            )
        return "\n".join(lines)

    @classmethod
    def merge(cls, others: "list[ConfusionMatrix]") -> "ConfusionMatrix":
        cm = sum((o.cm for o in others), np.zeros_like(others[0].cm))
        return cls(cm=cm, class_names=others[0].class_names, ignore_index=others[0].ignore_index)


class ConfusionMatrixAccumulator:
    """跨多帧累加混淆矩阵；帧数再多也只占 C×C 内存。"""

    def __init__(self, num_classes: int, class_names: list[str], ignore_index: int = 255):
        self.num_classes = num_classes
        self.class_names = class_names
        self.ignore_index = ignore_index
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        gt = np.asarray(gt, dtype=np.int64).reshape(-1)
        pred = np.asarray(pred, dtype=np.int64).reshape(-1)
        if gt.shape != pred.shape:
            raise ValueError(f"gt 与 pred 长度不一致：{gt.shape} vs {pred.shape}")

        valid = (gt >= 0) & (gt < self.num_classes)
        valid &= (pred >= 0) & (pred < self.num_classes)
        if not valid.any():
            return

        g = gt[valid]
        p = pred[valid]
        idx = g * self.num_classes + p
        counts = np.bincount(idx, minlength=self.num_classes ** 2)
        self.cm += counts.reshape(self.num_classes, self.num_classes)

    @property
    def total(self) -> int:
        return int(self.cm.sum())

    def result(self) -> ConfusionMatrix:
        return ConfusionMatrix(self.cm.copy(), list(self.class_names), self.ignore_index)

    def reset(self) -> None:
        self.cm[:] = 0


def evaluate_points(
    gt: np.ndarray,
    pred: np.ndarray,
    keep: np.ndarray,
    num_classes: int,
    class_names: list[str],
) -> ConfusionMatrix:
    """单帧评测：先按 ``keep`` 掩码筛选，再累加到混淆矩阵。"""
    keep = np.asarray(keep, dtype=bool)
    acc = ConfusionMatrixAccumulator(num_classes, class_names)
    acc.update(np.asarray(gt)[keep], np.asarray(pred)[keep])
    return acc.result()


def top_confusions(cm: ConfusionMatrix, k: int = 5) -> list[dict]:
    """最常见的 k 种错误（真实类别 → 预测类别），用于失败案例分析。"""
    pairs = []
    for g in range(cm.num_classes):
        for p in range(cm.num_classes):
            if g == p:
                continue
            n = int(cm.cm[g, p])
            if n:
                pairs.append(
                    {
                        "gt": cm.class_names[g],
                        "pred": cm.class_names[p],
                        "count": n,
                        "share_of_gt": round(n / max(int(cm.cm[g].sum()), 1), 4),
                    }
                )
    pairs.sort(key=lambda d: d["count"], reverse=True)
    return pairs[:k]
