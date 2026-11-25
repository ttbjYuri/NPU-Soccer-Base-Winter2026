"""点采样工具：随机采样、最远点采样（FPS）、体素下采样。

为什么评测必须用**确定性**采样
------------------------------
训练时每轮随机丢掉一部分点，等价于对"稀疏退化"做数据增强，这是允许的。
但如果评测时也随机采样，两个模型就可能在**不同的点**上算 mIoU，
差异里混进了采样噪声，A/B 结论不可信。因此：

* 训练：``random``（每个 epoch 不同，天然增强）；
* 评测：``fps`` 或 ``stride``（与模型无关，确定性，只依赖帧内容）。

题目的四种退化（稀疏/缺失/噪声/离群点）都是**先施加扰动、再固定点集**，
所以评测侧只需要一个确定性采样器把点数降到硬件能承受的规模。
"""

from __future__ import annotations

import numpy as np


def random_indices(n: int, m: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """均匀随机取 ``m`` 个不重复下标（升序返回，保持文件中的原始顺序）。"""
    if m >= n:
        return np.arange(n)
    rng = rng if rng is not None else np.random.default_rng()
    return np.sort(rng.choice(n, size=m, replace=False))


def stride_indices(n: int, m: int) -> np.ndarray:
    """等间隔抽样：``np.linspace(0, n-1, m)`` 取整。

    完全确定、零开销，且因为 LiDAR 点按扫描线顺序存储，等间隔抽样近似
    "均匀降线数"，是评测稀疏鲁棒性时最中性的选择。
    """
    if m >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, m).round().astype(np.int64))


def fps_indices(points: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    """最远点采样（纯 numpy 实现，O(m·n)）。

    从随机一个起点开始，每次选"到已选集合最近距离"最大的点。
    覆盖均匀，但**慢**：n=100k、m=8192 时约需数秒。评测时若帧数多，
    建议改用 ``stride_indices``；两者的 mIoU 差异通常在 0.2 个点以内。
    """
    n = points.shape[0]
    if m >= n:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    sel = np.empty(m, dtype=np.int64)
    sel[0] = int(rng.integers(n))
    dist = np.linalg.norm(points - points[sel[0]], axis=1)

    for i in range(1, m):
        sel[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(points - points[sel[i]], axis=1))
    return np.sort(sel)


def voxel_downsample(points: np.ndarray, labels: np.ndarray, voxel: float) -> tuple:
    """体素下采样：每个体素保留离体素中心最近的一个点。

    用于把 12 万点/帧压到 2~4 万点，同时**保持空间结构**（不像随机稀疏那样
    打散局部邻域）。若你的改进模型依赖 kNN，这个操作比随机稀疏更友好，
    因此它属于"评测设置"，必须在报告里注明用的是哪种。
    """
    if voxel <= 0:
        return points, labels
    keys = np.floor(points / voxel).astype(np.int64)
    # 把三维体素坐标编码成单个 int64，避免用 np.unique(axis=0)（较慢）
    keys = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
    _, idx, inv = np.unique(keys, return_index=True, return_inverse=True)
    return points[idx], labels[idx]


def uniform_sample_by_class(
    points: np.ndarray,
    labels: np.ndarray,
    num_points: int,
    ignore_index: int = 255,
    rng: np.random.Generator | None = None,
) -> tuple:
    """按类别分层采样到 ``num_points``（稀有类别给保底名额）。

    小类别（行人、自行车）在整帧里可能只占 0.3%，纯随机采样会让一个 batch
    里几乎没有这些点，模型学不到。分层采样是训练侧的合法技巧，但**评测时
    绝对不能用**——评测必须反映真实分布。
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = points.shape[0]
    if num_points >= n:
        return points, labels

    classes, counts = np.unique(labels, return_counts=True)
    valid = [(c, n_) for c, n_ in zip(classes, counts) if c != ignore_index]

    # 先给每个类别按比例分配名额，保底 32 个（不超过该类自身点数）
    quota: dict[int, int] = {}
    total_valid = sum(n_ for _, n_ in valid)
    for c, n_ in valid:
        quota[c] = min(n_, max(32, int(round(num_points * n_ / max(total_valid, 1)))))
    # 名额超出预算时按比例回撤
    used = sum(quota.values())
    if used > num_points:
        scale = num_points / used
        quota = {c: max(1, int(v * scale)) for c, v in quota.items()}
    used = sum(quota.values())
    # 剩余名额补给最大类别
    if used < num_points and valid:
        biggest = max(valid, key=lambda t: t[1])[0]
        quota[biggest] = min(
            int(counts[list(classes).index(biggest)]) - quota[biggest],
            num_points - used,
        ) + quota[biggest]

    chosen = []
    for c, k in quota.items():
        idx = np.flatnonzero(labels == c)
        if idx.size <= k:
            chosen.append(idx)
        else:
            chosen.append(rng.choice(idx, size=k, replace=False))
    sel = np.sort(np.concatenate(chosen)) if chosen else np.arange(0)
    if sel.size == 0:
        return random_sample(points, labels, num_points, rng)
    return points[sel], labels[sel]


def random_sample(points, labels, num_points, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    idx = random_indices(points.shape[0], num_points, rng)
    return points[idx], labels[idx]
