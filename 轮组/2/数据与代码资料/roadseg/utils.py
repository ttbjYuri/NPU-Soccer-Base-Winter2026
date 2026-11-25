"""公共工具：随机种子、坐标归一化、kNN 邻域、JSON 日志与计时。

本模块**只依赖 numpy**。kNN 用分块暴力法实现，不引入 scipy，
避免做题人在 4—5 周里被额外的编译依赖卡住。
"""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# 随机性
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """一次性固定 python / numpy / torch 的随机性。

    ``deterministic=True`` 时还会关闭 cudnn 自动调优。代价是速度略降，
    换来"同一台机器上两次训练结果逐位一致"，这是复现性的底线。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:                                  # PyTorch >= 1.8
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 多进程时给每个 worker 派生不同的种子。"""
    base = 0 if os.environ.get("PYTHONHASHSEED") is None else int(os.environ["PYTHONHASHSEED"])
    np.random.seed(base + worker_id)


# --------------------------------------------------------------------------- #
# 坐标归一化
# --------------------------------------------------------------------------- #
def normalize_points(points: np.ndarray) -> tuple[np.ndarray, dict]:
    """把一帧点云平移到"水平中心在原点"，并返回还原所需的参数。

    只做平移、**不做缩放**：缩放到单位球会改变"厘米级噪声"的物理含义
    （σ=5cm 缩放到单位球后可能变成 σ=0.2），鲁棒性实验会失去意义。
    平移不改变任何距离量，只让输入落在零附近、更好优化。
    """
    center = points[:, :2].mean(axis=0)
    shifted = points.copy()
    shifted[:, 0] -= center[0]
    shifted[:, 1] -= center[1]
    return shifted, {"center_xy": [float(center[0]), float(center[1])]}


def denormalize_points(points: np.ndarray, meta: dict) -> np.ndarray:
    out = points.copy()
    cx, cy = meta["center_xy"]
    out[:, 0] += cx
    out[:, 1] += cy
    return out


def unit_scale(points: np.ndarray) -> float:
    """返回点云的最大水平半径，用于把 ``outlier shell`` 的半径设得合理。"""
    return float(np.linalg.norm(points[:, :2], axis=1).max()) if points.size else 0.0


# --------------------------------------------------------------------------- #
# kNN
# --------------------------------------------------------------------------- #
def knn_indices(points: np.ndarray, k: int, chunk: int = 2048,
                include_self: bool = True) -> np.ndarray:
    """分块暴力 kNN，返回 ``(N, k)`` 的 int64 下标。

    Parameters
    ----------
    points : (N, 3) float32
    k : int
        邻居数，包含自身。
    chunk : int
        每块处理多少个查询点，用来控制 ``chunk × N`` 的临时内存。
    include_self : bool
        True 时第 0 列一定是点自己。

    Notes
    -----
    复杂度 O(N²)，n=20k 约 0.3 s，n=100k 约 8 s。
    训练时不要每步都算——请在**数据预处理阶段**算一次并存盘，
    或用模型内部的体素/kd 结构。这个函数面向离线分析与可视化。
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    n = points.shape[0]
    k = min(k, n)
    out = np.empty((n, k), dtype=np.int64)
    kth = k - 1 if include_self else min(k, n - 1)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = points[start:end]
        # (B, N) 距离平方，避免 sqrt 的开销
        d = (
            (block ** 2).sum(1)[:, None]
            + (points ** 2).sum(1)[None, :]
            - 2.0 * block @ points.T
        )
        np.maximum(d, 0.0, out=d)
        idx = np.argpartition(d, kth=kth, axis=1)[:, :k]
        # argpartition 只保证第 k 位有序，块内再按距离排序
        take = np.take_along_axis(d, idx, axis=1)
        order = np.argsort(take, axis=1)
        out[start:end] = np.take_along_axis(idx, order, axis=1)
    return out


def neighbor_coords(points: np.ndarray, knn: np.ndarray) -> np.ndarray:
    """按 kNN 下标取邻居坐标：``(N, k, 3)``。供离线可视化/单元测试用。"""
    return points[knn]


# --------------------------------------------------------------------------- #
# 计时与日志
# --------------------------------------------------------------------------- #
@contextmanager
def timer(name: str = "", verbose: bool = True):
    """``with timer("train epoch"):`` 打印耗时，同时可读 ``t.elapsed``。"""
    t0 = time.perf_counter()

    class _T:
        elapsed = 0.0

    t = _T()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - t0
        if verbose:
            print(f"[time] {name}: {t.elapsed:.3f}s")


class JsonlLogger:
    """极简 JSONL 日志：一行一条记录，方便 pandas / Excel 直接读。

    为什么不用 tensorboard：课程交付要求"训练日志"，纯文本既方便学生贴进
    报告，也方便出题方写脚本批量核对，不需要额外开服务。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")

    def log(self, **kwargs) -> None:
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs}
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_seconds(sec: float) -> str:
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}min"
    return f"{sec / 3600:.2f}h"
