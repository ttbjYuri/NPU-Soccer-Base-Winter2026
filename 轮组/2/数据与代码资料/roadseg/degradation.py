"""点云退化扰动库（只依赖 numpy）。

本模块实现题目"三、鲁棒性挑战"中要求的四类退化，并提供组合退化的入口。
设计要点
--------
1. 所有扰动都在**同一帧点云**上以确定性随机数生成器完成：给定相同的
   ``seed`` 与 ``spec``，任何机器上得到的结果完全一致，便于隐藏测试复现。
2. 扰动分两类：
   - **删点类**（稀疏、局部缺失）：只改变点的子集，保留点的坐标不变；
   - **改点类**（坐标噪声）：保留点的数量不变，但坐标被扰动。
3. 离群点插入会**新增**点。新增点不参与语义评测，因此 ``DegradedCloud.keep``
   对这新点恒为 ``False``，``labels`` 置为 ``IGNORE``（255）。
   这一点对应题目原文："测试只评价保留下来的原始点；新增离群点不计入语义指标"。

典型用法
--------
>>> import numpy as np
>>> from roadseg.degradation import degrade, SparsitySpec, NoiseSpec
>>> pts = np.random.rand(1000, 3).astype("float32")
>>> lab = np.random.randint(0, 5, size=1000).astype("int64")
>>> out = degrade(pts, lab, [SparsitySpec(0.25), NoiseSpec(sigma=0.05)], seed=0)
>>> out.points.shape, out.labels.shape, out.keep.sum()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

IGNORE: int = 255
"""不参与评测的标签值（离群点、未标注点）。"""


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class DegradedCloud:
    """一帧退化后的点云及其评测信息。"""

    points: np.ndarray          # (M, 3) float32，送入模型的坐标
    labels: np.ndarray          # (M,) int64，255 表示不参与评测
    keep: np.ndarray            # (M,) bool，True 表示"原始点且要求评测"
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.points.ndim == 2 and self.points.shape[1] == 3
        assert self.labels.shape == (self.points.shape[0],)
        assert self.keep.shape == (self.points.shape[0],)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_evaluated(self) -> int:
        return int(self.keep.sum())

    def summary(self) -> dict:
        return {
            "num_points": len(self),
            "num_evaluated": self.num_evaluated,
            "num_inserted": len(self) - self.num_evaluated,
            **self.info,
        }


# --------------------------------------------------------------------------- #
# 扰动配置
# --------------------------------------------------------------------------- #
@dataclass
class SparsitySpec:
    """随机稀疏：按比例保留点。``ratio`` 为保留比例（0, 1]。"""

    ratio: float

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio <= 1.0:
            raise ValueError(f"SparsitySpec.ratio 必须在 (0, 1] 内，收到 {self.ratio}")


@dataclass
class SectorSpec:
    """视角扇区缺失：删除方位角落在 [az_start, az_end) 内的点。

    Parameters
    ----------
    az_start, az_end:
        方位角，单位为度，取值 [-180, 180)，0 度指向 +x，逆时针为正。
        允许 az_start > az_end 表示跨 ±180 度边界的扇区。
    r_max:
        只删除半径小于 ``r_max`` 的点；``None`` 表示不限制。
    mode:
        ``"drop"`` 直接删除；``"keep"`` 反向保留（只留扇区内），便于构造极端测试。
    """

    az_start: float
    az_end: float
    r_max: float | None = None
    mode: str = "drop"


@dataclass
class BoxSpec:
    """区域缺失：删除立方体区域内的点。``center``/``size`` 为米。"""

    center: Sequence[float]
    size: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.center) != 3 or len(self.size) != 3:
            raise ValueError("BoxSpec.center 与 size 都必须是长度 3 的序列")
        if any(s <= 0 for s in self.size):
            raise ValueError("BoxSpec.size 的每个分量必须为正")


@dataclass
class NoiseSpec:
    """坐标噪声：对 xyz 施加零均值高斯扰动，``sigma`` 单位为米。"""

    sigma: float
    per_axis: str = "sigma"   # "sigma" 各轴同方差；"iso" 各轴 sigma/sqrt(3)，总模长近似 sigma

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("NoiseSpec.sigma 不能为负")


@dataclass
class OutlierSpec:
    """离群点插入：向场景中额外注入 ``n`` 个异常点。

    ``mode``
        * ``"local"``（默认）：以随机原始点为锚点，在 ``r_min``—``r_max`` 米的
          局部邻域内撒点。离群点会贴着地面、车辆表面出现，真正干扰局部特征；
          这与题面第四章"距某个原始点 0.5—2 m"的表述一致；
        * ``"uniform"``：在整帧轴对齐包围盒内均匀采样。SemanticPOSS 一帧最远
          可达 80 m，大部分点会落在没有真值的空区域，考核强度很弱，保留仅供对比；
        * ``"shell"``：在半径 ``radius`` 的球壳上采样（更"突兀"的干扰）；
        * ``"line"``：沿包围盒对角线注入（模拟传感器鬼影）。
    """

    n: int
    mode: str = "local"
    radius: float | None = None
    margin: float = 0.0
    """uniform 模式下把包围盒按比例外扩，0.0 表示严格落在原点云包围盒内。"""

    r_min: float = 0.5
    r_max: float = 2.0
    """local 模式下离群点到锚点的距离区间（米）。"""

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("OutlierSpec.n 不能为负")
        if self.r_max < self.r_min:
            raise ValueError("OutlierSpec.r_max 不能小于 r_min")
        if self.r_min < 0:
            raise ValueError("OutlierSpec.r_min 不能为负")


Spec = SparsitySpec | SectorSpec | BoxSpec | NoiseSpec | OutlierSpec


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def azimuth_deg(points: np.ndarray) -> np.ndarray:
    """返回每个点的方位角（度），范围 [-180, 180)。"""
    return np.degrees(np.arctan2(points[:, 1], points[:, 0])).astype(np.float32)


def _in_sector(az: np.ndarray, start: float, end: float) -> np.ndarray:
    """判断方位角是否落在 [start, end) 内，支持跨 ±180 度。"""
    az = (az - start) % 360.0
    span = (end - start) % 360.0
    if span == 0:            # 视为整圈
        return np.ones_like(az, dtype=bool)
    return az < span


def _in_box(points: np.ndarray, center: Sequence[float], size: Sequence[float]) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    half = np.asarray(size, dtype=np.float32) / 2.0
    return np.all(np.abs(points - c) <= half, axis=1)


# --------------------------------------------------------------------------- #
# 单步扰动
# --------------------------------------------------------------------------- #
def apply_sparsity(points, labels, keep, spec: SparsitySpec, rng) -> tuple:
    n = points.shape[0]
    m = int(round(n * spec.ratio))
    m = max(1, min(m, n))
    idx = np.sort(rng.choice(n, size=m, replace=False))
    return points[idx], labels[idx], keep[idx]


def apply_sector(points, labels, keep, spec: SectorSpec, rng) -> tuple:
    az = azimuth_deg(points)
    inside = _in_sector(az, spec.az_start, spec.az_end)
    if spec.r_max is not None:
        inside &= np.linalg.norm(points[:, :2], axis=1) < float(spec.r_max)
    sel = inside if spec.mode == "keep" else ~inside
    if not sel.any():                       # 防御：极端参数下不要产出空点云
        return points, labels, keep
    return points[sel], labels[sel], keep[sel]


def apply_box(points, labels, keep, spec: BoxSpec, rng) -> tuple:
    inside = _in_box(points, spec.center, spec.size)
    sel = ~inside
    if not sel.any():
        return points, labels, keep
    return points[sel], labels[sel], keep[sel]


def apply_noise(points, labels, keep, spec: NoiseSpec, rng) -> tuple:
    if spec.per_axis == "iso":
        sigma = spec.sigma / np.sqrt(3.0)
        noise = rng.normal(0.0, sigma, size=points.shape)
    else:
        noise = rng.normal(0.0, spec.sigma, size=points.shape)
    return (points + noise.astype(np.float32)), labels, keep


def apply_outlier(points, labels, keep, spec: OutlierSpec, rng) -> tuple:
    """插入离群点。

    默认模式 ``"local"`` 把新点撒在**已有点的局部邻域**里（距某个随机锚点
    ``spec.r_min``—``spec.r_max`` 米）。这是有意为之：早先的实现取整帧轴对齐
    包围盒均匀撒点，SemanticPOSS 一帧最远 80 m，64/256 个点几乎全落在
    40—80 m 的空地上，模型切掉的空区域本来就没有真值，等于没考。
    改成局部邻域后，离群点会贴着地面/车辆的表面出现，才真正干扰局部特征。

    另外两种模式保留给想做对比的同学：
    ``"uniform"`` 整帧包围盒均匀撒点（老行为），``"shell"`` 半径 r 的球面，
    ``"line"`` 沿包围盒对角线。
    """
    n = int(spec.n)
    if n <= 0 or points.shape[0] == 0:
        return points, labels, keep

    lo = points.min(axis=0)
    hi = points.max(axis=0)

    if spec.mode in ("local", "neighbor"):
        # 随机锚点 + 随机方向 + [r_min, r_max] 的半径
        anchor = points[rng.integers(0, points.shape[0], size=n)]
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        r = rng.uniform(float(spec.r_min), float(spec.r_max), size=(n, 1))
        new = anchor + v * r
    elif spec.mode == "shell":
        r = float(spec.radius if spec.radius is not None else np.linalg.norm(hi - lo) / 2.0)
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        new = v * r
    elif spec.mode == "line":
        direction = hi - lo
        direction /= np.linalg.norm(direction) + 1e-12
        t = rng.uniform(0.0, float(np.linalg.norm(hi - lo)), size=(n, 1))
        new = lo + direction * t
    else:  # uniform：整帧包围盒（含 margin 外扩）均匀撒点
        pad = (hi - lo) * float(spec.margin)
        new = rng.uniform(lo - pad, hi + pad, size=(n, 3))

    new = new.astype(np.float32)
    pts = np.concatenate([points, new], axis=0)
    lab = np.concatenate([labels, np.full(n, IGNORE, dtype=np.int64)])
    kp = np.concatenate([keep, np.zeros(n, dtype=bool)])
    return pts, lab, kp


_HANDLERS = {
    SparsitySpec: apply_sparsity,
    SectorSpec: apply_sector,
    BoxSpec: apply_box,
    NoiseSpec: apply_noise,
    OutlierSpec: apply_outlier,
}


# --------------------------------------------------------------------------- #
# 组合入口
# --------------------------------------------------------------------------- #
def degrade(
    points: np.ndarray,
    labels: np.ndarray,
    specs: Iterable[Spec] | None = None,
    seed: int = 0,
) -> DegradedCloud:
    """按给定顺序施加一组扰动。

    Parameters
    ----------
    points : (N, 3) float32
        原始坐标。
    labels : (N,) int
        原始语义标签，取值 0..C-1，未标注点用 ``IGNORE``。
    specs : 可迭代的 Spec
        按顺序施加；``None`` 或空表示不做任何扰动。
    seed : int
        随机种子。**同一 seed + 同一 specs 必须给出完全相同的结果。**

    Returns
    -------
    DegradedCloud

    Notes
    -----
    施加顺序被固定为"先删点、后改点、最后加点"，与题目描述一致：
    稀疏/缺失改变点的集合，噪声改变坐标，离群点只增不减。
    如果你确实需要别的顺序，请显式调用单步函数。
    """
    rng = np.random.default_rng(seed)
    pts = np.ascontiguousarray(points, dtype=np.float32)
    lab = np.ascontiguousarray(labels, dtype=np.int64)
    keep = lab != IGNORE
    info: dict = {}

    for spec in (specs or []):
        handler = _HANDLERS.get(type(spec))
        if handler is None:
            raise TypeError(f"不支持的扰动类型：{type(spec).__name__}")
        pts, lab, keep = handler(pts, lab, keep, spec, rng)
        info.setdefault("ops", []).append(type(spec).__name__)

    return DegradedCloud(points=pts, labels=lab, keep=keep, info=info)


# --------------------------------------------------------------------------- #
# 预置测试档位（与题目表格一一对应，隐藏测试可另行组合）
# --------------------------------------------------------------------------- #
def standard_suite() -> dict[str, list[Spec]]:
    """返回题目公开验证工具使用的默认档位表。

    键名为测试项名称，可直接用于报表与日志。
    """
    return {
        "clean": [],
        "sparse_50": [SparsitySpec(0.50)],
        "sparse_25": [SparsitySpec(0.25)],
        "sparse_12.5": [SparsitySpec(0.125)],
        "sector_60deg": [SectorSpec(az_start=-30.0, az_end=30.0)],
        "box_6m": [BoxSpec(center=(0.0, 0.0, 0.0), size=(6.0, 6.0, 3.0))],
        "noise_2cm": [NoiseSpec(0.02)],
        "noise_5cm": [NoiseSpec(0.05)],
        "noise_10cm": [NoiseSpec(0.10)],
        "outlier_64": [OutlierSpec(64)],
        "outlier_256": [OutlierSpec(256)],
        "sparse_25+noise_5cm": [SparsitySpec(0.25), NoiseSpec(0.05)],
        "sparse_12.5+noise_10cm": [SparsitySpec(0.125), NoiseSpec(0.10)],
        "sector+outlier": [SectorSpec(-30.0, 30.0), OutlierSpec(64)],
    }


def describe_suite() -> list[tuple[str, str]]:
    """人类可读的档位说明，用于生成评测报告的表头。"""
    table: dict[str, str] = {
        "clean": "原始测试集，无扰动",
        "sparse_50": "随机保留 50% 的点",
        "sparse_25": "随机保留 25% 的点",
        "sparse_12.5": "随机保留 12.5% 的点",
        "sector_60deg": "删除正前方 60°（±30°）扇区内的点",
        "box_6m": "删除原点处 6m×6m×3m 长方体内的点",
        "noise_2cm": "每轴独立叠加标准差 σ=2cm 的高斯噪声",
        "noise_5cm": "每轴独立叠加标准差 σ=5cm 的高斯噪声",
        "noise_10cm": "每轴独立叠加标准差 σ=10cm 的高斯噪声",
        "outlier_64": "在局部邻域（距锚点 0.5—2m）插入 64 个离群点",
        "outlier_256": "在局部邻域（距锚点 0.5—2m）插入 256 个离群点",
        "sparse_25+noise_5cm": "稀疏 25% + 5cm 噪声（组合退化）",
        "sparse_12.5+noise_10cm": "稀疏 12.5% + 10cm 噪声（组合退化）",
        "sector+outlier": "扇区缺失 + 局部离群点（组合退化）",
    }
    return list(table.items())
