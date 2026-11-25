"""SemanticPOSS 数据读取、课程划分与采样。

目录约定（由 ``tools/prepare_data.py`` 生成）
------------------------------------------------
::

    dataset/
      semanticposs/                 # 官方原始数据（不动它）
        sequences/00/velodyne/000000.bin
        sequences/00/labels/000000.label
        sequences/00/poses.txt
        ...
      course/                       # 本课程转换后的数据
        frames/00_000000.npz        # keys: points(float32 N,3), labels(uint8 N,), intensity(float32 N,)
        manifest.csv                # 每帧一行：frame_id, num_points, num_valid, ratio_*
      splits/
        train.txt                   # 每行一个 frame_id，如 00_000000
        val.txt
        test.txt

为什么先转成 npz
----------------
``.bin`` 是 4 列 float32（x, y, z, intensity），``.label`` 是 uint32，逐帧解码
一次约 2~5 ms。课程里评测会反复跑十几种退化档位，预先转成 npz（坐标与标签
一次解码、一次重映射）能省掉 90% 的 IO 与 CPU 时间，让 4—5 周内的实验可能
跑得完。转换只在本地做一次，不影响任何评测口径。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import DEFAULT_LUT, IGNORE, remap_labels
from .sampling import random_indices, stride_indices


# --------------------------------------------------------------------------- #
# 原始二进制读取
# --------------------------------------------------------------------------- #
def read_bin(path: str | Path, with_intensity: bool = True) -> np.ndarray:
    """读取 KITTI 风格 ``*.bin``：float32，(N, 4) = x, y, z, intensity。

    ``with_intensity=False`` 时返回 (N, 3)，只保留坐标。
    """
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path} 不是 4 的整数倍长度，疑似不是 KITTI 格式的 velodyne 文件")
    arr = raw.reshape(-1, 4)
    return arr if with_intensity else arr[:, :3]


def read_label(path: str | Path) -> np.ndarray:
    """读取 ``*.label``：uint32，(N,)，低 16 位语义、高 16 位实例。"""
    raw = np.fromfile(str(path), dtype=np.uint32)
    return raw


def load_frame(
    bin_path: str | Path,
    label_path: str | Path | None = None,
    lut: np.ndarray = DEFAULT_LUT,
) -> dict:
    """读取一帧原始数据并完成标签重映射。

    Returns
    -------
    dict with keys
        ``points`` (N,3) float32
        ``intensity`` (N,) float32
        ``labels`` (N,) int64   —— 0..4，或 255（忽略）
    """
    xyz_i = read_bin(bin_path)
    points = np.ascontiguousarray(xyz_i[:, :3], dtype=np.float32)
    intensity = np.ascontiguousarray(xyz_i[:, 3], dtype=np.float32)

    if label_path is None:
        labels = np.full(points.shape[0], IGNORE, dtype=np.int64)
    else:
        raw = read_label(label_path)
        if raw.shape[0] != points.shape[0]:
            raise ValueError(
                f"点数与标签数不一致：{bin_path} 有 {points.shape[0]} 点，"
                f"{label_path} 有 {raw.shape[0]} 个标签"
            )
        labels = remap_labels(raw, lut)
    return {"points": points, "intensity": intensity, "labels": labels}


# --------------------------------------------------------------------------- #
# 帧级数据集
# --------------------------------------------------------------------------- #
@dataclass
class SamplingScope:
    """评测空间范围。超出范围的点不送入模型，也不参与评测。

    建议直接用默认值（``None``），即"整帧全量"，这样最接近题目的描述：
    退化只来自我们施加的扰动，而不是我们自己又裁了一刀。
    只有在显存/时间吃紧时才收窄，并且必须在报告里写明。
    """

    r_max: float | None = None       # 水平半径上限（米）
    z_min: float | None = None
    z_max: float | None = None

    def mask(self, points: np.ndarray) -> np.ndarray:
        keep = np.ones(points.shape[0], dtype=bool)
        if self.r_max is not None:
            keep &= np.linalg.norm(points[:, :2], axis=1) <= self.r_max
        if self.z_min is not None:
            keep &= points[:, 2] >= self.z_min
        if self.z_max is not None:
            keep &= points[:, 2] <= self.z_max
        return keep


class CourseDataset:
    """按 frame_id 列表读取课程 npz，支持全帧与定长采样两种模式。

    Parameters
    ----------
    root : Path
        ``course/`` 目录，含 ``frames/``。
    frame_ids : list[str]
        ``["00_000000", ...]``。
    num_points : int | None
        ``None`` 表示用整帧（评测用）；否则随机/等距采样到固定点数（训练用）。
    scope : SamplingScope
        空间范围过滤。
    with_intensity : bool
        是否把 intensity 作为第 4 通道拼到坐标后（默认关闭，题目定义为 (x,y,z)）。
    """

    def __init__(
        self,
        root: str | Path,
        frame_ids: list[str],
        num_points: int | None = None,
        scope: SamplingScope | None = None,
        with_intensity: bool = False,
        training: bool = False,
    ):
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.frame_ids = list(frame_ids)
        self.num_points = num_points
        self.scope = scope or SamplingScope()
        self.with_intensity = with_intensity
        self.training = training
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(f"找不到 {self.frames_dir}，请先运行 tools/prepare_data.py")
        missing = [f for f in self.frame_ids[:5] if not (self.frames_dir / f"{f}.npz").exists()]
        if missing:
            raise FileNotFoundError(f"缺少帧文件，例如 {missing[0]}.npz")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> dict:
        frame_id = self.frame_ids[idx]
        z = np.load(self.frames_dir / f"{frame_id}.npz")
        points = z["points"].astype(np.float32)
        labels = z["labels"].astype(np.int64)
        intensity = z["intensity"].astype(np.float32) if "intensity" in z else None

        keep = self.scope.mask(points)
        if not keep.all():
            points, labels = points[keep], labels[keep]
            if intensity is not None:
                intensity = intensity[keep]

        if self.num_points is not None and points.shape[0] > self.num_points:
            if self.training:
                idx_sel = random_indices(points.shape[0], self.num_points)
            else:
                idx_sel = stride_indices(points.shape[0], self.num_points)
            points, labels = points[idx_sel], labels[idx_sel]
            if intensity is not None:
                intensity = intensity[idx_sel]

        feats = points
        if self.with_intensity and intensity is not None:
            # 强度通道的量化范围随传感器而异：SemanticPOSS 的 Pandora 侧原始值
            # 大致落在 0..255，而合成数据与部分导出工具给出 0..1。
            # 这里按"整帧最大值"自适应归一到 0..1，避免把 0..1 的数据再除 255
            # 变成几乎全 0 的死通道。
            iv = intensity.astype(np.float32)
            imax = float(iv.max()) if iv.size else 0.0
            if imax > 1.5:                     # 量级上像 0..255
                iv = iv / max(imax, 1e-6)
            feats = np.concatenate([points, iv[:, None]], axis=1).astype(np.float32)

        return {
            "frame_id": frame_id,
            "points": points,          # 原始 xyz，评测与可视化用
            "features": feats,         # 送进模型的输入
            "labels": labels,
        }


# --------------------------------------------------------------------------- #
# 划分读取
# --------------------------------------------------------------------------- #
MANIFEST_COLUMNS = [
    "frame_id", "num_points", "num_valid",
    "ratio_ground", "ratio_car", "ratio_pedestrian",
    "ratio_building", "ratio_vegetation", "ignored_ratio",
]
"""``manifest.csv`` 的**唯一**列定义。

写这份 manifest 的地方有两处（``tools/prepare_data.py`` 处理真实数据、
``tools/make_dummy_data.py`` 造合成数据），早期各写各的（一处用中文短名 +
``ignored_ratio``、一处用 ``num_valid`` + ``ratio_{short}``），读的地方
（``dataset_statistics``）只认后者，于是 dummy 数据一读就 KeyError。
现在两边都从这里取列名，读的时候也按这个顺序校验。
"""

_RATIO_KEYS = ["ground", "car", "pedestrian", "building", "vegetation"]


def read_split(path: str | Path) -> list[str]:
    """读取一行一个 frame_id 的划分文件，忽略空行与 ``#`` 注释。"""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def read_manifest(root: str | Path) -> list[dict]:
    """读取 ``manifest.csv``，返回每帧的统计信息。

    ``encoding="utf-8-sig"`` 是为了吞掉 Windows 上常见的 BOM：不吞的话第一列
    的键名会变成 ``'\\ufeffframe_id'``，下游一律 KeyError。
    """
    path = Path(root) / "manifest.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            # 只保留已知列，顺手把值转成需要的类型，避免下游到处 int()/float()
            row = {k: (v or "").strip() for k, v in r.items() if k}
            for key in ("num_points", "num_valid"):
                if key in row:
                    row[key] = int(float(row[key])) if row[key] else 0
            for key in _RATIO_KEYS:
                k = f"ratio_{key}"
                if k in row:
                    row[k] = float(row[k]) if row[k] else 0.0
            if "ignored_ratio" in row:
                row["ignored_ratio"] = float(row["ignored_ratio"] or 0.0)
            rows.append(row)
        return rows


def dataset_statistics(frame_ids: list[str], manifest: list[dict]) -> dict:
    """汇总所选划分的点数、各类别占比，用于数据说明表格。"""
    want = set(frame_ids)
    rows = [r for r in manifest if r.get("frame_id") in want]
    if not rows:
        return {"num_frames": 0}

    total = sum(int(r.get("num_points", 0)) for r in rows)
    # 兼容早期只有比例、没有 num_valid 的 manifest
    valid = sum(int(r["num_valid"]) for r in rows if "num_valid" in r)
    if not valid and "ignored_ratio" in rows[0]:
        valid = int(round(total * (1.0 - float(rows[0]["ignored_ratio"]))))
    stat = {
        "num_frames": len(rows),
        "total_points": total,
        "valid_points": valid,
        "ignore_ratio": 1.0 - valid / total if total else 0.0,
        "mean_points_per_frame": total / len(rows),
        "mean_valid_per_frame": valid / len(rows),
    }
    for c in _RATIO_KEYS:
        key = f"ratio_{c}"
        if key in rows[0]:
            stat[key] = sum(float(r[key]) for r in rows) / len(rows)
    return stat

