"""生成用于"管道自检"的合成数据（**不是**课程数据，不要用它算指标）。

用途
----
SemanticPOSS 有 2.4 GB，下载和解压都要时间。在真正跑实验之前，先用
这份合成数据把「读数据 → 训练 → 评估 → 出表」整条链路跑通，能省掉
大量"训练到第 30 个 epoch 才发现评估脚本崩了"的时间。

生成的东西
----------
::

    data/dummy/
      frames/00_000000.npz ...      与 course/frames 格式完全一致
      manifest.csv
      splits/{train,val,test}.txt

合成场景刻意做得像道路：地面是一个带轻微起伏的平面，两侧竖直的"建筑"，
零散的"植被"团块，少量"车辆"盒子与"行人"柱体。类别比例与真实数据
同量级，因此 lr / batch size 这些超参在合成数据上不会离谱。

用法
----
::

    python tools/make_dummy_data.py --out data/dummy --frames 12
    python train.py --config configs/pointnet.yaml --data-root data/dummy --epochs 2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roadseg.datasets import MANIFEST_COLUMNS
from roadseg.labels import CLASS_SHORTS, IGNORE, NUM_CLASSES

# 课程类别 id
GROUND, CAR, PED, BUILDING, VEG = range(NUM_CLASSES)


def _ground(rng: np.random.Generator, n: int, extent: float = 25.0) -> np.ndarray:
    """带缓坡与碎起伏的地面（不是纯平面，否则模型学不到几何）。"""
    xy = rng.uniform(-extent, extent, (n, 2))
    z = 0.02 * xy[:, 0] - 0.01 * xy[:, 1] + rng.normal(0, 0.03, n)
    return np.column_stack([xy, z])


def _building(rng: np.random.Generator, n: int) -> np.ndarray:
    """两个立面的长方体外壳：只在侧壁上撒点，屋顶几乎不可见。"""
    cx, cy = rng.choice([-18.0, 18.0]), rng.uniform(-10, 10)
    w, d, h = rng.uniform(6, 12), rng.uniform(8, 16), rng.uniform(4, 9)
    u = rng.uniform(0, 1, n)
    t = rng.uniform(0, 1, n)
    side = rng.integers(0, 4, n)
    x = np.where(side < 2, cx + np.where(side == 0, -w / 2, w / 2), cx - w / 2 + w * t)
    y = np.where(side < 2, cy - d / 2 + d * t, cy + np.where(side == 2, -d / 2, d / 2))
    z = h * u
    return np.column_stack([x, y, z])


def _vegetation(rng: np.random.Generator, n: int) -> np.ndarray:
    """若干球状树冠 + 一根树干。"""
    clusters = rng.uniform(-20, 20, (max(1, n // 400), 2))
    pick = rng.integers(0, clusters.shape[0], n)
    center = np.column_stack([clusters[pick], rng.uniform(2.0, 4.5, n)])
    r = rng.uniform(0.6, 1.2, n)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return center + direction * r[:, None]


def _car(rng: np.random.Generator, n: int) -> np.ndarray:
    """车：8 个角点撑起的扁盒子表面。"""
    c = np.column_stack([rng.uniform(-20, 20, n), rng.uniform(-20, 20, n), np.zeros(n)])
    L, W, H = rng.uniform(3.8, 5.0, n), rng.uniform(1.6, 2.0, n), rng.uniform(1.4, 1.8, n)
    face = rng.integers(0, 6, n)
    s, t = rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n)
    x = c[:, 0] + np.where(face < 2, np.where(face == 0, -0.5, 0.5) * L, s * L)
    y = c[:, 1] + np.where(face < 2, s * W, np.where(face < 4, np.where(face == 2, -0.5, 0.5) * W, t * W))
    z = c[:, 2] + np.where(face < 4, t * H + H / 2, np.where(face == 4, 0.0, H))
    return np.column_stack([x, y, z])


def _pedestrian(rng: np.random.Generator, n: int) -> np.ndarray:
    """行人：细长圆柱，半径 0.25 m、高 1.7 m。"""
    theta = rng.uniform(0, 2 * np.pi, n)
    r = 0.25 * np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([
        rng.uniform(-20, 20, n) + r * np.cos(theta),
        rng.uniform(-20, 20, n) + r * np.sin(theta),
        rng.uniform(0, 1.7, n),
    ])


def make_frame(seed: int, num_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """生成一帧 (points float32 (N,3), labels uint8 (N,))。"""
    rng = np.random.default_rng(seed)
    # 类别比例参考 SemanticPOSS：地面占绝对多数，行人/车辆很少
    ratio = {GROUND: 0.72, VEG: 0.14, BUILDING: 0.08, CAR: 0.05, PED: 0.01}
    pts, lab = [], []
    for cls, frac in ratio.items():
        n = max(1, int(num_points * frac))
        gen = {GROUND: _ground, VEG: _vegetation, BUILDING: _building, CAR: _car, PED: _pedestrian}[cls]
        pts.append(gen(rng, n))
        lab.append(np.full(n, cls, dtype=np.uint8))

    points = np.concatenate(pts).astype(np.float32)
    labels = np.concatenate(lab)
    # 模拟传感器：远处更稀、有强度通道、有地平线以下的噪声点
    dist = np.linalg.norm(points[:, :2], axis=1)
    keep_prob = np.clip(1.2 - dist / 30.0, 0.15, 1.0)
    keep = rng.uniform(size=points.shape[0]) < keep_prob
    points, labels = points[keep], labels[keep]

    order = rng.permutation(points.shape[0])          # 打乱顺序：模拟真实扫描次序
    points, labels = points[order], labels[order]
    intensity = np.clip(1.0 - dist[keep][order] / 40.0, 0, 1).astype(np.float32)
    return points, labels, intensity


def main() -> None:
    ap = argparse.ArgumentParser(description="生成管道自检用的合成点云数据")
    ap.add_argument("--out", default="data/dummy", help="输出目录")
    ap.add_argument("--frames", type=int, default=12, help="总帧数")
    ap.add_argument("--points", type=int, default=20000, help="每帧点数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    ids = []
    rows = []
    for i in range(args.frames):
        fid = f"00_{i:06d}"
        points, labels, intensity = make_frame(args.seed * 1000 + i, args.points)
        np.savez_compressed(out / "frames" / f"{fid}.npz",
                            points=points, labels=labels, intensity=intensity)
        ids.append(fid)
        # 列名与类型统一由 roadseg.datasets.MANIFEST_COLUMNS 定义，
        # 免得与 tools/prepare_data.py 的 schema 再次分叉。
        n_valid = int((labels != IGNORE).sum())
        row = {
            "frame_id": fid,
            "num_points": len(points),
            "num_valid": n_valid,
        }
        for c in range(NUM_CLASSES):
            row[f"ratio_{CLASS_SHORTS[c].lower()}"] = f"{float((labels == c).mean()):.6f}"
        row["ignored_ratio"] = f"{float((labels == IGNORE).mean()):.6f}"
        rows.append({k: row[k] for k in MANIFEST_COLUMNS})

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    n = len(ids)
    n_val = max(1, n // 6)
    n_test = max(1, n // 6)
    splits = {
        "train": ids[: n - n_val - n_test] or ids[:1],
        "val": ids[n - n_val - n_test: n - n_test],
        "test": ids[n - n_test:],
    }
    for name, list_ in splits.items():
        (out / "splits" / f"{name}.txt").write_text("\n".join(list_) + "\n", encoding="utf-8")

    print(f"合成数据已写到 {out.resolve()}")
    print(f"  frames: {n}  train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print("提醒：这份数据只用于验证代码能跑通，任何指标都不可写进报告。")


if __name__ == "__main__":
    main()
