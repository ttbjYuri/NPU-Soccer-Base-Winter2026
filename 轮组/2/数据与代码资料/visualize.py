"""预测可视化：把「原始点云 / 真实标签 / 模型预测 / 错误图」画成一张三联图。

用法
----
::

    # 画测试集前 6 帧（clean）
    python visualize.py --ckpt runs/pointnet_seed0/best.pth --num 6

    # 指定帧，并叠加退化档位（稀疏 25% + 5cm 噪声）
    python visualize.py --ckpt runs/pointnet_seed0/best.pth \
        --frames 03_000100,03_000233 --suite sparse_25+noise_5cm

    # 只画错误图，方便挑失败案例
    python visualize.py --ckpt runs/rpa_full_seed0/best.pth --num 12 --mode error

产物
----
::

    runs/<exp>_seed<seed>/predictions/
      <frame_id>_<suite>.png     三联图：真值 / 预测 / 错误
      index.csv                  每帧的 mIoU、错误点占比（挑成功/失败案例用）

为什么用 matplotlib 而不是 Open3D
---------------------------------
题目要求的是"能看清结构的静态图"，不是可交互窗口。matplotlib 无系统依赖、
纯 pip 可装、能在无显示器的服务器上 `Agg` 后端出图；Open3D 需要 OpenGL，
在很多教学机上装不起来。代价是没有旋转交互——所以每个视角单独出一张。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from roadseg.datasets import CourseDataset, read_split  # noqa: E402
from roadseg.degradation import degrade, standard_suite  # noqa: E402
from roadseg.evaluation import DEFAULT_SEED_BASE, frame_seed  # noqa: E402
from roadseg.labels import CLASSES, CLASS_NAMES, CLASS_SHORTS, IGNORE, NUM_CLASSES  # noqa: E402
from roadseg.sampling import stride_indices  # noqa: E402
from roadseg.utils import normalize_points, set_seed  # noqa: E402


def _palette() -> np.ndarray:
    """把课程五类的 RGB 做成 (NUM_CLASSES+1, 3) 的查表，末行给"忽略/错误"用。"""
    lut = np.zeros((NUM_CLASSES + 1, 3), dtype=np.float32)
    for i, c in enumerate(CLASSES):
        lut[i] = np.array(c.color, dtype=np.float32) / 255.0
    lut[NUM_CLASSES] = np.array([0.6, 0.6, 0.6], dtype=np.float32)
    return lut


def colorize(labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """标签 → 颜色。255（忽略）映射到最后一行灰色。"""
    idx = np.where(labels == IGNORE, NUM_CLASSES, labels).astype(np.int64)
    idx = np.clip(idx, 0, NUM_CLASSES)
    return lut[idx]


def _setup_axes(ax, xyz: np.ndarray, title: str, elev: float, azim: float,
                title_size: float = 10.0) -> None:
    """统一视角、等比例、去掉坐标轴刻度（点数云图上看刻度没有意义）。"""
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.8, linewidths=0)
    ax.set_title(title, fontsize=title_size)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.35))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    # 去掉 matplotlib 3D 默认的灰色背景板，否则深色点会被衬托得发白
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)


def _setup_font() -> None:
    """让 matplotlib 能显示中文类别名。

    不设的话五类名字在图上全是方框。这里按"Windows → macOS → Linux"的顺序
    挑第一个存在的中文字体，都没有就退回英文短名（不报错，不影响出图）。
    """
    import matplotlib
    matplotlib.use("Agg")                     # 无显示器也能出图
    from matplotlib import font_manager, rcParams

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "PingFang SC",
                 "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei"):
        if cand in available:
            rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            break


def labels_for_plot() -> list[str]:
    """图例文字：系统有中文字体就用中文名，否则用英文短名。"""
    import matplotlib
    from matplotlib import font_manager, rcParams

    fam = (rcParams["font.sans-serif"] or [""])[0]
    has_cjk = fam in {f.name for f in font_manager.fontManager.ttflist}
    return list(CLASS_NAMES) if has_cjk else list(CLASS_SHORTS)


def plot_frame(
    xyz: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    title: str,
    max_points: int = 60_000,
    elev: float = 22.0,
    azim: float = -60.0,
    mode: str = "all",
) -> dict:
    """画图：``mode="all"`` 出三联图（真值/预测/错误），``mode="error"`` 只出错误图。

    只画被评测的原始点——插入的离群点没有真值，画上去会让人误以为模型在乱标。
    """
    if mode not in ("all", "error"):
        raise ValueError(f"未知的 mode：{mode!r}（可选 all / error）")

    _setup_font()
    import matplotlib.pyplot as plt

    names = labels_for_plot()

    rng = np.random.default_rng(0)
    if xyz.shape[0] > max_points:             # 图上一百万点也看不清，抽稀即可
        sel = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz, gt, pred = xyz[sel], gt[sel], pred[sel]

    lut = _palette()
    wrong = pred != gt
    err_xyz = xyz[wrong]
    ok_xyz = xyz[~wrong]

    if mode == "error":
        # 单幅错误图：挑失败案例时一次能看十几帧，省时间
        fig = plt.figure(figsize=(4.8, 4.2), dpi=150)
        ax3 = fig.add_subplot(111, projection="3d")
    else:
        fig = plt.figure(figsize=(13.5, 4.2), dpi=150)
        ax1 = fig.add_subplot(131, projection="3d")
        _setup_axes(ax1, xyz, "Ground truth", elev, azim)
        ax1.collections[0].set_facecolors(colorize(gt, lut))

        ax2 = fig.add_subplot(132, projection="3d")
        _setup_axes(ax2, xyz, "Prediction", elev, azim)
        ax2.collections[0].set_facecolors(colorize(pred, lut))

        ax3 = fig.add_subplot(133, projection="3d")

    ax3.scatter(ok_xyz[:, 0], ok_xyz[:, 1], ok_xyz[:, 2],
                s=0.5, c="#d8d8d8", linewidths=0, label="correct")

    if err_xyz.size:
        ax3.scatter(err_xyz[:, 0], err_xyz[:, 1], err_xyz[:, 2],
                    s=1.2, c="#d62728", linewidths=0, label="wrong")
    ax3.set_title(f"Errors ({wrong.mean() * 100:.1f}% of points)", fontsize=10)
    ax3.view_init(elev=elev, azim=azim)
    ax3.set_box_aspect((1, 1, 0.35))
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
    ax3.grid(False)
    ax3.xaxis.pane.set_visible(False)
    ax3.yaxis.pane.set_visible(False)
    ax3.zaxis.pane.set_visible(False)
    if err_xyz.size:
        ax3.legend(loc="upper right", fontsize=8, markerscale=6)

    if mode == "error":
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
    else:
        # 图例只画一次（颜色 → 类别名），放在整张图底部
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                       markerfacecolor=lut[i], markeredgecolor="none", label=name)
            for i, name in enumerate(names)
        ]
        fig.legend(handles=handles, loc="lower center", ncol=NUM_CLASSES,
                   frameon=False, fontsize=9)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        m = gt == i
        per_class[name] = round(float((pred[m] == i).mean()), 4) if m.any() else None
    return {
        "num_points": int(xyz.shape[0]),
        "error_ratio": round(float(wrong.mean()), 4),
        "per_class_acc": per_class,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="预测可视化（真值 / 预测 / 错误）")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--frames", type=str, default=None,
                    help="逗号分隔的 frame_id；不给则从该 split 里取前 --num 帧")
    ap.add_argument("--num", type=int, default=6, help="画多少帧")
    ap.add_argument("--suite", type=str, default="clean",
                    help="退化档位名，见 evaluate.py --list-suites；默认 clean")
    ap.add_argument("--num-points", type=int, default=None,
                    help="每帧送进模型的点数上限（None=全量，最老实但慢）")
    ap.add_argument("--max-points", type=int, default=60_000,
                    help="单张图上最多画多少点（只影响画图，不影响指标）")
    ap.add_argument("--mode", type=str, default="all", choices=["all", "error"],
                    help="error 只画错误图，用于快速挑失败案例")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED_BASE,
                    help="退化随机种子基准值，与 evaluate.py 相同；第 i 帧用 seed+i")
    args = ap.parse_args()

    from evaluate import load_checkpoint         # 复用同一套权重加载逻辑

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(0)
    model, cfg, meta = load_checkpoint(args.ckpt, device)

    data_root = args.data_root or Path(cfg.get("data", {}).get("root", "data/course"))
    num_points = args.num_points
    if num_points is None:
        num_points = cfg.get("eval", {}).get("num_points")

    if args.frames:
        frames = [f.strip() for f in args.frames.split(",") if f.strip()]
    else:
        frames = read_split(data_root / "splits" / f"{args.split}.txt")[: args.num]

    suites = standard_suite()
    if args.suite not in suites:
        raise SystemExit(f"未知档位 {args.suite!r}；可用：{sorted(suites)}")
    specs = suites[args.suite]

    exp = cfg.get("exp_name", args.ckpt.parent.name)
    out_dir = args.out_dir or (args.ckpt.parent / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = CourseDataset(data_root, frames, num_points=None, training=False)
    rows = []
    for i, frame_id in enumerate(frames):
        sample = ds[i]
        xyz = sample["points"]
        gt = sample["labels"].astype(np.int64)

        if specs:                                   # 退化必须在归一化之前
            deg = degrade(xyz, gt, specs, seed=frame_seed(args.seed, i))
            xyz, gt, keep = deg.points, deg.labels, deg.keep
        else:
            keep = np.ones(xyz.shape[0], dtype=bool)

        # 与 evaluate.py 用同一个采样函数，保证"图上的点"就是"评测的点"
        if num_points and xyz.shape[0] > num_points:
            sel = stride_indices(xyz.shape[0], num_points)
            xyz, gt, keep = xyz[sel], gt[sel], keep[sel]

        pts_n, _ = normalize_points(xyz)
        x = torch.from_numpy(pts_n[None].astype(np.float32)).to(device)
        pred = model(x).argmax(-1)[0].cpu().numpy().astype(np.int64)

        # 只画被评测的原始点；插入的离群点没有真值，画上去会误导
        xyz_e, gt_e, pred_e = xyz[keep], gt[keep], pred[keep]
        tag = f"{frame_id}_{args.suite}"
        out_path = out_dir / f"{tag}.png"
        info = plot_frame(xyz_e, gt_e, pred_e, out_path,
                          title=f"{frame_id}  [{args.suite}]",
                          max_points=args.max_points, mode=args.mode)
        info.update({"frame_id": frame_id, "suite": args.suite, "png": out_path.name})
        rows.append(info)
        print(f"{frame_id:16s} {args.suite:22s} 错误点 {info['error_ratio'] * 100:5.2f}%  "
              f"→ {out_path.name}")

    index = out_dir / "index.csv"
    with index.open("w", encoding="utf-8", newline="") as f:
        fields = ["frame_id", "suite", "png", "num_points", "error_ratio"]
        fields += [f"acc_{c}" for c in CLASS_NAMES]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            flat = {k: r[k] for k in ("frame_id", "suite", "png", "num_points", "error_ratio")}
            for c in CLASS_NAMES:
                flat[f"acc_{c}"] = r["per_class_acc"].get(c)
            writer.writerow(flat)

    if meta.get("val_miou") is not None:
        print(f"\n该权重训练时最佳验证 mIoU: {meta['val_miou']:.4f} (epoch {meta['epoch']})")
    print(f"共 {len(rows)} 张图 → {out_dir}")
    print(f"案例索引（按错误率排序挑成功/失败案例）：{index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
