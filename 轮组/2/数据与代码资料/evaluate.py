"""评估脚本：加载权重，在原始测试集 + 全部退化档位上算 mIoU。

用法
----
::

    python evaluate.py --ckpt runs/pointnet_seed0/best.pth
    python evaluate.py --ckpt runs/rpa_full_seed0/best.pth --split test --save-json res.json
    python evaluate.py --ckpt ... --suite clean,sparse_50,noise_5cm    # 只跑指定档位
    python evaluate.py --ckpt ... --max-frames 50                      # 快速核对
    python evaluate.py --list-suites                                   # 看全部档位说明

输出的 JSON 结构（报告里的表都从这里生成）
------------------------------------------
::

    {
      "checkpoint": "...",
      "split": "test",
      "num_frames": 500,
      "num_points_rule": 8192,
      "num_params": 1234567,
      "suites": {
        "clean": {
          "miou": 0.312, "oa": 0.86, "fwiou": 0.44,
          "per_class_iou": {"地面": 0.9, ...},
          "top_confusions": [...],
          "num_points": 3900000, "num_frames": 500
        },
        "sparse_50": {...},
        ...
      },
      "seconds": 187.4
    }

评估纪律
--------
1. 只评**保留下来的原始点**：被删除的点、插入的离群点都不进混淆矩阵。
2. 采样规则固定（等间隔 stride），两模型比的是同一批点；
   某档退化后点数不足 ``num_points`` 时直接用全部剩余点，不上采样。
3. **先在原始坐标上施加退化，再归一化送进模型**，否则 6 m 盒子会变成无量纲数字。
4. 测试集只在最后评一次；任何"看到测试结果再回去调参"都会污染结论。
5. 本脚本与 train.py 共用 ``roadseg/evaluation.py``，退化种子与帧计数口径一致，
   训练日志里的 val mIoU 与这里的数字可以互相核对。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roadseg.models  # noqa: F401
from roadseg.datasets import read_split
from roadseg.evaluation import (
    DEFAULT_NUM_POINTS,
    DEFAULT_SEED_BASE,
    evaluate_suites,
    format_suite_table,
    resolve_suites,
)
from roadseg.labels import CLASS_NAMES
from roadseg.models import build_model
from roadseg.utils import format_seconds, set_seed


def load_checkpoint(path: str | Path, device: torch.device):
    """读取 ``.pth``，返回 ``(model, cfg, meta)``。兼容只存了 state_dict 的情况。"""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state, cfg = ckpt["model"], ckpt.get("config", {})
        epoch, val_miou = ckpt.get("epoch"), ckpt.get("val_miou")
    else:
        state, cfg, epoch, val_miou = ckpt, {}, None, None

    if "model" not in cfg:
        raise SystemExit(
            f"{path} 里没有模型配置，无法重建网络结构。"
            "请用 train.py 产出的权重，或手动指定 --model。"
        )
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, {"epoch": epoch, "val_miou": val_miou}


def main() -> int:
    ap = argparse.ArgumentParser(description="在原始与退化测试集上评估")
    ap.add_argument("--ckpt", type=Path, default=None, help="权重路径（.pth）")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--num-points", type=int, default=None,
                    help="每帧评测采样点数；默认读 checkpoint 的 eval.num_points")
    ap.add_argument("--suite", type=str, default=None,
                    help="逗号分隔的档位名，如 clean,sparse_50,noise_5cm；默认全部")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED_BASE,
                    help="退化随机种子的基准值，逐帧加帧序号；固定以便复跑")
    ap.add_argument("--list-suites", action="store_true", help="打印档位说明后退出")
    args = ap.parse_args()

    if args.list_suites:
        print(format_suite_table())
        return 0
    if args.ckpt is None:
        raise SystemExit("需要 --ckpt 指定权重（或用 --list-suites 只看档位说明）")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(0)  # 评估本身不需要随机性，固定住是为了让模型侧任何 dropout 也可复现

    model, cfg, meta = load_checkpoint(args.ckpt, device)
    data_root = args.data_root or Path(cfg.get("data", {}).get("root", "data/course"))

    num_points = args.num_points
    if num_points is None:
        num_points = (cfg.get("eval", {}).get("num_points")
                      or cfg.get("data", {}).get("num_points")
                      or DEFAULT_NUM_POINTS)

    frames = read_split(data_root / "splits" / f"{args.split}.txt")
    if not frames:
        raise SystemExit(f"{data_root}/splits/{args.split}.txt 为空")

    try:
        suites = resolve_suites(args.suite)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    n_used = min(len(frames), args.max_frames) if args.max_frames else len(frames)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 78)
    print(f"权重    : {args.ckpt}")
    print(f"划分    : {args.split}（{len(frames)} 帧，实际评估 {n_used} 帧）")
    print(f"档位    : {len(suites)} 个  {'、'.join(suites)}")
    print(f"设备    : {device}")
    print(f"每帧点数: {num_points}（不足则该帧用全部剩余点，不上采样）")
    print(f"退化种子: base={args.seed}，第 i 帧用 seed={args.seed}+i")
    print(f"总参数量: {n_params:,}")
    if meta["val_miou"] is not None:
        print(f"该权重训练时的最佳验证 mIoU: {meta['val_miou']:.4f} (epoch {meta['epoch']})")
    print("=" * 78)

    t_all = time.time()
    # 一次遍历算完所有档位：同一帧的全部档位共用同一个退化种子，
    # 而且数据集只读一遍（14 档各读一次会让 500 帧的测试集读 7000 次盘）。
    results = evaluate_suites(
        model, data_root, frames, device=device,
        suites=list(suites.items()), num_points=num_points,
        max_frames=n_used, seed_base=args.seed, verbose=False,
    )
    elapsed = time.time() - t_all
    for name in suites:
        stats = results[name]
        print(f"{name:24s} mIoU={stats['miou']:.4f}  OA={stats['oa']:.4f}  "
              f"点数={stats['num_points']:,}")

    # ---- 汇总表 ----
    print("\n" + "=" * 78)
    print("退化档位 × 各类别 IoU")
    header = f"{'档位':<20s}" + "".join(f"{c[:6]:>8s}" for c in CLASS_NAMES) + f"{'mIoU':>8s}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        cells = ""
        for c in CLASS_NAMES:
            v = r["per_class_iou"].get(c)
            cells += f"{'—':>8s}" if v is None else f"{v:>8.3f}"
        print(f"{name:<20s}{cells}{r['miou']:>8.3f}")
    print("=" * 78)
    print(f"总用时 {format_seconds(elapsed)}")

    payload = {
        "checkpoint": str(args.ckpt),
        "split": args.split,
        "num_frames": n_used,
        "num_points_rule": num_points,
        "num_params": n_params,
        "seed_base": args.seed,
        "seconds": round(elapsed, 1),
        "suites": results,
    }
    out = args.save_json or (args.ckpt.parent / f"eval_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
