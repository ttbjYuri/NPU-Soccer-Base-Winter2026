"""训练脚本：配置驱动，一条命令跑完一个实验。

用法
----
::

    python train.py --config configs/pointnet.yaml
    python train.py --config configs/rpa_full.yaml --seed 1
    python train.py --config configs/pointnet.yaml --epochs 2 --limit-frames 20   # 冒烟测试

产物
----
::

    runs/<exp_name>_seed<seed>/
      config.json              本次运行实际使用的完整配置（快照，可复现的关键）
      train.log.jsonl          每个 epoch 一行的训练/验证指标
      best.pth                 验证集 mIoU 最高的权重
      last.pth                 最后一个 epoch
      eval_<split>.json        （由 evaluate.py 写出）
      predictions/             （由 visualize.py 写出）

为什么日志用 JSONL 而不是 tensorboard
--------------------------------------
交付要求"训练日志"要能直接核查。一行一条 JSON，Excel / pandas / jq
都能直接打开，不需要额外起服务，也不会因为换机器而丢数据。

关于 ``--resume``
-----------------
``--resume runs/xxx/last.pth`` 会恢复模型权重、优化器状态、学习率调度器以及起始
epoch，从断点继续训练。不加这个参数就是从头训。

关于"用验证集选模型"
--------------------
``best.pth`` 是按**验证集** mIoU 挑的，不是测试集。测试集只在最后一次
评估里用一次。这是硬性纪律：用测试集调参的实验结论一律无效。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roadseg.models  # noqa: F401  触发模型注册
from roadseg.datasets import CourseDataset, read_split
from roadseg.evaluation import evaluate_suites
from roadseg.labels import IGNORE, NUM_CLASSES
from roadseg.models import build_model, count_parameters
from roadseg.utils import JsonlLogger, format_seconds, normalize_points, set_seed, worker_init_fn


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    """读 YAML。只依赖 PyYAML，缺了就报一句人话。

    用 ``utf-8-sig`` 而不是 ``utf-8``：Windows 上另存的 YAML 经常带 BOM，
    而 ``utf-8`` 解码器不会吞掉它，PyYAML 会把 ``\\ufeffmodel`` 当成键名，
    于是出现"明明写了 model 却说没有 model"这种莫名其妙的错误。
    """
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("缺少 PyYAML：pip install pyyaml") from e
    try:
        with open(path, encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"找不到配置文件：{path}") from None
    except Exception as e:  # yaml 的语法错误类型较多，统一报出原始信息
        raise SystemExit(f"配置文件 {path} 解析失败：{e}") from e
    if not isinstance(cfg, dict):
        raise SystemExit(f"配置文件 {path} 的内容不是一个字典，请检查缩进")
    return cfg


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# 数据集包装：把"归一化 + 数据增强"放进 __getitem__
# --------------------------------------------------------------------------- #
class TrainTransform:
    """训练侧增强。**必须与评测分布对得上**，否则鲁棒性实验失去意义。

    这里只做三类与题目退化同源的增强：
    * 绕 z 轴随机旋转 —— 场景中没有"上北下南"，旋转不改变语义；
    * 随机丢点 —— 直接对应"稀疏"退化，让模型见过 50%/25% 的密度；
    * 坐标抖动 —— 对应"坐标噪声"，σ 取 1cm 量级；
    * （可选）小块挖空 —— 对应"局部缺失"。

    不做：缩放（会改变噪声的物理量纲）、x/y 翻转（道路场景不对称）。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.jitter = float(self.cfg.get("jitter_std", 0.0))
        self.dropout = float(self.cfg.get("random_drop_ratio", 0.0))
        self.rotate = bool(self.cfg.get("rotate_z", False))
        self.scale_range = self.cfg.get("scale_range")  # 默认 None；开了要在报告里说明

    def __call__(self, points: np.ndarray, labels: np.ndarray, rng: np.random.Generator):
        pts = points.copy()

        if self.rotate:
            theta = rng.uniform(-np.pi, np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            pts = pts @ rot.T

        if self.scale_range:
            lo, hi = self.scale_range
            pts = pts * rng.uniform(lo, hi)

        if self.dropout > 0 and pts.shape[0] > 1:
            keep_n = max(1, int(round(pts.shape[0] * (1.0 - self.dropout))))
            idx = rng.choice(pts.shape[0], size=keep_n, replace=False)
            idx.sort()
            pts, labels = pts[idx], labels[idx]

        if self.jitter > 0:
            pts = pts + rng.normal(0.0, self.jitter, size=pts.shape).astype(np.float32)

        return pts, labels


class SegDataset(torch.utils.data.Dataset):
    """把 ``CourseDataset`` 包成返回定长 tensor 的 Dataset。

    点数规则（这是修过的 bug，改动前请先读完）
    ------------------------------------------
    早期版本把 ``num_points`` 传给 ``CourseDataset`` 降采样一次，``TrainTransform``
    再随机丢 30%，最后 ``collate`` 又按 batch 内最小点数截齐——三重丢点叠加，
    实际进模型的点数在 2800 上下浮动且逐 batch 不同，无法与评测侧固定的 8192 对齐。

    现在的规则是：**读整帧 → 做增强 → 重采样到恰好 ``num_points`` 点**。
    增强后的点数可能多于也可能少于目标值，重采样在点数不足时允许有放回抽样
    （重复点的标签相同，对语义分割训练没有副作用），从而保证：

    * 每个样本的点数恒定，``collate`` 不再需要随机截齐；
    * 训练与评测的"每帧采样点数"口径一致，只是采样方式不同（训练随机、评测等间隔）。
    """

    def __init__(self, base: CourseDataset, transform: TrainTransform | None = None,
                 seed: int = 0, num_points: int | None = None, epoch: int = 0):
        self.base = base
        self.transform = transform
        self.seed = seed
        self.num_points = num_points
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        """由训练循环在每轮开始时调用，让增强随 epoch 变化。"""
        self.epoch = int(epoch)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        pts, labels = sample["points"], sample["labels"]
        # 用 (seed, epoch, idx) 派生 rng：既保证可复现（不依赖 worker 数量），
        # 又保证每个 epoch 看到不同的旋转角与抖动。
        key = (self.seed * 1_000_003 + self.epoch * 7_919 + idx) % (2 ** 31)
        rng = np.random.default_rng(key)
        if self.transform is not None:
            pts, labels = self.transform(pts, labels, rng)

        if self.num_points and pts.shape[0] > 0:
            n = int(self.num_points)
            m = pts.shape[0]
            if m >= n:
                sel = rng.choice(m, size=n, replace=False)
            else:
                # 点数不足：先全取，再补重复点，保证定长
                extra = rng.choice(m, size=n - m, replace=True)
                sel = np.concatenate([np.arange(m), extra])
            sel.sort()
            pts, labels = pts[sel], labels[sel]

        pts, _meta = normalize_points(pts)
        return {
            "points": torch.from_numpy(np.ascontiguousarray(pts, dtype=np.float32)),
            "labels": torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
            "frame_id": sample["frame_id"],
        }


def collate(batch: list[dict]) -> dict:
    """把若干帧拼成一个 batch。

    现在每个样本的点数已经恒定（见 ``SegDataset``），所以这里只做 ``stack``。
    保留兜底分支：如果外部传来的样本长度不一致，才按最小点数随机对齐指标
    **同步**裁剪点和标签——早期版本对点做 ``randperm`` 却把标签取前 n 个，
    导致坐标与语义错位，模型学到的等价于随机标签，这类 bug 不报错、只让指标莫名很低。
    """
    lengths = {b["points"].shape[0] for b in batch}
    if len(lengths) == 1:
        return {
            "points": torch.stack([b["points"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "frame_id": [b["frame_id"] for b in batch],
        }

    n = min(lengths)
    pts_list, lab_list = [], []
    for b in batch:
        m = b["points"].shape[0]
        if m > n:
            sel = torch.randperm(m)[:n]
            pts_list.append(b["points"][sel])
            lab_list.append(b["labels"][sel])
        else:
            pts_list.append(b["points"])
            lab_list.append(b["labels"])
    return {
        "points": torch.stack(pts_list),
        "labels": torch.stack(lab_list),
        "frame_id": [b["frame_id"] for b in batch],
    }


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
# 评估逻辑已统一到 roadseg/evaluation.py。这里只保留一个薄封装，
# 避免 train.py 与 evaluate.py 各写一份、种子与计数口径却对不上的老问题。
def evaluate(model: nn.Module, data_root, frame_ids: list[str], device: torch.device,
             num_points: int, max_frames: int = 0) -> dict:
    """训练过程中的轻量验证：只评 clean 一档。

    退化档位的完整评估交给 evaluate.py，因为那是以"最终测试集"口径跑的，
    训练中途反复评退化档既慢又容易让人误用测试集。
    """
    return evaluate_suites(
        model, data_root, frame_ids, device=device,
        suites=[("clean", [])], num_points=num_points, max_frames=max_frames,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None,
                    help="从 last.pth 或 best.pth 断点续训（恢复权重/优化器/调度器/epoch）")
    ap.add_argument("--dry-run", action="store_true", help="只建模型、跑一步就退出")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = str(args.data_root)

    seed = int(cfg.get("train", {}).get("seed", 0))
    set_seed(seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data_root = Path(cfg["data"]["root"])
    num_points = int(cfg["data"].get("num_points", 4096))
    # 评测侧点数：配置里 eval.num_points 若为 null，则与训练侧保持一致，
    # 避免"训练 4096 / 评测 8192"这种两套口径。
    eval_points = cfg.get("eval", {}).get("num_points") or num_points
    ignore_index = int(cfg["data"].get("ignore_index", IGNORE))

    train_ids = read_split(data_root / "splits" / "train.txt")
    val_ids = read_split(data_root / "splits" / "val.txt")
    if args.limit_frames:
        train_ids, val_ids = train_ids[: args.limit_frames], val_ids[: max(1, args.limit_frames // 4)]
    if not train_ids:
        raise SystemExit(f"{data_root}/splits/train.txt 为空，请先运行 tools/prepare_data.py")
    if not val_ids:
        raise SystemExit(f"{data_root}/splits/val.txt 为空，无法按验证集选模型")

    transform = TrainTransform(cfg["data"].get("augment"))
    # 注意：CourseDataset 的 num_points 传 None（读整帧），点数的裁剪统一在
    # SegDataset 里做——"增强后再定长采样"是有意为之，见 SegDataset 的文档字符串。
    train_ds = SegDataset(CourseDataset(data_root, train_ids, num_points=None, training=True),
                          transform, seed=seed, num_points=num_points)

    bs = int(cfg["train"].get("batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
                              collate_fn=collate, worker_init_fn=worker_init_fn,
                              drop_last=False)

    model = build_model(cfg["model"]).to(device)
    n_params = count_parameters(model)

    exp_name = cfg.get("exp_name", Path(args.config).stem)
    run_dir = Path(cfg.get("output_dir", "runs")) / f"{exp_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["train"].get("epochs", 120))

    # lr_schedule 以前是个死键：配置里写着 cosine，代码里却无条件用余弦。
    # 现在真正按配置分派，并支持 step / none，方便做"学习率调度是否有效"的消融。
    schedule = str(cfg["train"].get("lr_schedule", "cosine")).lower()
    if schedule in ("cosine", "cos"):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))
    elif schedule in ("step", "multistep"):
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=max(epochs // 3, 1), gamma=0.1)
    elif schedule in ("none", "constant", "off"):
        sched = None
    else:
        raise SystemExit(f"未知的 lr_schedule：{schedule}（可选 cosine / step / none）")

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.0)

    start_epoch, best_miou = 1, -1.0
    if args.resume is not None:
        if not args.resume.exists():
            raise SystemExit(f"找不到断点文件：{args.resume}")
        ckpt = torch.load(str(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        # sched 可能是 None（训练时 lr_schedule=none，或旧版权重没存这个键），
        # 直接 load_state_dict(None) 会抛 TypeError，所以两边都必须是有效对象才恢复。
        if sched is not None and ckpt.get("sched") is not None:
            sched.load_state_dict(ckpt["sched"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_miou = float(ckpt.get("val_miou", -1.0))
        print(f"已从 {args.resume} 恢复：起始 epoch={start_epoch}，best mIoU={best_miou:.4f}")

    print("=" * 72)
    print(f"实验 {exp_name}  seed={seed}  device={device}")
    print(f"训练 {len(train_ids)} 帧 / 验证 {len(val_ids)} 帧")
    print(f"训练每帧 {num_points} 点 / 评测每帧 {eval_points} 点")
    print(f"模型 {cfg['model']['name']}  参数量 {n_params:,}")
    print(f"优化器 AdamW lr={cfg['train'].get('lr')} 调度={schedule} "
          f"ignore_index={ignore_index}")
    print(f"输出 {run_dir}")
    print("=" * 72)

    if args.dry_run:
        batch = next(iter(train_loader))
        x = batch["points"][:1].to(device)
        y = model(x)
        print(f"dry-run: 输入 {tuple(x.shape)} → 输出 {tuple(y.shape)}  OK")
        return 0

    logger = JsonlLogger(run_dir / "train.log.jsonl")
    t_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_ds.set_epoch(epoch)  # 让增强在每个 epoch 都变，而不是 120 轮同一份
        total_loss, total_n, t0 = 0.0, 0, time.time()

        for step, batch in enumerate(train_loader):
            x = batch["points"].to(device)
            y = batch["labels"].to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(cfg["train"].get("grad_clip", 1.0)))
            optim.step()

            valid = (y != ignore_index).sum().item()
            total_loss += loss.item() * valid
            total_n += valid

        if sched is not None:
            sched.step()
        train_loss = total_loss / max(total_n, 1)

        do_eval = (epoch % int(cfg["train"].get("eval_every", 5)) == 0) or epoch == epochs
        val = {}
        if do_eval:
            val = evaluate(model, data_root, val_ids, device,
                           num_points=eval_points, max_frames=len(val_ids))

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "lr": round(optim.param_groups[0]["lr"], 8),
            "epoch_time": round(time.time() - t0, 2),
            "val_miou": round(val["clean"]["miou"], 5) if val else None,
            "val_oa": round(val["clean"]["oa"], 5) if val else None,
        }
        logger.log(**rec)
        print(f"[{epoch:3d}/{epochs}] loss={train_loss:.4f}  "
              f"lr={rec['lr']:.2e}  {rec['epoch_time']:.1f}s"
              + (f"  val mIoU={rec['val_miou']:.4f}" if val else ""))

        if val and val["clean"]["miou"] > best_miou:
            best_miou = val["clean"]["miou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "config": cfg,
                    "epoch": epoch,
                    "val_miou": best_miou,
                    "num_params": n_params,
                },
                run_dir / "best.pth",
            )

        # last.pth 每轮覆盖，带上全部状态，供 --resume 续训
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict() if sched is not None else None,
                "config": cfg,
                "epoch": epoch,
                "val_miou": best_miou,
                "num_params": n_params,
            },
            run_dir / "last.pth",
        )
    logger.close()

    print("=" * 72)
    print(f"训练完成，用时 {format_seconds(time.time() - t_start)}，"
          f"最佳验证 mIoU={best_miou:.4f}")
    print(f"权重：{run_dir / 'best.pth'}")
    print(f"下一步：python evaluate.py --ckpt {run_dir / 'best.pth'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
