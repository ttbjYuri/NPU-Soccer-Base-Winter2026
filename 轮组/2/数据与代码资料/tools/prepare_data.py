"""数据集准备：SemanticPOSS → 课程 npz + 划分文件 + 数据统计。

用法
----
::

    # 1) 只检查原始数据是否完整、标签是否正常
    python tools/prepare_data.py --check --raw data/SemanticPOSS/dataset/sequences

    # 2) 全量转换（约 3000 帧，机械盘上 5~10 分钟）
    python tools/prepare_data.py --raw data/SemanticPOSS/dataset/sequences \
        --out data/course --split 00:train,01:train,02:train,03:test,04:val,05:train

    # 3) 只转前 50 帧，先用小样本跑通流程
    python tools/prepare_data.py ... --limit 50

输出
----
::

    data/course/
      frames/00_000000.npz        points(float32 N,3) labels(uint8 N,) intensity(float32 N,)
      manifest.csv                每帧点数、有效点、各类占比
      splits/{train,val,test}.txt
      label_stats.json            全局类别分布（写进报告"数据说明"一节）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadseg.labels import (  # noqa: E402
    CLASSES, IGNORE, RAW_LABELS, build_lut, DEFAULT_RULES, class_distribution,
)
from roadseg.datasets import MANIFEST_COLUMNS, read_bin, read_label  # noqa: E402


def parse_split(spec: str) -> dict[str, str]:
    """解析 ``"00:train,01:train,03:test"`` 形式的划分参数。"""
    mapping: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"划分项必须形如 'seq:split'，收到 {item!r}")
        seq, split = item.split(":", 1)
        if split not in ("train", "val", "test", "ignore"):
            raise ValueError(f"未知划分名 {split!r}")
        mapping[seq.strip()] = split.strip()
    return mapping


def discover_sequences(raw: Path) -> list[str]:
    """列出 ``sequences/`` 下的序列目录名（00, 01, ...）。"""
    if not raw.is_dir():
        raise FileNotFoundError(
            f"找不到 {raw}。SemanticPOSS 解压后应包含 sequences/00/velodyne 等目录。"
        )
    seqs = sorted(p.name for p in raw.iterdir() if p.is_dir() and p.name.isdigit())
    if not seqs:
        raise FileNotFoundError(f"{raw} 下没有数字命名的序列目录")
    return seqs


def check_sequence(seq_dir: Path, lut: np.ndarray) -> dict:
    """检查一个序列：帧数、点数一致性、标签取值是否越界。"""
    velo = seq_dir / "velodyne"
    labels = seq_dir / "labels"
    if not velo.is_dir():
        raise FileNotFoundError(f"{seq_dir} 下缺少 velodyne/")
    bins = sorted(velo.glob("*.bin"))
    result = {
        "sequence": seq_dir.name,
        "frames": len(bins),
        "has_labels": labels.is_dir(),
        "missing_label": 0,
        "length_mismatch": 0,
        "unknown_raw_ids": [],
        "total_points": 0,
    }

    seen_ids: set[int] = set()
    for b in bins[: min(len(bins), 20)]:          # 抽查 20 帧即可，不必全扫
        pts = read_bin(b)
        result["total_points"] += pts.shape[0]
        lab_path = labels / (b.stem + ".label")
        if not lab_path.exists():
            result["missing_label"] += 1
            continue
        raw = read_label(lab_path)
        if raw.shape[0] != pts.shape[0]:
            result["length_mismatch"] += 1
        seen_ids |= set(np.unique(raw & 0xFFFF).tolist())

    result["unknown_raw_ids"] = sorted(i for i in seen_ids if i not in RAW_LABELS)
    return result


def convert_frame(bin_path: Path, label_path: Path, lut: np.ndarray) -> dict:
    """读取一帧并转换标签，返回 npz 所需的三个数组。"""
    xyz_i = read_bin(bin_path)
    points = np.ascontiguousarray(xyz_i[:, :3], dtype=np.float32)
    intensity = np.ascontiguousarray(xyz_i[:, 3], dtype=np.float32)

    raw = read_label(label_path)
    if raw.shape[0] != points.shape[0]:
        raise ValueError(
            f"{bin_path.name}: 点 {points.shape[0]} 与标签 {raw.shape[0]} 不一致"
        )
    semantic = (raw & 0xFFFF).astype(np.int64)
    semantic = np.clip(semantic, 0, 255)
    labels = lut[semantic].astype(np.uint8)
    return {"points": points, "intensity": intensity, "labels": labels}


def frame_stats(frame_id: str, points: np.ndarray, labels: np.ndarray) -> dict:
    valid = labels != IGNORE
    # 列名与顺序由 roadseg.datasets.MANIFEST_COLUMNS 统一定义，
    # 免得与 tools/make_dummy_data.py 再分叉一次。
    row = {
        "frame_id": frame_id,
        "num_points": int(points.shape[0]),
        "num_valid": int(valid.sum()),
    }
    dist = class_distribution(labels)
    for c in CLASSES:
        row[f"ratio_{c.short}"] = round(dist.get(c.name, 0.0), 6)
    row["ignored_ratio"] = round(1.0 - valid.sum() / max(points.shape[0], 1), 6)
    return {k: row[k] for k in MANIFEST_COLUMNS}


def main() -> int:
    ap = argparse.ArgumentParser(description="SemanticPOSS → 课程数据格式")
    ap.add_argument("--raw", type=Path, required=True,
                    help="SemanticPOSS 的 sequences 目录")
    ap.add_argument("--out", type=Path, default=Path("data/course"))
    ap.add_argument("--split", type=str, default="",
                    help="形如 '00:train,01:train,02:train,03:test,04:val,05:train'；"
                         "留空则使用 SemanticPOSS 论文的官方划分（part3 测试）")
    ap.add_argument("--limit", type=int, default=0, help="每个序列最多转换多少帧（0=全部）")
    ap.add_argument("--check", action="store_true", help="只做检查，不写文件")
    ap.add_argument("--alt-mapping", action="store_true",
                    help="使用备选标签映射（见 roadseg/labels.py 的 ALT_RULES）")
    args = ap.parse_args()

    from roadseg.labels import ALT_RULES

    lut = build_lut(ALT_RULES if args.alt_mapping else DEFAULT_RULES)
    seqs = discover_sequences(args.raw)

    print("=" * 72)
    print(f"原始数据: {args.raw}")
    print(f"序列数量: {len(seqs)}  ({', '.join(seqs)})")
    print("=" * 72)

    reports = [check_sequence(args.raw / s, lut) for s in seqs]
    for r in reports:
        flag = "OK " if (r["has_labels"] and not r["length_mismatch"]) else "!! "
        print(f"{flag}seq {r['sequence']}: {r['frames']:5d} 帧, "
              f"标注={'有' if r['has_labels'] else '无'}, "
              f"抽查点数={r['total_points']:,}")
        if r["missing_label"]:
            print(f"    缺少标签 {r['missing_label']} 帧")
        if r["length_mismatch"]:
            print(f"    点数/标签长度不一致 {r['length_mismatch']} 帧")
        if r["unknown_raw_ids"]:
            print(f"    出现未知原始标签 id: {r['unknown_raw_ids']} → 已按忽略处理")

    if args.check:
        print("\n仅检查模式，未写出任何文件。")
        return 0

    # ---- 划分 ----
    if args.split:
        split_map = parse_split(args.split)
    else:
        # SemanticPOSS 论文：按车辆位置切 6 段，part 3 作测试集。
        # 官方发布为 00..05，我们把 03 作为测试、04 作为验证、其余训练。
        split_map = {s: ("test" if s == "03" else ("val" if s == "04" else "train"))
                     for s in seqs}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "frames").mkdir(exist_ok=True)
    (args.out / "splits").mkdir(exist_ok=True)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    manifest: list[dict] = []
    global_dist: dict[str, float] = {}
    n_frames = 0

    for seq in seqs:
        split = split_map.get(seq, "ignore")
        if split == "ignore":
            continue
        seq_dir = args.raw / seq
        velo_dir, lab_dir = seq_dir / "velodyne", seq_dir / "labels"
        bins = sorted(velo_dir.glob("*.bin"))
        if args.limit:
            bins = bins[: args.limit]

        for b in bins:
            lab_path = lab_dir / (b.stem + ".label")
            if not lab_path.exists():
                continue
            frame_id = f"{seq}_{b.stem}"
            data = convert_frame(b, lab_path, lut)
            np.savez_compressed(
                args.out / "frames" / f"{frame_id}.npz",
                points=data["points"],
                labels=data["labels"],
                intensity=data["intensity"],
            )
            row = frame_stats(frame_id, data["points"], data["labels"])
            manifest.append(row)
            splits[split].append(frame_id)

            dist = class_distribution(data["labels"].astype(np.int64))
            for c in CLASSES:
                global_dist[c.name] = global_dist.get(c.name, 0.0) + dist.get(c.name, 0.0)

            n_frames += 1
            if n_frames % 100 == 0:
                print(f"  已转换 {n_frames} 帧 ...")

    for name, ids in splits.items():
        (args.out / "splits" / f"{name}.txt").write_text(
            "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8"
        )

    if manifest:
        with (args.out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(manifest)

    n = max(n_frames, 1)
    label_stats = {
        "num_frames": n_frames,
        # global_dist 累加的是"每帧的类别占比"，再除以帧数，
        # 得到的是"按帧平均的类别占比"，不是"按点数统计的全局占比"——两者
        # 在长尾类别上会明显不同（点多的帧权重更大）。字段名如实标注，别误读。
        "class_ratio_frame_mean": {k: round(v / n, 6) for k, v in global_dist.items()},
        "splits": {k: len(v) for k, v in splits.items()},
        "mapping": "ALT_RULES" if args.alt_mapping else "DEFAULT_RULES",
        "sequences": seqs,
        "split_map": split_map,
    }
    (args.out / "label_stats.json").write_text(
        json.dumps(label_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print(f"完成：{n_frames} 帧 → {args.out}")
    print(f"  训练 {len(splits['train'])} 帧 / 验证 {len(splits['val'])} 帧 / "
          f"测试 {len(splits['test'])} 帧")
    print("  类别占比（按帧平均）：")
    for k, v in label_stats["class_ratio_frame_mean"].items():
        print(f"    {k:8s} {v * 100:6.2f}%")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

