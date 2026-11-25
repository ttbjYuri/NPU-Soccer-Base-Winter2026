"""单元测试：不依赖 torch 的部分（标签映射、退化、采样、指标）。

运行
----
::

    python -m pytest tests -q
    python tests/test_pipeline.py         # 不想装 pytest 就直接跑这个文件

这些用例只用 numpy，因此即使机器上还没装 torch、也还没下载 SemanticPOSS，
也能立刻验证"评测口径"这一层没写错——而口径写错是这类作业里最常见、
也最致命的失分点。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roadseg.datasets import MANIFEST_COLUMNS, dataset_statistics, read_manifest
from roadseg.degradation import OutlierSpec, degrade, standard_suite
from roadseg.labels import (
    DEFAULT_LUT,
    IGNORE,
    INVALID_RAW_IDS,
    NUM_CLASSES,
    RAW_LABELS,
    remap_labels,
)
from roadseg.metrics import ConfusionMatrix
from roadseg.sampling import stride_indices


def _fake_frame(n: int = 5000, seed: int = 0):
    """一帧"像样"的合成点云：地面 + 一块车，含少量 255。"""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 3), np.float32)
    m = int(n * 0.75)
    pts[:m, 0] = rng.uniform(-30, 30, m)
    pts[:m, 1] = rng.uniform(-30, 30, m)
    pts[:m, 2] = rng.normal(0, 0.03, m)
    pts[m:, 0] = rng.uniform(-2, 2, n - m)
    pts[m:, 1] = rng.uniform(5, 9, n - m)
    pts[m:, 2] = rng.uniform(0, 1.6, n - m)
    lab = np.zeros(n, np.int64)
    lab[m:] = 1
    lab[:100] = IGNORE
    return pts, lab


# --------------------------------------------------------------------------- #
# 标签映射
# --------------------------------------------------------------------------- #
def test_mapping_covers_whole_byte_range():
    """0—255 全部有定义，不抛异常——这是"查表而非 if/else"的意义。

    输出只允许是 0..C-1 或 255，不能出现别的数。
    """
    mapped = remap_labels(np.arange(256, dtype=np.int64), DEFAULT_LUT)
    assert mapped.shape == (256,)
    allowed = set(range(NUM_CLASSES)) | {IGNORE}
    assert set(np.unique(mapped).tolist()) <= allowed
    assert mapped.min() >= 0


def test_undefined_ids_map_to_ignore():
    """2/3/18/19/20 与 23—31 这些未定义区间必须全部落到 255。

    题目表 2-2 只列了"要保留的"和"明确忽略的"两类，剩下的 id 必须有兜底，
    否则实现者会漏掉其中几个，导致同一份数据在不同人手里算出不同的 mIoU。
    """
    mapped = remap_labels(np.arange(256, dtype=np.int64), DEFAULT_LUT)
    defined = set(RAW_LABELS) | set(INVALID_RAW_IDS)
    for raw_id in range(256):
        if raw_id not in defined:
            assert mapped[raw_id] == IGNORE, f"未定义的 id {raw_id} 被映射到了 {mapped[raw_id]}"
    # 再单独钉死题面点名的那几个
    for raw_id in (2, 3, 18, 19, 20):
        assert mapped[raw_id] == IGNORE


def test_mapping_keeps_the_five_classes():
    """五类的代表 id 各归其位。"""
    lut = DEFAULT_LUT
    assert lut[22] == 0   # ground   → 地面
    assert lut[7] == 1    # car      → 车辆
    assert lut[4] == 2 and lut[5] == 2   # person → 行人
    assert lut[15] == 3   # building → 建筑
    assert lut[9] == 4    # vegetation → 植被


# --------------------------------------------------------------------------- #
# 退化
# --------------------------------------------------------------------------- #
def test_suite_has_fourteen_and_all_lengths_match():
    pts, lab = _fake_frame()
    suite = standard_suite()
    assert len(suite) == 14
    for name, specs in suite.items():
        d = degrade(pts, lab, specs, seed=1234)
        assert d.points.shape[0] == d.labels.shape[0] == d.keep.shape[0], name
        assert d.num_evaluated <= d.points.shape[0], name
        assert d.num_evaluated >= 0, name


def test_degradation_is_deterministic():
    """同一 seed 必须给出同一结果，否则报告里的数字无法复现。"""
    pts, lab = _fake_frame()
    specs = standard_suite()["noise_5cm"]
    a = degrade(pts, lab, specs, seed=7)
    b = degrade(pts, lab, specs, seed=7)
    assert np.allclose(a.points, b.points)
    c = degrade(pts, lab, specs, seed=8)
    assert not np.allclose(a.points, c.points)


def test_inserted_outliers_are_not_evaluated():
    """题目 2.5 第 1 条：插入的离群点不进混淆矩阵。

    实现方式是给新点打 IGNORE 标签且 keep=False，两者都要成立——
    只做其中一个，下游一旦忘记过滤就会把离群点算成"错分"。
    """
    pts, lab = _fake_frame()
    d = degrade(pts, lab, [OutlierSpec(64)], seed=0)
    n_new = d.points.shape[0] - pts.shape[0]
    assert n_new == 64
    assert (d.labels[-64:] == IGNORE).all()
    assert not d.keep[-64:].any()
    assert d.num_evaluated == int((lab != IGNORE).sum()), (
        f"num_evaluated={d.num_evaluated}，期望 {int((lab != IGNORE).sum())}"
    )


def test_outliers_land_near_existing_points():
    """离群点要落在已有点的局部邻域里（题面第四章：距某个原始点 0.5—2 m）。

    早先的实现取整帧包围盒均匀撒点，一帧 80 m 时几乎全落在没有真值的空地上，
    等于没考鲁棒性。
    """
    pts, lab = _fake_frame(n=3000)
    d = degrade(pts, lab, [OutlierSpec(64)], seed=0)
    new = d.points[pts.shape[0]:]
    # 暴力求每个新点到最近原始点的距离
    dist = np.sqrt(((new[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(axis=1)
    assert dist.max() < 3.0, f"有离群点离最近原始点超过 3 m（最大 {dist.max():.2f} m）"
    assert np.median(dist) < 1.5


def test_sparsity_ratio_is_relative_to_frame():
    """sparse_25 保留约 25% 的点（对整帧计，含 255）。"""
    pts, lab = _fake_frame(n=4000)
    d = degrade(pts, lab, standard_suite()["sparse_25"], seed=0)
    ratio = d.points.shape[0] / pts.shape[0]
    assert 0.20 < ratio < 0.30, f"实际保留比例 {ratio:.3f}"


# --------------------------------------------------------------------------- #
# 采样
# --------------------------------------------------------------------------- #
def test_stride_returns_exactly_requested_or_all():
    assert len(stride_indices(20000, 8192)) == 8192
    # 点数不足时用全部，不上采样
    assert len(stride_indices(1158, 8192)) == 1158
    assert len(stride_indices(8192, 8192)) == 8192


def test_stride_is_deterministic_and_sorted():
    a = stride_indices(5000, 512)
    b = stride_indices(5000, 512)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def test_iou_matches_hand_calculation():
    """手算核对：gt=[0,1,2,3,4,255,0,1]、pred=[0,1,2,3,0,255,1,1]。

    * 类 0：TP=1（idx0）FP=1（idx4，gt 是 4）FN=1（idx6，gt0→pred1）→ IoU=1/3
    * 类 1：TP=2（idx1、idx7）FP=1（idx6）FN=0                      → IoU=2/3
    * 类 2、3：TP=1 FP=0 FN=0                                       → IoU=1
    * 类 4：TP=0 FP=0 FN=1                                          → IoU=0
    * mIoU = (1/3 + 2/3 + 1 + 1 + 0) / 5 = 0.6
    * OA   = 正确 5 点 / 有效 7 点 = 0.7143
    """
    gt = np.array([0, 1, 2, 3, 4, 255, 0, 1])
    pred = np.array([0, 1, 2, 3, 0, 255, 1, 1])
    cm = ConfusionMatrix.from_pairs(gt, pred, class_names=[str(i) for i in range(5)],
                                    ignore_index=255)
    assert abs(cm.miou() - 0.6) < 1e-6, f"实际 {cm.miou()}"
    assert abs(cm.oa() - 5 / 7) < 1e-6, f"实际 {cm.oa()}"
    # 255 既不算对也不算错
    assert cm.cm.sum() == 7


def test_ignore_index_never_enters_confusion_matrix():
    gt = np.array([0, 0, 255, 255, 1])
    pred = np.array([0, 1, 0, 1, 1])
    cm = ConfusionMatrix.from_pairs(gt, pred, class_names=["a", "b"], ignore_index=255)
    assert cm.cm.sum() == 3


# --------------------------------------------------------------------------- #
# manifest 往返
# --------------------------------------------------------------------------- #
def test_manifest_columns_are_stable():
    """manifest 的列定义只有一处（MANIFEST_COLUMNS），读写两侧都从这里取。

    历史上 tools/prepare_data.py 与 tools/make_dummy_data.py 各写各的列名
    （一处 num_valid + 英文短名、一处中文名 + ignored_ratio），
    于是 dataset_statistics 读 dummy 数据必定 KeyError。
    """
    assert MANIFEST_COLUMNS[0] == "frame_id"
    assert "num_valid" in MANIFEST_COLUMNS
    assert "ignored_ratio" in MANIFEST_COLUMNS
    for c in ("ground", "car", "pedestrian", "building", "vegetation"):
        assert f"ratio_{c}" in MANIFEST_COLUMNS


def test_read_manifest_tolerates_bom(tmp_path: Path):
    """带 BOM 的 manifest 也要能读——Windows 上 Excel 另存必带 BOM。"""
    p = tmp_path / "manifest.csv"
    cols = MANIFEST_COLUMNS
    row = {"frame_id": "00_000000", "num_points": 100, "num_valid": 90,
           "ratio_ground": 0.5, "ratio_car": 0.1, "ratio_pedestrian": 0.05,
           "ratio_building": 0.2, "ratio_vegetation": 0.15, "ignored_ratio": 0.1}
    header = ",".join(cols)
    values = ",".join(str(row[c]) for c in cols)
    p.write_text(header + "\n" + values + "\n", encoding="utf-8-sig")

    rows = read_manifest(tmp_path)
    assert len(rows) == 1
    assert "frame_id" in rows[0], f"键名被 BOM 污染：{list(rows[0])}"
    stat = dataset_statistics(["00_000000"], rows)
    assert stat["num_frames"] == 1
    assert stat["valid_points"] == 90


if __name__ == "__main__":
    import shutil

    failed = 0
    # 临时目录建在仓库里而不是系统 temp：某些受限环境（比如只允许写工作区的
    # 沙箱）下系统 temp 不可写，测试会以 PermissionError 失败而非断言失败。
    scratch = Path(__file__).resolve().parent / "_tmp"
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            if fn.__code__.co_argcount == 0:
                fn()
            else:
                shutil.rmtree(scratch, ignore_errors=True)
                scratch.mkdir(parents=True, exist_ok=True)
                fn(scratch)
                shutil.rmtree(scratch, ignore_errors=True)
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    shutil.rmtree(scratch, ignore_errors=True)
    print("全部通过" if failed == 0 else f"{failed} 个用例失败")
    sys.exit(1 if failed else 0)
