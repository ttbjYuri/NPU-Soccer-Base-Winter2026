"""单元测试：模型的形状、交换不变性、注册表。

运行
----
::

    python -m pytest tests -q
    python tests/test_models.py           # 不想装 pytest 就直接跑这个文件

这些测试不碰真实数据，只验证"网络结构本身兑现了题目要求"，因此可以
在任何有 torch 的机器上秒级跑完。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import roadseg.models  # noqa: F401  触发模型注册
from roadseg.models import ABLATIONS, MODEL_REGISTRY, build_model
from roadseg.models.pointnet import count_parameters


def _rand(batch: int = 2, num_points: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch, num_points, 3) * 2 - 1


def test_registry_has_baseline_and_improved():
    assert "pointnet" in MODEL_REGISTRY
    assert "rpa" in MODEL_REGISTRY


def test_output_shape_one_label_per_point():
    """题目要求：N 个点进，N 个类别出。"""
    x = _rand(2, 64)
    for name in ("pointnet", "rpa"):
        model = build_model({"name": name}).eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 64, 5), f"{name} 输出 {tuple(y.shape)}，期望 (2, 64, 5)"


def test_permutation_invariance_and_equivariance():
    """打乱输入点顺序，输出应当跟着同样的顺序被打乱（逐点预测的交换等变性）。

    这是"为什么点云不能当作有序序列"的代码级证据：模型对顺序不敏感，
    但也因此完全不知道「谁挨着谁」——这正是基线丢失局部结构的原因。
    """
    x = _rand(2, 64)
    perm = torch.randperm(64)
    for name in ("pointnet", "rpa"):
        model = build_model({"name": name}).eval()
        with torch.no_grad():
            a = model(x)
            b = model(x[:, perm])
        diff = (a[:, perm] - b).abs().max().item()
        assert diff < 1e-4, f"{name} 交换等变性被破坏，最大误差 {diff}"


def test_all_ablations_forward():
    """六个消融版本都要能前向，且输出形状一致。"""
    x = _rand(2, 64)
    shapes = {}
    for name, cfg in ABLATIONS.items():
        kwargs = {k: v for k, v in cfg.items() if k != "name"}
        model = build_model({"name": cfg["name"], "num_classes": 5, **kwargs}).eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 64, 5), f"{name} 输出形状不对"
        shapes[name] = count_parameters(model)
    # 改进模型应该比基线略大，但量级相同（不是"换个更大的网络"来骗指标）
    assert shapes["rpa_full"] > shapes["baseline"]
    assert shapes["rpa_full"] < shapes["baseline"] * 1.5


def test_small_input_smaller_than_k():
    """k=16 但只给 4 个点时不能崩（k 要被 min(k, N) 截断）。"""
    x = _rand(1, 4)
    model = build_model({"name": "rpa", "k": 16}).eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 5)


def test_attention_block_uses_relative_position_only():
    """注意力块内部只看**相对**坐标：点云整体平移，块输出必须完全不变。

    有 RPE 时位置信息以 ``neigh_xyz - xyz`` 的形式进入 K、V 和 attn bias，
    所以平移不变是设计的直接推论。注意整个 RPANet 对平移**并非**不变——
    块之前还有一层吃绝对坐标的卷积。为什么这会带来问题、该怎么改，
    是题面第三问要你自己回答的，这里不做结论。
    """
    from roadseg.models.rpa import LocalAttentionBlock

    torch.manual_seed(0)
    feats = torch.randn(1, 32, 64)
    xyz = torch.rand(1, 32, 3) * 2 - 1
    shift = torch.tensor([5.0, -3.0, 2.0])

    for use_rpe in (True, False):
        block = LocalAttentionBlock(dim=64, num_heads=4, k=16, use_rpe=use_rpe).eval()
        with torch.no_grad():
            a = block(feats, xyz)
            b = block(feats, xyz + shift)
        diff = (a - b).abs().max().item()
        assert diff < 1e-4, f"use_rpe={use_rpe} 时平移改变了块输出，差 {diff}"


def test_rpe_reacts_to_neighborhood_geometry():
    """有 RPE 时，把邻居点推开（改变局部几何、保持特征不变）必须改变输出。

    这条用来证明"位置信息真的参与了计算"，而不是挂了个没用上的模块。
    """
    from roadseg.models.rpa import LocalAttentionBlock

    torch.manual_seed(0)
    feats = torch.randn(1, 32, 64)
    xyz = torch.rand(1, 32, 3) * 2 - 1

    block = LocalAttentionBlock(dim=64, num_heads=4, k=16, use_rpe=True).eval()
    no_rpe = LocalAttentionBlock(dim=64, num_heads=4, k=16, use_rpe=False).eval()
    with torch.no_grad():
        base = block(feats, xyz)
        stretched = block(feats, xyz * 3.0)      # 同一组点，几何被拉伸 3 倍
        base_n = no_rpe(feats, xyz)
        stretched_n = no_rpe(feats, xyz * 3.0)

    with_rpe = (base - stretched).abs().max().item()
    without = (base_n - stretched_n).abs().max().item()
    # 无 RPE 时 kNN 邻域可能变化，所以不强制为 0；关键是开关必须改变行为
    assert with_rpe > 1e-4, "开了 RPE 却对几何不敏感，说明位置编码没接进计算"
    assert with_rpe > without, (
        f"RPE 版对几何的敏感度({with_rpe:.4f}) 应高于无 RPE 版({without:.4f})"
    )


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    print("全部通过" if failed == 0 else f"{failed} 个用例失败")
    sys.exit(1 if failed else 0)
