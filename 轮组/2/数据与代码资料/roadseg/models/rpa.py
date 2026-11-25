"""基于相对位置注意力的改进模型（供做题人参考/对照）。

本文件不要求做题人照抄，它的作用是：
1. 给出一个**满足题目"四、论文检索与模型改进"要求**的最小可行设计，
   让出题方和做题人对"什么算合格的改进"有共同标准；
2. 把 Q/K/V 的来源、相对位置如何进入注意力、张量形状如何变化，
   全部写成代码里的注释，答辩时可以直接指着讲。

对照的基线缺陷（实验现象见报告模板 3.2 节）
--------------------------------------------
PointNet 只有一次全局 max 聚合，等价于"每个点只看自己 + 一整帧的均值式摘要"：
* 相邻点之间的几何关系（局部平面、边缘、曲面法向）从未显式建模；
* max 聚合对**局部空洞**无感知——删掉一片区域后，剩下的点特征几乎不变，
  但它们的邻域已经变了，模型没有任何信号可用；
* 孤立离群点会直接参与 max，一个异常点就可能改写某一维全局特征。

改进核心
--------
把"一整帧一个全局向量"换成"**每个点在自己的 k 个近邻里做注意力**"：

    Q_i = W_q · f_i
    K_j = W_k · (f_j + φ(p_j − p_i))
    V_j = W_v · (f_j + φ(p_j − p_i))
    α_ij = softmax_j( Q_i·K_j / √d  +  b(p_j − p_i) )
    f_i' = Σ_j α_ij V_j

其中 ``p_j − p_i`` 是**相对坐标**，``φ`` 是高斯径向基（RBF）编码，
``b`` 是一个可学习的相对位置偏置。相对坐标而不是绝对坐标，是平移/旋转
不变性的来源，也是与视觉 Transformer（需要绝对位置编码）最本质的差别。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .pointnet import SharedMLP, count_parameters


# --------------------------------------------------------------------------- #
# 相对位置编码
# --------------------------------------------------------------------------- #
class RelativePositionEncoding(nn.Module):
    """把相对坐标 (Δx, Δy, Δz) 编码成 ``out_dim`` 维向量。

    Parameters
    ----------
    out_dim : int
        输出维度，需与注意力头维度对齐。
    num_rbf : int
        径向基的个数。用高斯 RBF 而不是直接线性投影，是因为
        "距离近"与"距离远"的语义是非线性的，RBF 让网络更容易区分
        1cm 与 1m 的差别。
    r_max : float
        RBF 中心覆盖的最远距离（米）。取 4~8 m 覆盖一个局部邻域即可；
        取太大等于退化成全局注意力。
    """

    def __init__(self, out_dim: int, num_rbf: int = 16, r_max: float = 4.0,
                 use_rbf: bool = True):
        super().__init__()
        self.use_rbf = use_rbf
        self.num_rbf = num_rbf
        self.r_max = r_max

        if use_rbf:
            # 高斯中心在对数空间均匀分布，近处密、远处疏
            centers = torch.linspace(0.0, r_max, num_rbf)
            self.register_buffer("centers", centers)
            self.register_buffer("widths", torch.full((num_rbf,), r_max / num_rbf))
            in_dim = num_rbf + 3          # RBF 距离编码 + 原始方向
        else:
            self.register_buffer("centers", torch.zeros(1))
            self.register_buffer("widths", torch.ones(1))
            in_dim = 3                    # 只用原始 Δxyz（消融用）

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        """rel: (..., 3) → (..., out_dim)。"""
        if self.use_rbf:
            d = rel.norm(dim=-1, keepdim=True)                       # (..., 1)
            diff = d - self.centers.view(*([1] * (d.dim() - 1)), -1)
            rbf = torch.exp(-0.5 * (diff / self.widths.view(self.widths.shape[0])) ** 2)
            feat = torch.cat([rbf, rel], dim=-1)
        else:
            feat = rel
        return self.mlp(feat)

    def bias(self, rel: torch.Tensor) -> torch.Tensor:
        """用于加在注意力 logits 上的标量偏置：``(..., 1)``。"""
        return self.mlp(rel if not self.use_rbf else self._rbf_feat(rel)).sum(-1, keepdim=True)

    def _rbf_feat(self, rel: torch.Tensor) -> torch.Tensor:
        d = rel.norm(dim=-1, keepdim=True)
        diff = d - self.centers.view(*([1] * (d.dim() - 1)), -1)
        rbf = torch.exp(-0.5 * (diff / self.widths) ** 2)
        return torch.cat([rbf, rel], dim=-1)


# --------------------------------------------------------------------------- #
# 局部邻域注意力
# --------------------------------------------------------------------------- #
class LocalAttentionBlock(nn.Module):
    """在一个点的 k 近邻上做多头注意力（含相对位置）。

    Parameters
    ----------
    dim : int
        通道数。
    num_heads : int
        头数，``dim`` 必须能被它整除。
    k : int
        邻居数。k 越大越接近全局注意力，但显存是 O(N·k)。
    r_max : float
        RBF 覆盖半径。
    use_rpe : bool
        是否使用相对位置编码。**这是消融的核心开关**：
        关掉后 K、V 只由特征构成，模型退化成"局部特征注意力"，
        用它来证明"收益来自位置信息"而不是"多了一个注意力层"。
    agg : {"attn", "max"}
        ``attn`` 用注意力加权；``max`` 在同一 kNN 上用 max 池化。
        这是**替换型消融**：两者邻域完全相同，只差聚合方式，
        用来证明收益来自注意力而不是 kNN 本身。
    dropout : float
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        k: int = 16,
        r_max: float = 4.0,
        use_rpe: bool = True,
        agg: str = "attn",
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} 不能被 num_heads={num_heads} 整除")
        if agg not in ("attn", "max"):
            raise ValueError(f"agg 只能是 'attn' 或 'max'，收到 {agg}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.k = k
        self.agg = agg
        self.use_rpe = use_rpe
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        if use_rpe:
            self.rpe_k = RelativePositionEncoding(dim, num_rbf=16, r_max=r_max)
            self.rpe_v = RelativePositionEncoding(dim, num_rbf=16, r_max=r_max)
            self.attn_bias = RelativePositionEncoding(num_heads, num_rbf=16, r_max=r_max)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # -- kNN ---------------------------------------------------------------
    @staticmethod
    def knn(xyz: torch.Tensor, k: int) -> torch.Tensor:
        """xyz: (B, N, 3) → idx: (B, N, k)，含自身。

        用 ``torch.cdist`` 分块算距离以控制显存；B·N·N 在 N=8192、B=8 时
        仍会占 512MB，因此按点分块（``chunk``）。
        """
        b, n, _ = xyz.shape
        k = min(k, n)
        idx = torch.empty(b, n, k, dtype=torch.long, device=xyz.device)
        chunk = max(1, min(n, 1024))
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            d = torch.cdist(xyz[:, s:e], xyz)                  # (B, chunk, N)
            idx[:, s:e] = d.topk(k, dim=-1, largest=False).indices
        return idx

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """feats: (B, N, C)，xyz: (B, N, 3) → (B, N, C)。"""
        b, n, c = feats.shape
        k = min(self.k, n)

        idx = self.knn(xyz, k)                                  # (B, N, k)
        h, d = self.num_heads, self.head_dim

        # 邻居特征与相对坐标：全部用 gather 从 (B,N,·) 展开到 (B,N,k,·)
        idx_f = idx.unsqueeze(-1).expand(-1, -1, -1, c)
        neigh = torch.gather(feats.unsqueeze(1).expand(-1, n, -1, -1), 2, idx_f)
        idx_p = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        neigh_xyz = torch.gather(xyz.unsqueeze(1).expand(-1, n, -1, -1), 2, idx_p)

        rel = neigh_xyz - xyz.unsqueeze(2)                      # (B, N, k, 3) 相对坐标

        x = self.norm1(feats)
        q = self.q(x).view(b, n, h, d)                          # (B, N, H, d)
        kf = self.k_proj(x)                                     # (B, N, C)
        vf = self.v_proj(x)                                     # (B, N, C)

        # 把邻居的 K、V 特征 gather 成 (B, N, k, C)：第 i 个点拿到它 k 个邻居的特征
        idx_c = idx.unsqueeze(-1).expand(-1, -1, -1, c)
        kf = torch.gather(kf.unsqueeze(1).expand(-1, n, -1, -1), 2, idx_c)
        vf = torch.gather(vf.unsqueeze(1).expand(-1, n, -1, -1), 2, idx_c)

        # 有 RPE 时把位置信息加进 K、V，让"邻居特征 + 它相对于我有多远"一起决定权重
        if self.use_rpe:
            kf = kf + self.rpe_k(rel)
            vf = vf + self.rpe_v(rel)

        kk = kf.view(b, n, k, h, d).permute(0, 1, 3, 2, 4)      # (B, N, H, k, d)
        vv = vf.view(b, n, k, h, d).permute(0, 1, 3, 2, 4)      # (B, N, H, k, d)

        if self.agg == "max":
            # 替换型消融：同样的邻域、同样的 K/V，但用 max 聚合
            agg = vv.max(dim=3).values.reshape(b, n, c)
        else:
            logits = (q.unsqueeze(3) * kk).sum(-1) * self.scale  # (B, N, H, k)
            if self.use_rpe:
                logits = logits + self.attn_bias(rel).permute(0, 1, 3, 2)  # (B,N,H,k)
            attn = torch.softmax(logits, dim=-1)
            attn = self.dropout(attn)
            agg = (attn.unsqueeze(-1) * vv).sum(dim=3)          # (B, N, H, d)
            agg = agg.reshape(b, n, c)

        out = self.out(agg)
        out = self.dropout(out)
        # 残差 + 第二层前馈，保证"关掉注意力"时网络退化成普通逐点 MLP 也能跑
        return self.norm2(feats + out)


# --------------------------------------------------------------------------- #
# 改进模型
# --------------------------------------------------------------------------- #
class RPANet(nn.Module):
    """Relative Position Attention Network：PointNet 基线 + 局部相对位置注意力。

    与基线的差异只有一处：在"逐点特征"和"对称聚合"之间插入了
    ``num_blocks`` 个 ``LocalAttentionBlock``。其余四段主干保持不变，
    这样任何性能变化都能归因到注意力模块，而不是"换了更大的网络"。

    Parameters
    ----------
    num_classes, in_channels : int
    k : int
        邻域大小。
    num_blocks : int
        注意力块个数，建议 1~2。再多显存会明显吃紧。
    num_heads : int
    use_rpe : bool
        消融开关，默认 True。
    agg : {"attn", "max"}
        消融开关，默认 "attn"。
    global_pool : {"max", "attn"}
        对称聚合方式。``attn`` 用可学习的注意力权重对 N 个点加权求和，
        仍然是交换不变量，但对离群点更鲁棒（离群点可以被分配趋近 0 的权重）。
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        local_dim: int = 64,
        global_dim: int = 1024,
        k: int = 16,
        num_blocks: int = 2,
        num_heads: int = 4,
        r_max: float = 4.0,
        use_rpe: bool = True,
        agg: str = "attn",
        global_pool: str = "max",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_blocks = num_blocks
        self.global_pool = global_pool

        # ---- 与基线完全相同的前两段 ----
        self.local_mlp = SharedMLP(in_channels, [64, local_dim])
        self.global_mlp = SharedMLP(local_dim, [128, 512, global_dim])

        # ---- 改进：局部相对位置注意力 ----
        self.attn_blocks = nn.ModuleList([
            LocalAttentionBlock(
                dim=local_dim, num_heads=num_heads, k=k, r_max=r_max,
                use_rpe=use_rpe, agg=agg, dropout=0.0,
            )
            for _ in range(num_blocks)
        ])

        # ---- 对称聚合（可选注意力池化）----
        if global_pool == "attn":
            self.pool_score = nn.Sequential(
                nn.Linear(global_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

        # ---- 与基线相同的分割头 ----
        self.seg_head = nn.Sequential(
            nn.Conv1d(local_dim + global_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, points: torch.Tensor, return_features: bool = False):
        """points: (B, N, 3) → logits: (B, N, C)。"""
        xyz = points

        f1 = self.local_mlp(xyz.transpose(1, 2).contiguous())     # (B, C1, N)
        f1 = f1.transpose(1, 2).contiguous()                      # (B, N, C1)

        for blk in self.attn_blocks:
            f1 = blk(f1, xyz)                                     # (B, N, C1)

        f2 = self.global_mlp(f1.transpose(1, 2).contiguous())     # (B, C2, N)

        if self.global_pool == "attn":
            w = torch.softmax(self.pool_score(f2.transpose(1, 2)), dim=1)  # (B, N, 1)
            g = (f2 * w.transpose(1, 2)).sum(dim=2)                        # (B, C2)
        else:
            g = f2.max(dim=2).values

        g_exp = g.unsqueeze(2).expand(-1, -1, points.shape[1])
        fused = torch.cat([f1.transpose(1, 2), g_exp], dim=1)
        logits = self.seg_head(fused).transpose(1, 2).contiguous()

        if return_features:
            return logits, {
                "local": f1,
                "pointwise": f2.transpose(1, 2),
                "global": g,
                "fused": fused.transpose(1, 2),
            }
        return logits


def build_rpa(num_classes: int = 5, **kwargs) -> RPANet:
    return RPANet(num_classes=num_classes, **kwargs)


#: 题目要求的消融组合，训练脚本按名字取用（注册在 models/__init__.py 里统一做）。
ABLATIONS: dict[str, dict] = {
    "baseline":        {"name": "pointnet"},
    "rpa_full":        {"name": "rpa", "use_rpe": True,  "agg": "attn", "num_blocks": 2},
    "rpa_no_rpe":      {"name": "rpa", "use_rpe": False, "agg": "attn", "num_blocks": 2},
    "rpa_local_max":   {"name": "rpa", "use_rpe": True,  "agg": "max",  "num_blocks": 2},
    "rpa_1block":      {"name": "rpa", "use_rpe": True,  "agg": "attn", "num_blocks": 1},
    "rpa_attnpool":    {"name": "rpa", "use_rpe": True,  "agg": "attn", "num_blocks": 2,
                        "global_pool": "attn"},
}


if __name__ == "__main__":
    # 快速自检：打印每个消融版本的参数量与输出形状
    # 用法：python -m roadseg.models.rpa
    import roadseg.models  # noqa: F401  触发注册
    from ..registry import MODEL_REGISTRY

    torch.manual_seed(0)
    x = torch.rand(2, 256, 3)
    for name, cfg in ABLATIONS.items():
        kwargs = {k: v for k, v in cfg.items() if k != "name"}
        m = MODEL_REGISTRY[cfg["name"]](**kwargs).eval()
        with torch.no_grad():
            y = m(x)
        print(f"{name:15s} params={count_parameters(m):>9,d}  out={tuple(y.shape)}")

