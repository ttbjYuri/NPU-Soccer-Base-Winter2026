# 数据与代码资料 · roadseg 参考实现

本目录是《三维点云的识别》项目题配套下发的**参考实现**，实现题目要求的全部基础组件：
五类标签映射、四类退化模拟、mIoU 评测、PointNet 基线、带相对位置注意力的改进模型、
训练与评估脚本、预测可视化。

> 使用边界：这份代码是出题方的参考实现，目的是让你有一个正确、可运行、口径统一的起点，
> 避免把时间花在"标签到底怎么对""mIoU 到底怎么算"这类与题目能力点无关的事情上。
> 具体边界：
>
> - **基线部分**（`roadseg/models/pointnet.py` 的四段主干）——题目要求自己写、自己讲。
>   不建议直接 import `PointNetSeg`，若确实用了，必须在报告里标注哪些行是自己写的。
> - **改进部分**（`roadseg/models/rpa.py`）——可以借鉴、修改、替换，但要讲清改了什么、
>   为什么这么改。原样照抄不算改进。

---

## 1. 环境

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

# 没有 GPU 时安装 CPU 版 torch（体积小很多）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

依赖只有 `torch / numpy / PyYAML / pytest`（`matplotlib` 仅 `visualize.py` 需要）。
不依赖 scipy，也不依赖任何点云专用库，普通笔记本可跑。

## 2. 数据

课程统一使用 **SemanticPOSS**（Pan et al., *SemanticPOSS: A Point Cloud Dataset with
Large Amount of Dynamic Objects*, arXiv:2002.09147），北大校园 LiDAR 序列，
2988 帧、14 类原始标签、约 2.41 GB，许可 **CC BY-NC-SA 3.0（仅非商业教学使用）**。

下载：<http://www.poss.pku.edu.cn/semanticposs.html>

原始格式与 SemanticKITTI 一致：

```
sequences/00/velodyne/000000.bin    float32 (N, 4) = x, y, z, intensity
sequences/00/labels/000000.label    uint32  (N,) 低 16 位语义、高 16 位实例
```

### 课程五类与官方划分

题目正文表 2-2 定义了五类映射（`roadseg/labels.py` 是**唯一权威定义**）：

| 课程类别 | 编号 | 来源原始标签 id |
|---|---|---|
| 地面 | 0 | 22 ground |
| 车辆 | 1 | 7 car |
| 行人 | 2 | 4 person, 5 persons |
| 建筑 | 3 | 15 building |
| 植被 | 4 | 9 plants |
| 忽略 | 255 | 其余全部（0,1,2,3,6,8,10-14,16-21,23-255 …） |

不在上表出现的 id 一律映射为 255（`roadseg/labels.py` 的 `DEFAULT_LUT` 长度为 256，
是查表而非 if/else，所以不存在漏分支的问题）。

**为什么是 5 类而不是 6 类**：SemanticPOSS 只有 `ground(22)` 一个地面语义类，
不区分车道与人行道。若强行保留"人行道"，该类在所有划分里点数恒为 0，
IoU 会被记作 nan 并从 mIoU 剔除——既无意义也不可复现。详见 `roadseg/labels.py` 的模块文档。

论文原口径（按车辆位置切 6 段，part 3 为测试集）：

```
train = 00, 01, 02, 04, 05
test  = 03
val   = 无
```

官方没有验证集，而题目禁止用测试集选参数，因此**课程固定**从训练部分切出 04 作验证集：

```
train = 00, 01, 02, 05     训练，可做数据增强
val   = 04                 选模型、调超参
test  = 03                 最终评测，只允许用一次
```

这也是 `tools/prepare_data.py` 在不传 `--split` 时的默认划分
（见 `tools/prepare_data.py:186`）。`--split` 参数可以覆盖它，但报告里必须写明用的是哪一种。

### 转换

```bash
# 1) 只检查：确认帧数、标签取值、点/标签长度一致
python tools/prepare_data.py --raw D:/SemanticPOSS/dataset/sequences --check

# 2) 全量转换（约 3000 帧，机械盘 5~10 分钟）
python tools/prepare_data.py --raw D:/SemanticPOSS/dataset/sequences \
    --out data/course --split 00:train,01:train,02:train,03:test,04:val,05:train

# 3) 只转前 50 帧，先跑通流程
python tools/prepare_data.py ... --limit 50
```

产物：

```
data/course/
  frames/00_000000.npz      points (N,3) float32 / labels (N,) uint8 / intensity (N,) float32
  manifest.csv              每帧点数、有效点、各类占比
  splits/{train,val,test}.txt
  label_stats.json          全局类别分布（写进报告的"数据说明"一节）
```

## 3. 三条命令

先用合成数据确认环境没问题（半分钟），再上真实数据：

```bash
# 1) 合成数据自检：不下载任何东西，只验证代码能跑通
python tools/make_dummy_data.py --out data/dummy --frames 15

python train.py --config configs/pointnet.yaml \
    --data-root data/dummy --epochs 2 --limit-frames 6

python evaluate.py --ckpt runs/pointnet_seed0/best.pth \
    --data-root data/dummy --max-frames 2

# 2) 正式训练 + 全档位评估
python train.py --config configs/pointnet.yaml
python evaluate.py --ckpt runs/pointnet_seed0/best.pth \
    --split test --save-json results/pointnet_test.json
```

改进模型与消融：把 `--config` 换成 `configs/rpa_full.yaml` /
`rpa_no_rpe.yaml` / `rpa_local_max.yaml` / `rpa_1block.yaml` / `rpa_attnpool.yaml`。

预测可视化（真值 / 预测 / 错误三联图，含挑案例用的 index.csv）：

```bash
python visualize.py --ckpt runs/rpa_full_seed0/best.pth --num 6
python visualize.py --ckpt runs/pointnet_seed0/best.pth \
    --frames 03_000100,03_000233 --suite sparse_25+noise_5cm
```

## 4. 十四档退化

`evaluate.py --list-suites` 可查看，`--suite clean,sparse_50` 可只跑指定档：

| 档位 | 含义 |
|---|---|
| `clean` | 原始测试数据 |
| `sparse_50/25/12.5` | 保留 50% / 25% / 12.5% 的点 |
| `sector_60deg` | 删除正前方 60°（±30°）扇区内所有点 |
| `box_6m` | 删除原点处 6 m × 6 m × 3 m 长方体内所有点 |
| `noise_2cm/5cm/10cm` | 对每轴独立叠加标准差 σ = 2 / 5 / 10 cm 的高斯噪声 |
| `outlier_64/256` | 在已有点的局部邻域内插入 64 / 256 个离群点（距锚点 0.5—2 m） |
| `sparse_25+noise_5cm` | 组合：保留 25% + 5 cm 噪声 |
| `sparse_12.5+noise_10cm` | 组合：保留 12.5% + 10 cm 噪声 |
| `sector+outlier` | 组合：删扇区 + 插离群点 |

评测只统计**保留下来的原始点**：被删的点不计入，插入的离群点标签为 255、`keep=False`，
既不影响指标也会参与模型前向——与题目第四章一致。

若某档退化后剩余点数不足 `--num-points`（默认 8192），该帧直接使用全部剩余点，
不做上采样、不做重复采样；报告里应一并记录实际参与评测的点数。

## 5. 目录结构

```
数据与代码资料/
├── train.py                    训练（配置驱动，支持 --resume 断点续训）
├── evaluate.py                 原始 + 14 个退化档位评估
├── visualize.py                预测可视化（真值 / 预测 / 错误）
├── requirements.txt
├── configs/
│   ├── pointnet.yaml           基线
│   ├── rpa_full.yaml           完整改进模型
│   ├── rpa_no_rpe.yaml         消融：去掉相对位置编码
│   ├── rpa_local_max.yaml      消融：注意力换成 max 池化
│   ├── rpa_1block.yaml         消融：只留一个注意力块
│   └── rpa_attnpool.yaml       消融：全局池化换成注意力池化
├── docs/
│   └── make_doc.py             生成题目文档（出题方用，学生可不看）
├── roadseg/
│   ├── labels.py               五类定义、原始标签 → 课程标签映射
│   ├── datasets.py             读 npz、空间范围过滤、定长采样、manifest 列定义
│   ├── sampling.py             随机 / 等间隔 / FPS / 体素 / 分层采样
│   ├── degradation.py          四类退化 + 14 个标准档位
│   ├── metrics.py              混淆矩阵、IoU、mIoU、OA、FW-IoU
│   ├── utils.py                种子、归一化、kNN、日志
│   ├── registry.py             模型注册表
│   ├── evaluation.py           退化评估循环（train.py 与 evaluate.py 共用）
│   └── models/
│       ├── pointnet.py         PointNet 基线（四段主干）
│       └── rpa.py              相对位置注意力改进模型 + 6 个消融配置
├── tools/
│   ├── prepare_data.py         .bin/.label → npz + manifest + splits
│   └── make_dummy_data.py      生成合成数据，用于跑通整条链路
└── tests/
    ├── conftest.py             让 pytest 能在仓库根目录 import roadseg
    ├── test_models.py          形状、交换不变性、RPE 是否真的生效（需要 torch）
    └── test_pipeline.py        标签映射、退化、采样、指标、manifest（纯 numpy）
```

`train.py` 与 `evaluate.py` 共用 `roadseg/evaluation.py`，因此训练日志里的 val mIoU
与 `evaluate.py` 的数字是同一套口径（同样的退化种子、同样的帧计数方式）。

## 6. 四段主干对照

| 题目要求 | 参考实现中的位置 | 张量形状 |
|---|---|---|
| 共享的逐点特征提取 | `PointNetSeg.local_mlp`（Conv1d k=1 实现的共享 MLP） | (B, 3, N) → (B, 64, N) |
| 对称的全局特征聚合 | `PointNetSeg.global_mlp` + 沿 N 取 max | (B, 64, N) → (B, 1024) |
| 局部与全局特征融合 | 沿通道拼接 local 与 broadcast 后的 global | (B, 1088, N) |
| 逐点类别预测 | `seg_head`（三层 Conv1d k=1） | (B, 1088, N) → (B, 5, N) |

改进模型只在这四段之间的一个位置插入注意力块：在"逐点特征"之后、"全局聚合"之前，
对每个点的 k 近邻做多头注意力。其余四段不变，因此性能变化可以归因到注意力模块。
所有消融版本参数量都在 **1.30 M — 1.49 M** 之间，量级相同。

## 7. 已知取舍

> 本节只说明参考实现的工程取舍，**不构成题面第三、五问的答案**。
> 答辩要问的“Q/K/V 从哪来、空间关系如何参与计算、代价与失效条件”，
> 需要你自己读代码后组织语言，不要照抄本节。

- **不做尺度归一化，只做平移。** 退化扰动的是物理坐标（厘米级噪声、米级扇区），
  一旦缩放，2 cm 噪声就变成无量纲的 0.05，与题目的物理含义脱钩。
  注意 `roadseg/utils.py:normalize_points` 只按 **xy 均值**平移，z 完全不动。
- **训练时不挖空、不加离群点。** 这两类增强属于改进方向，若加进基线，基线就不公平了。
- **kNN 用暴力分块，不引 scipy。** 实测（纯 numpy，k=16）：n=4096 约 0.42 s、
  n=8192 约 1.49 s、n=20000 约 8.6 s、n=60000 约 73 s。因此 rpa 系列在 CPU 上
  训练极慢，**建议先只在 4096 点上做冒烟验证**；100k 点以上必须先体素下采样。
- **T-Net 默认关闭。** 开启后坐标噪声下 T-Net 估计的变换矩阵会被噪声带偏；
  要开需在报告里给出你自己的对比数据，并要求基线与改进模型保持一致。

## 8. 自检清单

| 检查项 | 通过标准 |
|---|---|
| 环境能装起来 | `pip install -r requirements.txt` 后无报错（含 pytest） |
| 单元测试通过 | `python -m pytest tests -q` 全部通过 |
| 合成数据能跑通 | `make_dummy_data` → `train` → `evaluate` 三条命令全部成功 |
| 标签映射正确 | 五类占比之和接近 1，且地面占多数 |
| 评测口径一致 | 两模型用同一个 `--num-points`、同一个 `--split`、同一个 seed |
| 测试集只用一次 | 报告里的测试集数字来自最终那次 `evaluate` |
| 日志与配置齐全 | `runs/` 下每份实验都有 `config.json` 与 `train.log.jsonl` |
| 权重可复现 | 给定 `config.json` + seed，重训能得到相近指标 |

---

数据引用：本项目使用 **SemanticPOSS**（Pan et al., arXiv:2002.09147），
遵循 **CC BY-NC-SA 3.0** 许可，仅用于非商业教学。
