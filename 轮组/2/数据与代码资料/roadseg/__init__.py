"""roadseg —— 道路场景三维点云语义分割的课程工具包。

面向《面向稀疏、缺失与噪声的道路三维点云语义分割》项目，提供**评测基础设施**，
不提供可以直接交作业的成品模型：

=================  ====================================================
模块                 作用
=================  ====================================================
``labels``         课程 5 类的唯一定义、SemanticPOSS 原始标签 → 课程标签的映射
``datasets``       ``.bin``/``.label`` 读取、帧级 Dataset、划分读取
``degradation``    稀疏 / 局部缺失 / 坐标噪声 / 离群点四类退化 + 标准 14 档套件
``metrics``        混淆矩阵、mIoU、各类 IoU、OA、FWIoU、高频混淆对
``sampling``       随机 / 等间隔 / FPS / 体素 / 分层采样
``utils``          随机种子、坐标归一化、kNN、JSONL 日志
``models``         PointNet 基线 + 相对位置注意力改进模型（含消融开关）
``evaluation``     训练/评估共用的退化评测循环（需要 torch，故不在此处导入）
=================  ====================================================

用法::

    from roadseg.labels import CLASS_NAMES, NUM_CLASSES
    from roadseg.degradation import standard_suite, degrade
    from roadseg.metrics import ConfusionMatrixAccumulator

设计原则
--------
1. **只依赖 numpy**（``models`` 与 ``evaluation`` 子模块额外需要 torch）。
   任何会卡住环境安装的依赖（scipy、open3d、torch-scatter）都不使用。
2. **评测口径写死在代码里**：标签映射、忽略规则、退化参数、采样方式都以
   常量或默认参数形式固化，保证不同人的结果可比。
3. **一套评估只写一遍**：``roadseg/evaluation.py`` 是唯一的退化评测实现，
   ``train.py`` 与 ``evaluate.py`` 都调它。以前两边各写一份，退化种子
   （1000+i vs 10000+i）和计数口径（按 batch vs 按帧）都不一样，
   训练日志里的数字和最终报告的数字根本对不上。
4. **谁都可以改，但改了要留痕**：所有取舍都写成注释，说明"为什么这样"。
"""

from . import datasets, degradation, labels, metrics, sampling, utils

__all__ = ["labels", "datasets", "degradation", "metrics", "sampling", "utils"]

__version__ = "0.2.0"

