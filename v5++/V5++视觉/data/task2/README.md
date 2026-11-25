# Task 2 数据集

这里存放 Task 2 的训练集, 验证集和测试集, 图片均为 64x64 的灰度手写数字图片. 数据集包含数字 0 ~ 8 共 9 个类别 (不含 9).

## 目录结构

```text
task2/
├── train/
│   ├── images/        # 5400 张 64x64 灰度图 (0000.png ~ 5399.png)
│   └── labels.json    # 训练集答案标注
├── val/
│   ├── images/        # 900 张 64x64 灰度图 (0000.png ~ 0899.png)
│   └── labels.json    # 验证集答案标注
└── test/
    ├── images/        # 900 张 64x64 灰度图 (0000.png ~ 0899.png)
    └── labels.json    # 测试集答案标注
```

## `labels.json` 格式

与 Task 1 一致, 每个目录下的 `labels.json` 都是一个 JSON 列表:

```json
[
  {
    "id": 0,
    "image": "images/0000.png",
    "label": 1
  }
]
```

-   `id`: 样本编号
-   `image`: 图片相对当前子目录的路径
-   `label`: 对应的数字 (0 ~ 8)
