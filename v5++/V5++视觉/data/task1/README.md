# Task 1 验证集

这里存放 Task 1 的验证数据, 用于在 MNIST 训练集上训练好模型后做推理测试.

## 目录结构

```text
val/
├── images/        # 1000 张 28x28 的灰度数字图片 (0000.png ~ 0999.png)
└── labels.json    # 每张图片对应的答案标注
```

## `labels.json` 格式

一个 JSON 列表, 每条记录对应一张图片:

```json
[
  {
    "id": 0,
    "image": "images/0000.png",
    "label": 3
  }
]
```

-   `id`: 样本编号 (0 ~ 999)
-   `image`: 图片相对 `val/` 的路径
-   `label`: 对应的数字 (0 ~ 9)
