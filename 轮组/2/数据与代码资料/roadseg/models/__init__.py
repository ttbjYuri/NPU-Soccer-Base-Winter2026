"""模型子包：PointNet 基线 + 基于相对位置注意力的改进模型。

新增模型的三步：

1. 在本目录下新建 ``mynet.py``，定义 ``class MyNet(nn.Module)``；
2. 写一个 ``build_mynet(...)`` 工厂函数；
3. 在下面的 ``_REGISTER`` 里加一行 ``register_model("mynet")(build_mynet)``，
   然后配置文件写 ``model.name: mynet`` 即可，训练脚本不用改。
"""

from ..registry import MODEL_REGISTRY, build_model, register_model
from .pointnet import PointNetSeg, build_pointnet, count_parameters
from .rpa import ABLATIONS, LocalAttentionBlock, RPANet, build_rpa

__all__ = [
    "PointNetSeg", "build_pointnet", "count_parameters",
    "RPANet", "build_rpa", "LocalAttentionBlock", "ABLATIONS",
    "MODEL_REGISTRY", "build_model", "register_model",
]

#: 把基线与改进模型注册进全局表，让 ``build_model({"name": "..."})`` 生效。
register_model("pointnet")(build_pointnet)
register_model("rpa")(build_rpa)
