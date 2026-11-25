"""模型注册表：按名字构造模型，供配置文件驱动。

做题人新增自己的模型时，只需在 ``roadseg/models/`` 下写一个类并
``@register_model("myname")``，配置文件里改 ``model.name: myname`` 即可，
不需要动训练脚本。
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """装饰器：把一个构造函数注册为名为 ``name`` 的模型。"""

    def deco(fn: Callable[..., nn.Module]):
        if name in MODEL_REGISTRY:
            raise KeyError(f"模型名重复注册：{name}")
        MODEL_REGISTRY[name] = fn
        return fn

    return deco


def build_model(cfg: dict) -> nn.Module:
    """按配置字典构造模型。

    Parameters
    ----------
    cfg : dict
        ``{"name": "pointnet", "num_classes": 5, ...其余关键字参数}``

    Returns
    -------
    nn.Module

    Notes
    -----
    模型是在 ``roadseg/models/__init__.py`` 里注册的。如果调用方只
    ``from roadseg.registry import build_model`` 而不 import ``roadseg.models``，
    注册表还是空的。为避免这种"看起来能 import、一调用就 KeyError"的坑，
    这里在注册表为空时惰性导入一次 ``roadseg.models``（只依赖 torch，
    导入本身没有副作用）。
    """
    if not isinstance(cfg, dict) or "name" not in cfg:
        raise ValueError("model 配置必须是含 'name' 键的字典")
    if not MODEL_REGISTRY:                     # 懒加载，避免用前必须手动 import
        import roadseg.models  # noqa: F401
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"未知模型 {name!r}；已注册：{sorted(MODEL_REGISTRY)}。"
            " 请检查配置文件或确认已 import 相应模块。"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return MODEL_REGISTRY[name](**kwargs)

