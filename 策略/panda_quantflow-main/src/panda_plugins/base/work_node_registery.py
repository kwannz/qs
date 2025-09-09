from typing import Callable, Literal, Optional, Type, Dict
from panda_plugins.base.base_work_node import BaseWorkNode

ALL_WORK_NODES: Dict[str, Type[BaseWorkNode]] = {}

def work_node(
    name: Optional[str],
    group: Optional[str] = "自定义节点",
    order: Optional[int] = 1,
    type: Optional[str] = "general",
    box_color: Optional[
        Literal["red", "brown", "green", "blue", "cyan", "purple", "yellow", "black"]
    ] = "black",
) -> Callable[[Type[BaseWorkNode]], Type[BaseWorkNode]]:
    """
    Decorator for registering work nodes.
    Use @work_node() to register work nodes.
    Parameters:
    - name: The name of the work node.
    - group: The group of the work node, support multi-level directory structure separated by "/".
    - [Deprecated] order: The order of the work node.
    - type: The type of the work node.

    用于注册工作节点的装饰器。
    使用 @work_node() 来注册工作节点。
    参数：
    - name: 工作节点的名称。
    - group: 工作节点的分组,支持以"/"形式分割多层目录结构。
    - [Deprecated] order: 工作节点的顺序。
    - type: 工作节点的类型。
    """

    def decorator(cls: Type[BaseWorkNode]) -> Type[BaseWorkNode]:
        # 防御性检查类型
        if not issubclass(cls, BaseWorkNode):
            raise TypeError(f"Node {cls.__name__} must inherit from BaseWorkNode")

        # 如果 name 为空，则使用类名作为 name
        nonlocal name
        if name == "":
            name = cls.__name__

        # 设置类属性
        setattr(cls, "__work_node_name__", cls.__name__)
        setattr(cls, "__work_node_display_name__", name)
        setattr(cls, "__work_node_group__", group)
        setattr(cls, "__work_node_order__", order)
        setattr(cls, "__work_node_type__", type)
        setattr(cls, "__work_node_box_color__", box_color)

        ALL_WORK_NODES[cls.__name__] = cls
        return cls

    return decorator
