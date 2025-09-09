from abc import ABC, abstractmethod
import logging

class BaseNode(ABC):
    """
    所有节点的抽象基类
    定义输入输出契约、公共属性和执行接口
    """
    def __init__(self, params: dict):
        self.params = params      # 超参数字典
        self.inputs = []          # 输入Artifact类型列表
        self.outputs = []         # 输出Artifact类型列表
        self.status = "initialized"

    @abstractmethod
    def execute(self, context: dict):
        """
        通用执行入口，context 包含输入数据引用、运行时环境等信息
        """
        pass