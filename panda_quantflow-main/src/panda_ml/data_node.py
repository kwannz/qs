from .base_node import BaseNode
import logging

from abc import ABC, abstractmethod

class DataNode(BaseNode):
    """
    数据处理节点基类，负责对输入 data 进行转换、清洗、特征工程等
    """
    def __init__(self, params: dict):
        super().__init__(params)

    def fit(self, data):
        """如果节点自身需要训练（如Scaler），重写此方法"""
        pass

    @abstractmethod
    def transform(self, data):
        """对数据进行转换"""
        pass

    def execute(self, context: dict):
        """
        执行数据转换流程，根据是否含 fit 阶段调用对应方法
        context 可包含: {'mode': 'fit'/'transform', 'data': ...}
        """
        data = context.get('data')
        mode = context.get('mode', 'transform')
        if mode == 'fit':
            self.fit(data)
        return self.transform(data)
