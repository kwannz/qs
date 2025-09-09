from abc import ABC, abstractmethod
import logging

from .base_node import BaseNode

class ModelNode(BaseNode):
    """
    机器学习/深度学习模型节点基类
    实现训练、预测、在线更新等方法
    """
    def __init__(self, params: dict):
        super().__init__(params)
        # 可在子类中提取具体超参数
        # e.g. self.learning_rate = params.get('learning_rate', 0.01)

    @abstractmethod
    def fit(self, X, y):
        """离线训练模型"""
        pass

    @abstractmethod
    def predict(self, X):
        """模型推断/预测"""
        pass

    def update(self, X_new, y_new):
        """
        在线学习或增量更新接口
        默认抛出异常，需要支持增量训练的模型节点重写此方法
        """
        raise NotImplementedError("Online update not implemented")

    def execute(self, context: dict):
        """
        根据 context 决定是训练（fit）还是预测（predict）
        context 示例：{
          'mode': 'train' or 'predict' or 'update',
          'X': ..., 'y': ...
        }
        """
        mode = context.get('mode')
        X = context.get('X')
        y = context.get('y')
        if mode == 'train':
            model = self.fit(X, y)
            self.status = 'trained'
            return model
        elif mode == 'predict':
            preds = self.predict(X)
            self.status = 'predicted'
            return preds
        elif mode == 'update':
            model = self.update(X, y)
            self.status = 'updated'
            return model
        else:
            raise ValueError(f"Unknown mode: {mode}")