from datetime import datetime
import logging

from typing import Optional, Type, List, Union
import uuid
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, Field
from pathlib import Path
from panda_plugins.base.base_work_node import BaseWorkNode
from panda_plugins.base.work_node_registery import work_node
from panda_plugins.base import ui
from panda_factor.generate.macro_factor import MacroFactor
from panda_plugins.internal.models.common_models import MLModel, MLOutputModel, FeatureModel

logger = logging.getLogger(__name__)

"""
SVM节点
"""

@ui(
    feature={"input_type": "None"},
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
    kernel={"input_type": "select", "options": ["linear", "poly", "rbf", "sigmoid"],"allow_link": False},
    degree={"input_type": "number_field","allow_link": False},
    gamma={"input_type": "select", "options": ["scale", "auto"],"allow_link": False},
    C={"input_type": "slider", "min": 0.1, "max": 10,"allow_link": False},
    epsilon={"input_type": "slider", "min": 0.01, "max": 1,"allow_link": False},
    shrinking={"input_type": "checkbox","allow_link": False},
    max_iter={"input_type": "number_field","allow_link": False},
)
class SVMInputModel(BaseModel):
    """Input model for SVM node"""
    feature: FeatureModel = Field(default="", title="特征工程")
    start_date: str = Field(default="20250101", title="训练开始时间")
    end_date: str = Field(default="20250301", title="训练结束时间")
    kernel: str = Field(default="rbf", title="核函数类型")
    degree: int = Field(default=3, title="多项式核函数次数")
    gamma: str = Field(default="scale", title="核函数系数")
    coef0: float = Field(default=0.0, title="核函数独立项")
    C: float = Field(default=1.0, title="正则化参数")
    epsilon: float = Field(default=0.1, title="SVR模型中的Epsilon值")
    shrinking: bool = Field(default=True, title="是否使用收缩启发式")
    cache_size: int = Field(default=200, title="核缓存大小(MB)")
    max_iter: int = Field(default=1000, title="最大迭代次数")
    tol: float = Field(default=0.001, title="停止准则容差")

class SVMModelWrapper:
    """SVM模型包装类，封装模型和标准化器"""
    def __init__(self, model, scaler_X, scaler_y):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        predictions_scaled = self.model.predict(X_scaled)
        return self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    def save(self, model_path):
        """保存模型和scaler"""
        import pickle
        save_dict = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }
        with open(model_path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, model_path):
        """加载模型和scaler"""
        import pickle
        with open(model_path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # 创建包装器
        wrapper = cls(save_dict['model'], save_dict['scaler_X'], save_dict['scaler_y'])
        return wrapper

@work_node(
    name="SVM模型", 
    group="03-机器学习", 
    type="general", 
    box_color="red"
)
class SVMControl(BaseWorkNode):
    """Node for SVM model training and prediction"""

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return SVMInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return MLOutputModel

    def run(self, input: SVMInputModel) -> MLOutputModel:
        """
        Train SVM model and make predictions
        
        Args:
            input: SVMInputModel containing data and model parameters
            
        Returns:
            MLOutputModel with the path to the saved model
        """
        macro_factor = MacroFactor()

        # 批量获取因子值
        if input.feature.features:
            factors = input.feature.features.split("\n")
            factor_values = pd.DataFrame()
            factor_values = macro_factor.create_factor_from_formula_pro(
                factor_logger=logger,
                formulas=factors,
                start_date=input.start_date,
                end_date=input.end_date
            )
        else:
            raise ValueError("因子不能为空")
        
        if input.feature.label:
            label = macro_factor.create_factor_from_formula(
                factor_logger=logger,
                formula=input.feature.label,
                start_date=input.start_date,
                end_date=input.end_date
            )
            factor_values["label"] = label
            print("计算完成")
            print(factor_values)
        else:
            raise ValueError("标签不能为空")

        # 开始训练
        print("开始训练")
        
        # 初始化标准化器
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # 构造特征 X 和标签 y
        feature_cols = [col for col in factor_values.columns if col != 'label']
        X = factor_values[feature_cols]
        y = factor_values["label"]
        
        # 数据标准化
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        # 初始化并训练模型
        base_model = SVR(
            kernel=input.kernel,
            degree=input.degree,
            gamma=input.gamma,
            coef0=input.coef0,
            C=input.C,
            epsilon=input.epsilon,
            shrinking=input.shrinking,
            cache_size=input.cache_size,
            max_iter=input.max_iter,
            tol=input.tol
        )

        model = MultiOutputRegressor(base_model)
        model.fit(X_scaled, y_scaled)
        print("训练结束")

        # 设置模型保存路径 - 保存到models文件夹
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        short_uuid = str(uuid.uuid4())[:8]  # 生成8位短UUID
        model_path = model_dir / f"svm_model_{datetime.now().strftime('%Y%m%d%H%M')}_{short_uuid}.pkl"
        
        # 创建包装对象并保存
        svm_wrapper = SVMModelWrapper(model, scaler_X, scaler_y)
        svm_wrapper.save(model_path)

        ml_model = MLModel(model_path=str(model_path), model_type="svm")
        return MLOutputModel(model=ml_model)

if __name__ == "__main__":
    node = SVMControl() 