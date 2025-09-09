from typing import Optional, Type, Union, Annotated, Any
from common.utils.index_calculate import get_factors
from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field, ConfigDict, field_validator
import pandas as pd
from pandas import DataFrame
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
from pathlib import Path
import pickle
from panda_factor.generate.macro_factor import MacroFactor
from panda_plugins.internal.models.common_models import FeatureModel, MLModel
import lightgbm as lgb
from joblib import load, dump
import logging

from panda_plugins.internal.mtl_nn_node import MTLNNModelWrapper
from panda_plugins.internal.svm_node import SVMModelWrapper
from panda_plugins.internal.ml_lstm_node import LSTMModelWrapper
from panda_plugins.internal.ml_gru_node import GRUModelWrapper

logger = logging.getLogger(__name__)

"""
机器学习因子构建节点
"""

@ui(
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
)
class MLFactorBuildInputModel(BaseModel):
    model: MLModel = Field(title="机器学习模型",)
    feature: FeatureModel = Field(title="特征工程",)
    start_date: str = Field(default="20250101",title="开始时间",)
    end_date: str = Field(default="20250301",title="结束时间",)

class MLFactorBuildOutputModel(BaseModel):
    factor: Any = Field(default=None,title="因子值")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('factor')
    def validate_factor(cls, v):
        if not isinstance(v, DataFrame):
            raise ValueError('factor must be a pandas DataFrame')
        return v

@work_node(name="因子构建(机器学习)", group="04-因子相关", type="general", box_color="blue")
class MLFactorBuildControl(BaseWorkNode):
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return MLFactorBuildInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return MLFactorBuildOutputModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:

        factor_values = get_factors(input.feature,input.start_date, input.end_date)
        # 重命名最后一列
        # factor_values.columns = list(factor_values.columns[:-1]) + ['label']
        print(input.model.model_type)

        # 根据模型类型加载对应的模型
        if input.model.model_type == "xgboost":
            model = XGBRegressor()
            model.load_model(input.model.model_path)
        elif input.model.model_type == "lightgbm":
            model = lgb.Booster(model_file=input.model.model_path)
        elif input.model.model_type == "randomforest":
            model = load(input.model.model_path)
        elif input.model.model_type == "svm":
            model = SVMModelWrapper.load(input.model.model_path)
        elif input.model.model_type == "mtl_nn":
            model = MTLNNModelWrapper.load(input.model.model_path)
        elif input.model.model_type == "lstm":
            model = LSTMModelWrapper.load(input.model.model_path)
        elif input.model.model_type == "gru":
            model = GRUModelWrapper.load(input.model.model_path)
        else:
            raise ValueError(f"不支持的模型类型: {input.model.model_type}")

        print(model)
        print("开始预测计算")
        # 准备数据
        df = factor_values.copy()
        
        # 获取特征列（除了label列之外的所有列）
        feature_cols = [col for col in df.columns if col != 'label']
        
        # 创建预测结果列
        df['value'] = np.nan
        
        # 使用前一天的数据预测当天的值
        X_pred = df[feature_cols].shift(1).iloc[1:]  # 获取前一天特征数据并去掉第一行
        
        # 批量预测
        predictions = model.predict(X_pred)
        # 将预测结果填入对应日期
        df.iloc[1:, df.columns.get_loc('value')] = predictions
        print("预测计算完成")
        
        # 将多级索引转换为列并只保留需要的列
        df = df.reset_index()[['date', 'symbol', 'value']]
        return MLFactorBuildOutputModel(factor=df)

if __name__ == "__main__":
    pass
    # node = XgboostControl()
    # factors = "CLOSE\nLOW"
    # input = InputModel(feature=FeatureModel(factors=factors, label="RETURNS(CLOSE,1)"),start_date="20250101",end_date="20250301")
    # model = node.run(input)
    # node = FactorBuildControl()
    # factors = "CLOSE\nOPEN\nHIGH\nLOW"
    # input = InputModel(model=model)
    # print(node.run(input))
