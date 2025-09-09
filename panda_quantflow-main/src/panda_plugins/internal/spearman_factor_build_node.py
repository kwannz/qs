from typing import Optional, Type, Union, Annotated, Any
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

logger = logging.getLogger(__name__)

"""
Spearman因子构建节点
"""

@ui(
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
)

class SpearmanFactorBuildInputModel(BaseModel):
    model: MLModel = Field(title="机器学习模型",)
    feature: FeatureModel = Field(title="特征工程",)
    start_date: str = Field(default="20250101",title="因子回测开始时间",)
    end_date: str = Field(default="20250301",title="因子回测结束时间",)

class SpearmanFactorBuildOutputModel(BaseModel):
    factor: Any = Field(default=None)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('factor')
    def validate_factor(cls, v):
        if not isinstance(v, DataFrame):
            raise ValueError('factor must be a pandas DataFrame')
        return v

@work_node(name="Spearman因子构建", group="06-线下课专属", type="general", box_color="blue")
class SpearmanFactorBuildControl(BaseWorkNode):
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return SpearmanFactorBuildInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return SpearmanFactorBuildOutputModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:

        # 批量获取因子值
        macro_factor = MacroFactor()
        factors = input.feature.features.split("\n")
        # factors.append(input.feature.label)

        factor_values = macro_factor.create_factor_from_formula_pro(
            factor_logger=logger,
            formulas=factors,
            start_date=input.start_date,
            end_date=input.end_date
        )
        # 重命名最后一列
        # factor_values.columns = list(factor_values.columns[:-1]) + ['label']

        if input.model.model_type == "xgboost":
            # 加载XGBoost模型
            model = XGBRegressor()
            model.load_model(input.model.model_path)
            print("开始预测计算")
            # 准备数据
            df = factor_values.copy()
            
            # 获取特征列（除了label列之外的所有列）
            feature_cols = [col for col in df.columns if col != 'label']
            
            # 创建预测结果列
            df['value'] = np.nan
            
            # 使用前一天的数据预测当天的值
            # 获取所有前一天的特征数据
            X_pred = df[feature_cols].shift(1)
            # 去掉第一行(NaN)
            X_pred = X_pred.iloc[1:]
            # 批量预测
            predictions = model.predict(X_pred)
            # 将预测结果填入对应日期
            df.iloc[1:, df.columns.get_loc('value')] = predictions
            print("预测计算完成")
            
            # 将多级索引转换为列
            df = df.reset_index()
            # 只保留date、symbol、value三列
            df = df[['date', 'symbol', 'value']]
            return SpearmanFactorBuildOutputModel(factor=df)
        

# if __name__ == "__main__":
#     node = XgboostControl()
#     factors = "CLOSE\nLOW"
#     input = InputModel(feature=FeatureModel(factors=factors, label="RETURNS(CLOSE,1)"),start_date="20250101",end_date="20250301")
#     model = node.run(input)
#     node = FactorBuildControl()
#     factors = "CLOSE\nOPEN\nHIGH\nLOW"
#     input = InputModel(model=model)
#     print(node.run(input))
