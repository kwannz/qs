from typing import Optional, Type, Union, Annotated, Any
from common.utils.index_calculate import get_factors,get_factors_mutil
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

logger = logging.getLogger(__name__)

"""
机器学习因子构建节点
"""

@ui(
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
)

class MTLFactorBuildInputModel(BaseModel):
    model: MLModel = Field(title="机器学习模型",)
    feature1: FeatureModel | None = Field(title="特征工程1",default=None)
    feature2: FeatureModel | None = Field(title="特征工程2",default=None)
    feature3: FeatureModel | None = Field(title="特征工程3",default=None)
    feature4: FeatureModel | None = Field(title="特征工程4",default=None)
    feature5: FeatureModel | None = Field(title="特征工程5",default=None)
    start_date: str = Field(default="20250101",title="因子回测开始时间",)
    end_date: str = Field(default="20250301",title="因子回测结束时间",)

class MTLFactorBuildOutputModel(BaseModel):
    factor: Any = Field(default=None)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('factor')
    def validate_factor(cls, v):
        if not isinstance(v, DataFrame):
            raise ValueError('factor must be a pandas DataFrame')
        return v

@work_node(name="因子构建(机器学习-单模型多特征)", group="06-线下课专属", type="general", box_color="blue")
class MTLFactorBuildControl(BaseWorkNode):
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return MTLFactorBuildInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return MTLFactorBuildOutputModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:

        factors = [f for f in
                   [input.feature1, input.feature2, input.feature3, input.feature4, input.feature5] if
                   f and f.features and f.features.strip()]
        factor_values = get_factors_mutil(feature_list=factors, start_date=input.start_date, end_date=input.end_date)

        # 重命名最后一列
        # factor_values.columns = list(factor_values.columns[:-1]) + ['label']

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
        else:
            raise ValueError(f"不支持的模型类型: {input.model.model_type}")

        print("开始预测计算")
        # 准备数据
        df = factor_values.copy()
        
        # 获取特征列（除了label列之外的所有列）
        # feature_cols = [col for col in df.columns if col != 'label']
        feature_cols = [col for col in df.columns if col.startswith("factor")]
        # label_cols = [col for col in factor_values.columns if col.startswith("label")]
        # 创建预测结果列
        # df['value'] = np.nan
        
        # 使用前一天的数据预测当天的值
        X_pred = df[feature_cols].shift(1).iloc[1:]  # 获取前一天特征数据并去掉第一行
        
        # 批量预测
        predictions = model.predict(X_pred)

        for i in range(predictions.shape[1]):
            value_col = f"value{i + 1}"
            df[value_col] = np.nan  # 创建列
            df.iloc[1:, df.columns.get_loc(value_col)] = predictions[:, i]
        # 将预测结果填入对应日期
        # df.iloc[1:, df.columns.get_loc('value')] = predictions
        print("预测计算完成")
        # 去除所有与计算相关列含NaN的行
        relevant_cols = feature_cols + [f"value{i + 1}" for i in range(predictions.shape[1])]
        df_clean = df.dropna(subset=relevant_cols)
        weights = np.stack([df_clean[f"value{i + 1}"] for i in range(predictions.shape[1])], axis=1)
        weights_sum = weights.sum(axis=1).reshape(-1, 1)
        weights_normalized = weights / weights_sum  # shape: [N, M]

        factors = np.stack([df_clean[col] for col in feature_cols], axis=1)  # shape: [N, M]

        df_clean["value"] = (factors * weights_normalized).sum(axis=1)  # 加权求和
        # 将多级索引转换为列并只保留需要的列
        # 保留日期、symbol 和所有 valueN 列
        # value_cols = [f"value{i + 1}" for i in range(predictions.shape[1])]
        # df = df.reset_index()[["date", "symbol"] + value_cols]
        # df[feature_cols]
        df_clean = df_clean.reset_index()[["date", "symbol","value"]]
        return MTLFactorBuildOutputModel(factor=df_clean)

# if __name__ == "__main__":
#     node = XgboostControl()
#     factors = "CLOSE\nLOW"
#     input = InputModel(feature=FeatureModel(factors=factors, label="RETURNS(CLOSE,1)"),start_date="20250101",end_date="20250301")
#     model = node.run(input)
#     node = FactorBuildControl()
#     factors = "CLOSE\nOPEN\nHIGH\nLOW"
#     input = InputModel(model=model)
#     print(node.run(input))
