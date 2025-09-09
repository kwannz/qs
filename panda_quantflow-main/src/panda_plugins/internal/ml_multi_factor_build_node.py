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

logger = logging.getLogger(__name__)

"""
机器学习多因子因子构建节点
"""

@ui(
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
)

class MLMultiFactorBuildInputModel(BaseModel):
    model1: MLModel = Field(title="机器学习模型1",)
    features1: FeatureModel = Field(title="特征工程1",)
    model2: MLModel | None = Field(title="机器学习模型2",default=None) 
    features2: FeatureModel | None = Field(title="特征工程2",default=None)
    model3: MLModel | None = Field(title="机器学习模型3",default=None)
    features3: FeatureModel | None = Field(title="特征工程3",default=None)
    model4: MLModel | None = Field(title="机器学习模型4",default=None)
    features4: FeatureModel | None = Field(title="特征工程4",default=None)
    model5: MLModel | None = Field(title="机器学习模型5",default=None)
    features5: FeatureModel | None = Field(title="特征工程5",default=None)
    start_date: str = Field(default="20250101",title="因子回测开始时间",)
    end_date: str = Field(default="20250301",title="因子回测结束时间",)

class MLMultiFactorBuildOutputModel(BaseModel):
    factor: Any = Field(default=None)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('factor')
    def validate_factor(cls, v):
        if not isinstance(v, DataFrame):
            raise ValueError('factor must be a pandas DataFrame')
        return v

@work_node(name="多因子构建(机器学习)", group="06-线下课专属", type="general", box_color="blue")
class MLMultiFactorBuildControl(BaseWorkNode):
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return MLMultiFactorBuildInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return MLMultiFactorBuildOutputModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:

        factors = [f for f in [input.features1, input.features2, input.features3, input.features4, input.features5] if f and f.features and f.features.strip()]
        models = [m for m in [input.model1, input.model2, input.model3, input.model4, input.model5] if m]
        
        res = pd.DataFrame()
        for i in range(len(models)):
            df = get_factors(factors[i], input.start_date, input.end_date)
            ml_model = models[i]
            # 根据模型类型加载对应的模型
            if ml_model.model_type == "xgboost":
                model = XGBRegressor()
                model.load_model(ml_model.model_path)
            elif ml_model.model_type == "lightgbm":
                model = lgb.Booster(model_file=ml_model.model_path)
            elif ml_model.model_type == "randomforest":
                model = load(ml_model.model_path)
            elif ml_model.model_type == "svm":
                model = load(ml_model.model_path)
            elif ml_model.model_type == "mtl_nn":
                model = MTLNNModelWrapper.load(ml_model.model_path)
            else:
                raise ValueError(f"不支持的模型类型: {ml_model.model_type}")

            print(f"{i+1}模型开始预测计算")
            
            # 获取特征列（除了label列之外的所有列）
            feature_cols = [col for col in df.columns if col != 'label']
            
            # 创建权重列和因子列
            df[f'weight{i+1}'] = np.nan
            
            # 使用前一天的数据预测当天的值
            X_pred = df[feature_cols].shift(1).iloc[1:]  # 获取前一天特征数据并去掉第一行
            
            # 批量预测
            predictions = model.predict(X_pred)
            
            # 将预测结果填入对应日期
            df.iloc[1:, df.columns.get_loc(f'weight{i+1}')] = predictions

            # 将factor1列重命名为对应的factor列名
            df = df.rename(columns={'factor1': f'factor{i+1}'})
            
            print(f"{i+1}模型预测计算完成")
            
            # 将多级索引转换为列并只保留需要的列
            df = df.reset_index()[['date', 'symbol', f'weight{i+1}', f'factor{i+1}']]
            
            if res.empty:
                res = df
            else:
                res = pd.merge(res, df, on=['date', 'symbol'], how='outer')
        # 计算加权因子值
        weight_cols = [col for col in res.columns if col.startswith('weight')]
        factor_cols = [col for col in res.columns if col.startswith('factor')]
        
        # 初始化values列为0
        res['values'] = 0
        
        # 计算每个因子与权重的乘积之和
        for w_col, f_col in zip(weight_cols, factor_cols):
            res['values'] += res[w_col] * res[f_col]
        # 只保留需要的列
        res = res[['date', 'symbol', 'values']]
        return MLMultiFactorBuildOutputModel(factor=res)