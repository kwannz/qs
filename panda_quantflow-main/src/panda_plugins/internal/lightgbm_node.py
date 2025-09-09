from typing import Optional, Type, List
import uuid
import numpy as np
import pandas as pd
# import lightgbm as lgb
from pydantic import BaseModel, Field
from panda_plugins.base.base_work_node import BaseWorkNode
from panda_plugins.base.work_node_registery import work_node
from datetime import datetime
from panda_plugins.base import ui
from panda_factor.generate.macro_factor import MacroFactor
from pathlib import Path
from panda_plugins.internal.models.common_models import FeatureModel, MLModel, MLOutputModel

# import lightgbm as lgb
from lightgbm import LGBMRegressor
import logging

logger = logging.getLogger(__name__)

"""
LIGHTGBM节点
"""

@ui(
    feature={"input_type": "None"},
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
    n_estimators={"input_type": "number_field","allow_link": False},
    max_depth={"input_type": "number_field","allow_link": False},
    learning_rate={"input_type": "slider", "min": 0, "max": 1,"allow_link": False},
    num_leaves={"input_type": "number_field","allow_link": False},
    subsample={"input_type": "number_field","allow_link": False},
    colsample_bytree={"input_type": "number_field","allow_link": False},
    reg_alpha={"input_type": "number_field","allow_link": False},
    reg_lambda={"input_type": "number_field","allow_link": False},
)

class LightGBMInputModel(BaseModel):
    feature: FeatureModel = Field(default="",title="特征工程",)
    start_date: str = Field(default="20250101",title="训练开始时间",)
    end_date: str = Field(default="20250301",title="训练结束时间",)
    n_estimators: int = Field(default=100, title="决策树数量",description="越大越容易过拟合")
    max_depth: int = Field(default=3, title="最大深度",description="越大越容易过拟合")
    learning_rate: float = Field(default=0.1, title="学习率",description="越小越容易欠拟合")
    num_leaves: int = Field(default=31, title="叶子节点数量",description="越大越容易过拟合")
    subsample: float = Field(default=0.8, title="子样本比例",description="越大越容易过拟合")
    colsample_bytree: float = Field(default=0.8, title="列采样比例",description="越大越容易过拟合")
    reg_alpha: float = Field(default=0, title="L1正则化",description="越大越容易欠拟合")
    reg_lambda: float = Field(default=1, title="L2正则化",description="越大越容易欠拟合")

class LightGBMOutput(BaseModel):
    """Output model for LightGBM node"""
    predictions: object = Field(..., description="Model predictions")
    model: object = Field(..., description="Trained LightGBM model")
    feature_importance: pd.DataFrame = Field(..., description="Feature importance scores")

    class Config:
        arbitrary_types_allowed = True

@work_node(name="LightGBM模型", group="03-机器学习", type="general", box_color="red")
class LightGBMControl(BaseWorkNode):

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return LightGBMInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return MLOutputModel

    def run(self, input: BaseModel) -> MLOutputModel:
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
        model = LGBMRegressor(
            objective="regression",  # 回归任务
            n_estimators=input.n_estimators,
            max_depth=input.max_depth,
            learning_rate=input.learning_rate,
            num_leaves=input.num_leaves,
            subsample=input.subsample,
            colsample_bytree=input.colsample_bytree,
            reg_alpha=input.reg_alpha,
            reg_lambda=input.reg_lambda
        )
        # 1) 读数据
        df = factor_values

        # 2) 构造特征 X 和标签 y
        feature_cols = [col for col in factor_values.columns if col != 'label']
        X = df[feature_cols]
        y = df["label"]
        model.fit(
            X, y,
            eval_set=[(X, y)]            # 监控训练误差
        )
        print("训练结束")

        # 设置模型保存路径 - 保存到tests目录下的models文件夹        
        model_dir = Path(__file__).parent / 'models'        
        model_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在        
        short_uuid = str(uuid.uuid4())[:8]  # 生成8位短UUID        
        model_path = model_dir / f"lightgbm_model_{datetime.now().strftime('%Y%m%d%H%M')}_{short_uuid}.json"
        model.booster_.save_model(str(model_path))

        ml_model = MLModel(model_path=str(model_path), model_type="lightgbm")
        return MLOutputModel(model=ml_model)

if __name__ == "__main__":
    node = LightGBMControl() 