from datetime import datetime
import logging

logger = logging.getLogger(__name__)
from typing import Optional, Type, Union
import uuid
from common.utils.index_calculate import get_factors
from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field
import pandas as pd
from panda_plugins.internal.feature_engineering_node import FeatureModel
from panda_factor.generate.macro_factor import MacroFactor
from xgboost import XGBRegressor
from pathlib import Path
from panda_plugins.internal.models.common_models import MLModel, MLOutputModel

"""
XGBOOST节点
"""

@ui(
    feature={"input_type": "None"},
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
    n_estimators={"input_type": "number_field","allow_link": False,},
    max_depth={"input_type": "number_field","allow_link": False,},
    learning_rate={"input_type": "slider", "min": 0, "max": 1,"allow_link": False,},
    min_child_weight={"input_type": "number_field","allow_link": False,},
    gamma={"input_type": "number_field","allow_link": False,},
    subsample={"input_type": "number_field","allow_link": False,},
    colsample_bytree={"input_type": "number_field","allow_link": False,},
    reg_alpha={"input_type": "number_field","allow_link": False,},
    reg_lambda={"input_type": "number_field","allow_link": False,},
)

class XgboostInputModel(BaseModel):
    feature: FeatureModel = Field(default="",title="特征工程",)
    start_date: str = Field(default="20250101",title="训练开始时间",)
    end_date: str = Field(default="20250301",title="训练结束时间",)
    n_estimators: int = Field(default=100, title="决策树数量",description="越大越容易过拟合")
    max_depth: int = Field(default=3, title="最大深度",description="越大越容易过拟合")
    learning_rate: float = Field(default=0.1, title="学习率",description="越小越容易欠拟合")
    min_child_weight: int = Field(default=1, title="最小子权重",description="越大越容易欠拟合")
    gamma: float = Field(default=0, title="Gamma",description="越大越容易欠拟合")
    subsample: float = Field(default=1, title="子样本比例",description="越大越容易过拟合")
    colsample_bytree: float = Field(default=1, title="列采样比例",description="越大越容易过拟合")
    reg_alpha: float = Field(default=0, title="L1正则化",description="越大越容易欠拟合")
    reg_lambda: float = Field(default=1, title="L2正则化",description="越大越容易欠拟合")

@work_node(name="Xgboost模型(旧)", group="03-机器学习", type="general", box_color="red")
class XgboostControl(BaseWorkNode):

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return XgboostInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return MLOutputModel

    def run(self, input: XgboostInputModel) -> MLOutputModel:

        factor_values = get_factors(input.feature, input.start_date, input.end_date)

        # 开始训练
        print("开始训练")
        model = XGBRegressor(
            objective="reg:squarederror",  # 回归任务
            n_estimators=input.n_estimators,
            max_depth=input.max_depth,
            learning_rate=input.learning_rate,
            min_child_weight=input.min_child_weight,
            gamma=input.gamma,
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
        
        # 处理NaN值
        # 删除标签为NaN的样本
        valid_idx = ~y.isna()
        if sum(~valid_idx) > 0:
            print(f"警告：删除了 {sum(~valid_idx)} 个含有NaN标签的样本（占比 {sum(~valid_idx)/len(y):.2%}）")
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            
        if len(X) == 0:
            raise ValueError("处理NaN后没有可用的训练数据，请检查label列的计算是否正确")
            
        print(f"训练数据样本数: {len(X)}")
        print(f"特征数量: {X.shape[1]}")
        
        model.fit(
            X, y,
            eval_set=[(X, y)],            # 监控训练误差
            verbose=True
        )
        print("训练结束")

        # 设置模型保存路径 - 保存到tests目录下的models文件夹
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        short_uuid = str(uuid.uuid4())[:8]  # 生成8位短UUID
        model_path = model_dir / f"xgboost_model_{datetime.now().strftime('%Y%m%d%H%M')}_{short_uuid}.json"
        model.save_model(str(model_path))

        ml_model = MLModel(model_path=str(model_path), model_type="xgboost")
        return MLOutputModel(model=ml_model)

if __name__ == "__main__":
    node = XgboostControl()
