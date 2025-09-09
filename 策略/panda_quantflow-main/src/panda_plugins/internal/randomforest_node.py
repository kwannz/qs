from typing import Optional, Type, Union
import logging

from datetime import datetime
from pathlib import Path
import uuid
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pydantic import BaseModel, Field
from panda_plugins.base.base_work_node import BaseWorkNode
from panda_plugins.base.work_node_registery import work_node
from panda_plugins.base import ui
from panda_factor.generate.macro_factor import MacroFactor
from panda_plugins.internal.models.common_models import FeatureModel, MLModel, MLOutputModel

logger = logging.getLogger(__name__)

"""
随机森林节点
"""

@ui(
    feature={"input_type": "None"},
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
    n_estimators={"input_type": "number_field","allow_link": False},
    max_depth={"input_type": "number_field","allow_link": False},
    min_samples_split={"input_type": "number_field","allow_link": False},
    min_samples_leaf={"input_type": "number_field","allow_link": False},
    max_features={"input_type": "select", "options": ["auto", "sqrt", "log2"],"allow_link": False},
    bootstrap={"input_type": "checkbox","allow_link": False},
    oob_score={"input_type": "checkbox","allow_link": False},
)
class RandomForestInputModel(BaseModel):
    feature: FeatureModel = Field(default="", title="特征工程",)
    start_date: str = Field(default="20250101", title="训练开始时间",)
    end_date: str = Field(default="20250301", title="训练结束时间",)
    n_estimators: int = Field(default=100, title="决策树数量", description="越大越容易过拟合")
    max_depth: Optional[int] = Field(default=None, title="最大深度", description="越大越容易过拟合")
    min_samples_split: int = Field(default=2, title="最小分裂样本数", description="越大越容易欠拟合")
    min_samples_leaf: int = Field(default=1, title="叶节点最小样本数", description="越大越容易欠拟合")
    max_features: str = Field(default="sqrt", title="最大特征数", description="特征采样方式")
    bootstrap: bool = Field(default=True, title="自助采样", description="是否使用自助采样")
    oob_score: bool = Field(default=False, title="OOB评分", description="是否计算袋外评分")
    random_state: Optional[int] = Field(default=42, title="随机种子", description="设置随机数种子以复现结果")
    n_jobs: int = Field(default=-1, title="并行任务数", description="使用的CPU核数，-1表示全部")

@work_node(
    name="随机森林模型",
    group="03-机器学习",
    type="general",
    box_color="red"
)
class RandomForestControl(BaseWorkNode):
    """随机森林模型训练节点"""

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return RandomForestInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return MLOutputModel

    def run(self, input: RandomForestInputModel) -> MLOutputModel:
        """
        训练随机森林模型并返回模型信息
        
        Args:
            input: RandomForestInputModel模型，包含特征数据和模型参数
            
        Returns:
            MLOutputModel: 包含模型路径和类型的信息
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
        model = RandomForestRegressor(
            n_estimators=input.n_estimators,
            max_depth=input.max_depth,
            min_samples_split=input.min_samples_split,
            min_samples_leaf=input.min_samples_leaf,
            max_features=input.max_features,
            bootstrap=input.bootstrap,
            oob_score=input.oob_score,
            random_state=input.random_state,
            n_jobs=input.n_jobs
        )

        # 1) 读数据
        df = factor_values

        # 2) 构造特征 X 和标签 y
        feature_cols = [col for col in factor_values.columns if col != 'label']
        X = df[feature_cols]
        y = df["label"]
        
        # 训练模型
        model.fit(X, y)
        print("训练结束")

        # 输出特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        })
        print("特征重要性:")
        print(feature_importance.sort_values('importance', ascending=False))

        # 设置模型保存路径 - 保存到models目录
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        short_uuid = str(uuid.uuid4())[:8]  # 生成8位短UUID
        model_path = model_dir / f"randomforest_model_{datetime.now().strftime('%Y%m%d%H%M')}_{short_uuid}.pkl"
        
        # 保存模型
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # 返回模型信息
        ml_model = MLModel(model_path=str(model_path), model_type="randomforest")
        return MLOutputModel(model=ml_model)

if __name__ == "__main__":
    node = RandomForestControl() 