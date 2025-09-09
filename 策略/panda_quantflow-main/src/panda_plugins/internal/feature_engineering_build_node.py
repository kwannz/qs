from typing import Optional, Type, Union
import logging

from common.utils.index_calculate import get_factors
from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field
from pandas import DataFrame

from panda_plugins.internal.models.common_models import FeatureModel

"""
特征工程节构建节点
"""

@ui(
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
)
class FeatureInputBuildModel(BaseModel):
    formulas: str = Field(default="",title="特征",)
    label: str = Field(default="",title="标签",)
    start_date: str = Field(default="20250101",title="开始时间",)
    end_date: str = Field(default="20250301",title="结束时间",)

class FeatureOutputBuildModel(BaseModel):
    factor: DataFrame = Field(..., title="特征值")
    feature_model: FeatureModel = Field(default="",title="特征工程",)
    class Config:
        arbitrary_types_allowed = True

@work_node(name="特征工程构建", group="02-特征工程", type="general", box_color="brown")
class FeatureEngineeringBuildNode(BaseWorkNode):

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FeatureInputBuildModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FeatureOutputBuildModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:
        feature_model = FeatureModel(features=input.formulas, label=input.label)
        factor = get_factors(feature_model, input.start_date, input.end_date)
        return FeatureOutputBuildModel(factor=factor, feature_model=feature_model)

if __name__ == "__main__":
    node = FeatureEngineeringBuildNode()
    formulas = "CLOSE\nOPEN\nHIGH\nLOW"
    label = "RETURNS(CLOSE,1)"
    input = FeatureInputBuildModel(formulas=formulas, label=label, start_date="20250101", end_date="20250301")
    print(node.run(input))
