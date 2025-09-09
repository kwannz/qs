from typing import Optional, Type, Union
import logging

logger = logging.getLogger(__name__)
from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field
from pandas import DataFrame
from panda_factor.generate.macro_factor import MacroFactor

"""
普通因子构建节点
"""
@ui(
    start_date={"input_type": "date_picker"},
    end_date={"input_type": "date_picker"},
)
class FactorBuildInputModel(BaseModel):
    start_date: str = Field(default="20250101",title="因子开始时间",)
    end_date: str = Field(default="20250301",title="因子结束时间",)
    formulas: str = Field(default="",title="因子公式",)

class FactorBuildOutputModel(BaseModel):
    factor: DataFrame
    class Config:
        arbitrary_types_allowed = True

@work_node(name="因子构建节点", group="04-因子相关", type="general", box_color="blue")
class FactorBuildControl(BaseWorkNode):
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FactorBuildInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FactorBuildOutputModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:
        macro_factor = MacroFactor()
        factor_values = macro_factor.create_factor_from_formula_pro(
            factor_logger=logger,
            formulas=input.formulas.split("\n"),
            start_date=input.start_date,
            end_date=input.end_date
        )
        return FactorBuildOutputModel(factor=factor_values)

if __name__ == "__main__":
    node = FactorBuildControl()
    formulas = "CLOSE\nLOW"
    input = FactorBuildInputModel(start_date="20250101",end_date="20250301",formulas=formulas)
    res = node.run(input)
    print(res)
