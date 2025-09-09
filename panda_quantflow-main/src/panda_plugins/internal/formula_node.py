from typing import Optional, Type, Union
from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

"""
因子公式节点
"""
@ui(
    formulas={
        "input_type": "text_field",
        "min_lines": 1,
        "max_lines": 50,
        "width": "300",
        "allow_link": False,
        "placeholder": "Please enter factors",
    }
)
class FormulaInputModel(BaseModel):
    formulas: str = Field(default="",title="公式",)

class FormulaOutputModel(BaseModel):
    formulas: str= Field(default="",title="公式",)

@work_node(name="公式输入", group="02-特征工程", type="general",box_color="green")
class FormulaControl(BaseWorkNode):
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FormulaInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FormulaOutputModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:
        return FormulaOutputModel(formulas=input.formulas)

if __name__ == "__main__":
    node = FormulaControl()
    formulas = "CLOSE\nLOW"
    input = FormulaInputModel(formulas=formulas)
    res = node.run(input)
    print(res)
