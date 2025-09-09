from typing import Optional, Type, Union
from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

"""
因子公式节点
"""
@ui(
    code={
        "input_type": "text_field",
        "min_lines": 1,
        "max_lines": 10000,
        "allow_link": False,
        "placeholder": "Please enter code",
    }
)
class CodeInputModel(BaseModel):
    code: str = Field(default="",title="策略代码",)

class CodeOutputModel(BaseModel):
    code: str= Field(default="",title="策略代码",)

@work_node(name="Python代码输入", group="01-基础工具", type="code",box_color="green")
class CodeControl(BaseWorkNode):
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return CodeInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return CodeOutputModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:
        return CodeOutputModel(code=input.code)

if __name__ == "__main__":
    node = CodeControl()
    code = "CLOSE\nLOW"
    input = CodeInputModel(code=code)
    res = node.run(input)
    print(res)
