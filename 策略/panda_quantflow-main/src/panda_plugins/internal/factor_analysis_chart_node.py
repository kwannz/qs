from typing import Optional, Type, Any
import logging

from tornado.process import task_id

from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field, field_validator
from panda_factor.analysis.factor_analysis_workflow import factor_analysis_workflow
import pandas as pd

@ui(
    task_id={"input_type": "None"},
)
class FactorAnalysisChartInputModel(BaseModel):
    """
    Define the input model for the node.
    Use pydantic to define, which is a library for data validation and parsing.
    Reference: https://pydantic-docs.helpmanual.io

    为工作节点定义输入模型.
    使用 Pydantic 定义, Pydantic 是一个用于数据验证和解析的库.
    参考文档: https://pydantic-docs.helpmanual.io
    """
    task_id: str = Field(default="error", title="分析结果")

class FactorAnalysisChartOutputModel(BaseModel):
    """
    Define the output model for the node.
    Use pydantic to define, which is a library for data validation and parsing.
    Reference: https://pydantic-docs.helpmanual.io

    为工作节点定义输出模型.
    使用 Pydantic 定义, Pydantic 是一个用于数据验证和解析的库.
    参考文档: https://pydantic-docs.helpmanual.io
    """
    task_id: str = Field(default="error", title="图表绘制")

@work_node(name="因子分析结果", group="04-因子相关",type="factor_analysis", box_color="purple")
class FactorAnalysisChartControl(BaseWorkNode):
    """
    Implement a example node, which can add two numbers and return the result.
    实现一个示例节点, 完成一个简单的加法运算, 输入 2 个数值, 输出 2 个数值的和.
    """

    # Return the input model
    # 返回输入模型
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FactorAnalysisChartInputModel

    # Return the output model
    # 返回输出模型
    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FactorAnalysisChartOutputModel

    # Node running logic
    # 节点运行逻辑
    def run(self, input: BaseModel) -> BaseModel:
        print(input.task_id)
        return FactorAnalysisChartOutputModel(task_id=str(input.task_id))

if __name__ == "__main__":
    node = FactorAnalysisChartControl()
    input = FactorAnalysisChartInputModel(task_id="123456")
    node.run(input)
