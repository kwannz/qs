from typing import Optional, Type
import logging

from panda_plugins.base import BaseWorkNode, work_node,ui
from pydantic import BaseModel,Field, field_validator
import pandas as pd
from panda_factor.analysis.factor_ic_workflow import factor_ic_workflow

@ui(
    df_factor={"input_type": "None"},

    adjustment_cycle={"input_type": "combobox", "options": ["1", "3", "5", "10", "20", "30"],
                      "placeholder": "请输入调仓周期",
                      "allow_link": False},
    group_number={"input_type": "combobox",
                  "options": ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                              "17", "18", "19", "20"], "placeholder": "请输入因子分组数量", "allow_link": False},
    factor_direction={"input_type": "combobox", "options": ["0", "1"], "placeholder": "请输入因子方向(0:正向，1:负向)",
                      "allow_link": False},
)
class FactorICCalculationInputModel(BaseModel):
    """
    Define the input model for the node.
    Use pydantic to define, which is a library for data validation and parsing.
    Reference: https://pydantic-docs.helpmanual.io

    为工作节点定义输入模型.
    使用 Pydantic 定义, Pydantic 是一个用于数据验证和解析的库.
    参考文档: https://pydantic-docs.helpmanual.io
    """

    df_factor: object = Field(title="因子值")
    adjustment_cycle: str = Field(default="1", title="调仓周期")
    group_number: str = Field(default="5", title="分组数量")
    factor_direction: str = Field(default="0", title="因子方向(0:正向，1:负向)")

    model_config = {"arbitrary_types_allowed": True}

    @field_validator('df_factor')
    def validate_df_factor(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError('df_factor must be a pandas DataFrame')
        return v

class FactorICCalculationOutputModel(BaseModel):
    """
    Define the output model for the node.
    Use pydantic to define, which is a library for data validation and parsing.
    Reference: https://pydantic-docs.helpmanual.io

    为工作节点定义输出模型.
    使用 Pydantic 定义, Pydantic 是一个用于数据验证和解析的库.
    参考文档: https://pydantic-docs.helpmanual.io
    """
    ic_value: str = Field(default="[0.1,0.2,0.3]",title="因子权重列表")

@work_node(name="因子IC计算", group="06-线下课专属",type="general", box_color="blue")
class FactorICCalculationControl(BaseWorkNode):
    """
    Implement a example node, which can add two numbers and return the result.
    实现一个示例节点, 完成一个简单的加法运算, 输入 2 个数值, 输出 2 个数值的和.
    """

    # Return the input model
    # 返回输入模型
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FactorICCalculationInputModel

    # Return the output model
    # 返回输出模型
    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FactorICCalculationOutputModel

    # Node running logic
    # 节点运行逻辑
    def run(self, input: BaseModel) -> BaseModel:
        ic_value=factor_ic_workflow(int(input.df_factor),int(input.adjustment_cycle),int(input.group_number,input.factor_direction))

        return FactorICCalculationOutputModel(ic_value="["+str(ic_value)+"]")

if __name__ == "__main__":
    node = FactorICCalculationControl()
    df = pd.read_csv(
        '/Users/peiqi/code/python/panda_workflow/src/panda_plugins/internal/test_factor.csv',
        usecols=["date", "symbol", "CZ1a14685"],  # 只读取需要的列，节省内存
        dtype={"date": str}  # 明确指定date列为字符串类型
    )
    df["abcdef"]=df["CZ1a14685"]
    input = FactorICCalculationInputModel(df_factor=df)
    print(node.run(input))

