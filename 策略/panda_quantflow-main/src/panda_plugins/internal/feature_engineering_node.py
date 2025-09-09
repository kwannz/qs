from typing import Optional, Type, Union
import logging

from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field

from panda_plugins.internal.models.common_models import FeatureModel

"""
特征工程节点
此示例演示了：
1. 使用UI装饰器控制输入控件类型
2. 为不同输入类型设置适当的默认值
3. 配置输入属性，如最小/最大值、占位符等

支持的输入类型包括：
- text_field: 单行或多行文本输入框
- code_editor: 带有语法高亮的代码编辑器
- password_field: 用于敏感信息的掩码输入框
- number_field: 数值输入框
- slider: 用于在范围内选择值的拖动控件
- switch: 用于布尔值的开关控件
- combobox: 带有可选项的下拉框
- date_picker: 日期选择控件
- time_picker: 时间选择控件
- datetime_picker: 日期和时间选择控件
- file_picker: 文件选择界面
"""

@ui(
    label={
        "input_type": "text_field",
        "placeholder": "Please enter label",
        "allow_link": False,
    }
)
class FeatureInputModel(BaseModel):
    formulas: str = Field(default="",title="特征公式",)
    label: str = Field(default="",title="标签公式",)

class FeatureOutputModel(BaseModel):
    feature_model: FeatureModel = Field(default="",title="特征工程",)

@work_node(name="特征工程构建（旧）", group="02-特征工程", type="general", box_color="brown")
class FeatureEngineeringNode(BaseWorkNode):

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FeatureInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FeatureOutputModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:
        feature_model = FeatureModel(features=input.formulas, label=input.label)
        return FeatureOutputModel(feature_model=feature_model)

if __name__ == "__main__":
    node = FeatureEngineeringNode()
    factors = "CLOSE\nOPEN\nHIGH\nLOW"
    input = FeatureInputModel(factors=factors, label="RETURNS(CLOSE,1)")
    print(node.run(input))
