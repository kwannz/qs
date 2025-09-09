# -*- coding: utf-8 -*-
"""
File: factor_weight_calculation_node.py
Author: peiqi
Date: 2025/5/22
Description: 
"""
from typing import Optional, Type
import logging

from panda_plugins.base import BaseWorkNode, work_node
from pydantic import BaseModel, Field, field_validator
from panda_factor.analysis.factor_analysis_workflow import factor_analysis_workflow
import pandas as pd
import random
import string
import numpy as np
class FactorWeightCalculationInputModel(BaseModel):
    """
    Define the input model for the node.
    Use pydantic to define, which is a library for data validation and parsing.
    Reference: https://pydantic-docs.helpmanual.io

    为工作节点定义输入模型.
    使用 Pydantic 定义, Pydantic 是一个用于数据验证和解析的库.
    参考文档: https://pydantic-docs.helpmanual.io
    """

    df_factor: object = Field(title="因子值")
    factor_weight: str = Field(default="error", examples="[0.22,0.32,0.35]",title="因子权重列表")
    model_config = {"arbitrary_types_allowed": True}

    @field_validator('df_factor')
    def validate_df_factor(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError('df_factor must be a pandas DataFrame')
        return v

class FactorWeightCalculationOutputModel(BaseModel):
    """
    Define the output model for the node.
    Use pydantic to define, which is a library for data validation and parsing.
    Reference: https://pydantic-docs.helpmanual.io

    为工作节点定义输出模型.
    使用 Pydantic 定义, Pydantic 是一个用于数据验证和解析的库.
    参考文档: https://pydantic-docs.helpmanual.io
    """
    df_factor: object = Field(title="因子值")

    model_config = {"arbitrary_types_allowed": True}

    @field_validator('df_factor')
    def validate_df_factor(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError('df_factor must be a pandas DataFrame')
        return v

@work_node(name="因子权重组合", group="06-线下课专属",type="general", box_color="blue")
class FactorWeightCalculationControl(BaseWorkNode):
    """
    Implement a example node, which can add two numbers and return the result.
    实现一个示例节点, 完成一个简单的加法运算, 输入 2 个数值, 输出 2 个数值的和.
    """

    # Return the input model
    # 返回输入模型
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FactorWeightCalculationInputModel

    # Return the output model
    # 返回输出模型
    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FactorWeightCalculationOutputModel

    # Node running logic
    # 节点运行逻辑
    def run(self, input: BaseModel) -> BaseModel:
        weights = [float(w) for w in input.factor_weight.strip("[]").split(",")]
        normalized_weights = np.array(weights) / sum(weights)
        new_factor_id=generate_custom_uuid()
        cols = input.df_factor.columns[2:]  # 获取所有因子名
        if len(cols) != len(normalized_weights):
            raise ValueError("权重数量与因子列数不匹配！")

        input.df_factor[new_factor_id] = (input.df_factor[cols] * normalized_weights).sum(axis=1)

        return FactorWeightCalculationOutputModel(df_factor=input.df_factor[["date","symbol",f"{new_factor_id}"]])

def generate_custom_uuid():
    # First two uppercase letters
    prefix = ''.join(random.choices(string.ascii_uppercase, k=2))

    # One digit
    digit = random.choice(string.digits)

    # One lowercase letter
    lower_char = random.choice(string.ascii_lowercase)

    # Five digits
    suffix = ''.join(random.choices(string.digits, k=5))

    return f"{prefix}{digit}{lower_char}{suffix}"
if __name__ == "__main__":
    node = FactorWeightCalculationControl()
    df = pd.read_csv(
        '/Users/peiqi/code/python/panda_workflow/src/panda_plugins/internal/test_factor2.csv',
        usecols=["date", "symbol", "a","b","c"],  # 只读取需要的列，节省内存
        dtype={"date": str}  # 明确指定date列为字符串类型
    )

    input = FactorWeightCalculationInputModel(df_factor=df, factor_weight="[1,2,2]")
    print(node.run(input))

