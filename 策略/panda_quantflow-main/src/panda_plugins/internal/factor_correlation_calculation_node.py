# -*- coding: utf-8 -*-
"""
File: factor_correlation_calculation_node.py
Author: Bayoro
Date: 2025/7/1
Description: 因子相关性分析 - 对多个因子/因子组进行两两相关性分析
"""

from typing import Optional, Type
import logging
from panda_plugins.base import BaseWorkNode, work_node
from panda_plugins.base.ui_control import ui
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
import warnings

import uuid
from datetime import datetime
from panda_factor.analysis.factor_correlation_workflow import factor_correlation_workflow
logger = logging.getLogger(__name__)

@ui(
    factor={"input_type": "None"},
)
class FactorCorrelationInputModel(BaseModel):
    """
    因子相关性分析输入模型
    """
    factor: object | None = Field(title="因子组", description="因子DataFrame",default=None)

    model_config = {"arbitrary_types_allowed": True}


class FactorCorrelationOutputModel(BaseModel):
    """
    因子相关性分析输出模型
    """
    task_id: str = Field(default="error", title="分析结果")

    

@work_node(name="因子相关性分析", group="04-因子相关", type="general", box_color="purple")
class FactorCorrelationCalculationControl(BaseWorkNode):
    """
    因子相关性分析节点
    
    功能：
    - 对因子/因子组进行合并操作
    - 对合并后的因子组进行两两相关性分析
    """

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FactorCorrelationInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FactorCorrelationOutputModel

    def run(self, input: BaseModel) -> BaseModel:
        df_factor=input.factor
        task_id = factor_correlation_workflow(df_factor)

        return FactorCorrelationOutputModel(task_id=str(task_id))
        


if __name__ == "__main__":
    # 测试代码
    node = FactorCorrelationCalculationControl()
    
    # 创建测试数据 - 5个因子DataFrame (使用MultiIndex)
    dates = ['20240101', '20240102', '20240103']
    symbols = ['000001.SZ', '000002.SZ']
    
    # 创建MultiIndex
    index_data = []
    for date in dates:
        for symbol in symbols:
            index_data.append((date, symbol))
    


    # 因子5 - 成长因子
    index5 = pd.MultiIndex.from_tuples(index_data, names=['date', 'symbol'])
    factor5_df = pd.DataFrame({'growth1': np.random.normal(0, 0.11, len(index5)),
                               'growth2': np.random.normal(0, 0.15, len(index5)),
                               'growth3': np.random.normal(0, 0.2, len(index5))
                               }, 
                               index=index5)
    
    

    print("测试数据:")
    print("因子5 形状:", factor5_df.shape, "列:", list(factor5_df.columns))
    
    # 测试合并
    input_model = FactorCorrelationInputModel(
        factor=factor5_df
    )
    
    result = node.run(input_model)
    

    print(result)