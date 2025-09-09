# -*- coding: utf-8 -*-
"""
File: factor_to_group_node.py
Author: Bayoro
Date: 2025/6/30
Description: 单因子集合到因子组 - 将多个因子DataFrame合并到一个DataFrame中
"""

from typing import Optional, Type
import logging
from panda_plugins.base import BaseWorkNode, work_node
from panda_plugins.base.ui_control import ui
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)

@ui(
    factor1={"input_type": "None"},
    factor2={"input_type": "None"},
    factor3={"input_type": "None"},
    factor4={"input_type": "None"},
    factor5={"input_type": "None"},
)
class FactorToGroupInputModel(BaseModel):
    """
    因子转换器输入模型
    """
    factor1: object | None = Field(title="因子/因子组1", description="第一个因子DataFrame",default=None)
    factor2: object | None = Field(title="因子/因子组2", description="第二个因子DataFrame",default=None)
    factor3: object | None = Field(title="因子/因子组3", description="第三个因子DataFrame",default=None)
    factor4: object | None = Field(title="因子/因子组4", description="第四个因子DataFrame",default=None)
    factor5: object | None = Field(title="因子/因子组5", description="第五个因子DataFrame",default=None)

    model_config = {"arbitrary_types_allowed": True}

class FactorToGroupOutputModel(BaseModel):
    """
    因子转换器输出模型
    """
    merged_factors: object = Field(title="因子组", description="合并后的DataFrame")
    
    model_config = {"arbitrary_types_allowed": True}

    @field_validator('merged_factors')
    def validate_merged_factors(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError('merged_factors must be a pandas DataFrame')
        return v


@work_node(name="因子集合器", group="04-因子相关", type="general", box_color="green")
class FactorToGroupControl(BaseWorkNode):
    """
    单因子集合器节点
    
    功能：
    - 将多个因子DataFrame合并到一个DataFrame中
    """

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FactorToGroupInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FactorToGroupOutputModel

    def run(self, input: BaseModel) -> BaseModel:
        """
        执行多因子合并并计算平均因子值
        
        Args:
            input: 包含5个因子/因子组的DataFrame的输入模型
            
        Returns:
            包含全部因子值的DataFrame
        """
        try:
            # 获取所有非空的因子 DataFrame
            factors = [f for f in [input.factor1, input.factor2, input.factor3, input.factor4, input.factor5] if f is not None]

            if not factors:
                raise ValueError("至少需要一个因子或因子组 DataFrame")
            
            
            self.log_info(f"开始因子合并，共{len(factors)}个因子/因子组...")
            
            # 记录每个因子的原始信息
            for i, df in enumerate(factors):
                if hasattr(df, 'index') and hasattr(df.index, 'names'):
                    index_info = df.index.names
                else:
                    index_info = "无索引信息"
                self.log_info(f"因子{i+1}: 形状={df.shape}, 列={list(df.columns)}, 索引={index_info}")
            
            # 准备合并的 DataFrame 列表
            processed_factors = []
            factor_count = 1

            for df in factors:
                # 复制 DataFrame 避免修改原数据
                df_copy = df.copy()
                
                # 如果有 MultiIndex，重置索引
                if isinstance(df_copy.index, pd.MultiIndex):
                    df_copy = df_copy.reset_index()
                
                # 确保有 date 和 symbol 列
                if 'date' not in df_copy.columns or 'symbol' not in df_copy.columns:
                    raise ValueError(f"因子{i+1}必须包含 date 和 symbol 列")
                
                # 获取因子列名（除了 date 和 symbol 的列）
                factor_cols = [col for col in df_copy.columns if col not in ['date', 'symbol']]


                if len(factor_cols) < 1:
                    raise ValueError(f"因子{i+1}应该至少包含一个因子列，当前有{len(factor_cols)}个")
                
                # 一个列名相当于一个因子
                for col in factor_cols:
                    
                    # 重命名因子列为标准名称
                    df_copy = df_copy.rename(columns={col: f'factor{factor_count}'})

                    # 确保 date 和 symbol 列为字符串类型以便合并
                    df_copy['date'] = df_copy['date'].astype(str)
                    df_copy['symbol'] = df_copy['symbol'].astype(str)



                    self.log_info(f"处理因子{factor_count}: 重命名'{col}' -> '{f'factor{factor_count}'}', 数据点数={len(df_copy)}")
                    factor_count += 1
                
                processed_factors.append(df_copy)
                
            
            # 从第一个因子开始，逐步合并
            merged_df = processed_factors[0]
            
            for i in range(1, len(processed_factors)):
                before_merge = len(merged_df)
                merged_df = pd.merge(
                    merged_df, 
                    processed_factors[i], 
                    on=['date', 'symbol'], 
                    how='outer'  # 使用外连接保留所有数据
                )
                after_merge = len(merged_df)
                self.log_info(f"合并因子{i+1}: 合并前{before_merge}行, 合并后{after_merge}行")
            
            # 按 date 和 symbol 排序
            merged_df = merged_df.sort_values(['date', 'symbol']).reset_index(drop=True)
            
            # 记录合并后的数据形状
            self.log_info(f"合并完成! 最终形状: {merged_df.shape}")
            
            # 验证结果
            unique_combinations = merged_df[['date', 'symbol']].drop_duplicates().shape[0]
            self.log_info(f"唯一的 (date, symbol) 组合数: {unique_combinations}")
            
            return FactorToGroupOutputModel(merged_factors=merged_df)
            
        except Exception as e:
            logger.error(f"因子合并失败: {str(e)}")
            raise ValueError(f"因子合并失败: {str(e)}")

if __name__ == "__main__":
    # 测试代码
    node = FactorToGroupControl()
    
    # 创建测试数据 - 5个因子DataFrame (使用MultiIndex)
    dates = ['20240101', '20240102', '20240103']
    symbols = ['000001.SZ', '000002.SZ']
    
    # 创建MultiIndex
    index_data = []
    for date in dates:
        for symbol in symbols:
            index_data.append((date, symbol))
    
    # 因子1 - 动量因子
    index1 = pd.MultiIndex.from_tuples(index_data, names=['date', 'symbol'])
    factor1_df = pd.DataFrame({'momentum': np.random.normal(0, 0.1, len(index1))}, index=index1)
    
    # 因子2 - 反转因子 (缺少一些数据点)
    index2 = pd.MultiIndex.from_tuples(index_data[:-1], names=['date', 'symbol'])  # 少一个数据点
    factor2_df = pd.DataFrame({'reversal': np.random.normal(0, 0.08, len(index2))}, index=index2)
    
    # 因子3 - 价值因子
    index3 = pd.MultiIndex.from_tuples(index_data, names=['date', 'symbol'])
    factor3_df = pd.DataFrame({'value_score': np.random.normal(0, 0.12, len(index3))}, index=index3)
    
    # 因子4 - 质量因子 (多一些数据点)
    extra_data = [('20240104', '000001.SZ')]
    index4 = pd.MultiIndex.from_tuples(index_data + extra_data, names=['date', 'symbol'])
    factor4_df = pd.DataFrame({'quality': np.random.normal(0, 0.09, len(index4))}, index=index4)
    
    # 因子5 - 成长因子
    index5 = pd.MultiIndex.from_tuples(index_data, names=['date', 'symbol'])
    factor5_df = pd.DataFrame({'growth1': np.random.normal(0, 0.11, len(index5)),'growth2': np.random.normal(0, 0.15, len(index5))}, index=index5)
    
    

    print("测试数据:")
    print("因子1 形状:", factor1_df.shape, "列:", list(factor1_df.columns))
    print("因子2 形状:", factor2_df.shape, "列:", list(factor2_df.columns))
    print("因子3 形状:", factor3_df.shape, "列:", list(factor3_df.columns))
    print("因子4 形状:", factor4_df.shape, "列:", list(factor4_df.columns))
    print("因子5 形状:", factor5_df.shape, "列:", list(factor5_df.columns))
    
    # 测试合并
    input_model = FactorToGroupInputModel(
        factor1=factor1_df,
        factor2=factor2_df,
        factor3=factor3_df,
        factor4=factor4_df,
        factor5=factor5_df
    )
    
    result = node.run(input_model)
    
    print("\n合并后的结果:")
    print("形状:", result.merged_factors.shape)
    print("列名:", list(result.merged_factors.columns))
    print("\n前几行数据:")
    print(result.merged_factors.head(10))
    