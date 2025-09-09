# -*- coding: utf-8 -*-
"""
File: multi_factor_merge_node.py
Author: Assistant
Date: 2025/6/24
Description: 多因子合并节点 - 将5个因子DataFrame根据symbol和date对齐合并
"""
from typing import Optional, Type

from panda_plugins.base import BaseWorkNode, work_node
from panda_plugins.base.ui_control import ui
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np

@ui(
    factor1={"input_type": "None"},
    factor2={"input_type": "None"},
    factor3={"input_type": "None"},
    factor4={"input_type": "None"},
    factor5={"input_type": "None"},
)
class MultiFactorMergeInputModel(BaseModel):
    """
    多因子合并节点输入模型
    """
    factor1: object | None = Field(title="因子1", description="第一个因子DataFrame",default=None)
    factor2: object | None = Field(title="因子2", description="第二个因子DataFrame",default=None)
    factor3: object | None = Field(title="因子3", description="第三个因子DataFrame",default=None)
    factor4: object | None = Field(title="因子4", description="第四个因子DataFrame",default=None)
    factor5: object | None = Field(title="因子5", description="第五个因子DataFrame",default=None)
    
    model_config = {"arbitrary_types_allowed": True}

    # @field_validator('factor1', 'factor2', 'factor3', 'factor4', 'factor5')
    # def validate_factors(cls, v):
    #     if v is None:
    #         return v
    #     if not isinstance(v, pd.DataFrame):
    #         raise ValueError('All factors must be pandas DataFrames')
        
    #     # 检查必需的索引：date和symbol应该是索引
    #     if not isinstance(v.index, pd.MultiIndex):
    #         raise ValueError('Factor DataFrame must have MultiIndex with date and symbol')
        
    #     # 检查索引名称
    #     index_names = list(v.index.names)
    #     if 'date' not in index_names or 'symbol' not in index_names:
    #         raise ValueError('Factor DataFrame index must contain both date and symbol levels')
        
    #     # 检查是否只有一个因子列
    #     if len(v.columns) != 1:
    #         raise ValueError('Each factor DataFrame should contain exactly one factor column')
        
    #     return v

class MultiFactorMergeOutputModel(BaseModel):
    """
    多因子合并节点输出模型
    """
    merged_factors: object = Field(title="因子值", description="包含平均因子值的合并DataFrame")
    
    model_config = {"arbitrary_types_allowed": True}

    @field_validator('merged_factors')
    def validate_merged_factors(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError('merged_factors must be a pandas DataFrame')
        return v

@work_node(name="多因子合并(5-1)", group="04-因子相关", type="general", box_color="green")
class MultiFactorMergeControl(BaseWorkNode):
    """
    多因子合并节点
    
    功能：
    - 接收5个因子DataFrame
    - 根据date和symbol对齐合并
    - 计算所有因子的平均值作为合成因子
    - 输出列名为: date, symbol, factor_value
    - 使用外连接保留所有数据点，处理缺失值
    """

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return MultiFactorMergeInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return MultiFactorMergeOutputModel

    def run(self, input: BaseModel) -> BaseModel:
        """
        执行多因子合并并计算平均因子值
        
        Args:
            input: 包含5个因子DataFrame的输入模型
            
        Returns:
            包含平均因子值的DataFrame (列: date, symbol, factor_value)
        """
        try:
            # 获取所有非空的因子DataFrame
            factors = [f for f in [input.factor1, input.factor2, input.factor3, input.factor4, input.factor5] if f is not None]
            if not factors:
                raise ValueError("至少需要一个因子DataFrame")
            
            factor_names = [f'factor{i+1}' for i in range(len(factors))]
            
            self.log_info(f"开始多因子合并，共{len(factors)}个因子...")
            
            # 记录每个因子的原始信息
            for i, df in enumerate(factors):
                if hasattr(df, 'index') and hasattr(df.index, 'names'):
                    index_info = df.index.names
                else:
                    index_info = "无索引信息"
                self.log_info(f"因子{i+1}: 形状={df.shape}, 列={list(df.columns)}, 索引={index_info}")
            
            # 准备合并的DataFrame列表
            processed_factors = []
            
            for i, (df, new_name) in enumerate(zip(factors, factor_names)):
                # 复制DataFrame避免修改原数据
                df_copy = df.copy()
                
                # 如果有MultiIndex，重置索引
                if isinstance(df_copy.index, pd.MultiIndex):
                    df_copy = df_copy.reset_index()
                
                # 确保有date和symbol列
                if 'date' not in df_copy.columns or 'symbol' not in df_copy.columns:
                    raise ValueError(f"因子{i+1}必须包含date和symbol列")
                
                # 获取因子列名（除了date和symbol的列）
                factor_cols = [col for col in df_copy.columns if col not in ['date', 'symbol']]
                if len(factor_cols) != 1:
                    raise ValueError(f"因子{i+1}应该只包含一个因子列，当前有{len(factor_cols)}个")
                
                factor_col = factor_cols[0]
                
                # 重命名因子列为标准名称
                df_copy = df_copy.rename(columns={factor_col: new_name})
                
                # 确保date和symbol列为字符串类型以便合并
                df_copy['date'] = df_copy['date'].astype(str)
                df_copy['symbol'] = df_copy['symbol'].astype(str)
                
                # 选择需要的列
                df_copy = df_copy[['date', 'symbol', new_name]]
                
                processed_factors.append(df_copy)
                self.log_info(f"处理因子{i+1}: 重命名'{factor_col}' -> '{new_name}', 数据点数={len(df_copy)}")
            
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
            
            # 按date和symbol排序
            merged_df = merged_df.sort_values(['date', 'symbol']).reset_index(drop=True)
            
            # 统计合并前的缺失值
            missing_stats = {}
            actual_factor_cols = [col for col in merged_df.columns if col.startswith('factor')]
            for col in actual_factor_cols:
                missing_count = int(merged_df[col].isna().sum())
                missing_rate = float(missing_count / len(merged_df) * 100)
                missing_stats[col] = f"{missing_count}个 ({missing_rate:.1f}%)"
            
            self.log_info("合并前缺失值统计:")
            for col, stat in missing_stats.items():
                self.log_info(f"  {col}: {stat}")
            
            # 计算平均因子值 (忽略NaN值)
            merged_df['factor_value'] = merged_df[actual_factor_cols].mean(axis=1, skipna=True)
            
            # 选择最终输出的列
            final_df = merged_df[['date', 'symbol', 'factor_value']].copy()
            
            # 统计最终结果
            total_valid = int(final_df['factor_value'].notna().sum())
            total_invalid = int(final_df['factor_value'].isna().sum())
            
            self.log_info(f"合并完成! 最终形状: {final_df.shape}")
            self.log_info(f"有效因子值: {total_valid}个, 无效因子值: {total_invalid}个")
            
            if total_valid > 0:
                min_val = float(final_df['factor_value'].min())
                max_val = float(final_df['factor_value'].max())
                self.log_info(f"因子值范围: [{min_val:.4f}, {max_val:.4f}]")
            else:
                logger.warning("所有因子值都为空，无法计算范围")
            
            # 验证结果
            unique_combinations = final_df[['date', 'symbol']].drop_duplicates().shape[0]
            self.log_info(f"唯一的(date, symbol)组合数: {unique_combinations}")
            
            return MultiFactorMergeOutputModel(merged_factors=final_df)
            
        except Exception as e:
            logger.error(f"多因子合并失败: {str(e)}")
            raise ValueError(f"多因子合并失败: {str(e)}")

if __name__ == "__main__":
    # 测试代码
    node = MultiFactorMergeControl()
    
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
    factor5_df = pd.DataFrame({'growth': np.random.normal(0, 0.11, len(index5))}, index=index5)
    
    print("测试数据:")
    print("因子1 形状:", factor1_df.shape, "列:", list(factor1_df.columns))
    print("因子2 形状:", factor2_df.shape, "列:", list(factor2_df.columns))
    print("因子3 形状:", factor3_df.shape, "列:", list(factor3_df.columns))
    print("因子4 形状:", factor4_df.shape, "列:", list(factor4_df.columns))
    print("因子5 形状:", factor5_df.shape, "列:", list(factor5_df.columns))
    
    # 测试合并
    input_model = MultiFactorMergeInputModel(
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
    
    print("\n因子值统计:")
    valid_count = result.merged_factors['factor_value'].notna().sum()
    if valid_count > 0:
        print(f"因子值范围: [{result.merged_factors['factor_value'].min():.4f}, {result.merged_factors['factor_value'].max():.4f}]")
        print(f"因子值均值: {result.merged_factors['factor_value'].mean():.4f}")
    print(f"有效因子值: {valid_count}个")
    print(f"缺失因子值: {result.merged_factors['factor_value'].isna().sum()}个")