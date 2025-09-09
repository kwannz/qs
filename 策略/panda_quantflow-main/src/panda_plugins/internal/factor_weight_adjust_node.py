# -*- coding: utf-8 -*-
"""
File: factor_weight_adjust_node.py
Author: Assistant
Date: 2025/6/24
Description: 因子权重调整节点 - 对单个DataFrame中的因子值进行权重调整
"""
from typing import Optional, Type
import logging

from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field, field_validator
import pandas as pd

logger = logging.getLogger(__name__)

@ui(
    df_factor={"input_type": "None"},
    weight={"input_type": "number"}
)
class FactorWeightAdjustInputModel(BaseModel):
    """
    因子权重调整节点输入模型
    """
    df_factor: object = Field(title="因子值", description="包含因子值的DataFrame，需要有date、symbol和因子列")
    weight: float = Field(default=1.0, title="权重", description="调整权重，可以是正数或负数", ge=-10.0, le=10.0)
    
    model_config = {"arbitrary_types_allowed": True}

    @field_validator('df_factor')
    def validate_df_factor(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError('df_factor must be a pandas DataFrame')
        
        # 检查必需的列
        required_columns = ['date', 'symbol']
        for col in required_columns:
            if col not in v.columns:
                raise ValueError(f'df_factor must contain column: {col}')
        
        # 检查是否有因子列（除了date和symbol之外的列）
        factor_cols = [col for col in v.columns if col not in ['date', 'symbol']]
        if len(factor_cols) == 0:
            raise ValueError('df_factor must contain at least one factor column besides date and symbol')
        
        return v

class FactorWeightAdjustOutputModel(BaseModel):
    """
    因子权重调整节点输出模型
    """
    df_factor: object = Field(title="因子值", description="权重调整后的因子DataFrame")
    
    model_config = {"arbitrary_types_allowed": True}

    @field_validator('df_factor')
    def validate_df_factor(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError('df_factor must be a pandas DataFrame')
        return v

@work_node(name="因子权重调整（归一化）", group="04-因子相关", type="general", box_color="blue")
class FactorWeightAdjustControl(BaseWorkNode):
    """
    因子权重调整节点
    
    功能：
    - 对输入的DataFrame中的因子列进行权重调整
    - 保持date和symbol列不变
    - 所有因子列都会乘以指定的权重
    """

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FactorWeightAdjustInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FactorWeightAdjustOutputModel

    def run(self, input: BaseModel) -> BaseModel:
        """
        执行因子权重调整
        
        Args:
            input: 包含DataFrame和权重的输入模型
            
        Returns:
            调整后的DataFrame
        """
        try:
            # 复制原始DataFrame避免修改原数据
            df_result = input.df_factor.copy()
            
            # 获取因子列（排除date和symbol）
            factor_columns = [col for col in df_result.columns if col not in ['date', 'symbol']]
            
            logger.info(f"对因子列进行归一化和权重调整: {factor_columns}, 权重: {input.weight}")
            
            # 筛选数值类型的因子列
            numeric_factor_cols = [col for col in factor_columns 
                                 if pd.api.types.is_numeric_dtype(df_result[col])]
            non_numeric_cols = [col for col in factor_columns 
                              if not pd.api.types.is_numeric_dtype(df_result[col])]
            
            if non_numeric_cols:
                logger.warning(f"以下列不是数值类型，跳过归一化和权重调整: {non_numeric_cols}")
            
            if numeric_factor_cols:
                # 记录原始统计信息
                original_stats = df_result[numeric_factor_cols].agg(['mean', 'std'])
                
                # 向量化归一化：一次性对所有数值因子列进行Z-score标准化
                factor_data = df_result[numeric_factor_cols]
                std_values = factor_data.std()
                
                # 只对标准差大于0的列进行归一化
                valid_cols = std_values[std_values > 0].index.tolist()
                zero_std_cols = std_values[std_values == 0].index.tolist()
                
                if zero_std_cols:
                    logger.warning(f"以下列标准差为0，跳过归一化: {zero_std_cols}")
                
                if valid_cols:
                    # 向量化Z-score标准化和权重调整
                    df_result[valid_cols] = ((factor_data[valid_cols] - factor_data[valid_cols].mean()) / 
                                           factor_data[valid_cols].std()) * input.weight
                    
                    logger.info(f"向量化归一化和权重调整完成，处理列: {valid_cols}")
                    for col in valid_cols:
                        logger.info(f"列 {col} 原始统计: mean={original_stats.loc['mean', col]:.6f}, "
                                  f"std={original_stats.loc['std', col]:.6f}")
                
                # 对标准差为0的列只应用权重（不归一化）
                if zero_std_cols:
                    df_result[zero_std_cols] = df_result[zero_std_cols] * input.weight
            
            logger.info(f"归一化和权重调整完成，处理了 {len(factor_columns)} 个因子列")
            
            return FactorWeightAdjustOutputModel(df_factor=df_result)
            
        except Exception as e:
            logger.error(f"因子权重调整失败: {str(e)}")
            raise ValueError(f"因子权重调整失败: {str(e)}")

if __name__ == "__main__":
    # 测试代码
    node = FactorWeightAdjustControl()
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'date': ['20240101', '20240101', '20240102', '20240102'],
        'symbol': ['000001.SZ', '000002.SZ', '000001.SZ', '000002.SZ'],
        'factor1': [0.1, 0.2, 0.15, 0.25],
        'factor2': [-0.05, 0.08, -0.02, 0.12]
    })
    
    print("原始数据:")
    print(test_data)
    print(f"\n原始数据统计:")
    print(f"factor1: mean={test_data['factor1'].mean():.6f}, std={test_data['factor1'].std():.6f}")
    print(f"factor2: mean={test_data['factor2'].mean():.6f}, std={test_data['factor2'].std():.6f}")
    
    # 测试归一化和权重调整
    input_model = FactorWeightAdjustInputModel(df_factor=test_data, weight=2.0)
    result = node.run(input_model)
    
    print("\n归一化和权重调整后 (权重=2.0):")
    print(result.df_factor)
    print(f"\n处理后数据统计:")
    print(f"factor1: mean={result.df_factor['factor1'].mean():.6f}, std={result.df_factor['factor1'].std():.6f}")
    print(f"factor2: mean={result.df_factor['factor2'].mean():.6f}, std={result.df_factor['factor2'].std():.6f}")