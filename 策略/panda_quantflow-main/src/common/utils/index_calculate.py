from pyexpat import features
import logging

import pandas as pd
import numpy as np
from panda_factor.generate.macro_factor import MacroFactor

from panda_plugins.internal.models.common_models import FeatureModel

# 配置logger
logger = logging.getLogger(__name__)

def compute_cross_section_ic(exposures: pd.Series, returns: pd.Series, method: str = 'spearman') -> float:
    """
    计算单期截面 IC（信息系数）：因子暴露与下一期收益的排名相关/皮尔逊相关。

    参数：
        exposures: pd.Series, index=资产代码, 当期因子暴露 x_ij
        returns:   pd.Series,  index=资产代码, 下期收益 r_i
        method:    'spearman' 或 'pearson'
    返回：
        IC 值（float）
    """
    # 首先删除任一列中的NaN值
    valid_data = pd.DataFrame({'x': exposures, 'y': returns}).dropna()
    
    # 确保有足够的数据点计算相关性
    if len(valid_data) <= 5:
        return np.nan
        
    try:
        if method == 'spearman':
            return valid_data['x'].corr(valid_data['y'], method='spearman')
        else:
            return valid_data['x'].corr(valid_data['y'], method='pearson')
    except Exception as e:
        print(f"计算相关系数时出错: {str(e)}")
        return np.nan

def _compute_daily_ic(group, method='spearman'):
    """单日IC计算的辅助函数"""
    # 如果数据点太少，返回None
    if len(group) <= 5:  # 增加最小样本要求
        return np.nan
        
    try:
        # 确保这两列都存在
        if 'factor1' not in group.columns or 'factor2' not in group.columns:
            print(f"缺少必要的列: {group.columns}")
            return np.nan
            
        # 删除任一列中含NaN的行
        valid_data = group[['factor1', 'factor2']].dropna()
        if len(valid_data) <= 5:
            return np.nan
            
        return compute_cross_section_ic(valid_data['factor1'], valid_data['factor2'], method=method)
    except Exception as e:
        print(f"计算IC时出错: {str(e)}")
        return np.nan

def compute_ic_value(factor: str, start_date: str, end_date: str, method: str = 'spearman') -> pd.Series:
    """
    计算每日截面IC值 - 向量化版本
    
    参数：
        factor: 因子公式
        start_date: 开始日期
        end_date: 结束日期
        method: 'spearman' 或 'pearson'
    返回：
        pd.Series: 以日期为索引的每日IC值序列
    """
    macro_factor = MacroFactor()
    factor_list = [factor,"FUTURE_RETURNS(close,1)"]
    factor_values = macro_factor.create_factor_from_formula_pro(
                factor_logger=logger,
                formulas=factor_list,
                start_date=start_date,
                end_date=end_date
            )
    
    print(f"因子值形状: {factor_values.shape}")
    print(f"因子值列: {factor_values.columns}")
    print(f"因子值缺失率: \n{factor_values.isna().mean()}")
    
    # 使用groupby和apply进行向量化计算
    # 按日期分组，为每组计算IC值
    daily_ic = factor_values.groupby(level=0).apply(
        lambda x: _compute_daily_ic(x, method=method)
    )
    
    # 打印IC值有效率
    valid_ic_count = (~daily_ic.isna()).sum()
    total_days = len(daily_ic)
    print(f"有效IC天数: {valid_ic_count}/{total_days} ({valid_ic_count/total_days:.2%})")
    
    return daily_ic

def get_factors(feature: FeatureModel, start_date: str, end_date: str) -> pd.DataFrame:
    macro_factor = MacroFactor()
     # 批量获取因子值
    if feature.features:
        factors = feature.features.split("\n")
        factor_values = pd.DataFrame()
        factor_values = macro_factor.create_factor_from_formula_pro(
            factor_logger=logger,
            formulas=factors,
            start_date=start_date,
            end_date=end_date
        )
    else:
        raise ValueError("因子不能为空")
        
    if feature.label:
        if feature.label == "IC":                
            # 计算IC
            daily_ic = compute_ic_value(factors[0], start_date, end_date)
            
            # 填充缺失的IC值
            if daily_ic.isna().sum() > 0:
                # 如果所有值都是NaN，使用0填充
                if daily_ic.isna().all():
                    print("警告: 所有IC值都为NaN，使用0填充")
                    daily_ic = pd.Series(0, index=daily_ic.index)
                else:
                    # 用前一个有效值填充NaN
                    print(f"使用前向填充法处理缺失的IC值 ({daily_ic.isna().sum()}个)")
                    daily_ic = daily_ic.fillna(method='ffill')
                    # 对于开头的NaN，用后一个有效值填充
                    daily_ic = daily_ic.fillna(method='bfill')
            
            unique_dates = factor_values.index.get_level_values(0).unique()
            for date in unique_dates:
                if date in daily_ic.index:
                    # 为当天所有股票分配相同的IC值
                    idx = factor_values.index.get_level_values(0) == date
                    factor_values.loc[idx, 'label'] = daily_ic[date]
        else:
            label = macro_factor.create_factor_from_formula(
                factor_logger=logger,
                formula=feature.label,
                start_date=start_date,
                end_date=end_date
            )
            factor_values["label"] = label
        print("计算完成")
        
        # 检查标签列的有效性
        label_valid_count = (~factor_values['label'].isna()).sum()
        total_count = len(factor_values)
        print(f"标签列有效值: {label_valid_count}/{total_count} ({label_valid_count/total_count:.2%})")
    else:
        raise ValueError("标签不能为空")
    return factor_values

def get_factors_mutil(feature_list: list[FeatureModel], start_date: str, end_date: str) -> pd.DataFrame:
    macro_factor = MacroFactor()
    result_df = None

    for i, feature in enumerate(feature_list):
        # 获取因子名称（支持换行多个）
        if feature.features:
            factors = feature.features.split("\n")
            factor_values = macro_factor.create_factor_from_formula_pro(
                factor_logger=logger,
                formulas=factors,
                start_date=start_date,
                end_date=end_date
            )
        else:
            raise ValueError(f"第 {i+1} 个 FeatureModel 的因子不能为空")

        # 获取标签
        if feature.label:
            if feature.label == "IC":
                daily_ic = compute_ic_value(factors[0], start_date, end_date)

                # 缺失值处理
                if daily_ic.isna().sum() > 0:
                    if daily_ic.isna().all():
                        print("警告: 所有IC值都为NaN，使用0填充")
                        daily_ic = pd.Series(0, index=daily_ic.index)
                    else:
                        print(f"使用前向填充法处理缺失的IC值 ({daily_ic.isna().sum()}个)")
                        daily_ic = daily_ic.fillna(method='ffill').fillna(method='bfill')

                # 将IC值分配到对应股票横截面
                unique_dates = factor_values.index.get_level_values(0).unique()
                label_series = pd.Series(index=factor_values.index, dtype=float)
                for date in unique_dates:
                    if date in daily_ic.index:
                        idx = factor_values.index.get_level_values(0) == date
                        label_series.loc[idx] = daily_ic[date]
                factor_values[f'label{i+1}'] = label_series
            else:
                label = macro_factor.create_factor_from_formula(
                    factor_logger=logger,
                    formula=feature.label,
                    start_date=start_date,
                    end_date=end_date
                )
                factor_values[f'label{i+1}'] = label

            label_valid_count = (~factor_values[f'label{i+1}'].isna()).sum()
            total_count = len(factor_values)
            print(f"[第{i+1}组] 标签有效值: {label_valid_count}/{total_count} ({label_valid_count/total_count:.2%})")
        else:
            raise ValueError(f"第 {i+1} 个 FeatureModel 的标签不能为空")

        # 重命名因子列
        factor_col_names = [f"factor{i+1}" for _ in range(factor_values.shape[1] - 1)]
        factor_col_names.append(f"label{i+1}")
        factor_values.columns = factor_col_names

        # 合并结果
        if result_df is None:
            result_df = factor_values
        else:
            result_df = pd.concat([result_df, factor_values], axis=1)

    return result_df

if __name__ == "__main__":
    # 测试get_factors函数
    from panda_plugins.internal.feature_engineering_node import FeatureModel
    
    # 构造测试数据
    feature1 = FeatureModel(
        features="RETURNS(close,5)",  # 使用多个因子进行测试
        label="IC"  # 测试IC标签计算
    )
    feature2 = FeatureModel(
        features="RETURNS(open,5)",  # 使用多个因子进行测试
        label="IC"  # 测试IC标签计算
    )
    start_date = "20230101"
    end_date = "20230601"
    
    # 获取因子数据
    factor_values = get_factors_mutil([feature1,feature2], start_date, end_date)
    print("\nFactor values shape:", factor_values.shape)
    print("\nFactor values head:")
    print(factor_values.head())
    print("\nFactor values describe:")
    print(factor_values.describe())

