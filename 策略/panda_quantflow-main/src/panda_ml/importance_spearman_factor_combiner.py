import pandas as pd
import logging

import numpy as np
from scipy.stats import spearmanr

class ImportanceSpearmanFactorCombiner:
    """
    基于模型重要性指标与 Spearman 相关性的加权因子合成器

    核心思想：
      - 通过模型给出的三种重要性指标（weight/gain/cover）与后续收益的 Spearman 相关性，
        计算每个因子的综合权重 w_j。
      - 用 w_j 对各因子截面值 x_ij 加权，得到复合因子 F_i。
    """
    def __init__(self, metrics=['weight', 'gain', 'cover']):
        # 支持的模型重要性指标列表，默认 ["weight","gain","cover"]
        self.metrics = metrics
        # 用于保存 fit 阶段计算出的 Spearman 相关系数，DataFrame，
        # 行为因子名，列为指标名
        self.rho_df = None

    def fit(self,
            importance_history: pd.DataFrame,
            returns: pd.Series):
        """
        计算每个因子重要性指标与收益的 Spearman 排序相关系数。

        参数：
            importance_history: pd.DataFrame，
                index: 时间周期（如日期），
                columns: MultiIndex 第一层为指标名（"weight","gain","cover"），第二层为因子名
                示例：
                    ('weight','factor1'), ('gain','factor1'), ('cover','factor1'), ...
            returns: pd.Series，
                index 与 importance_history 相同，
                值为对应周期的收益率（或其他衡量因子效果的标量指标）

        返回：
            self (可链式调用)，并将 Spearman 相关系数存入 self.rho_df。
        """
        # 检查输入格式
        if not isinstance(importance_history.columns, pd.MultiIndex):
            raise ValueError("importance_history.columns 必须是 MultiIndex：第一层为指标名，第二层为因子名")
        # 提取去重后的因子列表，保持出现顺序
        _, factors = zip(*importance_history.columns)
        factors = list(dict.fromkeys(factors))

        # 初始化字典，用于累积各因子各指标的 Spearman 相关系数
        rho_dict = {m: [] for m in self.metrics}

        # 逐因子、逐指标计算相关系数
        for factor in factors:
            for metric in self.metrics:
                # 获取该因子在所有周期下该指标的时间序列
                series_m = importance_history[(metric, factor)]
                # 计算 Spearman 相关系数，忽略 NaN
                rho, _ = spearmanr(series_m, returns, nan_policy='omit')
                rho_dict[metric].append(rho)

        # 构建 DataFrame: 行 index=factors, 列 metrics
        self.rho_df = pd.DataFrame(rho_dict, index=factors)
        return self

    def compute_weights(self,
                        current_importance: pd.Series) -> pd.Series:
        """
        使用计算好的 Spearman 相关系数，结合当前期的模型重要性指标，
        计算每个因子的综合权重 w_j。

        参数：
            current_importance: pd.Series，
                index 为 MultiIndex (指标名, 因子名)，对应当前期 importance_history 中相同列名的数据
                例：current_importance[('weight','factor1')] 即当前期 weight 指标

        返回：
            pd.Series，index 为因子名，值为对应的综合权重 w_j。
        """
        # 确保已经 fit 过
        if self.rho_df is None:
            raise ValueError("请先调用 fit() 计算 Spearman 相关系数")

        factors = self.rho_df.index.tolist()
        w_dict = {}
        # 逐因子计算综合权重
        for factor in factors:
            # 取相关系数
            rho_vals = self.rho_df.loc[factor]
            # 取当前期该因子的三种重要性指标值
            weight_j = current_importance[('weight', factor)]
            gain_j = current_importance[('gain', factor)]
            cover_j = current_importance[('cover', factor)]

            # 分子：各指标值乘以对应的 Spearman 相关系数后求和
            numerator = (
                rho_vals['weight'] * weight_j +
                rho_vals['gain']   * gain_j   +
                rho_vals['cover']  * cover_j
            )
            # 分母：相关系数之和
            denominator = rho_vals.sum()

            # 防止除零
            if denominator == 0 or np.isnan(denominator):
                # 若分母为0，可考虑设为0或等权
                w_j = 0.0
            else:
                w_j = numerator / denominator
            w_dict[factor] = w_j

        # 返回因子权重 Series
        return pd.Series(w_dict, name='composite_weight')

    def transform(self,
                  x_df: pd.DataFrame,
                  weights: pd.Series) -> pd.Series:
        """
        对跨标的因子暴露进行加权，合成复合因子值 F_i。

        参数：
            x_df: pd.DataFrame，
                index 为标的（如股票代码），columns 为因子名，
                值为该标的上各因子暴露 x_ij。
            weights: pd.Series，
                index 为因子名，值为 compute_weights 输出的综合权重 w_j。

        返回：
            pd.Series，index 同 x_df.index，代表复合因子值 F_i = sum_j w_j * x_ij。
        """
        # 矩阵乘法：逐行（标的）与列（因子）点乘
        F = x_df.dot(weights)
        F.name = 'composite_factor'
        return F

    def fit_transform(self,
                      importance_history: pd.DataFrame,
                      returns: pd.Series,
                      current_importance: pd.Series,
                      x_df: pd.DataFrame) -> pd.Series:
        """
        一步完成 fit 和 transform，直接输出复合因子。
        """
        self.fit(importance_history, returns)
        weights = self.compute_weights(current_importance)
        return self.transform(x_df, weights)
