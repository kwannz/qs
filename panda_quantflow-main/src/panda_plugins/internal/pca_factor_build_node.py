from typing import Optional, Type, Union, Annotated, Any
from common.utils.index_calculate import get_factors
from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field, ConfigDict, field_validator
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
from panda_factor.generate.macro_factor import MacroFactor
from panda_plugins.internal.models.common_models import FeatureModel, MLModel
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)

"""
PCA因子构建节点
基于主成分分析方法构建复合因子
"""

def construct_composite_factor(factors_importance, X, standardize_X=True):
    """
    根据因子重要性矩阵和因子暴露数据构建复合因子
    
    参数：
    factors_importance : ndarray, 形状为 (n_factors, 3)
        每行对应一个因子的 weight, gain, cover 指标
    X : ndarray, 形状为 (n_stocks, n_factors)
        每行对应一个股票在各个因子上的暴露值
    standardize_X : bool, 可选
        是否对因子暴露值进行标准化（默认True）
        
    返回：
    F : ndarray, 形状为 (n_stocks,)
        复合因子得分
    weights : ndarray, 形状为 (n_factors,)
        各因子的权重
    """
    # 归一化因子重要性矩阵
    scaler_I = StandardScaler()
    I_normalized = scaler_I.fit_transform(factors_importance)
    
    # PCA降维提取第一主成分得分
    pca = PCA(n_components=1)
    Z = pca.fit_transform(I_normalized)  # 形状 (n_factors, 1)
    weights = Z.flatten()  # 形状 (n_factors,)
    
    # 构建复合因子
    F = np.dot(X, weights)
    
    return F, weights

@ui(
    predict_start_date={"input_type": "date_picker"},
    predict_end_date={"input_type": "date_picker"},
)

class PCAFactorBuildInputModel(BaseModel):
    model: MLModel = Field(title="机器学习模型",)
    feature: FeatureModel = Field(title="特征工程",)
    test_start_date: str = Field(default="20250101",title="因子回测开始时间",)
    test_end_date: str = Field(default="20250301",title="因子回测结束时间",)
    predict_start_date: str = Field(default="20250301",title="因子预测开始时间",)
    predict_end_date: str = Field(default="20250501",title="因子预测结束时间",)

class PCAFactorBuildOutputModel(BaseModel):
    factor: Any = Field(default=None, title="复合因子得分")
    weights: list = Field(default=[], title="因子权重")

    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('factor')
    def validate_factor(cls, v):
        if not isinstance(v, DataFrame):
            raise ValueError('factor must be a pandas DataFrame')
        return v

@work_node(name="PCA因子构建", group="06-线下课专属", type="general", box_color="blue")
class PCAFactorBuildControl(BaseWorkNode):
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return PCAFactorBuildInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return PCAFactorBuildOutputModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:
        
        print("开始PCA因子构建")
        
        # 1. 获取测试期间的数据用于计算因子暴露度
        from panda_factor.generate.macro_factor import MacroFactor
        macro_factor = MacroFactor()
        
        # 准备特征公式列表
        if input.feature.features:
            factors = input.feature.features.split("\n")
        else:
            raise ValueError("因子不能为空")
        
        # 添加标签公式到因子列表中
        if input.feature.label:
            factor_list = factors + [input.feature.label]
        else:
            raise ValueError("标签不能为空")
        
        # 获取测试期间的数据用于计算暴露度
        test_factor_values = macro_factor.create_factor_from_formula_pro(
            factor_logger=logger,
            formulas=factor_list,
            start_date=input.test_start_date,
            end_date=input.test_end_date
        )
        
        print(f"测试期间因子数据形状: {test_factor_values.shape}")
        
        # 获取预测期间的数据
        predict_factor_values = macro_factor.create_factor_from_formula_pro(
            factor_logger=logger,
            formulas=factor_list,
            start_date=input.predict_start_date,
            end_date=input.predict_end_date
        )
        
        print(f"预测期间因子数据形状: {predict_factor_values.shape}")
        print(f"因子列名: {predict_factor_values.columns.tolist()}")
        
        # 2. 分离特征和标签
        feature_cols = test_factor_values.columns[:-1].tolist()  # 除最后一列外的所有列
        label_col = test_factor_values.columns[-1]  # 最后一列是标签
        
        print(f"特征列: {feature_cols}")
        print(f"标签列: {label_col}")
        
        # 3. 使用sklearn LinearRegression计算因子暴露度
        # 准备训练数据
        X_test = test_factor_values[feature_cols].dropna()
        y_test = test_factor_values[label_col].dropna()
        
        # 确保X和y有相同的索引
        common_index = X_test.index.intersection(y_test.index)
        X_test = X_test.loc[common_index]
        y_test = y_test.loc[common_index]
        
        print(f"用于暴露度计算的样本数: {len(X_test)}")
        
        # 拟合线性回归模型
        lr = LinearRegression(fit_intercept=True)
        lr.fit(X_test, y_test)
        
        # 回归系数就是各因子的暴露度
        exposures = pd.Series(lr.coef_, index=feature_cols, name='exposure')
        print("各因子暴露度（sklearn）：")
        print(exposures)
        print(f"截距（常数项）：{lr.intercept_:.6f}")
        
        # 4. 从训练好的模型中获取因子重要性指标
        if input.model.model_type == "xgboost":
            from xgboost import XGBRegressor
            model = XGBRegressor()
            model.load_model(input.model.model_path)
            
            # 获取特征重要性
            weight_scores = model.get_booster().get_score(importance_type='weight')
            gain_scores = model.get_booster().get_score(importance_type='gain')
            cover_scores = model.get_booster().get_score(importance_type='cover')
            
        elif input.model.model_type == "lightgbm":
            import lightgbm as lgb
            model = lgb.Booster(model_file=input.model.model_path)
            
            # 获取特征重要性
            weight_scores = model.feature_importance(importance_type='split')  # weight相当于split
            gain_scores = model.feature_importance(importance_type='gain')
            # LightGBM没有cover，用split代替
            cover_scores = model.feature_importance(importance_type='split')
            
            # 转换为字典格式
            feature_names = model.feature_name()
            weight_scores = dict(zip(feature_names, weight_scores))
            gain_scores = dict(zip(feature_names, gain_scores))
            cover_scores = dict(zip(feature_names, cover_scores))
            
        elif input.model.model_type == "randomforest":
            import joblib
            model = joblib.load(input.model.model_path)
            
            # RandomForest特征重要性
            feature_importances = model.feature_importances_
            
            # 获取特征名称（假设与feature_cols一致）
            feature_names = feature_cols
            
            # RandomForest主要提供基于不纯度的特征重要性
            # weight_scores: 使用feature_importances_（基于不纯度减少）
            # gain_scores: 同样使用feature_importances_
            # cover_scores: 使用每个特征在所有树中被使用的次数
            weight_scores = dict(zip(feature_names, feature_importances))
            gain_scores = dict(zip(feature_names, feature_importances))
            
            # 计算每个特征在所有树中的使用频次作为cover
            n_features = len(feature_names)
            feature_usage = np.zeros(n_features)
            for tree in model.estimators_:
                tree_features = tree.tree_.feature
                # 统计非叶子节点使用的特征
                valid_features = tree_features[tree_features >= 0]
                for feature_idx in valid_features:
                    if feature_idx < n_features:
                        feature_usage[feature_idx] += 1
            
            cover_scores = dict(zip(feature_names, feature_usage))
            
        elif input.model.model_type == "svm":
            import joblib
            from sklearn.svm import SVR
            model = joblib.load(input.model.model_path)
            
            # SVM不直接提供特征重要性，我们使用系数绝对值作为重要性指标
            if hasattr(model, 'coef_') and model.coef_ is not None:
                # 线性SVM的情况
                feature_coefs = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                feature_names = feature_cols
                
                # 对于SVM，我们将系数绝对值作为所有三种重要性的代理
                weight_scores = dict(zip(feature_names, feature_coefs))
                gain_scores = dict(zip(feature_names, feature_coefs))
                cover_scores = dict(zip(feature_names, feature_coefs))
            else:
                # 非线性SVM的情况，无法直接获取特征重要性
                # 使用置换重要性或者设为均等权重
                feature_names = feature_cols
                n_features = len(feature_names)
                equal_importance = np.ones(n_features) / n_features
                
                weight_scores = dict(zip(feature_names, equal_importance))
                gain_scores = dict(zip(feature_names, equal_importance))
                cover_scores = dict(zip(feature_names, equal_importance))
                
                logger.warning("非线性SVM模型无法直接获取特征重要性，使用均等权重")
            
        else:
            raise ValueError(f"暂不支持从{input.model.model_type}模型中获取特征重要性")
        
        # 5. 构建因子重要性矩阵
        weights = []
        gains = []
        covers = []
        
        for feature in feature_cols:
            # 获取特征重要性，如果不存在则设为0
            weight = weight_scores.get(feature, 0)
            gain = gain_scores.get(feature, 0)
            cover = cover_scores.get(feature, 0)
            
            weights.append(weight)
            gains.append(gain)
            covers.append(cover)
        
        # 构建因子重要性矩阵 (n_factors, 3)
        factors_importance = np.array([weights, gains, covers]).T
        
        print(f"因子重要性矩阵形状: {factors_importance.shape}")
        print(f"特征重要性统计:")
        print(f"Weight: {weights}")
        print(f"Gain: {gains}")
        print(f"Cover: {covers}")
        
        # 6. 准备预测数据并使用暴露度计算复合因子
        predict_df = predict_factor_values.copy()
        predict_df['values'] = np.nan  # 使用'values'列名与ml_multi_factor_build保持一致
        
        # 存储因子权重
        final_weights = None
        
        # 对每天的数据应用复合因子构建
        for date, group in predict_df.groupby(level=0):  # 按日期分组（多级索引的第0级）
            # 获取特征数据
            X_features = group[feature_cols].values
            
            if np.isnan(X_features).any():
                logger.warning(f"日期 {date} 的数据包含缺失值，跳过")
                continue
            
            # 使用暴露度作为权重计算暴露值矩阵
            # X = 特征值 * 暴露度权重
            X_exposures = X_features * exposures.values.reshape(1, -1)  # 广播乘法
            
            # 使用task3.py中的复合因子构建方法
            F, pca_weights = construct_composite_factor(factors_importance, X_exposures, standardize_X=True)
            
            # 存储权重（所有日期应该使用相同的权重）
            if final_weights is None:
                final_weights = pca_weights
            
            # 将复合因子得分填入对应的行
            predict_df.loc[predict_df.index.get_level_values(0) == date, 'values'] = F
        
        # 7. 整理输出结果
        # 重置索引，将多级索引转换为列
        result_df = predict_df.reset_index()
        
        # 只保留需要的列：date, symbol, values（与ml_multi_factor_build保持一致）
        if 'date' in result_df.columns and 'symbol' in result_df.columns:
            result_df = result_df[['date', 'symbol', 'values']]
        else:
            # 如果列名不同，使用索引级别名称
            level_names = predict_df.index.names
            result_df = result_df[[level_names[0], level_names[1], 'values']]
            result_df.columns = ['date', 'symbol', 'values']
        
        # 移除包含NaN的行
        result_df = result_df.dropna()
        
        # 准备输出结果
        weights_list = final_weights.tolist() if final_weights is not None else []
        
        print("PCA因子构建完成")
        
        return PCAFactorBuildOutputModel(
            factor=result_df, 
            weights=weights_list
        )
        
