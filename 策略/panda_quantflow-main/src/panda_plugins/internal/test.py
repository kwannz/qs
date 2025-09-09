from panda_plugins.internal.xgboost_node import XgboostControl, XgboostInputModel
import logging

from panda_ml.importance_spearman_factor_combiner import ImportanceSpearmanFactorCombiner
from panda_plugins.internal.feature_engineering_node import FeatureEngineeringNode, FeatureInputModel, FeatureModel
from panda_plugins.internal.formula_node import FormulaControl, FormulaInputModel
from panda_plugins.internal.ml_factor_build_node import MLFactorBuildControl, MLFactorBuildInputModel
from panda_plugins.internal.lightgbm_node import LightGBMControl, LightGBMInputModel
from panda_plugins.internal.ml_multi_factor_build_node import MLMultiFactorBuildControl, MLMultiFactorBuildInputModel
# from panda_plugins.internal.factor_analysis_node import FactorAnalysis, InputModel
import pandas as pd
from panda_plugins.internal.mtl_nn_node import MTLNNControl, MTLNNInputModel
from xgboost import XGBRegressor
from panda_plugins.internal.mtl_factor_build_node import MTLFactorBuildControl, MTLFactorBuildInputModel
if __name__ == "__main__":
    # # 公式节点 构建公式
    # formulas = "CLOSE\nLOW\nOPEN"
    # formula_node = FormulaControl()
    # input = FormulaInputModel(formulas=formulas)
    # res = formula_node.run(input)
    # print(res)

    # 特征工程节点 构建特征
    feature_node1 = FeatureEngineeringNode()
    label1 = "IC"
    input1 = FeatureInputModel(formulas="CLOSE", label=label1)
    feature_output1 = feature_node1.run(input1)
    print(feature_output1)
    feature_node2 = FeatureEngineeringNode()
    label2 = "IC"
    input2 = FeatureInputModel(formulas="OPEN", label=label2)
    feature_output2 = feature_node2.run(input2)
    print(feature_output2)
    mtl_nn_node=MTLNNControl()
    mtlnninput=MTLNNInputModel(feature1=feature_output1.feature_model, feature2=feature_output2.feature_model)
    mtlnnmodel=mtl_nn_node.run(mtlnninput)
    # # lightgbm节点 训练模型
    # lightgbm_node = LightGBMControl()
    # input = LightGBMInputModel(feature=feature_output.feature_model, start_date="20250101", end_date="20250301", n_estimators=100, max_depth=3, learning_rate=0.1, num_leaves=31, subsample=1.0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0)
    # lightgbm_model = lightgbm_node.run(input)

    # xgboot节点 训练模型
    # xgboost_node = XgboostControl()
    # input = XgboostInputModel(feature=feature_output.feature_model)
    # xgboost_model = xgboost_node.run(input)
    #
    # model = XGBRegressor()
    # model.load_model(xgboost_model.model.model_path)
    # # 获取每个因子在模型中的重要指标
    # booster=model.get_booster()
    # imp_weight=booster.get_score(importance_type='weight')
    # imp_gain=booster.get_score(importance_type='gain')
    # imp_cover=booster.get_score(importance_type='cover')
    # df_imp=pd.DataFrame({
    #     'weight': pd.Series(imp_weight),
    #     'gain': pd.Series(imp_gain),
    #     'cover': pd.Series(imp_cover)
    # }).fillna(0)
    # print(df_imp)
    # ── 5. 实例化合成器 & 计算──────────────────────────────────────────────────
    # combiner = ImportanceSpearmanFactorCombiner()
    # （1）fit：计算每个因子三种指标与收益的 Spearman 相关系数
    # combiner.fit(df_imp, returns)

    #
    # 因子构建节点 构建因子
    # ml_node = MLMultiFactorBuildControl()
    # input = MLMultiFactorBuildInputModel(model1=mtlnnmodel.model,features1=feature_output1.feature_model,model2=mtlnnmodel.model,features2=feature_output2.feature_model, start_date="20250301", end_date="20250501")
    # factors = ml_node.run(input)
    # print(factors)
    # 因子构建节点 构建因子
    mtl_node = MTLFactorBuildControl()
    input = MTLFactorBuildInputModel(feature2=feature_output2.feature_model, feature1=feature_output1.feature_model, model=mtlnnmodel.model, start_date="20250301", end_date="20250501")
    factors = mtl_node.run(input)
    print(factors)

    # 因子分析
    factor_analysis_node = FactorAnalysis()
    input = InputModel(df_factor=factors.factor, adjustment_cycle=1,group_number=5,factor_direction=0)
    res = factor_analysis_node.run(input)
    print(res)