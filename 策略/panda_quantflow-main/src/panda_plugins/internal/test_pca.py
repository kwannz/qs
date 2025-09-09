# from panda_plugins.internal.feature_engineering_node import FeatureEngineeringNode, FeatureInputModel, FeatureModel
import logging

# from panda_plugins.internal.formula_node import FormulaControl, FormulaInputModel
# from panda_plugins.internal.pca_factor_build_node import PCAFactorBuildControl, PCAFactorBuildInputModel
# from panda_plugins.internal.lightgbm_node import LightGBMControl, LightGBMInputModel
# from panda_plugins.internal.factor_analysis_node import FactorAnalysisControl, FactorAnalysisInputModel, FactorAnalysisOutputModel
# import numpy as np

# if __name__ == "__main__":
#     # 公式节点 构建公式
#     formulas = "CLOSE\nOPEN\nHIGH\nLOW\nVOLUME"
#     formula_node = FormulaControl()
#     input = FormulaInputModel(formulas=formulas)
#     res = formula_node.run(input)
#     print("公式节点结果:", res)

#     # 特征工程节点 构建特征
#     feature_node = FeatureEngineeringNode()
#     label = "RETURNS(CLOSE,1)"
#     input = FeatureInputModel(formulas=res.formulas, label=label)
#     feature_output = feature_node.run(input)
#     print("特征工程节点完成")

#     # lightgbm节点 训练模型（为了符合输入模型的要求）
#     lightgbm_node = LightGBMControl()
#     input = LightGBMInputModel(feature=feature_output.feature_model, start_date="20250101", end_date="20250301", n_estimators=100, max_depth=3, learning_rate=0.1, num_leaves=31, subsample=1.0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0)
#     lightgbm_model = lightgbm_node.run(input)
#     print("LightGBM模型训练完成")

#     # PCA因子构建节点 构建因子（按照task3.py的逻辑）
#     pca_node = PCAFactorBuildControl()
#     input = PCAFactorBuildInputModel(
#         model=lightgbm_model.model,  # 从训练好的模型中获取特征重要性
#         feature=feature_output.feature_model, 
#         test_start_date="20250101",     # 测试期间用于计算因子暴露度
#         test_end_date="20250301",
#         predict_start_date="20250301",  # 预测期间用于构建复合因子
#         predict_end_date="20250501"
#     )
#     pca_factors = pca_node.run(input)
#     print("\nPCA因子构建完成")
#     print("因子权重:", pca_factors.weights)
#     print("因子数据形状:", pca_factors.factor.shape)
#     print("因子数据前5行:")
#     print(pca_factors.factor.head())

#     # 因子分析
#     factor_analysis_node = FactorAnalysisControl()
#     input = FactorAnalysisInputModel(df_factor=pca_factors.factor, adjustment_cycle=1, group_number=5, factor_direction=0)
#     res = factor_analysis_node.run(input)
#     print("\nPCA因子分析结果:", res)

#     # 显示复合因子构建的核心结果
#     print("\n=== 复合因子构建结果 ===")
#     print("因子权重:", pca_factors.weights)
#     print("复合因子数据统计:")
#     print(pca_factors.factor['values'].describe())
    