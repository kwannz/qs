# from typing import Optional, Type, Dict, Any
# import logging

# import os
# import joblib
# from pydantic import BaseModel, Field
# from panda_plugins.base.base_work_node import BaseWorkNode
# from panda_plugins.base.work_node_registery import work_node


# class LoadModelInput(BaseModel):
#     """Input model for LoadModel node"""

#     model_name: str = Field(
#         ..., description="Name of the model file (without extension)"
#     )
#     model_type: str = Field(
#         ...,
#         description="Type of the model (e.g., 'svm', 'randomforest', 'xgboost', 'lightgbm')",
#     )
#     load_scalers: bool = Field(True, description="Whether to load associated scalers")

#     model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}


# class LoadModelOutput(BaseModel):
#     """Output model for LoadModel node"""

#     model: object = Field(..., description="Loaded model")
#     scaler_X: Optional[object] = Field(
#         None, description="Loaded feature scaler if available"
#     )
#     scaler_y: Optional[object] = Field(
#         None, description="Loaded target scaler if available"
#     )
#     model_params: Dict[str, Any] = Field(..., description="Model parameters")

#     model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}


# @work_node(name="load_model", group="99-测试节点", order=3, type="utility")
# class LoadModelNode(BaseWorkNode):
#     """Node for loading saved models"""

#     @classmethod
#     def input_model(cls) -> Type[BaseModel]:
#         return LoadModelInput

#     @classmethod
#     def output_model(cls) -> Type[BaseModel]:
#         return LoadModelOutput

#     def run(self, input: LoadModelInput) -> LoadModelOutput:
#         """
#         Load a saved model and its associated scalers

#         Args:
#             input: LoadModelInput containing model name and type

#         Returns:
#             LoadModelOutput containing loaded model and scalers
#         """
#         # 记录开始加载模型
#         self.log_info(
#             "开始加载模型",
#             model_name=input.model_name,
#             model_type=input.model_type,
#             load_scalers=input.load_scalers,
#         )

#         model_dir = "/user_data/model"

#         # Load model
#         model_path = os.path.join(
#             model_dir, f"{input.model_name}_{input.model_type}.joblib"
#         )

#         self.log_debug("检查模型文件路径", model_path=model_path)

#         if not os.path.exists(model_path):
#             self.log_error("模型文件不存在", model_path=model_path)
#             raise FileNotFoundError(f"Model file not found: {model_path}")

#         self.log_info("开始加载模型文件")
#         model = joblib.load(model_path)
#         self.log_info("模型文件加载成功", model_type=type(model).__name__)

#         # Load scalers if requested
#         scaler_X = None
#         scaler_y = None

#         if input.load_scalers:
#             self.log_info("开始加载缩放器")
#             scaler_X_path = os.path.join(
#                 model_dir, f"{input.model_name}_{input.model_type}_scaler_X.joblib"
#             )
#             scaler_y_path = os.path.join(
#                 model_dir, f"{input.model_name}_{input.model_type}_scaler_y.joblib"
#             )

#             if os.path.exists(scaler_X_path):
#                 scaler_X = joblib.load(scaler_X_path)
#                 self.log_info("特征缩放器加载成功", scaler_type=type(scaler_X).__name__)
#             else:
#                 self.log_warning("特征缩放器文件不存在", scaler_path=scaler_X_path)

#             if os.path.exists(scaler_y_path):
#                 scaler_y = joblib.load(scaler_y_path)
#                 self.log_info("目标缩放器加载成功", scaler_type=type(scaler_y).__name__)
#             else:
#                 self.log_warning("目标缩放器文件不存在", scaler_path=scaler_y_path)
#         else:
#             self.log_debug("跳过缩放器加载")

#         # Extract model parameters
#         model_params = {}
#         if hasattr(model, "get_params"):
#             model_params = model.get_params()
#             self.log_debug("提取模型参数成功", param_count=len(model_params))
#         else:
#             self.log_warning("模型不支持参数提取")

#         self.log_info(
#             "模型加载完成",
#             has_scaler_X=scaler_X is not None,
#             has_scaler_y=scaler_y is not None,
#             param_count=len(model_params),
#         )

#         return LoadModelOutput(
#             model=model, scaler_X=scaler_X, scaler_y=scaler_y, model_params=model_params
#         )
