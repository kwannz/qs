# from typing import Optional, Type, Dict, Any
# import logging

# import os
# import joblib
# from pydantic import BaseModel, Field
# from panda_plugins.base.base_work_node import BaseWorkNode
# from panda_plugins.base.work_node_registery import work_node

# class SaveModelInput(BaseModel):
#     """Input model for SaveModel node"""
#     model: object = Field(..., description="Trained model to save")
#     model_name: str = Field(..., description="Name of the model file (without extension)")
#     model_type: str = Field(..., description="Type of the model (e.g., 'svm', 'randomforest', 'xgboost', 'lightgbm')")
#     scaler_X: Optional[object] = Field(None, description="Feature scaler to save")
#     scaler_y: Optional[object] = Field(None, description="Target scaler to save")
#     model_params: Optional[Dict[str, Any]] = Field(None, description="Model parameters to save")

#     model_config = {
#         'arbitrary_types_allowed': True,
#         'protected_namespaces': ()
#     }
        

# class SaveModelOutput(BaseModel):
#     """Output model for SaveModel node"""
#     model_path: str = Field(..., description="Path where the model was saved")
#     scaler_X_path: Optional[str] = Field(None, description="Path where the feature scaler was saved")
#     scaler_y_path: Optional[str] = Field(None, description="Path where the target scaler was saved")
#     params_path: Optional[str] = Field(None, description="Path where the model parameters were saved")

#     model_config = {
#         'arbitrary_types_allowed': True,
#         'protected_namespaces': ()
#     }
        
# @work_node(
#     name="save_model",
#     group="99-测试节点",
#     order=4,
#     type="utility"
# )
# class SaveModelNode(BaseWorkNode):
#     """Node for saving trained models and their associated scalers"""

#     @classmethod
#     def input_model(cls) -> Type[BaseModel]:
#         return SaveModelInput

#     @classmethod
#     def output_model(cls) -> Type[BaseModel]:
#         return SaveModelOutput

#     def run(self, input: SaveModelInput) -> SaveModelOutput:
#         """
#         Save a trained model and its associated scalers
        
#         Args:
#             input: SaveModelInput containing model and scalers to save
            
#         Returns:
#             SaveModelOutput containing paths where files were saved
#         """
#         # Create model directory if it doesn't exist
#         model_dir = "/user_data/model"
#         os.makedirs(model_dir, exist_ok=True)

#         # Save model
#         model_path = os.path.join(model_dir, f"{input.model_name}_{input.model_type}.joblib")
#         joblib.dump(input.model, model_path)

#         # Save scalers if provided
#         scaler_X_path = None
#         scaler_y_path = None
#         params_path = None

#         if input.scaler_X is not None:
#             scaler_X_path = os.path.join(model_dir, f"{input.model_name}_{input.model_type}_scaler_X.joblib")
#             joblib.dump(input.scaler_X, scaler_X_path)

#         if input.scaler_y is not None:
#             scaler_y_path = os.path.join(model_dir, f"{input.model_name}_{input.model_type}_scaler_y.joblib")
#             joblib.dump(input.scaler_y, scaler_y_path)

#         # Save model parameters if provided
#         if input.model_params is not None:
#             params_path = os.path.join(model_dir, f"{input.model_name}_{input.model_type}_params.joblib")
#             joblib.dump(input.model_params, params_path)

#         return SaveModelOutput(
#             model_path=model_path,
#             scaler_X_path=scaler_X_path,
#             scaler_y_path=scaler_y_path,
#             params_path=params_path
#         ) 