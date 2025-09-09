from pydantic import BaseModel, Field
import logging

from pandas import DataFrame

class FeatureModel(BaseModel):
    features: str = Field(default="")
    label: str = Field(default="")
    
class MLModel(BaseModel):
    model_path: str = Field(default="")
    model_type: str = Field(default="xgboost")
    
    model_config = {
        'protected_namespaces': ()
    }

class MLOutputModel(BaseModel):
    """通用的机器学习模型输出"""
    model: MLModel = Field(default=None, title="模型", description="训练好的机器学习模型")
