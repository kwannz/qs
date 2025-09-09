from typing import Optional
from pydantic import BaseModel, Field, field_validator
from ..enums.llm_model_type import MODEL_CONFIG


class FactorAssistantRequest(BaseModel):
    """
    Factor Assistant Request
    因子助手请求模型，包含用户消息、会话ID、原始代码和运行日志等字段
    """
    
    session_id: Optional[str] = Field(default=None, description="会话ID，可选")
    message: Optional[str] = Field(default=None, description="用户消息，可选")
    original_code: Optional[str] = Field(default=None, description="需要AI参考的当前版本因子代码，可选")
    last_run_log: Optional[str] = Field(default=None, description="上次运行日志，用于给 AI 提供debug信息，可选")
    model: str = Field(..., description="要使用的模型名称")

    class Config:
        extra = "forbid"  # 只允许传入定义的字段
        json_schema_extra = {
            "example": {
                "session_id": "123456",
                "message": "帮我优化这个因子",
                "original_code": "def calculate(factors): return factors.close",
                "last_run_log": None,
                "model": "DeepSeek-V3"
            }
        }

    @field_validator("model")
    def validate_model(cls, v):
        if v not in MODEL_CONFIG:
            raise ValueError(f"不支持的模型类型: {v}。支持的模型类型: {list(MODEL_CONFIG.keys())}")
        return v 