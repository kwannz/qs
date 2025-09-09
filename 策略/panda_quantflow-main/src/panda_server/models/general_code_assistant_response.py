from pydantic import BaseModel, Field
from .base_api_response import BaseAPIResponse

class GeneralCodeAssistantMessage(BaseModel):
    """代码类助手消息结构"""
    new_code: str = Field(None, description="新代码，如果为空，则表示不需要新代码")
    explanation: str = Field(..., description="AI对新代码的解释")

class GeneralCodeAssistantData(BaseModel):
    """"LLM 会话响应数据"""
    session_id: str
    message: GeneralCodeAssistantMessage

class GeneralCodeAssistantResponse(BaseAPIResponse[GeneralCodeAssistantData]):
    """通用代码类LLM助手的会话响应: CodeAssistant, BacktestAssistant, TradeAssistant 等共用"""
    pass
