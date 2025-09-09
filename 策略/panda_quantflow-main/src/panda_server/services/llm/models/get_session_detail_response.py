from pydantic import BaseModel, Field
from panda_server.models.base_api_response import BaseAPIResponse
from .chat_session_model import ChatSessionModel

class GetSessionDetailResponseData(BaseModel):
    """获取会话详情响应数据"""
    session: ChatSessionModel = Field(description="聊天会话详情")

class GetSessionDetailResponse(BaseAPIResponse[GetSessionDetailResponseData]):
    """获取会话详情响应"""
    pass 