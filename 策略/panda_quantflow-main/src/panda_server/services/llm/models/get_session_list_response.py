from typing import List
from pydantic import BaseModel, Field
from panda_server.models.base_api_response import BaseAPIResponse
from .chat_session_model import ChatSessionModel

class GetSessionListResponseData(BaseModel):
    """获取会话列表响应数据"""
    sessions: List[ChatSessionModel] = Field(description="聊天会话列表")
    total_count: int = Field(description="总数量")

class GetSessionListResponse(BaseAPIResponse[GetSessionListResponseData]):
    """获取会话列表响应"""
    pass 