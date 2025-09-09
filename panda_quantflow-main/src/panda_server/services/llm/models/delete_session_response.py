from pydantic import BaseModel, Field
from panda_server.models.base_api_response import BaseAPIResponse

class DeleteSessionResponseData(BaseModel):
    """删除会话响应数据"""
    session_id: str = Field(description="被删除的会话ID")
    deleted: bool = Field(description="是否删除成功")

class DeleteSessionResponse(BaseAPIResponse[DeleteSessionResponseData]):
    """删除会话响应"""
    pass 