from typing import List, Optional
import logging

from pydantic import BaseModel, Field
from .base_api_response import BaseAPIResponse
from common.logging.user_log_model import UserLog

class QueryWorkflowLogsResponseData(BaseModel):
    """查询工作流日志响应数据"""
    logs: List[UserLog] = Field(description="工作流日志列表")
    has_more: bool = Field(description="是否还有更多日志")
    next_log_id: Optional[str] = Field(description="下一页查询的起始日志ID（兼容旧版本）", default=None)
    next_sequence: Optional[int] = Field(description="下一页查询的起始序列号（推荐使用）", default=None)
    total_count: Optional[int] = Field(description="总数量（可选）", default=None)

class QueryWorkflowLogsResponse(BaseAPIResponse[QueryWorkflowLogsResponseData]):
    """查询工作流日志响应"""
    pass 