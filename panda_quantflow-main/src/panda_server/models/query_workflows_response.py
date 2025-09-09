from typing import List
from pydantic import BaseModel, Field
from .base_api_response import BaseAPIResponse
from panda_server.models.workflow_model import WorkflowModel

class QueryWorkflowsResponseData(BaseModel):
    """查询工作流列表响应数据"""
    workflows: List[WorkflowModel] = Field(description="工作流列表")
    total_count: int = Field(description="总数量")

class QueryWorkflowsResponse(BaseAPIResponse[QueryWorkflowsResponseData]):
    """查询工作流列表响应"""
    pass 