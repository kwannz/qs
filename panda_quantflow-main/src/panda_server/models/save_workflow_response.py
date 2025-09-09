from pydantic import BaseModel
import logging

from .base_api_response import BaseAPIResponse

class SaveWorkflowResponseData(BaseModel):
    """保存节点"""
    workflow_id: str

class SaveWorkflowResponse(BaseAPIResponse[SaveWorkflowResponseData]):
    """保存节点"""
    pass
