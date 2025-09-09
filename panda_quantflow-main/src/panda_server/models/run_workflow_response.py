from pydantic import BaseModel
import logging

from .base_api_response import BaseAPIResponse

class RunWorkflowResponseData(BaseModel):
    """运行节点"""
    workflow_run_id: str

class RunWorkflowResponse(BaseAPIResponse[RunWorkflowResponseData]):
    """运行节点"""
    pass
