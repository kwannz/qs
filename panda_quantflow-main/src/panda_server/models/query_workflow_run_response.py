from typing import Dict, List
import logging

from pydantic import BaseModel, Field
from panda_server.enums.workflow_run_status import WorkflowStatus
from .base_api_response import BaseAPIResponse

class QueryWorkflowRunResponseData(BaseModel):
    """查询 workflow 运行状态"""

    status: WorkflowStatus
    progress: float  # 0-100
    running_node_ids: List[str]
    success_node_ids: List[str]
    failed_node_ids: List[str]
    passed_link_ids: List[str]
    output_data_obj: Dict[str, str] = Field(description="key: uuid of the node, value: output_db_id")

class QueryWorkflowRunResponse(BaseAPIResponse[QueryWorkflowRunResponseData]):
    """查询 workflow 运行状态"""

    pass
