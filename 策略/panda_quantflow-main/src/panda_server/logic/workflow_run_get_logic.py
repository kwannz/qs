import logging
from bson import ObjectId
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from panda_server.models.query_workflow_run_response import (
    QueryWorkflowRunResponse,
    QueryWorkflowRunResponseData,
)
from panda_server.models.workflow_run_model import WorkflowRunModel

logger = logging.getLogger(__name__)


async def workflow_run_get_logic(
    workflow_run_id: str, user_id: str
) -> QueryWorkflowRunResponse:
    """
    查询指定 workflow run 运行状态

    Args:
        workflow_run_id: 工作流运行ID
        user_id: 用户ID

    Returns:
        QueryWorkflowRunResponse: 工作流运行状态响应

    Raises:
        HTTPException: 当工作流运行不存在或用户无权限访问时
    """
    # 从 mongodb 中获取 workflow run 信息
    workflow_run_collection = mongodb.get_collection("workflow_run")
    result = await workflow_run_collection.find_one({"_id": ObjectId(workflow_run_id)})
    if not result:
        logger.error(f"No workflow run found, id:{workflow_run_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow run with id {workflow_run_id} not found",
        )

    workflow_run = WorkflowRunModel(**result)

    # 校验权限：
    # 如果既不是本人的 workflow_run，也不是公共的 workflow_run（owner="*"），则拒绝访问
    if workflow_run.owner != user_id and workflow_run.owner != "*":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to query this workflow run",
        )

    #  返回 workflow_run 信息
    response_data = QueryWorkflowRunResponseData(
        status=workflow_run.status,
        progress=workflow_run.progress,
        running_node_ids=workflow_run.running_node_ids,
        success_node_ids=workflow_run.success_node_ids,
        failed_node_ids=workflow_run.failed_node_ids,
        passed_link_ids=workflow_run.passed_link_ids,
        output_data_obj=workflow_run.output_data_obj
    )
    return QueryWorkflowRunResponse(data=response_data) 