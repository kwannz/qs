import logging
from bson import ObjectId
from fastapi import HTTPException, status
from panda_server.config.env import RUN_MODE
from panda_server.config.database import mongodb
from panda_server.models.base_api_response import BaseAPIResponse
from panda_server.models.run_workflow_request import TerminateWorkflowRunRequest
from panda_server.models.workflow_run_model import WorkflowRunModel
from panda_server.enums.workflow_run_status import WorkflowStatus

# 获取 logger
logger = logging.getLogger(__name__)


async def workflow_terminate_logic(
    request: TerminateWorkflowRunRequest, user_id: str
) -> BaseAPIResponse:
    """
    终止指定 workflow 运行的业务逻辑

    Args:
        request: 终止工作流运行请求
        user_id: 用户ID

    Returns:
        BaseAPIResponse: 终止操作的结果

    Raises:
        HTTPException: 当工作流不存在、权限不足或状态不正确时
    """
    # 从请求中获取 workflow_run_id
    workflow_run_id = request.workflow_run_id

    # 从 mongodb 中获取 workflow_run 信息
    workflow_run_collection = mongodb.get_collection("workflow_run")
    query_result = await workflow_run_collection.find_one(
        {"_id": ObjectId(workflow_run_id)}
    )
    if not query_result:
        logger.error(f"No workflow run found, id:{workflow_run_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow run not found",
        )

    workflow_run = WorkflowRunModel(**query_result)

    # 校验 workflow 的 owner 是否和 user_id 一致 (防止攻击者终止他人 workflow)
    if workflow_run.owner != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to terminate this workflow run",
        )

    # 校验 workflow run 当前状态
    if workflow_run.status not in [WorkflowStatus.RUNNING, WorkflowStatus.PENDING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow run is not running or pending",
        )

    # 确定运行模式
    if RUN_MODE == "CLOUD":
        # TODO 调用 RabitMQ 发送终止消息
        # 数据库插入终止记号
        await workflow_run_collection.update_one(
            {"_id": ObjectId(workflow_run_id)},
            {"$set": {"status": WorkflowStatus.MANUAL_STOP}},
        )
    elif RUN_MODE == "LOCAL":
        # 数据库插入终止记号
        await workflow_run_collection.update_one(
            {"_id": ObjectId(workflow_run_id)},
            {"$set": {"status": WorkflowStatus.MANUAL_STOP}},
        )

    return BaseAPIResponse(data=None) 