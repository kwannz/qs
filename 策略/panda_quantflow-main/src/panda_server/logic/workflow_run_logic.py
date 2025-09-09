import datetime
import json
import logging
from bson import ObjectId
from fastapi import HTTPException, status, BackgroundTasks
from panda_server.config.database import mongodb
from panda_server.config.env import (
    RUN_MODE,
    WORKFLOW_EXCHANGE_NAME,
    WORKFLOW_ROUTING_KEY,
)
from panda_server.models.run_workflow_request import RunWorkflowRequest
from panda_server.models.run_workflow_response import (
    RunWorkflowResponse,
    RunWorkflowResponseData,
)
from panda_server.models.workflow_run_model import WorkflowRunCreateModel
from panda_server.enums.workflow_run_status import WorkflowStatus
from panda_server.messaging.rabbitmq_client import AsyncRabbitMQ
from panda_server.utils.run_workflow_utils import run_workflow_in_background

logger = logging.getLogger(__name__)


async def workflow_run_logic(
    background_tasks: BackgroundTasks,
    request: RunWorkflowRequest,
    user_id: str,
    quantflow_auth: str,
) -> RunWorkflowResponse:
    """
    运行指定 workflow

    Args:
        background_tasks: FastAPI BackgroundTasks 实例
        request: 运行工作流的请求数据
        user_id: 用户ID
        quantflow_auth: 认证凭据

    Returns:
        RunWorkflowResponse: 包含工作流运行ID的响应

    Raises:
        HTTPException: 当权限不足、工作流不存在或用户无权限访问时

    TODO 稍后处理: 因为用户在运行过程中如果再次保存 workflow 会造成运行的代码版本不一致. 所以还要保存一个工作流的备份版本.
    """
    # 鉴权检查：只有当 quantflow-auth 为 "2" 时才允许执行
    if quantflow_auth != "2":
        # 后端控制台记录未授权访问尝试
        logger.warning(
            f"Unauthorized workflow run attempt, user_id: {user_id}, auth: {quantflow_auth}"
        )
        # 返回给前端403状态码和错误信息
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="权限不足，无法执行工作流"
        )

    # 从请求中获取 workflow_id
    workflow_id = request.workflow_id

    # 从 mongodb 中获取 workflow 信息
    workflow_collection = mongodb.get_collection("workflow")
    workflow = await workflow_collection.find_one({"_id": ObjectId(workflow_id)})

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with id {workflow_id} not found",
        )

    # 校验 workflow 的 owner 是否和 user_id 一致 (防止攻击者调用他人 workflow)
    if workflow.get("owner") != user_id:
        logger.error(
            f"User {user_id} does not have permission to run workflow {workflow_id}, who is owned by {workflow.get('owner')}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to run this workflow",
        )

    # MongoDB 中创建 workflow_run 记录
    workflow_run_collection = mongodb.get_collection("workflow_run")
    workflow_run = WorkflowRunCreateModel(
        workflow_id=workflow_id,
        owner=user_id,
        status=WorkflowStatus.PENDING,
        progress=0.0,
        running_node_ids=[],
        success_node_ids=[],
        failed_node_ids=[],
        passed_link_ids=[],
        output_data_obj={},
        log_obj_id="",  # 日志对象 ID 初始为空
    )

    # 使用事务同时执行插入工作流运行记录和更新工作流的last_run_id
    async with await mongodb.client.start_session() as session:
        async with session.start_transaction():
            # 插入工作流运行记录
            result = await workflow_run_collection.insert_one(
                workflow_run.model_dump(), session=session
            )
            workflow_run_id = str(result.inserted_id)

            # 更新工作流的last_run_id
            await workflow_collection.find_one_and_update(
                {"_id": ObjectId(workflow_id)},
                {"$set": {"last_run_id": workflow_run_id}},
                return_document=False,
                session=session,
            )

    # 确定运行模式
    if RUN_MODE == "CLOUD":
        # CLOUD模式：使用RabbitMQ队列
        rabbitMq = AsyncRabbitMQ()
        message = json.dumps(
            {
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "run_workflow",
                "user_id": user_id,
                "content": workflow_run_id,
            }
        )
        logger.info(f"CLOUD mode: Adding workflow task {workflow_run_id} to RabbitMQ queue")
        logger.info(message)
        await rabbitMq.publish(
            exchange_name=WORKFLOW_EXCHANGE_NAME,
            routing_key=WORKFLOW_ROUTING_KEY,
            message=message,
        )
    elif RUN_MODE == "LOCAL":
        # LOCAL模式：直接执行工作流
        logger.info(f"LOCAL mode: Executing workflow task {workflow_run_id} directly")
        background_tasks.add_task(run_workflow_in_background, workflow_run_id)

    return RunWorkflowResponse(
        data=RunWorkflowResponseData(workflow_run_id=workflow_run_id)
    )
