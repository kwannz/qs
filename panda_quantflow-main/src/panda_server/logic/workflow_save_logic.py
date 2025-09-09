from bson import ObjectId
from panda_server.models.workflow_model import (
    WorkflowCreateModel,
    WorkflowUpdateModel,
)
from fastapi import HTTPException, status
import logging
from panda_server.config.database import mongodb
from panda_server.models.save_workflow_request import SaveWorkflowRequest
from panda_server.models.save_workflow_response import (
    SaveWorkflowResponse,
    SaveWorkflowResponseData,
)

logger = logging.getLogger(__name__)

# 定义 collection 名称
COLLECTION_NAME = "workflow"


async def workflow_save_logic(
    request: SaveWorkflowRequest, user_id: str
) -> SaveWorkflowResponse:
    workflow_collection = mongodb.get_collection(COLLECTION_NAME)

    # 如果 request.id 不存在, 则执行工作流创建逻辑
    if request.id is None:
        return await create_workflow(workflow_collection, request, user_id)

    # 检查 request.id 对应工作流是否存在
    existing_workflow = await workflow_collection.find_one(
        {"_id": ObjectId(request.id)}
    )
    if not existing_workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with id {request.id} not found",
        )

    # 如果 owner 为 * 表示从工作流模板新建
    if existing_workflow.get("owner") == "*":
        request.name += "-克隆"
        return await create_workflow(workflow_collection, request, user_id)

    # 修改现有工作流
    return await update_workflow(
        workflow_collection, request, user_id, existing_workflow
    )


async def create_workflow(
    workflow_collection, request: SaveWorkflowRequest, user_id: str
):
    # 新建工作流
    workflow_data = request.get_workflow_data()
    workflow_data["owner"] = user_id
    workflow_data = WorkflowCreateModel(**workflow_data).model_dump()

    # 创建新工作流记录
    result = await workflow_collection.insert_one(workflow_data)
    workflow_id = str(result.inserted_id)

    return SaveWorkflowResponse(data=SaveWorkflowResponseData(workflow_id=workflow_id))


async def update_workflow(
    workflow_collection,
    request: SaveWorkflowRequest,
    user_id: str,
    existing_workflow: dict,
):
    # 检查权限
    if existing_workflow.get("owner") != user_id:
        logger.error(
            f"User {user_id} does not have permission to modify workflow {request.id}, "
            f"which is owned by {existing_workflow.get('owner')}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to modify this workflow",
        )

    # 准备更新数据
    update_data = request.get_workflow_data()
    if update_data:  # 只有在有数据需要更新时才进行更新
        update_data = WorkflowUpdateModel(**update_data).model_dump(exclude_unset=True)
        # 执行更新
        await workflow_collection.update_one(
            {"_id": ObjectId(request.id)},
            {"$set": update_data},
        )

    return SaveWorkflowResponse(data=SaveWorkflowResponseData(workflow_id=request.id))
