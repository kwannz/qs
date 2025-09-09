import logging
from bson import ObjectId
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from panda_server.models.workflow_model import WorkflowModel

logger = logging.getLogger(__name__)

# 定义 collection 名称
COLLECTION_NAME = "workflow"


async def workflow_get_logic(workflow_id: str, user_id: str) -> WorkflowModel:
    """
    获取指定 workflow

    Args:
        workflow_id: 工作流ID
        user_id: 用户ID

    Returns:
        WorkflowModel: 工作流信息

    Raises:
        HTTPException: 当工作流不存在或用户无权限访问时
        ValueError: 当工作流ID格式无效时
    """
    # 查询指定的工作流
    workflow_collection = mongodb.get_collection(COLLECTION_NAME)
    workflow_doc = await workflow_collection.find_one({"_id": ObjectId(workflow_id)})
    if not workflow_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with id {workflow_id} not found",
        )
    # 校验权限
    if workflow_doc["owner"] not in [user_id, "*"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to access this workflow",
        )

    # 将查询结果转换为 WorkflowModel
    workflow = WorkflowModel.model_validate(workflow_doc)
    return workflow
