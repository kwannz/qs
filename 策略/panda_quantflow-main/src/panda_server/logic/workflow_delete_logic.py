import logging
from bson import ObjectId
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from panda_server.models.delete_workflow_request import DeleteWorkflowRequest
from panda_server.models.base_api_response import BaseAPIResponse

logger = logging.getLogger(__name__)

# 定义 collection 名称
COLLECTION_NAME = "workflow"


async def workflow_delete_logic(
    request: DeleteWorkflowRequest, user_id: str
) -> BaseAPIResponse:
    """
    批量删除工作流

    Args:
        request: 删除工作流请求，包含要删除的工作流ID列表
        user_id: 用户ID

    Returns:
        BaseAPIResponse: 删除操作的结果

    Raises:
        HTTPException: 当权限不足或操作失败时
        ValueError: 当请求数据格式错误时
    """
    workflow_collection = mongodb.get_collection(COLLECTION_NAME)

    # 验证请求数据
    if not request.workflow_id_list:
        raise ValueError("工作流ID列表不能为空")

    # 验证所有ID格式
    try:
        workflow_object_ids = [ObjectId(wf_id) for wf_id in request.workflow_id_list]
    except Exception as e:
        raise ValueError(f"工作流ID格式无效: {str(e)}")

    # 查询要删除的工作流，验证权限
    workflows_to_delete = await workflow_collection.find(
        {"_id": {"$in": workflow_object_ids}}
    ).to_list(length=None)

    # 检查是否所有工作流都存在
    found_ids = {str(wf["_id"]) for wf in workflows_to_delete}
    requested_ids = set(request.workflow_id_list)
    missing_ids = requested_ids - found_ids

    if missing_ids:
        logger.warning(f"用户 {user_id} 尝试删除不存在的工作流: {missing_ids}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"以下工作流不存在: {', '.join(missing_ids)}",
        )

    # 检查权限：只能删除自己拥有的工作流
    unauthorized_workflows = [
        str(wf["_id"]) for wf in workflows_to_delete if wf.get("owner") != user_id
    ]

    if unauthorized_workflows:
        logger.warning(
            f"用户 {user_id} 尝试删除无权限的工作流: {unauthorized_workflows}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"您无权删除以下工作流: {', '.join(unauthorized_workflows)}",
        )

    # 执行批量删除
    delete_result = await workflow_collection.delete_many(
        {"_id": {"$in": workflow_object_ids}, "owner": user_id}
    )

    # 检查删除结果
    if delete_result.deleted_count != len(request.workflow_id_list):
        logger.error(
            f"删除工作流时发生意外：期望删除 {len(request.workflow_id_list)} 个，"
            f"实际删除 {delete_result.deleted_count} 个"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除操作未完全成功，请重试",
        )

    logger.info(f"用户 {user_id} 成功删除了 {delete_result.deleted_count} 个工作流")

    return BaseAPIResponse(data={"deleted_count": delete_result.deleted_count}) 