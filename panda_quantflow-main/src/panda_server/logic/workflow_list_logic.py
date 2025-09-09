import logging
from datetime import datetime
from fastapi import HTTPException
from panda_server.config.database import mongodb
from panda_server.config.env import RUN_MODE
from panda_server.models.query_workflows_response import (
    QueryWorkflowsResponse,
    QueryWorkflowsResponseData,
)
from panda_server.models.workflow_model import WorkflowModel
from panda_server.enums.feature_tag import FeatureTag
from bson import ObjectId
from typing import List, Optional

logger = logging.getLogger(__name__)

# 定义 collection 名称
COLLECTION_NAME = "workflow"


async def workflow_list_logic(
    user_id: str, limit: int, page: int, filter: Optional[List[str]]
) -> QueryWorkflowsResponse:
    """
    查询指定用户的所有 workflow (支持分页和feature_tag筛选)

    Args:
        user_id: 用户ID
        limit: 每页返回的数量
        page: 第几页，从1开始
        filter: 按 filter 筛选工作流, 可传入多个值，当前支持: backtest, signal, factor, trade

    Returns:
        QueryWorkflowsResponse: 包含工作流列表和总数的响应
    """
    workflow_collection = mongodb.get_collection(COLLECTION_NAME)
    
    # 对参数filter进行校验
    filter = filter or []
    valid_filters = [tag for tag in filter if tag in [t.value for t in FeatureTag]]
    if filter and not valid_filters:
        raise HTTPException(status_code=400, detail="Invalid filter value")

    return await get_user_workflows(workflow_collection, user_id, limit, page, filter=valid_filters)
    
async def get_user_workflows(
    workflow_collection, 
    user_id: str, 
    limit: int, 
    page: int, 
    filter: Optional[List[str]]
) -> QueryWorkflowsResponse:
    """获取用户的工作流列表，支持filter过滤"""
    # 构建查询条件
    query = {"owner": {"$in": [user_id, "*"]}}
    
    # 如果需要按feature_tag过滤
    if filter:
        # 使用$and查询多个feature_tag
        query["$and"] = [{"feature_tag." + tag: {"$exists": True}} for tag in filter]
    
    # 查询总数
    total_count = await workflow_collection.count_documents(query)

    # 计算跳过的数量
    skip = (page - 1) * limit

    # 添加排序（按更新时间倒序）和分页
    cursor = (
        workflow_collection.find(query)
        .sort("update_at", -1)
        .skip(skip)
        .limit(limit)
    )

    # 将查询结果转换为 WorkflowModel 列表
    workflows = []
    async for doc in cursor:
        workflow = WorkflowModel.model_validate(doc)
        workflows.append(workflow)

    # 构造响应数据
    response_data = QueryWorkflowsResponseData(
        workflows=workflows, total_count=total_count
    )

    return QueryWorkflowsResponse(data=response_data) 