import logging
import math
from typing import Any
from panda_server.enums.feature_tag import FeatureTag
from panda_server.enums.workflow_run_status import WorkflowStatus
from panda_server.models.workflow_model import WorkflowModel
from fastapi import HTTPException
from panda_server.models.base_api_response import BaseAPIResponse
from panda_server.models.workflow_run_model import WorkflowRunModel
from panda_server.utils.db_storage import get_from_gridfs, get_gridfs_metadata
from pydantic import BaseModel
from panda_server.config.database import mongodb
from bson import ObjectId

# 获取 logger
logger = logging.getLogger(__name__)

# 业务逻辑异常情况
class WorkflowRunningException(Exception):
    code = -1001
    message = "Workflow is running, please wait for it to finish"
class WorkflowRunFailedException(Exception):
    code = -1002
    message = "Workflow run failed, please fix and run it again"
class WorkflowNoLastRunException(Exception):
    code = -1003
    message = "Workflow has not been run yet"
    
async def workflow_run_output_by_last_run_logic(
    workflow_id: str,
    feature_tag: FeatureTag,
    locator: str,
    user_id: str,
    page: int | None = None,
    limit: int | None = None,
) -> BaseAPIResponse:
    # 获取 workflow 信息
    workflow_collection = mongodb.get_collection("workflow")
    query_result = await workflow_collection.find_one({"_id": ObjectId(workflow_id)})
    if not query_result:
        raise HTTPException(status_code=404, detail="Workflow not found")
    workflow = WorkflowModel(**query_result)
    # 校验用户权限
    if workflow.owner not in [user_id, "*"]:
        raise HTTPException(status_code=403, detail="Permission denied")
    # 检查 feature tag 是否合法
    if feature_tag not in workflow.feature_tag.keys():
        raise HTTPException(status_code=400, detail=f"Feature tag {feature_tag} not found in this workflow")
    # 检查 workflow 是否运行过
    if not workflow.last_run_id:
        raise WorkflowNoLastRunException
    # 获取 workflow 最后一次运行
    workflow_run_collection = mongodb.get_collection("workflow_run")
    query_result = await workflow_run_collection.find_one({"_id": ObjectId(workflow.last_run_id)})
    if not query_result:
        logger.error(f"workflow {workflow_id} has no last run record, but last_run_id is {workflow.last_run_id}")
        raise HTTPException(status_code=500, detail="Internal server data error")
    workflow_run = WorkflowRunModel(**query_result)
    # 检查最后一次运行结果是否可以提取数据
    if workflow_run.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
        raise WorkflowRunningException
    elif workflow_run.status in [WorkflowStatus.FAILED, WorkflowStatus.MANUAL_STOP]:
        raise WorkflowRunFailedException
    # 获取 feature tag 的输出
    feature_tag_detail = workflow.feature_tag[feature_tag]
    response_data = []
    for detail in feature_tag_detail:
        output_obj_id = workflow_run.output_data_obj.get(detail.node_id)
        # 获取文件内容
        object, _ = await get_from_gridfs("workflow_node_output_fs", output_obj_id)
        if not object:
            raise HTTPException(status_code=500, detail="Internal server data error")
        # 如果是 pydantic 模型对象, 转换为 dict
        if isinstance(object, BaseModel):
            object = object.model_dump()
        # 根据 locator 检索对象
        if locator:
            try:
                object = get_by_locator(object, locator)
            except (KeyError, IndexError, TypeError):
                raise HTTPException(status_code=400, detail="Invalid locator")
            except Exception as e:
                logger.error(f"Error getting object by locator: {e}")
                raise HTTPException(status_code=500, detail="Internal server data error")
        # 分页
        if page and limit:
            # 只有 list 类型可以分页
            if not isinstance(object, list):
                raise HTTPException(
                    status_code=400, detail="Invalid object type for pagination"
                )
            # 判断查询是否超过最大页数
            max_page = math.ceil(len(object) / limit)
            if page > max_page:
                raise HTTPException(
                    status_code=400, detail=f"Page out of range, max page: {max_page}"
                )
            object = object[(page - 1) * limit : min(page * limit, len(object))]
            
            # 添加分页相关信息
            object= {
                "pagination": {
                    "max_page": max_page,
                    "has_more": page < max_page,
                },
                "data": object,
            }
        # 将结果添加到 response_data 中
        response_data.append({
            "node_id": detail.node_id,
            "node_title": detail.node_title,
            "node_output": object,
        })

    return BaseAPIResponse(data=response_data)


def get_by_locator(object: dict, locator: str) -> Any:
    """
    根据 locator 检索对象
    locator 格式: 'a.b.0' 表示检索 a.b[0] 字段
    """
    keys = locator.split(".")
    for key in keys:
        if key.isdigit():
            key = int(key)
        object = object[key]
    return object
