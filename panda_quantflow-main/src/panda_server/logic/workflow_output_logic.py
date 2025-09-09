import logging
import math
from typing import Any
from fastapi import HTTPException
from panda_server.models.base_api_response import BaseAPIResponse
from panda_server.utils.db_storage import get_from_gridfs, get_gridfs_metadata
from pydantic import BaseModel

# 获取 logger
logger = logging.getLogger(__name__)


async def workflow_output_get_logic(
    output_obj_id: str,
    locator: str,
    user_id: str,
    page: int | None = None,
    limit: int | None = None,
) -> BaseAPIResponse:
    # 获取文件 metadata
    metadata = await get_gridfs_metadata("workflow_node_output_fs", output_obj_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Output object not found")
    if metadata == {}:
        raise HTTPException(status_code=500, detail="Internal server data error")
    # 校验 uid 是否有权限
    if metadata.get("owner") not in [user_id, "*"]:
        raise HTTPException(status_code=403, detail="Permission denied")
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

    return BaseAPIResponse(data=object)


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
