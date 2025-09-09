from fastapi import APIRouter, HTTPException, status, Header
import logging

from panda_server.logic.get_all_plugins_logic import get_all_plugins_logic
from panda_server.logic.userPlugin.plugin_save_logic import user_plugin_save_logic
from panda_server.models.all_plugins_response import AllPluginsResponse
from panda_server.models.userPlugin.save_user_plugin_request import SaveUserPluginRequest
from panda_server.models.userPlugin.save_user_plugin_response import SaveUserPluginResponse
import traceback
import logging

# 获取 logger
logger = logging.getLogger(__name__)

# 创建路由实例，设置前缀和标签
router = APIRouter(prefix="/api/plugins", tags=["plugins"])

@router.get("/all", response_model=AllPluginsResponse)
async def get_all_plugins():
    """
    获取所有可用的 plugins (work-nodes)
    """

    try:
        return await get_all_plugins_logic()
    except HTTPException:
        raise
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"unexpected error in get_all_plugins: {e}\n{stack_trace}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.post("/save", response_model=SaveUserPluginResponse, status_code=status.HTTP_201_CREATED)
async def save_user_plugin(
    request: SaveUserPluginRequest,
    user_id: str = Header(..., alias="uid", description="用户ID")
) -> SaveUserPluginResponse:
    """
    保存用户自定义插件
    
    支持新建和更新两种操作：
    - 如果不传 plugin_id，则为新建插件（需要插件名称不重复）
    - 如果传了 plugin_id，则为更新插件（需要验证插件存在性和权限）
    
    Args:
        request: 保存用户插件的请求数据
        user_id: 从 headers 中获取的用户ID
        
    Returns:
        SaveUserPluginResponse: 保存结果响应
        
    Raises:
        409: 插件名称重复
        403: 没有权限修改插件
        404: 插件不存在
    """
    try:
        
        # 调用保存逻辑
        return await user_plugin_save_logic(request, user_id)
        
    except HTTPException:
        raise
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"保存插件时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="保存插件时发生意外错误"
        )
