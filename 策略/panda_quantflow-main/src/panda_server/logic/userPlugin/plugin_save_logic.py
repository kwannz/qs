import logging
import time
from typing import Dict, Any
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from panda_server.models.userPlugin.user_plugin_model import UserPluginModel, UserPluginCreateModel, UserPluginUpdateModel
from panda_server.models.userPlugin.save_user_plugin_request import SaveUserPluginRequest
from panda_server.models.userPlugin.save_user_plugin_response import SaveUserPluginResponse, SaveUserPluginResponseData
from panda_server.utils.userPlugin.user_plugin_validator import PluginValidator

logger = logging.getLogger(__name__)

# 定义 collection 名称
USER_PLUGIN_COLLECTION = "user_plugin"


async def user_plugin_save_logic(
    request: SaveUserPluginRequest,
    user_id: str
) -> SaveUserPluginResponse:
    """
    保存用户自定义插件的业务逻辑
    
    Args:
        request: 保存用户插件的请求数据
        user_id: 用户ID
        
    Returns:
        SaveUserPluginResponse: 保存结果响应
        
    Raises:
        HTTPException: 验证失败或保存失败时抛出
    """
    # 1. 验证代码
    if not request.code or not request.code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="插件代码不能为空"
        )
    
    # 进行代码验证
    validation_result = PluginValidator.validate_plugin_code(request.code)
    
    # 检查验证结果
    if not validation_result.get('success', False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_result.get('error_message', '插件代码验证失败')
        )
    
    # 2. 获取数据库集合
    collection = mongodb.get_collection(USER_PLUGIN_COLLECTION)
    
    current_time = int(time.time() * 1000)
    
    # 3. 根据是否有plugin_id决定新建或更新
    if request.plugin_id is None:
        # 新建插件逻辑
        return await create_user_plugin(collection, request, user_id, current_time)
    else:
        # 更新插件逻辑
        return await update_user_plugin(collection, request, user_id, current_time)


async def create_user_plugin(
    collection,
    request: SaveUserPluginRequest,
    user_id: str,
    current_time: int
) -> SaveUserPluginResponse:
    """
    创建新的用户插件
    
    Args:
        collection: 数据库集合
        request: 保存用户插件的请求数据
        user_id: 用户ID
        current_time: 当前时间戳
        
    Returns:
        SaveUserPluginResponse: 创建结果响应
        
    Raises:
        HTTPException: 插件名称重复时抛出
    """
    # 检查插件名称是否重复
    existing_plugin = await collection.find_one({
        "name": request.name,
        "creator": user_id
    })
    
    if existing_plugin:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"插件名称 '{request.name}' 已存在，请使用其他名称"
        )
    
    # 创建新插件
    plugin_data = UserPluginCreateModel(
        name=request.name,
        creator=user_id,
        code=request.code,
        create_at=current_time,
        update_at=current_time
    ).model_dump()
    
    result = await collection.insert_one(plugin_data)
    plugin_id = str(result.inserted_id)
    
    return SaveUserPluginResponse(
        code=201,
        message="用户插件创建成功",
        data=SaveUserPluginResponseData(plugin_id=plugin_id)
    )


async def update_user_plugin(
    collection,
    request: SaveUserPluginRequest,
    user_id: str,
    current_time: int
) -> SaveUserPluginResponse:
    """
    更新现有的用户插件
    
    Args:
        collection: 数据库集合
        request: 保存用户插件的请求数据
        user_id: 用户ID
        current_time: 当前时间戳
        
    Returns:
        SaveUserPluginResponse: 更新结果响应
        
    Raises:
        HTTPException: 插件不存在或权限不足时抛出
    """
    from bson import ObjectId
    
    # 查找要更新的插件
    existing_plugin = await collection.find_one({
        "_id": ObjectId(request.plugin_id)
    })
    
    if not existing_plugin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"插件ID '{request.plugin_id}' 不存在"
        )
    
    # 检查权限：只有创建者才能更新
    if existing_plugin.get("creator") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="您没有权限修改此插件"
        )
    
    # 如果要修改插件名称，需要检查新名称是否重复
    if existing_plugin.get("name") != request.name:
        name_conflict = await collection.find_one({
            "name": request.name,
            "creator": user_id,
            "_id": {"$ne": ObjectId(request.plugin_id)}
        })
        
        if name_conflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"插件名称 '{request.name}' 已存在，请使用其他名称"
            )
    
    # 更新插件
    update_data = UserPluginUpdateModel(
        name=request.name,
        code=request.code,
        update_at=current_time
    ).model_dump(exclude_unset=True)
    
    await collection.update_one(
        {"_id": ObjectId(request.plugin_id)},
        {"$set": update_data}
    )
    
    return SaveUserPluginResponse(
        code=200,
        message="用户插件更新成功",
        data=SaveUserPluginResponseData(plugin_id=request.plugin_id)
    ) 