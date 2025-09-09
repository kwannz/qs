import logging
from typing import Optional
from bson import ObjectId
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from panda_server.services.llm.models.get_session_detail_response import (
    GetSessionDetailResponse,
    GetSessionDetailResponseData,
)
from panda_server.services.llm.models.chat_session_model import ChatSessionModel

logger = logging.getLogger(__name__)

# 定义 collection 名称
CHAT_SESSIONS_COLLECTION = "chat_sessions"


async def get_session_detail_visible_logic(
    user_id: str, session_id: str, message_id: Optional[str] = None, limit: int = 50
) -> GetSessionDetailResponse:
    """
    获取指定用户的单个聊天会话详情

    Args:
        user_id: 用户ID
        session_id: 会话ID
        message_id: 可选的消息ID（UUID7），如果提供则从该消息开始获取（包含该消息）
        limit: 限制返回的消息数量，默认50

    Returns:
        GetSessionDetailResponse: 包含聊天会话详情的响应

    Raises:
        ValueError: 当session_id格式无效或message_id格式无效时
        HTTPException: 当会话不存在或用户无权限访问时
    """
    chat_session_collection = mongodb.get_collection(CHAT_SESSIONS_COLLECTION)

    # 验证 session_id 格式
    if not ObjectId.is_valid(session_id):
        raise ValueError(f"无效的会话ID格式: {session_id}")

    # 构建查询条件
    query = {"_id": ObjectId(session_id), "user_id": user_id}

    # 查询会话详情
    session_doc = await chat_session_collection.find_one(query)

    if not session_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话不存在或您无权限访问该会话: {session_id}",
        )

    # 转换 ObjectId 为字符串
    session_doc["_id"] = str(session_doc["_id"])

    # 过滤掉内部消息，只返回可见消息
    if "messages" in session_doc:
        visible_messages = [
            msg for msg in session_doc["messages"] 
            if not msg.get("is_internal", False)
        ]
        
        # 如果提供了 message_id，找到该消息后的消息
        if message_id:
            # 找到 message_id 在列表中的位置
            start_index = None
            for i, msg in enumerate(visible_messages):
                if msg.get("message_id") == message_id:
                    start_index = i  # 从该消息开始（包含该消息）
                    break
            
            if start_index is None:
                # 如果没找到该消息ID，返回空消息列表
                logger.warning(f"消息ID不存在: {message_id}, 返回空消息列表")
                visible_messages = []
            else:
                # 从该消息开始获取消息（包含该消息），应用 limit 限制
                visible_messages = visible_messages[start_index:start_index + limit]
        else:
            # 如果没有 message_id，直接应用 limit 限制
            visible_messages = visible_messages[:limit]
        
        session_doc["messages"] = visible_messages

    # 使用模型验证数据
    session_model = ChatSessionModel(**session_doc)

    # 构造响应数据
    response_data = GetSessionDetailResponseData(session=session_model)

    return GetSessionDetailResponse(data=response_data) 