import logging
from bson import ObjectId
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from panda_server.services.llm.models.delete_session_response import (
    DeleteSessionResponse,
    DeleteSessionResponseData,
)

logger = logging.getLogger(__name__)

# 定义 collection 名称
CHAT_SESSIONS_COLLECTION = "chat_sessions"


async def delete_session_logic(
    user_id: str, session_id: str
) -> DeleteSessionResponse:
    """
    删除指定用户的单个聊天会话（物理删除）

    Args:
        user_id: 用户ID
        session_id: 会话ID

    Returns:
        DeleteSessionResponse: 包含删除结果的响应

    Raises:
        ValueError: 当session_id格式无效时
        HTTPException: 当会话不存在或用户无权限删除时
    """
    chat_session_collection = mongodb.get_collection(CHAT_SESSIONS_COLLECTION)

    # 验证 session_id 格式
    if not ObjectId.is_valid(session_id):
        raise ValueError(f"无效的会话ID格式: {session_id}")

    # 构建查询条件（确保用户只能删除自己的会话）
    query = {"_id": ObjectId(session_id), "user_id": user_id}

    # 先检查会话是否存在并属于该用户
    existing_session = await chat_session_collection.find_one(query)
    
    if not existing_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话不存在或您无权限删除该会话: {session_id}",
        )

    # 执行物理删除
    delete_result = await chat_session_collection.delete_one(query)

    # 检查删除结果
    if delete_result.deleted_count == 0:
        logger.error(f"删除会话失败，会话ID: {session_id}, 用户ID: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除会话失败",
        )

    logger.info(f"成功删除会话，会话ID: {session_id}, 用户ID: {user_id}")

    # 构造响应数据
    response_data = DeleteSessionResponseData(
        session_id=session_id,
        deleted=True
    )

    return DeleteSessionResponse(data=response_data) 