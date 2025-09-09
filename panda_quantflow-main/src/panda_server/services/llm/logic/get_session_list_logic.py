import logging
from panda_server.config.database import mongodb
from panda_server.services.llm.models.get_session_list_response import (
    GetSessionListResponse,
    GetSessionListResponseData,
)
from panda_server.services.llm.models.chat_session_model import ChatSessionModel

logger = logging.getLogger(__name__)

# 定义 collection 名称
COLLECTION_NAME = "chat_sessions"

async def get_session_list_logic(
    user_id: str, limit: int, page: int
) -> GetSessionListResponse:
    """
    查询指定用户的所有 chat session (支持分页)

    Args:
        user_id: 用户ID
        limit: 每页返回的数量
        page: 第几页，从1开始

    Returns:
        GetSessionListResponse: 包含聊天会话列表和总数的响应
    """
    chat_session_collection = mongodb.get_collection(COLLECTION_NAME)

    # 构建查询条件
    query = {"user_id": user_id}

    # 查询总数
    total_count = await chat_session_collection.count_documents(query)

    # 计算跳过的数量
    skip = (page - 1) * limit

    # 简单查询，在应用层处理
    cursor = chat_session_collection.find(query).sort("update_at", -1).skip(skip).limit(limit)

    # 处理查询结果并在应用层过滤内部消息
    sessions = []
    async for doc in cursor:
        # 转换 ObjectId 为字符串
        doc["_id"] = str(doc["_id"])
        
        # 直接倒序查找最后一条用户消息（用户消息永远可见）
        last_user_message = None
        for msg in reversed(doc.get("messages", [])):
            if msg.get("role") == "user":
                last_user_message = msg
                break
        doc["messages"] = [last_user_message] if last_user_message else []
        
        # 直接使用ChatSessionModel
        session = ChatSessionModel(**doc)
        sessions.append(session)

    # 构造响应数据
    response_data = GetSessionListResponseData(
        sessions=sessions, total_count=total_count
    )

    return GetSessionListResponse(data=response_data)
