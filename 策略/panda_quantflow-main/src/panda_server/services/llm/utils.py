import json
import logging
from bson import ObjectId
from typing import Optional, AsyncGenerator, Tuple
from panda_server.config.database import mongodb
from panda_server.services.llm.enums.role_type import RoleType
from panda_server.services.llm.models.chat_session_model import (
    ChatSessionModel,
    ChatSessionCreateModel,
    ChatSessionUpdateModel,
)
from panda_server.services.llm.models.message_model import Message
from panda_server.services.llm.base.llm_service import LLMService

logger = logging.getLogger(__name__)

# 定义 collection 名称
CHAT_SESSION_COLLECTION_NAME = "chat_sessions"


class LLMServiceUtils:

    def __init__(self, model_name: str, system_prompt: str | None = None):
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.llm = LLMService(self.system_prompt, model_name)

    async def process_message(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
        from_role: RoleType = RoleType.USER,
        stream: bool = False,
        json_mode: bool = False,
    ) -> Tuple[str, str] | Tuple[str, AsyncGenerator[str, None]]:
        """处理用户消息并返回 AI 响应

        Args:
            user_id: 用户 ID
            user_message: 用户消息
            session_id: 会话 ID, 如果为 None, 则创建一个新的会话
            from_role: 消息来源角色, 默认是 USER
            stream: 是否流式返回
            json_mode: 是否返回 JSON 模式

        Returns:
            Tuple[str, str]: 如果非流式模式, 第一个 str 是 session_id, 第二个 str 是 AI 响应
            Tuple[str, AsyncGenerator[str, None]]: 如果流式模式, 第一个 str 是 session_id, 第二个 AsyncGenerator 是异步生成器
        """

        # 准备会话 session
        session = await self._session_setup(
            user_id, user_message, session_id, from_role=from_role
        )

        # 非流式模式, 返回字符串
        if not stream:
            response_msg = await self.llm.chat_completion(
                session.messages, json_mode=json_mode
            )
            await self._session_teardown(session, response_msg)
            return str(session.id), response_msg

        # 流式模式，返回异步生成器
        async def stream_generator():
            full_content = ""
            full_reasoning = ""
            async for chunk in self.llm.chat_completion_stream(
                session.messages, json_mode=json_mode
            ):
                
                # 处理新的chunk格式
                reasoning_content = chunk.choices[0].delta.reasoning_content if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content else ""
                
                # 收集reasoning内容
                if reasoning_content:
                    full_reasoning += reasoning_content
                
                content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                full_content += content
                yield chunk  # 返回完整的chunk对象供调用方处理
                
            await self._session_teardown(session, full_content, full_reasoning)

        return str(session.id), stream_generator()

    async def _session_setup(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
        from_role: RoleType = RoleType.USER,
    ) -> ChatSessionModel:
        """准备会话 session
        返回检索到的已有 session, 或者创建新的 session 返回
        """
        # 判断是否需要创建一个新的 session 对话
        if session_id:
            try:
                # 获取会话时不过滤内部消息，保留完整的消息历史
                session = await self.get_session_detail_full(user_id, session_id)
            except ValueError as e:
                logger.error(f"Failed to get session: {e}")
                raise
        else:
            session = ChatSessionCreateModel(
                user_id=user_id,
                messages=[],
            )
            session: ChatSessionModel = await self.create_new_message_to_new_session(
                session
            )

        # 添加并存储新消息 - 系统自动判断是否为内部消息
        # 简单规则：只有USER消息可见，其他角色都默认不可见，最后再决定哪条可见
        is_internal = (from_role != RoleType.USER)
        
        user_msg = Message(
            role=from_role, 
            content=user_message,
            is_internal=is_internal
        )
        return await self.create_new_message_to_existing_session(
            session, user_msg, return_updated_session=True
        )

    async def _session_teardown(self, session: ChatSessionModel, content: str, reasoning: str = None) -> None:
        """session 结束, 存储新消息响应到 session 中"""
        # AI响应消息默认不可见，最后再通过代码决定哪条可见
        ai_message = Message(
            role=RoleType.ASSISTANT, 
            content=content,
            is_internal=True  # 默认不可见，最后再决定哪条可见
        )
        
        # 如果有reasoning内容，添加到消息中
        if reasoning:
            ai_message.reasoning = reasoning
            
        await self.create_new_message_to_existing_session(
            session, ai_message, return_updated_session=False
        )


    async def get_session_detail_full(
        self, user_id: str, session_id: str
    ) -> ChatSessionModel:
        """获取指定用户的完整聊天会话，包含所有内部消息"""
        # 验证 session_id 格式
        if not ObjectId.is_valid(session_id):
            raise ValueError(f"无效的会话ID格式: {session_id}")

        # 查询会话详情
        chat_session_collection = mongodb.get_collection(CHAT_SESSION_COLLECTION_NAME)
        session_doc = await chat_session_collection.find_one(
            {"_id": ObjectId(session_id), "user_id": user_id}
        )

        if not session_doc:
            raise ValueError(f"session_id not found: {session_id}")

        # 不过滤任何消息，保留完整的消息历史
        # 转换 ObjectId 为字符串并返回模型
        session_doc["_id"] = str(session_doc["_id"])
        return ChatSessionModel(**session_doc)


    async def create_new_message_to_new_session(
        self, chat_session: ChatSessionCreateModel
    ) -> ChatSessionModel:
        """创建新的session（用于内部调用，输入已验证）"""
        # 直接插入到数据库（输入已在调用方验证）
        session_dict = chat_session.model_dump(by_alias=True, exclude_unset=True)
        collection = mongodb.get_collection(CHAT_SESSION_COLLECTION_NAME)
        result = await collection.insert_one(session_dict)

        # 构造返回的ChatSessionModel对象
        session_dict["_id"] = str(result.inserted_id)
        return ChatSessionModel(**session_dict)

    async def create_new_message_to_existing_session(
        self,
        session: ChatSessionModel,
        message: Message,
        return_updated_session: bool = False,
    ) -> Optional[ChatSessionModel]:
        """统一的消息添加逻辑：向session中添加消息并更新数据库

        Args:
            session: 目标会话
            message: 要添加的消息
            return_updated_session: 是否返回更新后的session对象

        Returns:
            如果 return_updated_session 为 True，返回更新后的 ChatSessionModel，否则返回 None
        """
        # 将消息添加到session的副本中（避免直接修改原对象）
        updated_messages = session.messages + [message]

        # 使用 ChatSessionUpdateModel 进行数据验证
        session_update_data = ChatSessionUpdateModel(messages=updated_messages)

        # 更新数据库
        collection = mongodb.get_collection(CHAT_SESSION_COLLECTION_NAME)
        session_object_id = ObjectId(session.id)

        # 使用验证后的数据进行数据库更新
        update_dict = session_update_data.model_dump(by_alias=True, exclude_unset=True)

        await collection.update_one({"_id": session_object_id}, {"$set": update_dict})

        if return_updated_session:
            # 查询并返回更新后的session
            updated_doc = await collection.find_one({"_id": session_object_id})
            updated_doc["_id"] = str(updated_doc["_id"])
            return ChatSessionModel(**updated_doc)
        else:
            # 即使不返回，也要更新原session对象的messages，保持状态一致性
            session.messages = updated_messages

        return None


    # 添加方法用于消息状态管理
    async def mark_message_as_external(self, session_id: str, message_id: str):
        """将指定消息标记为外部消息"""
        if not ObjectId.is_valid(session_id):
            raise ValueError(f"无效的会话ID格式: {session_id}")
        
        collection = mongodb.get_collection(CHAT_SESSION_COLLECTION_NAME)
        result = await collection.update_one(
            {"_id": ObjectId(session_id), "messages.message_id": message_id},
            {"$set": {"messages.$.is_internal": False}}
        )
        
        if result.matched_count == 0:
            logger.warning(f"Message not found: session_id={session_id}, message_id={message_id}")


    async def get_last_assistant_message_id(self, session_id: str) -> Optional[str]:
        """获取会话中最后一条AI助手消息的ID"""
        try:
            collection = mongodb.get_collection(CHAT_SESSION_COLLECTION_NAME)
            session_doc = await collection.find_one(
                {"_id": ObjectId(session_id)},
                {"messages": 1}
            )
            
            if session_doc and session_doc.get("messages"):
                # 倒序查找最后一条assistant消息（性能更好）
                for msg in reversed(session_doc["messages"]):
                    if msg.get("role") == "assistant":
                        return msg.get("message_id")
            return None
        except Exception as e:
            logger.error(f"Error getting last assistant message ID: {e}")
            return None
