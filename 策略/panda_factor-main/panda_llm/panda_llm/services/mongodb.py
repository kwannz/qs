from typing import Optional, List, Dict, Any
from panda_common.handlers.database_handler import DatabaseHandler
from panda_common.config import config
from panda_common.logger_config import logger
from panda_llm.models.chat import ChatSession
from bson import ObjectId


class MongoDBService:
    def __init__(self):
        self.db_handler = DatabaseHandler(config)
        self.collection = self.db_handler.get_mongo_collection("panda","chat_sessions")
        self.logger = logger

    async def create_chat_session(self, session: ChatSession) -> str:
        """创建新的聊天会话"""
        try:
            # 将业务 id 作为文档 _id，避免 ObjectId/字符串不一致
            doc: Dict[str, Any] = session.dict()
            doc.setdefault("_id", session.id)
            doc.setdefault("id", session.id)
            self.collection.insert_one(doc)
            return session.id
        except Exception as e:
            self.logger.error(f"创建会话失败: {str(e)}")
            raise

    async def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """获取聊天会话"""
        try:
            # 优先按字符串 _id 查询（新写入策略）
            session = self.collection.find_one({"_id": session_id})
            if not session:
                # 兼容历史数据：尝试按 ObjectId 查询
                try:
                    session = self.collection.find_one({"_id": ObjectId(session_id)})
                except Exception:
                    session = None

            if session:
                # 兼容模型：填充 id 字段并移除 _id
                session_doc = dict(session)
                session_doc.setdefault("id", str(session_doc.get("_id", "")))
                session_doc.pop("_id", None)
                return ChatSession(**session_doc)
            return None
        except Exception as e:
            self.logger.error(f"获取会话失败: {str(e)}")
            raise

    async def update_chat_session(self, session_id: str, session: ChatSession):
        """更新聊天会话"""
        try:
            data = session.dict()
            data.setdefault("id", session_id)

            # 优先字符串 _id 更新
            result = self.collection.update_one({"_id": session_id}, {"$set": data})
            if result.matched_count == 0:
                # 兼容历史：尝试 ObjectId
                try:
                    self.collection.update_one({"_id": ObjectId(session_id)}, {"$set": data})
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(f"更新会话失败: {str(e)}")
            raise

    async def delete_chat_session(self, session_id: str):
        """删除聊天会话"""
        try:
            # 优先按字符串 _id 删除
            result = self.collection.delete_one({"_id": session_id})
            if result.deleted_count == 0:
                try:
                    self.collection.delete_one({"_id": ObjectId(session_id)})
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(f"删除会话失败: {str(e)}")
            raise

    async def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        """获取用户的所有会话"""
        try:
            sessions = list(self.collection.find({"user_id": user_id}))
            result: List[ChatSession] = []
            for s in sessions:
                doc = dict(s)
                doc.setdefault("id", str(doc.get("_id", "")))
                doc.pop("_id", None)
                try:
                    result.append(ChatSession(**doc))
                except Exception as e:
                    self.logger.error(f"反序列化会话失败: {e}")
            return result
        except Exception as e:
            self.logger.error(f"获取用户会话失败: {str(e)}")
            raise
