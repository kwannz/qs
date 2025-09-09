from pydantic import BaseModel, Field, model_validator
import time
from uuid6 import uuid7
from typing import Optional
from ..enums.role_type import RoleType


class Message(BaseModel):
    """
    Message Model
    表示聊天会话中的一条消息, 包含: 角色, 内容, 时间戳等.
    用于LLM服务中管理对话中的单条消息.
    """

    role: RoleType = Field(
        description="Message role: user, system, assistant, or developer"
    )
    content: str = Field(
        description="Message content text"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning content for assistant messages - internal thought process"
    )
    timestamp: int = Field(
        description="The unix timestamp of the message creation time, unit: milliseconds"
    )
    is_internal: bool = Field(
        default=False, 
        description="Whether this message is internal (hidden from user) - system managed"
    )
    message_id: str = Field(
        default_factory=lambda: str(uuid7()), 
        description="Unique message identifier for internal tracking (UUID7 - timestamp based)"
    )

    # Set default timestamp and other fields
    @model_validator(mode="before")
    @classmethod
    def set_defaults(cls, data):
        now = int(time.time() * 1000)
        if "timestamp" not in data:
            data["timestamp"] = now
        if "message_id" not in data:
            data["message_id"] = str(uuid7())
        if "is_internal" not in data:
            data["is_internal"] = False
        return data

    class Config:
        populate_by_name = True
        from_attributes = True
        json_schema_extra = {"description": "Message Model"}


class MessageCreateModel(Message):
    """
    Message Create Model
    用于新增一条消息到聊天会话中，验证用户传入的消息字段是否完整
    继承自 Message 类，确保 role、content、timestamp 字段都存在
    注意：timestamp、is_internal、message_id 由系统自动添加，用户不应手动设置
    """

    class Config:
        extra = "forbid"  # 禁止额外字段
        json_schema_extra = {
            "description": "Message Create Model - for adding new messages to chat sessions",
            "example": {
                "role": "user",
                "content": "Hello, how are you?"
                # timestamp、is_internal、message_id 将由系统自动添加
            }
        }

