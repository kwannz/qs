from bson import ObjectId
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, List, Optional
import time
from .message_model import Message


class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, *_):
        from pydantic_core import core_schema

        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(ObjectId),
                    core_schema.chain_schema(
                        [
                            core_schema.str_schema(),
                            core_schema.no_info_plain_validator_function(cls.validate),
                        ]
                    ),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x),
                return_schema=core_schema.str_schema(),
                when_used="json",
            ),
        )

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


class ChatSessionBaseModel(BaseModel):
    """
    Chat Session Model
    表示用户的一个聊天会话, 包含: 用户ID, 消息列表, 创建和更新时间等.
    用于LLM服务中管理用户的对话历史记录.
    """

    # Metadata
    user_id: str = Field(
        description="User ID who owns this chat session"
    )
    messages: List[Message] = Field(
        default_factory=list, 
        description="List of messages in the chat session"
    )
    create_at: int = Field(
        description="The unix timestamp of the chat session creation time, unit: milliseconds"
    )
    update_at: int = Field(
        description="The unix timestamp of the chat session update time, unit: milliseconds"
    )

    # Set values for create_at and update_at
    @model_validator(mode="before")
    @classmethod
    def set_defaults(cls, data):
        now = int(time.time() * 1000)
        if "create_at" not in data:
            data["create_at"] = now
        if "update_at" not in data:
            data["update_at"] = now
        return data

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "Chat Session Model"}


class ChatSessionModel(ChatSessionBaseModel):
    id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="Chat session unique identifier, consistent with MongoDB's ObjectId",
        ),
    ]


class ChatSessionCreateModel(ChatSessionModel):
    """
    Chat Session Create Model
    用于创建新的聊天会话，包含自动生成的ID字段
    继承ChatSessionModel，包含所有字段并自动生成ObjectId
    """
    @model_validator(mode="before")
    @classmethod
    def set_create_timestamps(cls, data):
        now = int(time.time() * 1000)
        data["create_at"] = now
        data["update_at"] = now
        return data


class ChatSessionUpdateModel(ChatSessionBaseModel):
    user_id: Optional[str] = None
    messages: Optional[List[Message]] = None
    update_at: Optional[int] = None
    create_at: Optional[int] = Field(default=None, exclude=True)  # 在更新时完全排除这个字段

    @model_validator(mode="before")
    @classmethod
    def set_update_timestamp(cls, data):
        data["update_at"] = int(time.time() * 1000)
        return data