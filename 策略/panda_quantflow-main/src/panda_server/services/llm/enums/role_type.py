from enum import Enum


class RoleType(str, Enum):
    """
    Message Role Type Enum
    消息角色类型枚举，用于定义聊天会话中消息的角色类型
    ref: https://api-docs.deepseek.com/api/create-chat-completion
    """

    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"
