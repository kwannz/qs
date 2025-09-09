from typing import AsyncGenerator, List
from panda_server.services.llm.enums.llm_model_type import LLMModelType, MODEL_CONFIG
from panda_server.services.llm.models.message_model import Message
import openai
import traceback
import logging
import asyncio

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, system_prompt: str, model_name: str):
        self.system_prompt = system_prompt
        
        # 获取模型配置
        model_config = MODEL_CONFIG[model_name]
        
        # 使用异步客户端
        self.client = openai.AsyncOpenAI(
            api_key=model_config["api_key_name"],
            base_url=model_config["base_url"]
        )
        self.model = model_config["model"]

        # 定义系统提示词
        self.system_message = {"role": "system", "content": self.system_prompt}

    def _prepare_messages(self, messages: List[Message]):
        """转换消息格式以适配 OpenAI API 格式 (json)"""
        formatted_messages = []

        # 添加历史消息
        for msg in messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})

        # 添加(或覆盖)系统提示词
        if messages[0].role == "system":
            formatted_messages[0] = self.system_message
        else:
            formatted_messages.insert(0, self.system_message)
        
        return formatted_messages

    async def chat_completion(self, messages: List[Message], json_mode: bool = False) -> str:
        """发送聊天请求到 LLM API"""
        try:
            # 格式化消息
            formatted_messages = self._prepare_messages(messages)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=0,
                stream=False,
                response_format={"type": "json_object"} if json_mode else None,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            traceback.print_exc()
            logger.error(f"调用 LLM API 失败: {str(e)}")
            raise

    async def chat_completion_stream(self, messages, json_mode: bool = False) -> AsyncGenerator[str, None]:
        """发送流式聊天请求到 LLM API"""
        try:
            # 格式化消息
            formatted_messages = self._prepare_messages(messages)

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=0,
                response_format={"type": "json_object"} if json_mode else None,
                stream=True,
            )

            # 使用异步迭代
            async for chunk in stream:
                # 返回 chunk 本身，让调用方决定如何处理
                yield chunk
        except Exception as e:
            traceback.print_exc()
            logger.error(f"调用 LLM API 流式请求失败: {str(e)}")
            raise
