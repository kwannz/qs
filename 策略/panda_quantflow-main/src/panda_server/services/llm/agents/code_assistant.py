import logging
from typing import Tuple, AsyncGenerator
from panda_server.services.llm.utils import LLMServiceUtils
from panda_server.services.llm.agents.prompts_provider import PromptsProvider as pp
from panda_server.services.llm.enums.role_type import RoleType
import json


logger = logging.getLogger(__name__)


class CodeAssistant(LLMServiceUtils):
    def __init__(self, old_code: str, user_message: str, model_name: str):
        self.old_code = old_code
        self.user_message = user_message
        system_prompt = pp.join(
            pp.role_and_context_code_assistant,
            pp.response_format,
            pp.basic_code_requirements,
        )
        super().__init__(model_name, system_prompt)

    async def process_message(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
        old_code: str | None = None,
        last_run_log: str | None = None,
    ) -> Tuple[str, str]:
        user_message = f"""
        # Old Code
        {old_code}
        # Last Run Log
        {last_run_log}
        # User Message
        {user_message}
        """
        response = await super().process_message(
            user_id, user_message, session_id, stream=False, json_mode=True
        )
        return response

    async def process_message_stream(
        self,
        user_id: str,
        user_message: str,
        session_id: str | None = None,
        old_code: str | None = None,
        last_run_log: str | None = None,
    ) -> AsyncGenerator[tuple, None]:
        """
        流式处理消息：
        1. 对 reasoning 做流式输出
        2. 最后一次性返回 code 和 explanation
        """
        user_message_formatted = f"""
        # Old Code
        {old_code}
        # Last Run Log
        {last_run_log}
        # User Message
        {user_message}
        """

        current_session_id = session_id

        yield current_session_id, {
            "status": "代码助手启动中..."
        }

        yield current_session_id, {
            "status": "代码需求分析：" + (user_message or "暂无具体需求") + "\n"
        }

        yield current_session_id, {
            "status": "代码算法设计中...\n"
        }

        current_session_id, response_stream = await super().process_message(
            user_id,
            user_message_formatted,
            current_session_id,
            stream=True,
            json_mode=True,
            from_role=RoleType.USER,
        )

        yield current_session_id, {
            "status": "代码生成推理分析过程：\n"
        }

        complete_response = ""
        reasoning_content = ""
        content_started = False
        
        async for chunk in response_stream:
            # 处理reasoning内容的流式输出
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_chunk = chunk.choices[0].delta.reasoning_content
                
                if reasoning_chunk.strip():
                    reasoning_content += reasoning_chunk
                    yield current_session_id, {
                        "reasoning": reasoning_chunk
                    }
            
            # 处理content内容的流式输出
            if chunk.choices[0].delta.content:
                if not content_started:
                    yield current_session_id, {"status": "正在生成代码...\n"}
                    content_started = True
                yield current_session_id, {"content": chunk.choices[0].delta.content}
                complete_response += chunk.choices[0].delta.content

        # json 解析最终完整响应
        try:
            response_json = json.loads(complete_response)
            final_code = response_json.get("code", "")
            final_explanation = response_json.get("explanation", "")

            yield current_session_id, {
                "status": "代码生成完成，正在整理代码与解释说明...\n"
            }
            
            # 输出最终结果（用于前端显示）
            yield current_session_id, {
                "session_id": current_session_id,
                "code": final_code,
                "explanation": final_explanation
            }

        except Exception as e:
            logger.error(f"Error json parsing response: {e}")
            yield current_session_id, {
                "error": True,
                "message": f"代码生成失败: JSON解析错误"
            }

        # 处理消息可见性
        await self._finalize_message_visibility(current_session_id)

    async def _finalize_message_visibility(self, session_id: str) -> None:
        """
        处理结束后，将最后一条AI消息设为对用户可见
        
        Args:
            session_id: 会话ID
        """
        if not session_id:
            return
            
        try:
            last_assistant_message_id = await self.get_last_assistant_message_id(session_id)
            if last_assistant_message_id:
                await self.mark_message_as_external(session_id, last_assistant_message_id)
                logger.info(f"Code assistant interaction completed, final AI message set to visible: session_id={session_id}, message_id={last_assistant_message_id}")
            else:
                logger.warning(f"AI message not found for visibility setting: session_id={session_id}")
        except Exception as e:
            logger.error(f"Failed to set message visibility: {e}")
