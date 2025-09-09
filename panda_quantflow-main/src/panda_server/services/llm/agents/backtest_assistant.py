import json
import logging
from typing import Tuple, AsyncGenerator
from panda_server.services.llm.agents.prompts_provider import PromptsProvider as pp
from panda_server.services.llm.enums.role_type import RoleType
from panda_server.services.llm.code_checker.backtest_code_checker import (
    BacktestCodeChecker,
)
from panda_server.services.llm.utils import LLMServiceUtils
from panda_backtest.api.api import *
from panda_backtest.api.stock_api import *

logger = logging.getLogger(__name__)

class BacktestAssistant(LLMServiceUtils):
    def __init__(self, model_name: str):
        system_prompt = pp.join(
            pp.role_and_context_backtest_assistant,
            pp.response_format,
            pp.backtest_code_requirements,
            pp.get_backtest_engine_doc(),
        )
        super().__init__(model_name, system_prompt)


    async def process_message_nonstream(self, user_id: str, user_message: str, session_id: str | None = None,
                                      original_user_code: str | None = None, last_run_log: str | None = None,
                                      max_interaction_rounds: int = 10) -> Tuple[str, str]:

        formatted_message = await self.format_user_message(user_message, original_user_code, last_run_log)

        for round_count in range(max_interaction_rounds):
            if round_count >= max_interaction_rounds:
                logger.warning(f"Max interaction rounds reached: {max_interaction_rounds}, session_id: {session_id}")
                break
            
            session_id, response = await super().process_message(
                user_id,
                formatted_message if round_count == 0 else developer_message,
                session_id,
                stream=False,
                json_mode=True,
                from_role=RoleType.USER if round_count == 0 else RoleType.SYSTEM,
            )

            code, explanation, developer_message, should_break = await self.process_response_and_check_code(
                response, session_id, round_count
            )
            
            if should_break:
                break

        await self.finalize_message_visibility(session_id)
        return session_id, response

    async def process_message_stream(self, user_id: str, user_message: str, session_id: str | None = None,
                                   original_user_code: str | None = None, last_run_log: str | None = None,
                                   max_interaction_rounds: int = 10) -> AsyncGenerator[tuple, None]:
        """
        流式处理消息：
        1. 只对 reasoning 做流式输出
        2. 进行代码检查
        3. 最后一次性返回 code 和 explanation
        """
        formatted_message = await self.format_user_message(user_message, original_user_code, last_run_log)
        current_session_id = session_id

        yield current_session_id, {"status": "量化回测策略引擎启动中..."}
        yield current_session_id, {"status": "回测策略需求分析：" + (user_message or "暂无具体需求") + "\n"}

        final_code = None
        final_explanation = None

        for round_count in range(max_interaction_rounds):
            if round_count > 0:
                yield current_session_id, {"status": "回测策略代码修改中...\n"}

            if round_count >= max_interaction_rounds:
                logger.warning(f"Max interaction rounds reached: {max_interaction_rounds}, session_id: {current_session_id}")
                break
            
            current_session_id, response_stream = await super().process_message(
                user_id,
                formatted_message if round_count == 0 else developer_message,
                current_session_id,
                stream=True,
                json_mode=True,
                from_role=RoleType.USER if round_count == 0 else RoleType.SYSTEM,
            )

            complete_response = ""
            yield current_session_id, {"status": "思考中...\n"}

            # 这是流式版本独有的处理部分
            content_started = False
            async for chunk in response_stream:
                if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                    if chunk.choices[0].delta.reasoning_content.strip():
                        yield current_session_id, {"reasoning": chunk.choices[0].delta.reasoning_content}
                
                if chunk.choices[0].delta.content:
                    if not content_started:
                        yield current_session_id, {"status": "正在编写代码...\n"}
                        content_started = True
                    yield current_session_id, {"content": chunk.choices[0].delta.content}
                    complete_response += chunk.choices[0].delta.content
            
            # 使用共同的响应处理逻辑
            code, explanation, developer_message, should_break = await self.process_response_and_check_code(
                complete_response, current_session_id, round_count
            )
            
            if explanation:
                final_explanation = explanation

            # 流式版本的状态输出和结果处理
            if code:
                yield current_session_id, {"status": "回测策略代码静态分析与语法验证中...\n"}
                
                if should_break:
                    final_code = code
                    break
                elif developer_message:  # 有代码检查错误，继续优化
                    yield current_session_id, {"status": "检测到回测策略代码存在问题，执行自动修复与优化...\n"}
            elif should_break:
                break

        if current_session_id:
            await self.finalize_message_visibility(current_session_id)
        
        yield current_session_id, {"status": "回测策略构建完成，正在整理代码与解释说明...\n"}
        
        if final_code or final_explanation:
            yield current_session_id, {
                "session_id": current_session_id,
                "code": final_code,
                "explanation": final_explanation
            }

    async def format_user_message(self, user_message: str, original_user_code: str | None, last_run_log: str | None) -> str:
        """格式化用户消息"""
        return f"""
        # User Message
        {user_message}
        
        # Original User Code
        {original_user_code}
        
        # Last Run Log
        {last_run_log}
        """

    async def process_response_and_check_code(self, response: str, session_id: str, round_count: int):
        """处理AI响应和代码检查的通用逻辑"""
        # JSON解析
        try:
            response_json = json.loads(response)
            code = response_json.get("code", "")
            explanation = response_json.get("explanation", "")
        except Exception as e:
            logger.error(f"Error json parsing response: {e}")
            return None, None, pp.join(pp.json_parsed_error_response, pp.response_format), False

        # 检查是否忽略代码检查警告
        if response_json.get("ignore_code_checker_warning", False):
            logger.info(f"AI ignored code checker warning, session_id: {session_id}\n round: {round_count+1}")
            return code, explanation, None, True  # 结束循环

        # 检查代码
        if code:
            code_checker = BacktestCodeChecker(code)
            code_checker_response = code_checker.complete_check()

            if not code_checker_response:  # 没有问题
                return code, explanation, None, True  # 结束循环
            
            logger.debug(f"Code checker worked:\n session_id: {session_id}\n round: {round_count+1}\n code_checker_response: {code_checker_response}")
            developer_message = pp.join(
                pp.generate_code_checker_response(code_checker_response),
                pp.response_format_allow_ignore_code_checker,
            )
            return code, explanation, developer_message, False  # 继续循环

        return code, explanation, None, True  # 结束循环
    

    async def finalize_message_visibility(self, session_id: str) -> None:
        """
        多轮交互结束后，将最后一条AI消息设为对用户可见
        
        Args:
            session_id: 会话ID
        """
        if not session_id:
            return
            
        try:
            last_assistant_message_id = await self.get_last_assistant_message_id(session_id)
            if last_assistant_message_id:
                await self.mark_message_as_external(session_id, last_assistant_message_id)
                logger.info(f"Multi-turn interaction completed, final AI message set to visible: session_id={session_id}, message_id={last_assistant_message_id}")
            else:
                logger.warning(f"AI message not found for visibility setting: session_id={session_id}")
        except Exception as e:
            logger.error(f"Failed to set message visibility: {e}")