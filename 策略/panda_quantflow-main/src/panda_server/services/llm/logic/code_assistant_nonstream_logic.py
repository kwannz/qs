import json
import logging
from typing import Tuple, Optional
from panda_server.models.general_code_assistant_response import (
    GeneralCodeAssistantResponse, 
    GeneralCodeAssistantData, 
    GeneralCodeAssistantMessage
)
from panda_server.services.llm.agents.code_assistant import CodeAssistant
from panda_server.services.llm.models.code_assistant_request import CodeAssistantRequest

logger = logging.getLogger(__name__)


async def code_assistant_nonstream_logic(
    uid: str,
    request_data: CodeAssistantRequest,
) -> GeneralCodeAssistantResponse:
    """
    处理代码助手逻辑（非流式）
    
    Args:
        uid: 用户ID
        request_data: 代码助手请求数据
        
    Returns:
        GeneralCodeAssistantResponse: 代码助手响应
    """
    try:
        # 实例化代码助手
        assistant = CodeAssistant(
            old_code=request_data.original_code or "",
            user_message=request_data.message,
            model_name=request_data.model
        )
        
        # 处理消息
        session_id, response = await assistant.process_message(
            user_id=uid,
            user_message=request_data.message,
            session_id=request_data.session_id,
            old_code=request_data.original_code,
            last_run_log=request_data.last_run_log,
        )
        
        # 解析JSON响应
        try:
            response_json = json.loads(response)
            new_code = response_json.get("code", "")
            explanation = response_json.get("explanation", "")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse assistant response as JSON: {e}")
            raise ValueError("AI响应格式错误，无法解析JSON")
        
        # 构建响应消息
        assistant_message = GeneralCodeAssistantMessage(
            new_code=new_code,
            explanation=explanation
        )
        
        # 构建响应数据
        response_data = GeneralCodeAssistantData(
            session_id=session_id,
            message=assistant_message
        )
        
        # 返回完整响应
        return GeneralCodeAssistantResponse.success(
            data=response_data,
            message="代码助手处理成功"
        )
        
    except ValueError:
        # 重新抛出值错误，保持原始错误信息
        raise
    except Exception as e:
        logger.error(f"代码助手处理时发生错误: {str(e)}")
        raise Exception(f"代码助手处理失败: {str(e)}") 