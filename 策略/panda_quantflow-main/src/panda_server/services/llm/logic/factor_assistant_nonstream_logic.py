import logging
from typing import Dict, Any
from panda_server.models.general_code_assistant_response import (
    GeneralCodeAssistantResponse, 
    GeneralCodeAssistantData, 
    GeneralCodeAssistantMessage
)
from panda_server.services.llm.agents.factor_assistant import FactorAssistant
from panda_server.services.llm.models.factor_assistant_request import FactorAssistantRequest
import json

logger = logging.getLogger(__name__)

async def factor_assistant_nonstream_logic(
    uid: str,
    request_data: FactorAssistantRequest,
) -> GeneralCodeAssistantResponse:
    """
    处理因子助手逻辑
    
    Args:
        uid: 用户ID
        request_data: 因子助手请求数据
        
    Returns:
        GeneralCodeAssistantResponse: 因子助手响应
    """
    try:
        # 实例化因子助手
        assistant = FactorAssistant(model_name=request_data.model)
        
        # 处理消息
        session_id, response = await assistant.process_message_nonstream(
            user_id=uid,
            user_message=request_data.message,
            session_id=request_data.session_id,
            original_user_code=request_data.original_code,
            last_run_log=request_data.last_run_log,
        )
        
        # 解析JSON响应
        try:
            response_json = json.loads(response)
            new_code = response_json.get("code", "")
            explanation = response_json.get("explanation", "")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse factor assistant response as JSON: {e}")
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
            message="因子助手处理成功"
        )
        
    except ValueError:
        # 重新抛出值错误，保持原始错误信息
        raise
    except Exception as e:
        logger.error(f"因子助手处理时发生错误: {str(e)}")
        raise Exception(f"因子助手处理失败: {str(e)}") 