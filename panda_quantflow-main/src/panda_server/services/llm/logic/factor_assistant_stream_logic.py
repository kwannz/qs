import logging
from typing import Dict, Any, AsyncGenerator
from panda_server.services.llm.agents.factor_assistant import FactorAssistant
from panda_server.services.llm.models.factor_assistant_request import FactorAssistantRequest

logger = logging.getLogger(__name__)

async def factor_assistant_stream_logic(
    uid: str,
    request_data: FactorAssistantRequest,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    流式处理因子助手逻辑
    
    Args:
        uid: 用户ID
        request_data: 因子助手请求数据
        
    Yields:
        Dict[str, Any]: 流式响应数据
    """
    try:
        # 实例化因子助手
        assistant = FactorAssistant(model_name=request_data.model)
        async for response_tuple in assistant.process_message_stream(
            user_id=uid,
            user_message=request_data.message,
            session_id=request_data.session_id,
            original_user_code=request_data.original_code,
            last_run_log=request_data.last_run_log,
        ):
            # 处理新的tuple格式: (session_id, data)
            if response_tuple and len(response_tuple) == 2:
                session_id, data = response_tuple

                # 直接返回字典对象，让路由层负责序列化
                yield data
                        
    except Exception as e:
        logger.error(f"流式因子助手处理时发生错误: {str(e)}")
        yield {
            "error": True,
            "message": f"因子助手处理失败: {str(e)}"
        } 