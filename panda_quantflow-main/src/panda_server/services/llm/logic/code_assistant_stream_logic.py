import logging
from typing import Dict, Any, AsyncGenerator
from panda_server.services.llm.agents.code_assistant import CodeAssistant
from panda_server.services.llm.models.code_assistant_request import CodeAssistantRequest

logger = logging.getLogger(__name__)

async def code_assistant_stream_logic(
    uid: str,
    request_data: CodeAssistantRequest,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    流式处理代码助手逻辑
    
    Args:
        uid: 用户ID
        request_data: 代码助手请求数据
        
    Yields:
        Dict[str, Any]: 流式响应数据
    """
    try:
        # 实例化代码助手
        assistant = CodeAssistant(
            old_code=request_data.original_code or "",
            user_message=request_data.message,
            model_name=request_data.model
        )
        
        async for response_tuple in assistant.process_message_stream(
            user_id=uid,
            user_message=request_data.message,
            session_id=request_data.session_id,
            old_code=request_data.original_code,
            last_run_log=request_data.last_run_log,
        ):
            # 处理新的tuple格式: (session_id, data)
            if response_tuple and len(response_tuple) == 2:
                session_id, data = response_tuple

                # 直接返回字典对象，让路由层负责序列化
                yield data
                        
    except Exception as e:
        logger.error(f"流式代码助手处理时发生错误: {str(e)}")
        yield {
            "error": True,
            "message": f"代码助手处理失败: {str(e)}"
        } 