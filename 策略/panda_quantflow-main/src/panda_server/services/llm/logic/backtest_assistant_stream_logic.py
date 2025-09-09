import logging
from typing import Dict, Any, AsyncGenerator
from panda_server.services.llm.agents.backtest_assistant import BacktestAssistant
from panda_server.services.llm.models.backtest_assistant_request import BacktestAssistantRequest

logger = logging.getLogger(__name__)

async def backtest_assistant_stream_logic(
    uid: str,
    request_data: BacktestAssistantRequest,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    流式处理回测助手逻辑
    
    Args:
        uid: 用户ID
        request_data: 回测助手请求数据
        
    Yields:
        Dict[str, Any]: 流式响应数据
    """
    try:
        # 实例化回测助手
        assistant = BacktestAssistant(model_name=request_data.model)
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
        logger.error(f"流式回测助手处理时发生错误: {str(e)}")
        yield {
            "error": True,
            "message": f"回测助手处理失败: {str(e)}"
        } 