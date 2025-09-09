import logging
from typing import Optional
from fastapi import APIRouter, Body, Header, Query, status, HTTPException
from fastapi.responses import StreamingResponse
import traceback
import json
from panda_server.models.general_code_assistant_response import GeneralCodeAssistantResponse
from panda_server.services.llm.models.get_session_list_response import GetSessionListResponse
from panda_server.services.llm.logic.get_session_list_logic import get_session_list_logic
from panda_server.services.llm.logic.get_session_detail_visible_logic import get_session_detail_visible_logic
from panda_server.services.llm.models.get_session_detail_response import GetSessionDetailResponse
from panda_server.services.llm.logic.delete_session_logic import delete_session_logic
from panda_server.services.llm.models.delete_session_response import DeleteSessionResponse
from panda_server.services.llm.logic.code_assistant_nonstream_logic import code_assistant_nonstream_logic
from panda_server.services.llm.logic.code_assistant_stream_logic import code_assistant_stream_logic
from panda_server.services.llm.models.code_assistant_request import CodeAssistantRequest
from panda_server.services.llm.logic.backtest_assistant_nonstream_logic import backtest_assistant_nonstream_logic
from panda_server.services.llm.logic.backtest_assistant_stream_logic import backtest_assistant_stream_logic
from panda_server.services.llm.models.backtest_assistant_request import BacktestAssistantRequest
from panda_server.services.llm.logic.factor_assistant_nonstream_logic import factor_assistant_nonstream_logic
from panda_server.services.llm.logic.factor_assistant_stream_logic import factor_assistant_stream_logic
from panda_server.services.llm.models.factor_assistant_request import FactorAssistantRequest
from panda_server.services.llm.socketioUtils import (
    create_temporary_socketio_connection,
    close_temporary_connection,
    emit_to_user,
    is_user_connected,
    active_tasks
)
import asyncio
import time


# 获取 logger
logger = logging.getLogger(__name__)

# 创建路由实例，设置前缀和标签
router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/code-assistant-nonstream", response_model=GeneralCodeAssistantResponse, status_code=status.HTTP_200_OK)
async def chat_code_assistant_nonstream(
    uid: str = Header(..., alias="uid", description="用户ID"),
    request_data: CodeAssistantRequest = Body(..., description="代码助手请求数据")
) -> GeneralCodeAssistantResponse:
    """
    代码助手非流式聊天接口
    
    通用代码助手，专门用于开发和优化量化交易代码。
    支持根据用户需求生成、修改和改进代码。
    
    Args:
        uid: 从 headers 中获取的用户ID
        request_data: 代码助手请求数据，包含消息、会话ID、原始代码和运行日志等
        
    Returns:
        GeneralCodeAssistantResponse: 包含生成代码和解释的响应
    """
    try:
        return await code_assistant_nonstream_logic(
            uid=uid,
            request_data=request_data,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数错误: {str(e)}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"代码助手处理时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="代码助手处理时发生意外错误",
        )


@router.post("/code-assistant-stream")
async def chat_code_assistant_stream(
    uid: str = Header(..., alias="uid", description="用户ID"),
    request_data: CodeAssistantRequest = Body(..., description="代码助手请求数据")
):
    """
    代码助手流式聊天接口
    
    通用代码助手的流式接口，专门用于开发和优化量化交易代码。
    支持根据用户需求生成、修改和改进代码。
    使用流式响应提供实时的推理过程。
    
    Args:
        uid: 从 headers 中获取的用户ID
        request_data: 代码助手请求数据，包含消息、会话ID、原始代码和运行日志等
        
    Returns:
        StreamingResponse: 包含推理过程的流式响应
    """
    
    async def code_assistant_stream_generator():
        async for message in code_assistant_stream_logic(uid, request_data):
            # message已经是JSON字符串了，直接加上SSE格式的前缀
            yield f"data: {json.dumps(message,ensure_ascii=False)}\n\n"

    SSE_HEADERS = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
    }

    try:                
        return StreamingResponse(
            code_assistant_stream_generator(),
            media_type="text/event-stream",
            headers=SSE_HEADERS
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数错误: {str(e)}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"代码助手流式处理时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="代码助手流式处理时发生意外错误",
        )

@router.post("/backtest-assistant-nonstream", response_model=GeneralCodeAssistantResponse, status_code=status.HTTP_200_OK)
async def chat_backtest_assistant_nonstream(
    uid: str = Header(..., alias="uid", description="用户ID"),
    request_data: BacktestAssistantRequest = Body(..., description="回测助手请求数据")
) -> GeneralCodeAssistantResponse:
    """
    回测助手聊天接口
    
    专业的量化交易回测代码助手，专门用于开发和优化回测策略代码。
    支持根据用户需求生成、修改和改进回测策略，包含多轮代码验证机制。
    
    Args:
        uid: 从 headers 中获取的用户ID
        request_data: 回测助手请求数据，包含消息、会话ID、原始代码和运行日志等
        
    Returns:
        GeneralCodeAssistantResponse: 包含生成代码和解释的响应
    """
    try:
        return await backtest_assistant_nonstream_logic(
            uid=uid,
            request_data=request_data,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数错误: {str(e)}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"回测助手处理时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="回测助手处理时发生意外错误",
        )


@router.post("/backtest-assistant-stream")
async def chat_backtest_assistant_stream(
    uid: str = Header(..., alias="uid", description="用户ID"),
    request_data: BacktestAssistantRequest = Body(..., description="回测助手请求数据")
):
    """
    回测助手流式聊天接口
    
    专业的量化交易回测代码助手的流式接口，专门用于开发和优化回测策略代码。
    支持根据用户需求生成、修改和改进回测策略，包含多轮代码验证机制。
    使用流式响应提供实时的推理过程。
    
    Args:
        uid: 从 headers 中获取的用户ID
        request_data: 回测助手请求数据，包含消息、会话ID、原始代码和运行日志等
        
    Returns:
        StreamingResponse: 包含推理过程的流式响应
    """
    
    async def backtest_assistant_stream_generator():
        async for message in backtest_assistant_stream_logic(uid, request_data):
            # message已经是JSON字符串了，直接加上SSE格式的前缀
            yield f"data: {json.dumps(message,ensure_ascii=False)}\n\n"

    SSE_HEADERS = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
    }

    try:                
        return StreamingResponse(
            backtest_assistant_stream_generator(),
            media_type="text/event-stream",
            headers=SSE_HEADERS
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数错误: {str(e)}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"回测助手流式处理时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="回测助手流式处理时发生意外错误",
        )


@router.post("/factor-assistant-nonstream", response_model=GeneralCodeAssistantResponse, status_code=status.HTTP_200_OK)
async def chat_factor_assistant_nonstream(
    uid: str = Header(..., alias="uid", description="用户ID"),
    request_data: FactorAssistantRequest = Body(..., description="因子助手请求数据")
) -> GeneralCodeAssistantResponse:
    """
    因子助手聊天接口
    
    专业的量化因子开发助手，专门用于开发和优化因子代码。
    支持根据用户需求生成、修改和改进因子，包含多轮代码验证机制。
    
    Args:
        uid: 从 headers 中获取的用户ID
        request_data: 因子助手请求数据，包含消息、会话ID、原始代码和运行日志等
        
    Returns:
        GeneralCodeAssistantResponse: 包含生成代码和解释的响应
    """
    try:
        return await factor_assistant_nonstream_logic(
            uid=uid,
            request_data=request_data,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数错误: {str(e)}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"因子助手处理时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="因子助手处理时发生意外错误",
        )


@router.post("/factor-assistant-stream")
async def chat_factor_assistant_stream(
    uid: str = Header(..., alias="uid", description="用户ID"),
    request_data: FactorAssistantRequest = Body(..., description="因子助手请求数据")
):
    """
    因子助手流式聊天接口
    
    专业的量化因子开发助手的流式接口，专门用于开发和优化因子代码。
    支持根据用户需求生成、修改和改进因子，包含多轮代码验证机制。
    使用流式响应提供实时的推理过程。
    
    Args:
        uid: 从 headers 中获取的用户ID
        request_data: 因子助手请求数据，包含消息、会话ID、原始代码和运行日志等
        
    Returns:
        StreamingResponse: 包含推理过程的流式响应
    """
    
    async def factor_assistant_stream_generator():
        async for message in factor_assistant_stream_logic(uid, request_data):
            # message已经是JSON字符串了，直接加上SSE格式的前缀
            yield f"data: {json.dumps(message,ensure_ascii=False)}\n\n"

    SSE_HEADERS = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
    }

    try:                
        return StreamingResponse(
            factor_assistant_stream_generator(),
            media_type="text/event-stream",
            headers=SSE_HEADERS
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数错误: {str(e)}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"因子助手流式处理时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="因子助手流式处理时发生意外错误",
        )


@router.get(
    "/session/all", response_model=GetSessionListResponse, status_code=status.HTTP_200_OK
)
async def get_session_list(
    uid: str = Header(..., alias="uid", description="用户ID"),
    page: int = Query(1, description="第几页，从1开始", ge=1),
    limit: int = Query(10, description="每页返回的数量", ge=1, le=100),
) -> GetSessionListResponse:
    """
    获取用户的聊天会话列表（支持分页）

    Args:
        uid: 从 headers 中获取的用户ID
        page: 第几页，从1开始，默认1
        limit: 每页返回的数量，默认10，最大100

    Returns:
        GetSessionListResponse: 包含聊天会话列表和总数的响应
    """
    try:
        return await get_session_list_logic(uid, limit, page)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数错误: {str(e)}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"获取聊天会话列表时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取聊天会话列表时发生意外错误",
        )


@router.get(
    "/session/detail", response_model=GetSessionDetailResponse, status_code=status.HTTP_200_OK
)
async def get_session_detail_visible(
    session_id: str = Query(..., description="会话ID"),
    uid: str = Header(..., alias="uid", description="用户ID"),
    message_id: Optional[str] = Query(None, description="消息ID，如果提供则从该消息开始获取（包含该消息）"),
    limit: int = Query(50, description="限制返回的消息数量", ge=1, le=500),
) -> GetSessionDetailResponse:
    """
    获取指定会话的详细信息

    Args:
        session_id: 会话ID（查询参数）
        uid: 从 headers 中获取的用户ID
        message_id: 可选的消息ID（UUID7），如果提供则从该消息开始获取（包含该消息）
        limit: 限制返回的消息数量，默认50，最大500

    Returns:
        GetSessionDetailResponse: 包含会话详情的响应
    """
    try:
        return await get_session_detail_visible_logic(uid, session_id, message_id, limit)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数错误: {str(e)}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"获取会话详情时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取会话详情时发生意外错误",
        )


@router.delete(
    "/session/delete", response_model=DeleteSessionResponse, status_code=status.HTTP_200_OK
)
async def delete_session(
    session_id: str = Query(..., description="会话ID"),
    uid: str = Header(..., alias="uid", description="用户ID"),
) -> DeleteSessionResponse:
    """
    删除指定会话（物理删除）

    Args:
        session_id: 会话ID（查询参数）
        uid: 从 headers 中获取的用户ID

    Returns:
        DeleteSessionResponse: 包含删除结果的响应
    """
    try:
        return await delete_session_logic(uid, session_id)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数错误: {str(e)}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"删除会话时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除会话时发生意外错误",
        ) 
    




# ================================= Backup ==================================

# @router.post("/backtest-assistant-stream-socketio")
# async def chat_backtest_assistant_stream_socketio(
#     uid: str = Header(..., alias="uid", description="用户ID"),
#     request_data: BacktestAssistantRequest = Body(..., description="回测助手请求数据")
# ):
#     """
#     回测助手 Socket.IO 流式接口
    
#     与 SSE 版本功能完全相同，使用 Socket.IO 传输解决断连问题。
#     直接返回流式数据，业务逻辑与原接口100%相同。
    
#     Args:
#         uid: 用户ID
#         request_data: 回测助手请求数据
        
#     Returns:
#         StreamingResponse: 通过 Socket.IO 协议的流式数据
#     """
#     try:
#         if not request_data.message:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="消息内容不能为空",
#             )
        
#         # 创建一个唯一的流 ID
#         stream_id = f"{uid}_{int(time.time() * 1000)}"
        
#         async def socketio_stream_generator():
#             """动态创建 Socket.IO 连接的流式生成器"""
#             connection_created = False
#             task_registered = False
            
#             try:
#                 # 1. 动态创建 Socket.IO 连接
#                 sio_client = await create_temporary_socketio_connection(uid)
#                 if sio_client:
#                     connection_created = True
                    
#                     # 通过 Socket.IO 发送连接事件
#                     await emit_to_user(uid, 'connection_established', {
#                         'stream_id': stream_id, 
#                         'uid': uid,
#                         'timestamp': time.time()
#                     })
#                 else:
#                     # 连接创建失败，但仍然可以通过 SSE 提供服务
#                     yield f"data: {json.dumps({'type': 'connection_fallback', 'stream_id': stream_id, 'uid': uid, 'transport': 'sse_only'}, ensure_ascii=False)}\n\n"
                
#                 # 2. 注册当前任务（用于外部取消）
#                 current_task = asyncio.current_task()
#                 if current_task:
#                     active_tasks[uid] = current_task
#                     task_registered = True
                                
#                 if connection_created:
#                     await emit_to_user(uid, 'stream_started', {
#                         'stream_id': stream_id,
#                         'message': 'Backtest stream started',
#                         'timestamp': time.time()
#                     })
                
#                 chunk_count = 0
                
#                 # 3. 直接调用现有的流式逻辑，100% 复用！
#                 async for response_data in backtest_assistant_stream_logic(uid, request_data):
#                     # 检查连接状态（断线检测）
#                     if connection_created and not is_user_connected(uid):
#                         logger.info(f"Socket.IO connection lost for user {uid}, stopping stream")
#                         break
                    
#                     # 通过 SSE 格式返回数据（保持兼容性）
#                     yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                    
#                     # 同时通过 Socket.IO 发送数据
#                     if connection_created:
#                         await emit_to_user(uid, 'stream_data', response_data)
                    
#                     chunk_count += 1
#                     await asyncio.sleep(0.001)
                
#                 # 4. 发送完成信号
#                 # completion_data = {
#                 #     'type': 'stream_completed', 
#                 #     'stream_id': stream_id, 
#                 #     'chunks_sent': chunk_count, 
#                 #     'message': 'Stream completed successfully'
#                 # }
#                 # yield f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"
                
#                 if connection_created:
#                     await emit_to_user(uid, 'stream_completed', {
#                         'stream_id': stream_id,
#                         'chunks_sent': chunk_count,
#                         'message': 'Stream completed successfully',
#                         'timestamp': time.time()
#                     })
                
#                 logger.info(f"Backtest stream completed for user {uid}, sent {chunk_count} chunks")
                
#             except asyncio.CancelledError:
#                 # 用户主动取消或连接断开
#                 cancel_data = {'type': 'stream_cancelled', 'stream_id': stream_id, 'message': 'Stream was cancelled'}
#                 yield f"data: {json.dumps(cancel_data, ensure_ascii=False)}\n\n"
                
#                 if connection_created:
#                     await emit_to_user(uid, 'stream_cancelled', {
#                         'stream_id': stream_id,
#                         'message': 'Stream was cancelled',
#                         'timestamp': time.time()
#                     })
                
#                 logger.info(f"Backtest stream cancelled for user {uid}")
                
#             except Exception as e:
#                 # 处理过程中的错误
#                 error_data = {
#                     'type': 'stream_error', 
#                     'stream_id': stream_id, 
#                     'code': 'PROCESSING_ERROR', 
#                     'message': str(e)
#                 }
#                 yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                
#                 if connection_created:
#                     await emit_to_user(uid, 'stream_error', {
#                         'stream_id': stream_id,
#                         'code': 'PROCESSING_ERROR',
#                         'message': str(e),
#                         'timestamp': time.time()
#                     })
                
#                 logger.error(f"Backtest stream error for user {uid}: {e}")
                
#             finally:
#                 # 5. 自动清理和断开连接
#                 try:
#                     if task_registered and uid in active_tasks:
#                         del active_tasks[uid]
                    
#                     if connection_created:

#                         await emit_to_user(uid, 'connection_closed', {
#                             'stream_id': stream_id,
#                             'message': 'Connection automatically closed',
#                             'timestamp': time.time()
#                         })
                        
#                         # 关闭临时连接
#                         await close_temporary_connection(uid)
#                         logger.info(f"Socket.IO connection for user {uid} automatically closed")
                        
#                 except Exception as cleanup_error:
#                     logger.error(f"Error during cleanup for user {uid}: {cleanup_error}")
        
#         # 返回流式响应
#         return StreamingResponse(
#             socketio_stream_generator(),
#             media_type="text/event-stream",
#             headers={
#                 "Cache-Control": "no-cache",
#                 "Connection": "keep-alive",
#                 "Access-Control-Allow-Origin": "*",
#                 "Access-Control-Allow-Headers": "*",
#                 "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
#             }
#         )
        
#     except ValueError as e:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"请求参数错误: {str(e)}",
#         )
#     except Exception as e:
#         stack_trace = traceback.format_exc()
#         logger.error(f"Socket.IO 回测助手流式处理时发生意外错误: {e}\n{stack_trace}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Socket.IO 回测助手流式处理时发生意外错误",
#         )

# @router.post("/backtest-assistant-stream-json")
# async def chat_backtest_assistant_stream_json(
#     uid: str = Header(..., alias="uid", description="用户ID"),
#     request_data: BacktestAssistantRequest = Body(..., description="回测助手请求数据")
# ):
#     """
#     回测助手纯JSON流式接口
    
#     使用纯HTTP JSON流式传输，无需SSE或Socket.IO连接。
#     每行输出一个JSON对象，直接输出业务数据。
    
#     Args:
#         uid: 用户ID
#         request_data: 回测助手请求数据
        
#     Returns:
#         StreamingResponse: 纯JSON流式响应
#     """
#     try:
#         if not request_data.message:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="消息内容不能为空",
#             )
        
#         # 生成唯一任务ID
#         task_id = f"backtest_{uid}_{int(time.time() * 1000)}"
        
#         async def json_stream():
#             """纯业务数据JSON流式：每行一个JSON对象"""
#             chunk_count = 0
            
#             try:
#                 logger.info(f"Starting JSON stream for user {uid}, task {task_id}")
                
#                 # 发送开始信号
#                 yield json.dumps({"_meta": "stream_started", "task_id": task_id}, ensure_ascii=False) + "\n"
                
#                 # 执行核心业务逻辑
#                 async for response_data in backtest_assistant_stream_logic(uid, request_data):
#                     # 直接输出业务数据
#                     yield json.dumps(response_data, ensure_ascii=False) + "\n"
#                     chunk_count += 1
                    
#                     # 定期输出进度信息
#                     if chunk_count % 10 == 0:
#                         yield json.dumps({"_meta": f"progress_{chunk_count}", "task_id": task_id}, ensure_ascii=False) + "\n"
                    
#                     await asyncio.sleep(0.001)
                
#                 # 发送完成信号
#                 yield json.dumps({"_meta": "stream_completed", "total": chunk_count, "task_id": task_id}, ensure_ascii=False) + "\n"
                
#                 logger.info(f"JSON stream task {task_id} completed for user {uid}, {chunk_count} chunks sent")
                
#             except asyncio.CancelledError:
#                 yield json.dumps({"_error": "stream_cancelled", "task_id": task_id}, ensure_ascii=False) + "\n"
#                 logger.info(f"JSON stream task {task_id} cancelled for user {uid}")
                
#             except Exception as e:
#                 yield json.dumps({"_error": str(e), "task_id": task_id}, ensure_ascii=False) + "\n"
#                 logger.error(f"JSON stream task {task_id} failed for user {uid}: {e}")
        
#         return StreamingResponse(
#             json_stream(),
#             media_type="application/x-ndjson",
#             headers={
#                 "Cache-Control": "no-cache",
#                 "Connection": "keep-alive", 
#                 "Transfer-Encoding": "chunked",
#                 "Access-Control-Allow-Origin": "*",
#                 "Access-Control-Allow-Headers": "*",
#                 "X-Task-ID": task_id,
#                 "X-Stream-Type": "json-newline-delimited"
#             }
#         )
        
#     except ValueError as e:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"请求参数错误: {str(e)}",
#         )
#     except Exception as e:
#         stack_trace = traceback.format_exc()
#         logger.error(f"纯Socket.IO回测助手启动失败: {e}\n{stack_trace}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="回测助手启动失败",
#         )
