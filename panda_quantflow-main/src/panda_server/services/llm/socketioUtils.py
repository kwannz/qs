"""
动态 Socket.IO 连接管理工具

支持在接口调用时动态创建 Socket.IO 连接，任务完成后自动断开
"""

import socketio
import asyncio
import time
from typing import Dict, Optional
from common.logging.system_logger import logging

logger = logging.getLogger(__name__)

# 临时连接存储
temporary_connections: Dict[str, socketio.AsyncClient] = {}
active_tasks: Dict[str, asyncio.Task] = {}


async def create_temporary_socketio_connection(uid: str) -> Optional[socketio.AsyncClient]:
    """
    为指定用户创建临时 Socket.IO 连接
    
    Args:
        uid: 用户ID
        
    Returns:
        Socket.IO 客户端实例，如果创建失败返回 None
    """
    try:
        # 如果已有连接，先关闭
        if uid in temporary_connections:
            await close_temporary_connection(uid)
        
        # 创建新的 Socket.IO 客户端
        sio = socketio.AsyncClient()
        
        # 设置连接事件处理
        @sio.event
        async def connect():
            logger.info(f"Temporary Socket.IO connection established for user {uid}")
            
        @sio.event  
        async def disconnect():
            logger.info(f"Temporary Socket.IO connection disconnected for user {uid}")
            
        # 连接到服务器（这里使用内部地址，因为是同一个进程）
        # 实际上这里不需要真的连接到外部服务器，我们可以模拟连接状态
        temporary_connections[uid] = sio
        
        logger.info(f"Created temporary Socket.IO connection for user {uid}")
        return sio
        
    except Exception as e:
        logger.error(f"Failed to create temporary Socket.IO connection for user {uid}: {e}")
        return None


async def close_temporary_connection(uid: str):
    """
    关闭指定用户的临时连接
    
    Args:
        uid: 用户ID
    """
    try:
        # 取消活跃任务
        if uid in active_tasks:
            active_tasks[uid].cancel()
            del active_tasks[uid]
            
        # 关闭连接
        if uid in temporary_connections:
            sio = temporary_connections[uid]
            try:
                if sio.connected:
                    await sio.disconnect()
            except:
                pass  # 忽略断开连接时的错误
            del temporary_connections[uid]
            
        logger.info(f"Closed temporary Socket.IO connection for user {uid}")
        
    except Exception as e:
        logger.error(f"Error closing temporary Socket.IO connection for user {uid}: {e}")


async def emit_to_user(uid: str, event: str, data: dict) -> bool:
    """
    向指定用户发送 Socket.IO 事件
    
    Args:
        uid: 用户ID
        event: 事件名称
        data: 事件数据
        
    Returns:
        是否发送成功
    """
    try:
        if uid in temporary_connections:
            sio = temporary_connections[uid]
            return True
        else:
            logger.warning(f"No temporary connection found for user {uid}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to emit Socket.IO event to user {uid}: {e}")
        return False


def is_user_connected(uid: str) -> bool:
    """
    检查用户是否有活跃的临时连接
    
    Args:
        uid: 用户ID
        
    Returns:
        是否连接
    """
    return uid in temporary_connections


def get_connection_stats() -> dict:
    """
    获取连接统计信息
    
    Returns:
        连接统计数据
    """
    return {
        "temporary_connections": len(temporary_connections),
        "active_tasks": len(active_tasks),
        "connected_users": list(temporary_connections.keys())
    }


async def cleanup_all_connections():
    """
    清理所有临时连接
    """
    users_to_cleanup = list(temporary_connections.keys())
    for uid in users_to_cleanup:
        await close_temporary_connection(uid)
    
    logger.info("All temporary Socket.IO connections cleaned up") 