from datetime import datetime
import logging
from typing import Optional, Any, Dict
from pydantic import Field
from bson import ObjectId
import asyncio
import json
from .user_log_model import UserLog
from panda_server.config.database import mongodb
from panda_server.config.env import (
    RUN_MODE,
    WORKFLOW_EXCHANGE_NAME,
    WORKFLOW_LOG_ROUTING_KEY,
)


class UserLogger:
    """用户日志记录器（统一处理日志记录和存储）"""
    
    # 类级别的共享RabbitMQ实例，避免频繁创建连接
    _shared_rabbitmq = None
    
    def __init__(self, user_id: str, workflow_run_id: Optional[str] = None, work_node_id: Optional[str] = None):
        self.user_id = user_id
        self.workflow_run_id = workflow_run_id
        self.work_node_id = work_node_id
        self.sys_logger = logging.getLogger(__name__)
    
    @classmethod
    async def _get_rabbitmq_instance(cls):
        """获取共享的RabbitMQ实例（仅在CLOUD模式下创建）"""
        # 只有在CLOUD模式下才创建RabbitMQ连接
        if RUN_MODE == "CLOUD" and cls._shared_rabbitmq is None:
            # 延迟导入避免循环依赖
            from panda_server.messaging.rabbitmq_client import AsyncRabbitMQ
            if AsyncRabbitMQ is not None:
                cls._shared_rabbitmq = AsyncRabbitMQ()
        return cls._shared_rabbitmq
    
    async def _get_next_sequence(self, workflow_run_id: str) -> int:
        """获取指定workflow的下一个序列号（使用原子操作确保并发安全）"""
        if mongodb.db is None:
            return 0
        counter_collection = mongodb.db["workflow_sequence_counters"]
        result = await counter_collection.find_one_and_update(
            {"workflow_run_id": workflow_run_id},
            {"$inc": {"sequence": 1}},  # 原子递增，第一次会从0变成1
            upsert=True,
            return_document=True  # 返回更新后的文档
        )
        return result["sequence"]
    
    async def _publish_to_queue(self, user_log: UserLog):
        """将user_log发送到队列"""
        try:
            # 使用共享的RabbitMQ实例，避免频繁创建连接
            rabbit_mq = await self._get_rabbitmq_instance()
            if rabbit_mq is None:
                raise Exception("AsyncRabbitMQ not available")
                
            message = json.dumps({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "insert_workflow_log",
                "user_id": user_log.user_id,
                "content": user_log.model_dump(by_alias=True, mode='json')
            })
            
            self.sys_logger.info(f"将user_log消息加入rabbitMQ队列: {user_log.workflow_run_id}")
            
            await rabbit_mq.publish(
                exchange_name=WORKFLOW_EXCHANGE_NAME,
                routing_key=WORKFLOW_LOG_ROUTING_KEY,
                message=message,
            )
            
            self.sys_logger.debug(f"User log published to queue successfully: {user_log.workflow_run_id}")
            
        except Exception as e:
            # 如果队列发送失败，降级到直接插入数据库
            self.sys_logger.error(f"Failed to publish user_log to queue, fallback to direct insert: {e}")
            try:
                if mongodb.db is not None:
                    collection = mongodb.db["workflow_logs"]
                    await collection.insert_one(user_log.model_dump(by_alias=True))
                    self.sys_logger.debug(f"User log inserted to database directly as fallback")
            except Exception as db_error:
                self.sys_logger.error(f"Failed to insert user_log to database as fallback: {db_error}")
        # 不关闭连接，让共享实例保持连接复用
    
    async def _insert_to_database(self, user_log: UserLog):
        """直接插入数据库"""
        if mongodb.db is None:
            self.sys_logger.warning("MongoDB not connected, cannot insert user log")
            return
        collection = mongodb.db["workflow_logs"]
        return await collection.insert_one(user_log.model_dump(by_alias=True))
    
    async def _log(self, level: str, message: str, workflow_id: Optional[str] = None, work_node_id: Optional[str] = None, error_detail: Optional[str] = None, **kwargs):
        """内部日志记录方法（直接处理所有逻辑）"""
        if mongodb.db is None:
            # 数据库不可用，只记录到系统日志
            self.sys_logger.info(f"[USER_LOG] {level} - {message} (workflow_id: {workflow_id}, work_node_id: {work_node_id}, error_detail: {error_detail}, kwargs: {kwargs})")
            return
        
        # 创建UserLog对象
        user_log = UserLog(
            user_id=self.user_id,
            workflow_run_id=self.workflow_run_id,
            work_node_id=work_node_id if work_node_id is not None else self.work_node_id,
            level=level,
            message=message,
            type="workflow_run",
            workflow_id=workflow_id,
            error_detail=error_detail
        )
        
        # 如果有workflow_run_id且sequence为0，自动生成序列号
        if user_log.workflow_run_id and user_log.sequence == 0:
            user_log.sequence = await self._get_next_sequence(user_log.workflow_run_id)
        
        try:
            # 添加调试日志
            self.sys_logger.info(f"Current RUN_MODE: {RUN_MODE}, workflow_run_id: {user_log.workflow_run_id}")
            
            # 根据运行模式选择处理方式
            if RUN_MODE == "CLOUD":
                # CLOUD模式：通过队列存储到数据库
                self.sys_logger.info(f"CLOUD mode: publishing to queue")
                await self._publish_to_queue(user_log)
            elif RUN_MODE == "LOCAL":
                # LOCAL模式：直接存储到数据库
                self.sys_logger.info(f"LOCAL mode: inserting to database directly")
                await self._insert_to_database(user_log)
            else:
                # 其他模式：直接存数据库（fallback）
                self.sys_logger.info(f"Other mode ({RUN_MODE}): inserting to database directly")
                await self._insert_to_database(user_log)
        except Exception as e:
            # 日志记录失败不应该影响主流程，记录到系统日志
            self.sys_logger.error(f"Failed to insert user log: {e}")
    
    # 简单的async日志API，直接存储到数据库
    async def debug(self, message: str, workflow_id: Optional[str] = None, work_node_id: Optional[str] = None, **kwargs):
        await self._log("DEBUG", message, workflow_id, work_node_id, **kwargs)
    
    async def info(self, message: str, workflow_id: Optional[str] = None, work_node_id: Optional[str] = None, **kwargs):
        await self._log("INFO", message, workflow_id, work_node_id, **kwargs)
    
    async def warning(self, message: str, workflow_id: Optional[str] = None, work_node_id: Optional[str] = None, **kwargs):
        await self._log("WARNING", message, workflow_id, work_node_id, **kwargs)
    
    async def error(self, message: str, workflow_id: Optional[str] = None, work_node_id: Optional[str] = None, **kwargs):
        await self._log("ERROR", message, workflow_id, work_node_id, **kwargs)
    
    async def critical(self, message: str, workflow_id: Optional[str] = None, work_node_id: Optional[str] = None, **kwargs):
        await self._log("CRITICAL", message, workflow_id, work_node_id, **kwargs)
    
    @classmethod
    async def shutdown(cls):
        """关闭共享的RabbitMQ连接（应用退出时调用）"""
        if cls._shared_rabbitmq is not None:
            try:
                await cls._shared_rabbitmq.close()
                cls._shared_rabbitmq = None
                logging.getLogger(__name__).info("Shared RabbitMQ connection closed")
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to close shared RabbitMQ connection: {e}") 