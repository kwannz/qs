import logging
from typing import Dict, Any
from panda_server.config.database import mongodb
from common.logging.user_log_model import UserLog


class WorkflowLogQueueConsumer:
    """工作流日志队列消费者（CLOUD模式专用）"""
    
    @staticmethod
    async def process_workflow_log_message(message_data: Dict[str, Any]):
        """处理workflow_log队列消息"""
        try:
            message_type = message_data.get("type")
            content = message_data.get("content")
            user_id = message_data.get("user_id")
            
            logger = logging.getLogger(__name__)
            logger.info(f"Processing workflow_log message: type={message_type}, user_id={user_id}")
            logger.debug(f"Message content: {content}")
            
            if message_type == "insert_workflow_log":
                await WorkflowLogQueueConsumer._handle_insert_workflow_log(content)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Error processing workflow_log message: {e}")
            raise

    @staticmethod
    async def _handle_insert_workflow_log(log_data: Dict[str, Any]):
        """处理插入workflow_log的消息"""
        try:
            # 重新创建UserLog对象
            user_log = UserLog(**log_data)
            
            # 获取数据库连接
            if mongodb.db is None:
                logging.getLogger(__name__).error("MongoDB not connected, cannot insert workflow_log")
                return
                
            # 如果有workflow_run_id且sequence为0，自动生成序列号
            if user_log.workflow_run_id and user_log.sequence == 0:
                # 获取下一个序列号
                counter_collection = mongodb.db["workflow_sequence_counters"]
                result = await counter_collection.find_one_and_update(
                    {"workflow_run_id": user_log.workflow_run_id},
                    {"$inc": {"sequence": 1}},
                    upsert=True,
                    return_document=True
                )
                user_log.sequence = result["sequence"]
            
            # 直接插入数据库（绕过队列）
            collection = mongodb.db["workflow_logs"]
            result = await collection.insert_one(user_log.model_dump(by_alias=True, exclude_unset=True))
            logging.getLogger(__name__).info(f"User log inserted with id: {result.inserted_id}, workflow_run_id: {user_log.workflow_run_id}")
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to insert workflow_log from queue: {e}")
            raise 