import asyncio
import logging
from panda_server.config.env import (
    WORKFLOW_EXCHANGE_NAME,
    WORKFLOW_LOG_ROUTING_KEY,
    WORKFLOW_LOG_QUEUE,
)
from panda_server.messaging.rabbitmq_client import AsyncRabbitMQ
from panda_server.messaging.log_processor import WorkflowLogQueueConsumer

logger = logging.getLogger(__name__)


async def process_message_with_full_data(message_data):
    """统一的消息处理入口，根据type分发到不同的处理器"""
    try:
        message_type = message_data.get("type")
        content = message_data.get("content")
        user_id = message_data.get("user_id")
        
        logger.info(f"📨 Processing log message: type={message_type}, user_id={user_id}")
        
        if message_type == "insert_workflow_log":
            # 处理workflow_log插入消息
            logger.info(f"📨 Received workflow_log message for user {user_id}")
            await WorkflowLogQueueConsumer.process_workflow_log_message(message_data)
        else:
            logger.warning(f"Unknown log message type: {message_type}")
            
    except Exception as e:
        logger.error(f"Error processing log message: {e}")
        raise


class LogConsumer:
    """工作流日志消费者"""
    
    def __init__(self):
        pass

    async def single_worker(self, worker_id: int, client: AsyncRabbitMQ):
        """单个日志处理worker"""
        try:
            logger.info(
                f"workflow log worker {worker_id} started listening to queue: {WORKFLOW_LOG_QUEUE}"
            )
            logger.info(f"workflow log worker {worker_id} config: exchange={WORKFLOW_EXCHANGE_NAME}, routing_key={WORKFLOW_LOG_ROUTING_KEY}")
            await client.consume_full_message(
                queue_name=WORKFLOW_LOG_QUEUE,
                exchange_name=WORKFLOW_EXCHANGE_NAME,
                routing_key=WORKFLOW_LOG_ROUTING_KEY,
                callback=process_message_with_full_data,
            )
        except Exception as e:
            logger.error(f"workflow log worker {worker_id} execution failed: {str(e)}")

    async def start_workers(self, client: AsyncRabbitMQ, worker_count: int):
        """启动多个日志处理worker"""
        logger.info(f"Creating {worker_count} workflow log workers")
        logger.info(f"WorkflowLog queue config: exchange={WORKFLOW_EXCHANGE_NAME}, routing_key={WORKFLOW_LOG_ROUTING_KEY}, queue={WORKFLOW_LOG_QUEUE}")
        for i in range(worker_count):
            asyncio.create_task(self.single_worker(i, client)) 