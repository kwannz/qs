import asyncio
import json
import logging
import os
from typing import Any, Callable, Optional
import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractIncomingMessage, AbstractChannel
from panda_server.config.env import RABBITMQ_URL,RABBITMQ_MAX_RETRIES,RABBITMQ_RETRY_INTERVAL,RABBITMQ_PREFETCH_COUNT

logger = logging.getLogger(__name__)


class AsyncRabbitMQ:
    def __init__(
            self,
            url: str = RABBITMQ_URL,
            max_retries: int = RABBITMQ_MAX_RETRIES,
            retry_interval: int = RABBITMQ_RETRY_INTERVAL
    ):
        """
        异步RabbitMQ客户端

        :param url: RabbitMQ连接URL
        :param max_retries: 最大重试次数
        :param retry_interval: 重试间隔(秒)
        """
        self.url = url
        self.max_retries = int(max_retries)
        self.retry_interval = retry_interval
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[AbstractChannel] = None

    async def connect(self) -> None:
        """
        建立连接和通道
        """
        if self.connection and not self.connection.is_closed:
            return

        for attempt in range(self.max_retries):
            try:
                self.connection = await aio_pika.connect_robust(self.url)
                self.channel = await self.connection.channel()
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed to connect to RabbitMQ after {self.max_retries} attempts"
                    )
                    raise
                logger.warning(
                    f"Connection attempt {attempt + 1} failed, retrying in {self.retry_interval} seconds..."
                )
                await asyncio.sleep(self.retry_interval)

    async def test_connect(self):
        """
        测试与RabbitMQ服务器的连接是否正常
        """
        temp_connection = None
        try:
            temp_connection = await aio_pika.connect_robust(self.url)
            async with temp_connection:
                channel = await temp_connection.channel()
                queue = await channel.declare_queue(auto_delete=True)
                await queue.delete(if_unused=False, if_empty=False)
                logger.info("RabbitMQ connection test successful")
        except Exception as e:
            logger.error(f"RabbitMQ connection test failed: {e}")
            raise Exception(f"RabbitMQ connection test failed: {e}")
        finally:
            if temp_connection and not temp_connection.is_closed:
                await temp_connection.close()

    async def close(self) -> None:
        """
        关闭连接
        """
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("RabbitMQ connection closed")

    async def publish(
            self,
            exchange_name: str,
            routing_key: str,
            message: Any,
            exchange_type: ExchangeType = ExchangeType.DIRECT,
            durable: bool = True,
            delivery_mode: DeliveryMode = DeliveryMode.PERSISTENT,
    ) -> None:
        """
        发布消息到指定的exchange

        :param exchange_name: 交换机名称
        :param routing_key: 路由键
        :param message: 消息内容
        :param exchange_type: 交换机类型
        :param durable: 是否持久化
        :param delivery_mode: 传递模式
        """
        if not self.channel or self.channel.is_closed:
            await self.connect()

        try:
            exchange = await self.channel.declare_exchange(
                exchange_name, exchange_type, durable=durable
            )

            if not isinstance(message, bytes):
                message = str(message).encode()

            await exchange.publish(
                Message(body=message, delivery_mode=delivery_mode),
                routing_key=routing_key,
            )
            logger.debug(f"Message published to {exchange_name} with key {routing_key}")
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise

    async def consume(
            self,
            queue_name: str,
            callback: Callable[[AbstractIncomingMessage], Any],
            exchange_name: Optional[str] = None,
            routing_key: Optional[str] = None,
            exchange_type: ExchangeType = ExchangeType.DIRECT,
            durable: bool = True,
            prefetch_count: int = RABBITMQ_PREFETCH_COUNT,
            no_ack: bool = False,
    ) -> None:
        """
        消费指定队列的消息

        :param queue_name: 队列名称
        :param callback: 消息处理回调函数
        :param exchange_name: 交换机名称(可选)
        :param routing_key: 路由键(可选)
        :param exchange_type: 交换机类型
        :param durable: 是否持久化
        :param prefetch_count: 预取消息数量
        :param no_ack: 是否自动确认消息
        """
        if not self.channel or self.channel.is_closed:
            await self.connect()

        try:
            await self.channel.set_qos(prefetch_count=int(prefetch_count))

            if exchange_name:
                exchange = await self.channel.declare_exchange(
                    exchange_name, exchange_type, durable=durable
                )
                queue = await self.channel.declare_queue(queue_name, durable=durable)
                await queue.bind(exchange, routing_key or queue_name)
            else:
                queue = await self.channel.declare_queue(queue_name, durable=durable)

            async with queue.iterator(no_ack=no_ack) as queue_iter:
                async for message in queue_iter:
                    try:
                        # 处理消息前检查是否已被处理
                        if message.processed:
                            continue
                        data = json.loads(message.body.decode())
                        await callback(data["content"])
                        if not no_ack and not message.processed:
                            await message.ack()
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        if not no_ack and not message.processed:
                            await message.nack(requeue=False)
        except Exception as e:
            logger.error(f"Failed to consume messages: {e}")
            raise

    async def consume_full_message(
            self,
            queue_name: str,
            callback: Callable[[dict], Any],
            exchange_name: Optional[str] = None,
            routing_key: Optional[str] = None,
            exchange_type: ExchangeType = ExchangeType.DIRECT,
            durable: bool = True,
            prefetch_count: int = RABBITMQ_PREFETCH_COUNT,
            no_ack: bool = False,
    ) -> None:
        """
        消费指定队列的消息（传递完整消息对象）

        :param queue_name: 队列名称
        :param callback: 消息处理回调函数，接收完整的消息字典
        :param exchange_name: 交换机名称(可选)
        :param routing_key: 路由键(可选)
        :param exchange_type: 交换机类型
        :param durable: 是否持久化
        :param prefetch_count: 预取消息数量
        :param no_ack: 是否自动确认消息
        """
        if not self.channel or self.channel.is_closed:
            await self.connect()

        try:
            await self.channel.set_qos(prefetch_count=int(prefetch_count))

            if exchange_name:
                exchange = await self.channel.declare_exchange(
                    exchange_name, exchange_type, durable=durable
                )
                queue = await self.channel.declare_queue(queue_name, durable=durable)
                await queue.bind(exchange, routing_key or queue_name)
            else:
                queue = await self.channel.declare_queue(queue_name, durable=durable)

            async with queue.iterator(no_ack=no_ack) as queue_iter:
                async for message in queue_iter:
                    try:
                        # 处理消息前检查是否已被处理
                        if message.processed:
                            continue
                        data = json.loads(message.body.decode())
                        await callback(data)  # 传递完整的消息对象
                        if not no_ack and not message.processed:
                            await message.ack()
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        if not no_ack and not message.processed:
                            await message.nack(requeue=False)
        except Exception as e:
            logger.error(f"Failed to consume messages: {e}")
            raise

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close() 