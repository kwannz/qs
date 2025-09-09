import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, status
from fastapi.websockets import WebSocket
from pydantic import ValidationError

from freqtrade.enums import RPCMessageType, RPCRequestType
from freqtrade.exceptions import FreqtradeException
from freqtrade.rpc.api_server.api_auth import validate_ws_token
from freqtrade.rpc.api_server.deps import get_api_config, get_message_stream, get_rpc
from freqtrade.rpc.api_server.ws.channel import WebSocketChannel, create_channel
from freqtrade.rpc.api_server.ws.message_stream import MessageStream
from freqtrade.rpc.api_server.ws_schemas import (
    WSAnalyzedDFMessage,
    WSErrorMessage,
    WSMessageSchema,
    WSRequestSchema,
    WSWhitelistMessage,
)
from freqtrade.rpc.rpc import RPC


logger = logging.getLogger(__name__)

# Private router, protected by API Key authentication
router = APIRouter()


async def channel_reader(channel: WebSocketChannel, rpc: RPC, *, on_activity=None, max_msg_per_sec: int = 10, max_msg_bytes: int = 65536):
    """
    Iterate over the messages from the channel and process the request
    """
    # Simple frequency control
    timestamps: list[float] = []
    async for message in channel:
        now = time.time()
        # Update activity
        if on_activity:
            on_activity(now)
        # Frequency limit (messages/sec)
        timestamps = [t for t in timestamps if now - t <= 1.0]
        if len(timestamps) >= max_msg_per_sec:
            logger.warning(f"Rate limit exceeded from {channel}")
            await channel.close()
            break
        timestamps.append(now)

        # Size limit (approximate)
        try:
            import orjson

            size = len(orjson.dumps(message))
        except Exception:
            size = 0
        if max_msg_bytes and size and size > max_msg_bytes:
            logger.warning(f"Message too large from {channel}: {size} bytes")
            await channel.close()
            break
        try:
            await _process_consumer_request(message, channel, rpc)
        except FreqtradeException:
            logger.exception(f"Error processing request from {channel}")
            response = WSErrorMessage(data="Error processing request")

            await channel.send(response.dict(exclude_none=True))


async def channel_broadcaster(channel: WebSocketChannel, message_stream: MessageStream, *, on_activity=None):
    """
    Iterate over messages in the message stream and send them
    """
    async for message, ts in message_stream:
        if channel.subscribed_to(message.get("type")):
            # Log a warning if this channel is behind
            # on the message stream by a lot
            if (time.time() - ts) > 60:
                logger.warning(
                    f"Channel {channel} is behind MessageStream by 1 minute,"
                    " this can cause a memory leak if you see this message"
                    " often, consider reducing pair list size or amount of"
                    " consumers."
                )

            await channel.send(message, use_timeout=True)
            if on_activity:
                on_activity(time.time())


async def _process_consumer_request(request: dict[str, Any], channel: WebSocketChannel, rpc: RPC):
    """
    Validate and handle a request from a websocket consumer
    """
    # Validate the request, makes sure it matches the schema
    try:
        websocket_request = WSRequestSchema.model_validate(request)
    except ValidationError as e:
        logger.error(f"Invalid request from {channel}: {e}")
        return

    type_, data = websocket_request.type, websocket_request.data
    response: WSMessageSchema

    logger.debug(f"Request of type {type_} from {channel}")

    # If we have a request of type SUBSCRIBE, set the topics in this channel
    if type_ == RPCRequestType.SUBSCRIBE:
        # If the request is empty, do nothing
        if not data:
            return

        # If all topics passed are a valid RPCMessageType, set subscriptions on channel
        if all([any(x.value == topic for x in RPCMessageType) for topic in data]):
            channel.set_subscriptions(data)

        # We don't send a response for subscriptions
        return

    elif type_ == RPCRequestType.WHITELIST:
        # Get whitelist
        whitelist = rpc._ws_request_whitelist()

        # Format response
        response = WSWhitelistMessage(data=whitelist)
        await channel.send(response.model_dump(exclude_none=True))

    elif type_ == RPCRequestType.ANALYZED_DF:
        # Limit the amount of candles per dataframe to 'limit' or 1500
        limit = int(min(data.get("limit", 1500), 1500)) if data else None
        pair = data.get("pair", None) if data else None

        # For every pair in the generator, send a separate message
        for message in rpc._ws_request_analyzed_df(limit, pair):
            # Format response
            response = WSAnalyzedDFMessage(data=message)
            await channel.send(response.model_dump(exclude_none=True))


_WS_CONN_COUNT = 0
_WS_CONN_LOCK = asyncio.Lock()


@router.websocket("/message/ws")
async def message_endpoint(
    websocket: WebSocket,
    token: str = Depends(validate_ws_token),
    rpc: RPC = Depends(get_rpc),
    message_stream: MessageStream = Depends(get_message_stream),
    api_config: dict[str, Any] = Depends(get_api_config),
):
    global _WS_CONN_COUNT
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    # Connection cap
    max_conns = int(api_config.get("max_ws_connections", 100))
    async with _WS_CONN_LOCK:
        if _WS_CONN_COUNT >= max_conns:
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
            return
        _WS_CONN_COUNT += 1

    last_activity = {"ts": time.time()}

    def _on_activity(ts: float):
        last_activity["ts"] = ts

    idle_timeout = float(api_config.get("ws_idle_timeout", 120))
    heartbeat_interval = float(api_config.get("ws_heartbeat_interval", 30))
    max_msg_per_sec = int(api_config.get("ws_max_messages_per_sec", 10))
    max_msg_bytes = int(api_config.get("ws_max_message_bytes", 65536))

    async def _idle_monitor():
        try:
            while True:
                await asyncio.sleep(heartbeat_interval)
                if time.time() - last_activity["ts"] > idle_timeout:
                    logger.info(f"Closing idle websocket {websocket.client}")
                    await websocket.close(code=status.WS_1001_GOING_AWAY)
                    break
        except Exception:
            pass

    try:
        async with create_channel(websocket) as channel:
            await channel.run_channel_tasks(
                channel_reader(
                    channel,
                    rpc,
                    on_activity=_on_activity,
                    max_msg_per_sec=max_msg_per_sec,
                    max_msg_bytes=max_msg_bytes,
                ),
                channel_broadcaster(channel, message_stream, on_activity=_on_activity),
                _idle_monitor(),
            )
    finally:
        async with _WS_CONN_LOCK:
            _WS_CONN_COUNT = max(0, _WS_CONN_COUNT - 1)
