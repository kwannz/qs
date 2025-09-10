import time
import uuid
import logging
from typing import Callable
from starlette.types import ASGIApp, Receive, Scope, Send
from common.logging.log_context import set_request_id, reset_request_id


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.log = logging.getLogger(__name__)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method")
        path = scope.get("path")
        req_id = None

        async def send_wrapper(message):
            nonlocal req_id
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                # ensure request id
                if not req_id:
                    req_id = uuid.uuid4().hex[:12]
                headers.append((b"x-request-id", req_id.encode()))
            await send(message)

        start = time.perf_counter()
        req_id = uuid.uuid4().hex[:12]
        token = set_request_id(req_id)
        self.log.debug(f"HTTP start id={req_id} {method} {path}")
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            dur = (time.perf_counter() - start) * 1000.0
            self.log.info(f"HTTP end id={req_id} {method} {path} took={dur:.1f}ms")
            reset_request_id(token)
