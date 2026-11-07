"""Pure-ASGI middleware: request id propagation, access logging, latency metrics."""
from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import request_id_ctx
from app.core.metrics import HTTP_LATENCY, HTTP_REQUESTS

log = logging.getLogger("app.access")
_VALID_ID = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            route = scope.get("route")
            route_path = getattr(route, "path", "unmatched")  # template, not raw path: bounded cardinality
            method = scope["method"]
            HTTP_REQUESTS.labels(method, route_path, str(status_code)).inc()
            HTTP_LATENCY.labels(method, route_path).observe(elapsed)
            if route_path not in ("/health/live", "/health/ready", "/metrics"):
                log.info("request", extra={"method": method, "path": scope["path"], "status": status_code,
                                           "duration_ms": round(elapsed * 1000, 2)})
            request_id_ctx.reset(token)
