"""Uniform error envelope: {"error": {"code", "message", "details"}, "request_id"}."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException  # base class: also catches router 404/405

from app.core.logging import request_id_ctx

log = logging.getLogger(__name__)


class ServiceError(Exception):
    """Domain error raised by the service layer, mapped to an HTTP status here."""

    def __init__(self, code: str, message: str, status_code: int = status.HTTP_400_BAD_REQUEST,
                 details: Any = None):
        super().__init__(message)
        self.code, self.message, self.status_code, self.details = code, message, status_code, details


def _body(code: str, message: str, details: Any = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details}, "request_id": request_id_ctx.get()}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(_body(exc.code, exc.message, exc.details), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(jsonable_encoder(_body("validation_error", "Request validation failed", details)),
                            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        code = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed",
                413: "payload_too_large", 503: "service_unavailable"}.get(exc.status_code, "http_error")
        return JSONResponse(_body(code, str(exc.detail)), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(_body("internal_error", "Internal server error"),
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
