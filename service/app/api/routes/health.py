"""Kubernetes-style probes and Prometheus metrics."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.schemas.prediction import Health

router = APIRouter(tags=["ops"])


@router.get("/health/live", response_model=Health, summary="Liveness: process is up")
def live() -> Health:
    return Health(status="ok")


@router.get("/health/ready", response_model=Health, summary="Readiness: model loaded, ready for traffic",
            responses={503: {"model": Health}})
def ready(request: Request):
    service = getattr(request.app.state, "pd_service", None)
    if service is None:
        return JSONResponse(Health(status="unavailable").model_dump(), status_code=503)
    return Health(status="ok", model_version=service.model_version)


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
