"""Application factory. `uvicorn app.main:app` for serving; `create_app(settings)` for tests."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app import __version__
from app.api.deps import require_api_key
from app.api.routes import health, predict
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.ml.artifact import load_artifact
from app.ml.model import LogisticPDModel
from app.services.pd_service import PDService

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast: a missing or invalid artifact must stop the pod from becoming ready.
        artifact = load_artifact(settings.model_path)
        app.state.pd_service = PDService(LogisticPDModel(artifact), settings.reason_codes_top_n)
        log.info("model loaded", extra={"model_version": artifact.model_version, "sha256": artifact.sha256,
                                        "environment": settings.environment})
        yield
        app.state.pd_service = None
        log.info("shutdown complete")

    app = FastAPI(
        title="PD Scoring Service",
        version=__version__,
        description="Probability-of-default scoring for retail loan applications.",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    app.state.pd_service = None
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(predict.router, dependencies=[Depends(require_api_key)])
    return app


app = create_app()
