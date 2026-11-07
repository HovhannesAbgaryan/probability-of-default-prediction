"""FastAPI dependencies: service lookup and optional API-key auth."""
from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.core.config import Settings
from app.services.pd_service import PDService

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_pd_service(request: Request) -> PDService:
    service = getattr(request.app.state, "pd_service", None)
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return service


def require_api_key(settings: Annotated[Settings, Depends(get_settings_dep)],
                    api_key: Annotated[str | None, Depends(_api_key_header)]) -> None:
    expected = settings.api_key
    if expected is None:
        return
    if not api_key or not secrets.compare_digest(api_key.encode(), expected.get_secret_value().encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key",
                            headers={"WWW-Authenticate": "ApiKey"})


ServiceDep = Annotated[PDService, Depends(get_pd_service)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
