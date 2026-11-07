"""Application settings, read from environment variables (prefix ``PD_``) or a ``.env`` file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PD_", env_file=".env", extra="ignore")

    app_name: str = "pd-scoring-service"
    environment: Literal["local", "dev", "stage", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    model_path: Path = SERVICE_ROOT / "artifacts" / "model_card.json"
    max_batch_size: int = Field(default=1000, ge=1, le=10_000)
    reason_codes_top_n: int = Field(default=3, ge=0, le=7)

    # If set, every /v1/* request must send it in the X-API-Key header.
    api_key: SecretStr | None = None
    docs_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
